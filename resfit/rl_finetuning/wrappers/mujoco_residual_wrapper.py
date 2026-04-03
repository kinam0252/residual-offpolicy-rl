"""
Residual environment wrapper for MuJoCo + GR00T.

Combines a GR00T base policy (absolute EEF actions) with an RL residual
agent.  The wrapper:

1. Queries GR00T for a 16-token action chunk (absolute pos + quat + grip).
2. The RL agent outputs a 7-D residual delta (pos3 + euler3 + grip1).
3. Residual is applied on top of the base absolute action:
   - ``combined_pos = base_pos + residual_pos × scale``
   - ``combined_quat = base_quat ⊗ euler2quat(residual_euler × scale)``
   - ``combined_grip = clamp(base_grip + residual_grip × scale, 0, 1)``
4. The combined absolute target is passed to ``MuJoCoVecEnv.step()``.

Exposes the same interface as ``IsaacLabResidualWrapper`` so the training
loop and QAgent work without modification.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

# ── Isaac-GR00T imports ──
_GROOT_ROOT = str(Path(__file__).resolve().parents[4] / "Isaac-GR00T")
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

try:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
except ImportError:
    Gr00tPolicy = None
    EmbodimentTag = None


class _FakeImageFeaturesConfig:
    """Minimal shim so that downstream code that reads
    ``base_policy.config.image_features`` keeps working."""

    def __init__(self):
        self.image_features = {
            "observation.images.front": None,
            "observation.images.back": None,
            "observation.images.wrist": None,
        }


class MuJoCoResidualWrapper:
    """Combines GR00T base policy with residual RL on top of MuJoCoVecEnv.

    The RL agent sees ``observation.base_action`` (8-D absolute) and outputs
    a 7-D residual.
    """

    def __init__(
        self,
        vec_env: MuJoCoVecEnv,
        groot_checkpoint: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        policy_device: str = "cuda:0",
        task_description: str = "lift the cube",
        action_horizon: int = 16,
        open_loop_horizon: int = 16,
        residual_pos_scale: float = 0.02,   # metres
        residual_rot_scale: float = 0.05,   # radians
        residual_grip_scale: float = 0.1,
        action_clip: float = 1.0,
    ):
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.device = vec_env.device
        self.action_dim = 7  # RL residual: pos3 + euler3 + grip1
        self.action_horizon = action_horizon
        self.open_loop_horizon = open_loop_horizon
        self.residual_pos_scale = residual_pos_scale
        self.residual_rot_scale = residual_rot_scale
        self.residual_grip_scale = residual_grip_scale
        self.action_clip = action_clip
        self.task_description = task_description

        # ── Load GR00T policy ──
        self._skip_groot = str(
            os.environ.get("RESFIT_SKIP_GROOT_MODEL_LOAD", "0")
        ).lower() in {"1", "true", "yes"}

        if self._skip_groot or Gr00tPolicy is None:
            self.policy = None
            print("[MuJoCoResidualWrapper] GR00T policy SKIPPED (returns zeros)")
        else:
            print(f"[MuJoCoResidualWrapper] Loading GR00T from {groot_checkpoint}...")
            t0 = time.time()
            tag = getattr(EmbodimentTag, embodiment_tag, EmbodimentTag.NEW_EMBODIMENT)
            self.policy = Gr00tPolicy(
                embodiment_tag=tag,
                model_path=groot_checkpoint,
                device=policy_device,
            )
            print(f"[MuJoCoResidualWrapper] GR00T loaded in {time.time() - t0:.1f}s")

        # Config shim (for code that reads base_policy.config.image_features)
        self.config = _FakeImageFeaturesConfig()

        # ── Per-env chunk cache ──
        self._cached_chunks: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
        self._chunk_idx: list[int] = [self.open_loop_horizon] * self.num_envs  # force first query
        self._held_base_action = np.zeros((self.num_envs, 8), dtype=np.float32)

        # ── Build observation space (augment with base_action) ──
        spaces = dict(vec_env.observation_space.spaces)
        spaces["observation.base_action"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_envs, 8),  # absolute action: pos3+quat4+grip1
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

        # RL action space: 7-D residual
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )

        # Internal state
        self._last_obs: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset env and base policy, return augmented obs."""
        raw_obs, info = self.vec_env.reset(**kwargs)

        # Reset chunk caches
        self._cached_chunks = [None] * self.num_envs
        self._chunk_idx = [self.open_loop_horizon] * self.num_envs
        self._held_base_action[:] = 0.0

        # Get first base action
        base_action = self._get_base_actions(force_infer=True)
        augmented_obs = self._augment_obs(raw_obs, base_action)
        self._last_obs = raw_obs
        return augmented_obs, info

    def step(
        self, residual_action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step with residual action.

        Parameters
        ----------
        residual_action : (num_envs, 7) in [-1, 1]
            Residual delta from RL agent.
        """
        if isinstance(residual_action, torch.Tensor):
            residual_np = residual_action.detach().cpu().numpy()
        else:
            residual_np = np.asarray(residual_action)

        # Get current base actions (absolute)
        base_action = self._get_base_actions()  # (N, 8)

        # Combine base + residual → absolute target
        combined = self._combine_actions(base_action, residual_np)  # (N, 8)

        # Step MuJoCo env with combined absolute actions
        combined_t = torch.as_tensor(combined, device=self.device, dtype=torch.float32)
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(combined_t)

        # Store combined action for replay buffer
        info["scaled_action"] = torch.as_tensor(
            combined, device=self.device, dtype=torch.float32,
        )

        # Advance chunk indices
        for i in range(self.num_envs):
            self._chunk_idx[i] += 1

        # Reset chunks for terminated/truncated envs
        done = terminated | truncated
        if done.any():
            done_ids = torch.where(done)[0].tolist()
            for eid in done_ids:
                self._cached_chunks[eid] = None
                self._chunk_idx[eid] = self.open_loop_horizon

        # Get next base action for augmented obs
        next_base_action = self._get_base_actions()
        if done.any():
            for eid in torch.where(done)[0].tolist():
                next_base_action[eid] = 0.0

        self._last_obs = raw_obs
        augmented_obs = self._augment_obs(raw_obs, next_base_action)

        return augmented_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Base policy querying
    # ------------------------------------------------------------------

    def _get_base_actions(self, force_infer: bool = False) -> np.ndarray:
        """Get current base actions for all envs, querying GR00T as needed.

        Returns (num_envs, 8) absolute actions: [pos3, quat4, grip1].
        """
        actions = np.zeros((self.num_envs, 8), dtype=np.float32)

        for i in range(self.num_envs):
            # Check if we need a new chunk
            if (
                force_infer
                or self._cached_chunks[i] is None
                or self._chunk_idx[i] >= self.open_loop_horizon
            ):
                self._query_groot(i)

            # Serve current token from cache
            chunk = self._cached_chunks[i]
            idx = self._chunk_idx[i]
            if chunk is not None and idx < chunk["eef_pos"].shape[0]:
                actions[i, :3] = chunk["eef_pos"][idx]
                actions[i, 3:7] = chunk["eef_quat"][idx]
                actions[i, 7] = chunk["gripper_width"][idx, 0]
            else:
                # Fallback: current TCP pose
                groot_obs = self.vec_env.get_groot_obs(i, self.task_description)
                actions[i, :3] = groot_obs["state"]["proprio.eef_pos"][0, 0]
                actions[i, 3:7] = groot_obs["state"]["proprio.eef_quat"][0, 0]
                actions[i, 7] = groot_obs["state"]["proprio.gripper_width"][0, 0, 0]

        self._held_base_action = actions.copy()
        return actions

    def _query_groot(self, env_idx: int) -> None:
        """Query GR00T for a fresh action chunk for one environment."""
        if self.policy is None:
            # Skip mode: current pose as action (no movement)
            groot_obs = self.vec_env.get_groot_obs(env_idx, self.task_description)
            pos = groot_obs["state"]["proprio.eef_pos"][0, 0]
            quat = groot_obs["state"]["proprio.eef_quat"][0, 0]
            gw = groot_obs["state"]["proprio.gripper_width"][0, 0]
            self._cached_chunks[env_idx] = {
                "eef_pos": np.tile(pos, (self.action_horizon, 1)),
                "eef_quat": np.tile(quat, (self.action_horizon, 1)),
                "gripper_width": np.tile(gw, (self.action_horizon, 1)),
            }
            self._chunk_idx[env_idx] = 0
            return

        groot_obs = self.vec_env.get_groot_obs(env_idx, self.task_description)
        action_result, _info = self.policy.get_action(groot_obs)

        self._cached_chunks[env_idx] = {
            "eef_pos": np.asarray(action_result["action.eef_pos"][0], dtype=np.float32),
            "eef_quat": np.asarray(action_result["action.eef_quat"][0], dtype=np.float32),
            "gripper_width": np.asarray(action_result["action.gripper_width"][0], dtype=np.float32),
        }
        self._chunk_idx[env_idx] = 0

    # ------------------------------------------------------------------
    # Action combining
    # ------------------------------------------------------------------

    def _combine_actions(
        self,
        base_action: np.ndarray,   # (N, 8) [pos3, quat4, grip1]
        residual: np.ndarray,       # (N, 7) [dpos3, deuler3, dgrip1]
    ) -> np.ndarray:
        """Combine base absolute action with residual delta.

        Returns (N, 8) combined absolute action.
        """
        combined = np.zeros_like(base_action)

        for i in range(self.num_envs):
            base_pos = base_action[i, :3]
            base_quat_xyzw = base_action[i, 3:7]
            base_grip = base_action[i, 7]

            res_pos = residual[i, :3] * self.residual_pos_scale
            res_euler = residual[i, 3:6] * self.residual_rot_scale
            res_grip = residual[i, 6] * self.residual_grip_scale

            # Position: additive
            combined[i, :3] = base_pos + res_pos

            # Orientation: compose base quat with residual rotation
            base_rot = Rotation.from_quat(base_quat_xyzw)
            delta_rot = Rotation.from_euler("xyz", res_euler)
            combined_rot = base_rot * delta_rot
            combined[i, 3:7] = combined_rot.as_quat()  # xyzw

            # Gripper: additive, clamped
            combined[i, 7] = np.clip(base_grip + res_grip, 0.0, 1.0)

        return combined

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def _augment_obs(
        self,
        raw_obs: dict[str, torch.Tensor],
        base_action: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Add ``observation.base_action`` to the observation dict."""
        out = dict(raw_obs)
        out["observation.base_action"] = torch.as_tensor(
            base_action, device=self.device, dtype=torch.float32,
        )
        return out

    # ------------------------------------------------------------------
    # Pass-through
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        return self.vec_env.render()

    def close(self) -> None:
        self.vec_env.close()

    @property
    def fps(self) -> int:
        return self.vec_env.fps

    def reset_envs(self, env_ids: list[int]) -> None:
        self.vec_env.reset_envs(env_ids)
        for eid in env_ids:
            self._cached_chunks[eid] = None
            self._chunk_idx[eid] = self.open_loop_horizon

    def get_frame(self, env_id: int, camera: str = "front", size: tuple[int, int] = (128, 160)) -> np.ndarray:
        return self.vec_env.get_frame(env_id, camera, size)

    def set_active_env_ids(self, env_ids: list[int]) -> None:
        """No-op compatibility with IfaceEnvWrapper."""
        pass

    def _build_obs(self) -> dict[str, torch.Tensor]:
        """Rebuild observations (used after auto-reset)."""
        raw_obs = self.vec_env._build_obs_dict()
        base_action = self._held_base_action.copy()
        return self._augment_obs(raw_obs, base_action)

    @property
    def cube(self):
        """Compatibility shim for training script cube access."""
        return self

    @property
    def data(self):
        """Compatibility shim."""
        return self

    @property
    def root_state_w(self):
        """Return cube root state (N, 7+) for training script compatibility."""
        states = []
        for i in range(self.num_envs):
            pos = self.vec_env.get_cube_pos(i)
            quat = self.vec_env.get_cube_quat_xyzw(i)
            states.append(np.concatenate([pos, quat]))
        return torch.as_tensor(np.stack(states), device=self.device, dtype=torch.float32)

    @property
    def _initial_cube_z(self):
        """Return initial cube z for training script compatibility."""
        return torch.as_tensor(
            self.vec_env._initial_cube_z, device=self.device, dtype=torch.float32,
        )

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)
