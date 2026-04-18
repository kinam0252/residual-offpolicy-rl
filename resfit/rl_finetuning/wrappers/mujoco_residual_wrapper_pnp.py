"""
Residual environment wrapper for MuJoCo PnP + GR00T.

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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv

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


class MuJoCoResidualWrapperPnP:
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
        task_description: str = "Pick up the red cube and place it onto the plate.",
        action_horizon: int = 16,
        open_loop_horizon: int = 16,
        residual_pos_scale: float = 0.02,   # metres
        residual_rot_scale: float = 0.05,   # radians
        residual_grip_scale: float = 0.1,
        action_clip: float = 1.0,
        ema_alpha: float = 0.0,
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

        # ── EMA smoothing for base action ──
        self.ema_alpha = ema_alpha
        self._ema_pos: list[np.ndarray | None] = [None] * self.num_envs
        self._ema_quat: list[np.ndarray | None] = [None] * self.num_envs

        # ── Build observation space (augment with base_action) ──
        spaces = dict(vec_env.observation_space.spaces)
        spaces["observation.base_action"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_envs, 7),  # 7D: pos3 + euler_rpy3 + grip1
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
        self._ema_pos = [None] * self.num_envs
        self._ema_quat = [None] * self.num_envs

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
        _t0 = time.time()
        base_action = self._get_base_actions()  # (N, 8)
        _dt_groot = time.time() - _t0

        # Combine base + residual → absolute target
        combined = self._combine_actions(base_action, residual_np)  # (N, 8)

        # Step MuJoCo env with combined absolute actions
        combined_t = torch.as_tensor(combined, device=self.device, dtype=torch.float32)
        _t0 = time.time()
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(combined_t, render_mode="rl_only")
        _dt_mujoco = time.time() - _t0

        # Internal timing stats (exposed via info)
        if not hasattr(self, "_step_timers"):
            self._step_timers = {"groot_query": [], "mujoco_step": []}
        self._step_timers["groot_query"].append(_dt_groot)
        self._step_timers["mujoco_step"].append(_dt_mujoco)
        if len(self._step_timers["groot_query"]) % 500 == 0:
            gq = self._step_timers["groot_query"][-500:]
            ms = self._step_timers["mujoco_step"][-500:]
            print(f"[wrapper-timing] groot_query={sum(gq)/len(gq)*1000:.1f}ms "
                  f"mujoco_step={sum(ms)/len(ms)*1000:.1f}ms", flush=True)

        # Store combined action for replay buffer
        info["scaled_action"] = torch.as_tensor(
            combined, device=self.device, dtype=torch.float32,
        )

        # Advance chunk indices
        for i in range(self.num_envs):
            self._chunk_idx[i] += 1

        # Reset chunks for terminated/truncated envs + auto-reset env
        done = terminated | truncated
        if done.any():
            done_ids = torch.where(done)[0].tolist()
            for eid in done_ids:
                self._cached_chunks[eid] = None
                self._chunk_idx[eid] = self.open_loop_horizon
                self._ema_pos[eid] = None
                self._ema_quat[eid] = None

            # Auto-reset done environments so next step starts fresh
            self.vec_env.reset_envs(done_ids)
            # Rebuild obs from reset state
            raw_obs = self.vec_env._build_obs_dict(render_mode="rl_only")

        # Get next base action for augmented obs
        next_base_action = self._get_base_actions()
        if done.any():
            # After reset, zero out base action for done envs
            # (will be populated on next GR00T query)
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

        Uses **batched inference**: collects all envs that need a new chunk,
        builds a single batched observation (B=N_needing), and runs one
        ``policy.get_action()`` call on the GPU.

        Returns (num_envs, 8) absolute actions: [pos3, quat4, grip1].
        """
        # Identify envs that need fresh inference
        need_infer = []
        for i in range(self.num_envs):
            if (
                force_infer
                or self._cached_chunks[i] is None
                or self._chunk_idx[i] >= self.open_loop_horizon
            ):
                need_infer.append(i)

        # Batch-infer all needed envs at once
        if need_infer:
            _t0 = time.time()
            self._query_groot_batch(need_infer)
            _dt = (time.time() - _t0) * 1000
            if not hasattr(self, "_groot_infer_times"):
                self._groot_infer_times = []
            self._groot_infer_times.append(_dt)
            if len(self._groot_infer_times) % 10 == 0:
                avg = sum(self._groot_infer_times[-10:]) / 10
                print(f"[groot-infer] batch={len(need_infer)} envs, time={_dt:.0f}ms, avg={avg:.0f}ms", flush=True)

        # Serve current token from cache
        actions = np.zeros((self.num_envs, 8), dtype=np.float32)
        for i in range(self.num_envs):
            chunk = self._cached_chunks[i]
            idx = self._chunk_idx[i]
            if chunk is not None and idx < chunk["eef_pos"].shape[0]:
                actions[i, :3] = chunk["eef_pos"][idx]
                actions[i, 3:7] = chunk["eef_quat"][idx]
                actions[i, 7] = chunk["gripper_width"][idx, 0]
            else:
                # Fallback: current TCP pose (shouldn't happen after batch infer)
                groot_obs = self.vec_env.get_groot_obs(i, self.task_description)
                actions[i, :3] = groot_obs["state"]["proprio.eef_pos"][0, 0]
                actions[i, 3:7] = groot_obs["state"]["proprio.eef_quat"][0, 0]
                actions[i, 7] = groot_obs["state"]["proprio.gripper_width"][0, 0, 0]

        # Apply EMA smoothing to base action
        if self.ema_alpha > 0:
            for i in range(self.num_envs):
                pos = actions[i, :3]
                quat = actions[i, 3:7]
                if self._ema_pos[i] is None:
                    self._ema_pos[i] = pos.copy()
                    self._ema_quat[i] = quat.copy()
                else:
                    self._ema_pos[i] = self.ema_alpha * self._ema_pos[i] + (1 - self.ema_alpha) * pos
                    self._ema_quat[i] = self.ema_alpha * self._ema_quat[i] + (1 - self.ema_alpha) * quat
                    self._ema_quat[i] = self._ema_quat[i] / np.linalg.norm(self._ema_quat[i])
                actions[i, :3] = self._ema_pos[i].copy()
                actions[i, 3:7] = self._ema_quat[i].copy()

        self._held_base_action = actions.copy()
        return actions

    def _query_groot_batch(self, env_ids: list[int]) -> None:
        """Batch-query GR00T for fresh action chunks for multiple envs at once.

        Builds a single observation with batch dim B=len(env_ids) and runs
        one GPU forward pass instead of N sequential calls.
        """
        if not env_ids:
            return

        if self.policy is None:
            # Skip mode
            for eid in env_ids:
                groot_obs = self.vec_env.get_groot_obs(eid, self.task_description)
                pos = groot_obs["state"]["proprio.eef_pos"][0, 0]
                quat = groot_obs["state"]["proprio.eef_quat"][0, 0]
                gw = groot_obs["state"]["proprio.gripper_width"][0, 0]
                self._cached_chunks[eid] = {
                    "eef_pos": np.tile(pos, (self.action_horizon, 1)),
                    "eef_quat": np.tile(quat, (self.action_horizon, 1)),
                    "gripper_width": np.tile(gw, (self.action_horizon, 1)),
                }
                self._chunk_idx[eid] = 0
            return

        B = len(env_ids)

        # Collect per-env observations (each is B=1)
        per_env_obs = [self.vec_env.get_groot_obs(eid, self.task_description) for eid in env_ids]

        # Stack into batched observation (B=len(env_ids))
        batched_obs = {
            "video": {},
            "state": {},
            "language": {},
        }
        # Video: (1,1,H,W,3) per env → (B,1,H,W,3)
        for vk in per_env_obs[0]["video"]:
            batched_obs["video"][vk] = np.concatenate(
                [o["video"][vk] for o in per_env_obs], axis=0,
            )
        # State: (1,1,D) per env → (B,1,D)
        for sk in per_env_obs[0]["state"]:
            batched_obs["state"][sk] = np.concatenate(
                [o["state"][sk] for o in per_env_obs], axis=0,
            )
        # Language: [[str]] per env → [[str]] * B
        for lk in per_env_obs[0]["language"]:
            batched_obs["language"][lk] = [o["language"][lk][0] for o in per_env_obs]

        # Single batched inference call
        action_result, _info = self.policy.get_action(batched_obs)

        # Distribute results to per-env caches
        for bi, eid in enumerate(env_ids):
            self._cached_chunks[eid] = {
                "eef_pos": np.asarray(action_result["action.eef_pos"][bi], dtype=np.float32),
                "eef_quat": np.asarray(action_result["action.eef_quat"][bi], dtype=np.float32),
                "gripper_width": np.asarray(action_result["action.gripper_width"][bi], dtype=np.float32),
            }
            self._chunk_idx[eid] = 0

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
        """Add ``observation.base_action`` (7D) to the observation dict.

        Converts 8D absolute (pos3+quat4+grip1) to 7D (pos3+euler3+grip1)
        for QAgent compatibility.
        """
        out = dict(raw_obs)
        ba_7d = np.zeros((base_action.shape[0], 7), dtype=np.float32)
        for i in range(base_action.shape[0]):
            ba_7d[i, :3] = base_action[i, :3]  # pos
            quat_xyzw = base_action[i, 3:7]
            qn = np.linalg.norm(quat_xyzw)
            if qn > 1e-6:
                ba_7d[i, 3:6] = Rotation.from_quat(quat_xyzw / qn).as_euler("xyz")
            # else: leave euler as zeros
            ba_7d[i, 6] = base_action[i, 7]  # grip
        out["observation.base_action"] = torch.as_tensor(
            ba_7d, device=self.device, dtype=torch.float32,
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
