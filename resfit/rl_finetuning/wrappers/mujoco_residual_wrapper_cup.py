"""
Residual environment wrapper for MuJoCo Stand Cup + GR00T.

Nearly identical to MuJoCoResidualWrapperDrawer, adapted for:
- MuJoCoVecEnvCup (cup with freejoint physics)
- Task description for cup standing
- Gripper controlled by GR00T (not forced closed)
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup

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
    def __init__(self):
        self.image_features = {
            "observation.images.cam_base": None,
            "observation.images.cam_wrist": None,
        }


_GRIP_MIN = 0.0
_GRIP_MAX = 0.04
_GRIP_CLOSE_LATCH_THRESH = 0.015  # below this → latch closed
_GRIP_OPEN_LATCH_THRESH = 0.035   # above this → unlatch (open)


class MuJoCoResidualWrapperCup:
    """Combines GR00T base policy with residual RL on top of MuJoCoVecEnvCup."""

    def __init__(
        self,
        vec_env: MuJoCoVecEnvCup,
        groot_checkpoint: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        policy_device: str = "cuda:0",
        task_description: str = "Pick up the cup lying on its side and stand it upright",
        action_horizon: int = 16,
        open_loop_horizon: int = 16,
        residual_pos_scale: float = 0.02,
        residual_rot_scale: float = 0.05,
        residual_grip_scale: float = 0.004,
        action_clip: float = 1.0,
        ema_alpha: float = 0.0,
        torch_compile: bool = False,
    ):
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.device = vec_env.device
        self.action_dim = 7
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
            print("[MuJoCoResidualWrapperCup] GR00T policy SKIPPED")
        else:
            print(f"[MuJoCoResidualWrapperCup] Loading GR00T from {groot_checkpoint}...")
            t0 = time.time()
            tag = getattr(EmbodimentTag, embodiment_tag, EmbodimentTag.NEW_EMBODIMENT)
            self.policy = Gr00tPolicy(
                embodiment_tag=tag,
                model_path=groot_checkpoint,
                device=policy_device,
            )
            print(f"[MuJoCoResidualWrapperCup] GR00T loaded in {time.time() - t0:.1f}s")

            # ── Optional torch.compile for inference acceleration ──
            if torch_compile and hasattr(self.policy, 'model'):
                import torch as _torch
                print("[MuJoCoResidualWrapperCup] Applying torch.compile to GR00T model...")
                t1 = time.time()
                _torch._dynamo.config.suppress_errors = True
                self.policy.model = _torch.compile(
                    self.policy.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print(f"[MuJoCoResidualWrapperCup] torch.compile applied in {time.time() - t1:.1f}s")

        self.config = _FakeImageFeaturesConfig()

        # Per-env chunk cache
        self._cached_chunks: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
        self._chunk_idx: list[int] = [self.open_loop_horizon] * self.num_envs
        self._held_base_action = np.zeros((self.num_envs, 8), dtype=np.float32)
        self.ema_alpha = ema_alpha
        self._ema_pos: list[np.ndarray | None] = [None] * self.num_envs
        self._ema_quat: list[np.ndarray | None] = [None] * self.num_envs
        # Gripper latch: once closed, stay closed until GR00T explicitly opens
        self._grip_latched: list[bool] = [False] * self.num_envs
        self._grip_open_count: list[int] = [0] * self.num_envs  # consecutive open steps needed to unlatch

        # Observation space (augment with base_action)
        spaces = dict(vec_env.observation_space.spaces)
        spaces["observation.base_action"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_envs, 7),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self._last_obs: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        raw_obs, info = self.vec_env.reset(**kwargs)
        self._cached_chunks = [None] * self.num_envs
        self._chunk_idx = [self.open_loop_horizon] * self.num_envs
        self._held_base_action[:] = 0.0
        self._ema_pos = [None] * self.num_envs
        self._ema_quat = [None] * self.num_envs
        self._grip_latched = [False] * self.num_envs
        self._grip_open_count = [0] * self.num_envs
        base_action = self._get_base_actions(force_infer=True)
        augmented_obs = self._augment_obs(raw_obs, base_action)
        self._last_obs = raw_obs
        return augmented_obs, info

    def step(
        self, residual_action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(residual_action, torch.Tensor):
            residual_np = residual_action.detach().cpu().numpy()
        else:
            residual_np = np.asarray(residual_action)

        base_action = self._get_base_actions()
        combined = self._combine_actions(base_action, residual_np)
        combined_t = torch.as_tensor(combined, device=self.device, dtype=torch.float32)

        # Skip rendering if no env needs GR00T inference next step
        needs_infer_next = any(
            self._cached_chunks[i] is None or self._chunk_idx[i] + 1 >= self.open_loop_horizon
            for i in range(self.num_envs)
        )
        render = "rl_only" if needs_infer_next else "none"
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(combined_t, render_mode=render)

        info["scaled_action"] = torch.as_tensor(combined, device=self.device, dtype=torch.float32)

        for i in range(self.num_envs):
            self._chunk_idx[i] += 1

        done = terminated | truncated
        if done.any():
            done_ids = torch.where(done)[0].tolist()
            for eid in done_ids:
                self._cached_chunks[eid] = None
                self._chunk_idx[eid] = self.open_loop_horizon
                self._ema_pos[eid] = None
                self._ema_quat[eid] = None
                self._grip_latched[eid] = False
                self._grip_open_count[eid] = 0
            self.vec_env.reset_envs(done_ids)
            raw_obs = self.vec_env._build_obs_dict(render_mode="rl_only")

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
        need_infer = []
        for i in range(self.num_envs):
            if (force_infer or self._cached_chunks[i] is None
                    or self._chunk_idx[i] >= self.open_loop_horizon):
                need_infer.append(i)

        if need_infer:
            self._query_groot_batch(need_infer)

        actions = np.zeros((self.num_envs, 8), dtype=np.float32)
        for i in range(self.num_envs):
            chunk = self._cached_chunks[i]
            idx = self._chunk_idx[i]
            if chunk is not None and idx < chunk["eef_pos"].shape[0]:
                actions[i, :3] = chunk["eef_pos"][idx]
                actions[i, 3:7] = chunk["eef_quat"][idx]
                actions[i, 7] = chunk["gripper_width"][idx, 0]
            else:
                groot_obs = self.vec_env.get_groot_obs(i, self.task_description)
                actions[i, :3] = groot_obs["state"]["proprio.eef_pos"][0, 0]
                actions[i, 3:7] = groot_obs["state"]["proprio.eef_quat"][0, 0]
                actions[i, 7] = groot_obs["state"]["proprio.gripper_width"][0, 0, 0]

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
        if not env_ids:
            return
        if self.policy is None:
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

        per_env_obs = [self.vec_env.get_groot_obs(eid, self.task_description) for eid in env_ids]
        batched_obs = {"video": {}, "state": {}, "language": {}}
        for vk in per_env_obs[0]["video"]:
            batched_obs["video"][vk] = np.concatenate([o["video"][vk] for o in per_env_obs], axis=0)
        for sk in per_env_obs[0]["state"]:
            batched_obs["state"][sk] = np.concatenate([o["state"][sk] for o in per_env_obs], axis=0)
        for lk in per_env_obs[0]["language"]:
            batched_obs["language"][lk] = [o["language"][lk][0] for o in per_env_obs]

        action_result, _info = self.policy.get_action(batched_obs)
        for bi, eid in enumerate(env_ids):
            self._cached_chunks[eid] = {
                "eef_pos": np.asarray(action_result["action.eef_pos"][bi], dtype=np.float32),
                "eef_quat": np.asarray(action_result["action.eef_quat"][bi], dtype=np.float32),
                "gripper_width": np.asarray(action_result["action.gripper_width"][bi], dtype=np.float32),
            }
            self._chunk_idx[eid] = 0

    def _combine_actions(self, base_action, residual):
        combined = np.zeros_like(base_action)
        # Position: direct add (vectorized)
        combined[:, :3] = base_action[:, :3] + residual[:, :3] * self.residual_pos_scale
        # Gripper: clip + latch
        raw_grip = np.clip(
            base_action[:, 7] + residual[:, 6] * self.residual_grip_scale,
            _GRIP_MIN, _GRIP_MAX)
        for i in range(self.num_envs):
            if self._grip_latched[i]:
                # Latched closed — require N consecutive open commands to unlatch
                if raw_grip[i] > _GRIP_OPEN_LATCH_THRESH:
                    self._grip_open_count[i] += 1
                    if self._grip_open_count[i] >= 32:  # ~2 full action chunks
                        self._grip_latched[i] = False
                        self._grip_open_count[i] = 0
                        combined[i, 7] = raw_grip[i]
                    else:
                        combined[i, 7] = _GRIP_MIN  # still latched
                else:
                    self._grip_open_count[i] = 0
                    combined[i, 7] = _GRIP_MIN  # keep closed
            else:
                # Not latched — check if close command
                if raw_grip[i] < _GRIP_CLOSE_LATCH_THRESH:
                    self._grip_latched[i] = True
                    self._grip_open_count[i] = 0
                    combined[i, 7] = _GRIP_MIN  # close
                else:
                    combined[i, 7] = raw_grip[i]
        # Rotation: must use Rotation (batch-capable)
        base_rots = Rotation.from_quat(base_action[:, 3:7])
        delta_rots = Rotation.from_euler("xyz", residual[:, 3:6] * self.residual_rot_scale)
        combined[:, 3:7] = (base_rots * delta_rots).as_quat()
        return combined

    def _augment_obs(self, raw_obs, base_action):
        out = dict(raw_obs)
        ba_7d = np.zeros((base_action.shape[0], 7), dtype=np.float32)
        ba_7d[:, :3] = base_action[:, :3]
        quats = base_action[:, 3:7]
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        valid = (norms > 1e-6).ravel()
        if valid.any():
            safe_quats = quats[valid] / norms[valid]
            ba_7d[valid, 3:6] = Rotation.from_quat(safe_quats).as_euler("xyz")
        ba_7d[:, 6] = base_action[:, 7]
        out["observation.base_action"] = torch.as_tensor(ba_7d, device=self.device, dtype=torch.float32)
        return out

    def close(self):
        self.vec_env.close()

    def reset_envs(self, env_ids):
        self.vec_env.reset_envs(env_ids)
        for eid in env_ids:
            self._cached_chunks[eid] = None
            self._chunk_idx[eid] = self.open_loop_horizon
