"""
Residual environment wrapper for MuJoCo Close Drawer + GR00T.

Nearly identical to MuJoCoResidualWrapperStack, adapted for:
- MuJoCoVecEnvDrawer (cabinet with drawers instead of cubes)
- Task description for drawer closing
- Same gripper raw meters [0, 0.04]
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

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer

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


class MuJoCoResidualWrapperDrawer:
    """Combines GR00T base policy with residual RL on top of MuJoCoVecEnvDrawer.

    Supports ActionScaler mode (data-driven pos+grip normalization) matching
    the cube lift wrapper implementation.
    """

    def __init__(
        self,
        vec_env: MuJoCoVecEnvDrawer,
        groot_checkpoint: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        policy_device: str = "cuda:0",
        task_description: str = "Close the drawer",
        action_horizon: int = 16,
        open_loop_horizon: int = 16,
        residual_pos_scale: float = 0.02,
        residual_rot_scale: float = 0.05,
        residual_grip_scale: float = 0.004,
        action_clip: float = 1.0,
        ema_alpha: float = 0.0,
        action_scaler=None,
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
        self.action_scaler = action_scaler

        if action_scaler is not None:
            print("[MuJoCoResidualWrapperDrawer] Using ActionScaler mode (data-driven pos+grip normalization)")

        # ── Load GR00T policy ──
        self._skip_groot = str(
            os.environ.get("RESFIT_SKIP_GROOT_MODEL_LOAD", "0")
        ).lower() in {"1", "true", "yes"}

        if self._skip_groot or Gr00tPolicy is None:
            self.policy = None
            print("[MuJoCoResidualWrapperDrawer] GR00T policy SKIPPED")
        else:
            print(f"[MuJoCoResidualWrapperDrawer] Loading GR00T from {groot_checkpoint}...")
            t0 = time.time()
            tag = getattr(EmbodimentTag, embodiment_tag, EmbodimentTag.NEW_EMBODIMENT)
            self.policy = Gr00tPolicy(
                embodiment_tag=tag,
                model_path=groot_checkpoint,
                device=policy_device,
            )
            print(f"[MuJoCoResidualWrapperDrawer] GR00T loaded in {time.time() - t0:.1f}s")

        self.config = _FakeImageFeaturesConfig()

        # Per-env chunk cache
        self._cached_chunks: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
        self._chunk_idx: list[int] = [self.open_loop_horizon] * self.num_envs
        self._held_base_action = np.zeros((self.num_envs, 8), dtype=np.float32)
        self.ema_alpha = ema_alpha
        self._ema_pos: list[np.ndarray | None] = [None] * self.num_envs
        self._ema_quat: list[np.ndarray | None] = [None] * self.num_envs

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
        combined, combined_naction_7d = self._combine_actions(base_action, residual_np)
        combined_t = torch.as_tensor(combined, device=self.device, dtype=torch.float32)
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(combined_t, render_mode="rl_only")

        # ActionScaler mode: provide normalized combined for replay buffer
        if self.action_scaler is not None and combined_naction_7d is not None:
            info["scaled_action"] = torch.as_tensor(
                combined_naction_7d, device=self.device, dtype=torch.float32)
        else:
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
    # Base policy querying (identical logic to stack wrapper)
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
        combined_naction_7d = None

        if self.action_scaler is not None:
            import torch as _torch
            scaler = self.action_scaler
            combined_naction_7d = np.zeros((self.num_envs, 7), dtype=np.float32)

            for i in range(self.num_envs):
                base_pos = base_action[i, :3]
                base_quat_xyzw = base_action[i, 3:7]
                base_grip = base_action[i, 7:8]

                # Build 4D vector [pos3, grip1] for scaling
                base_pg = np.concatenate([base_pos, base_grip])
                base_pg_t = _torch.from_numpy(base_pg).float().unsqueeze(0)
                base_pg_norm = scaler.scale(base_pg_t)

                # Residual pos+grip in [-action_scale, action_scale]
                res_pg = np.concatenate([residual[i, :3], residual[i, 6:7]])
                res_pg_t = _torch.from_numpy(res_pg).float().unsqueeze(0)

                # Combine in normalized space
                combined_pg_norm = base_pg_norm + res_pg_t
                combined_pg = scaler.unscale(combined_pg_norm)
                combined_pg_np = combined_pg.squeeze(0).numpy()

                combined[i, :3] = combined_pg_np[:3]
                # Drawer: gripper always closed (push task)
                combined[i, 7] = 0.0

                # Rotation: delta composition with physical scale
                res_euler = residual[i, 3:6] * self.residual_rot_scale
                base_rot = Rotation.from_quat(base_quat_xyzw)
                delta_rot = Rotation.from_euler("xyz", res_euler)
                combined_rot = base_rot * delta_rot
                combined[i, 3:7] = combined_rot.as_quat()

                # Build 7D normalized combined for replay
                # Bug fix: re-scale the EXECUTED physical action to get the
                # clamped normalized value (matches what env actually ran)
                executed_pg = _torch.from_numpy(
                    np.concatenate([combined[i, :3], [0.0]])  # grip=0.0 (always closed)
                ).float().unsqueeze(0)
                executed_pg_norm = scaler.scale(executed_pg)
                executed_pg_norm_np = executed_pg_norm.squeeze(0).numpy()
                combined_naction_7d[i, :3] = executed_pg_norm_np[:3]
                combined_naction_7d[i, 3:6] = residual[i, 3:6]
                combined_naction_7d[i, 6] = executed_pg_norm_np[3]  # grip=const
        else:
            for i in range(self.num_envs):
                base_pos = base_action[i, :3]
                base_quat_xyzw = base_action[i, 3:7]
                base_grip = base_action[i, 7]
                res_pos = residual[i, :3] * self.residual_pos_scale
                res_euler = residual[i, 3:6] * self.residual_rot_scale
                res_grip = residual[i, 6] * self.residual_grip_scale
                combined[i, :3] = base_pos + res_pos
                base_rot = Rotation.from_quat(base_quat_xyzw)
                delta_rot = Rotation.from_euler("xyz", res_euler)
                combined_rot = base_rot * delta_rot
                combined[i, 3:7] = combined_rot.as_quat()
                # Close Drawer: gripper always closed (push task)
                combined[i, 7] = 0.0

        return combined, combined_naction_7d

    def _augment_obs(self, raw_obs, base_action):
        out = dict(raw_obs)
        ba_7d = np.zeros((base_action.shape[0], 7), dtype=np.float32)
        for i in range(base_action.shape[0]):
            ba_7d[i, :3] = base_action[i, :3]
            quat_xyzw = base_action[i, 3:7]
            qn = np.linalg.norm(quat_xyzw)
            if qn > 1e-6:
                ba_7d[i, 3:6] = Rotation.from_quat(quat_xyzw / qn).as_euler("xyz")
            ba_7d[i, 6] = base_action[i, 7]

        if self.action_scaler is not None:
            # Normalize pos(3D) + grip(1D) via ActionScaler
            pg = np.concatenate([ba_7d[:, :3], ba_7d[:, 6:7]], axis=-1)  # (N, 4)
            pg_t = torch.as_tensor(pg, dtype=torch.float32)
            pg_norm = self.action_scaler.scale(pg_t).numpy()
            ba_7d[:, :3] = pg_norm[:, :3]
            ba_7d[:, 6] = pg_norm[:, 3]
            ba_7d[:, 3:6] = 0.0

        out["observation.base_action"] = torch.as_tensor(ba_7d, device=self.device, dtype=torch.float32)
        return out

    def close(self):
        self.vec_env.close()

    def reset_envs(self, env_ids):
        self.vec_env.reset_envs(env_ids)
        for eid in env_ids:
            self._cached_chunks[eid] = None
            self._chunk_idx[eid] = self.open_loop_horizon
