"""
Minimal IsaacLab + GR00T rollout wrapper for residual learning bootstrap.

This wrapper is intentionally simple:
- queries GR00T base action every step (with internal chunk cache in policy)
- adds residual action (defaults to zero)
- forwards combined action to IsaacLab vec env
- augments observation with `observation.base_action`
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from resfit.rl_finetuning.policies.base_policy_interface import BaseChunkPolicy


class IsaacLabGrootRolloutEnv:
    def __init__(self, vec_env, base_policy: BaseChunkPolicy, action_clip: float = 1.0):
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_clip = action_clip

        self.num_envs = vec_env.num_envs
        self.action_dim = 7

        spaces = dict(vec_env.observation_space.spaces)
        spaces["observation.base_action"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self._last_obs: dict[str, torch.Tensor] | None = None
        self._cached_chunks: list[np.ndarray | None] = [None] * self.num_envs
        self._chunk_idx: list[int] = [0] * self.num_envs

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()
        self._invalidate_chunks()
        with torch.no_grad():
            base = self._next_base_action(obs)
        self._last_obs = obs
        return self._augment(obs, base), info

    def step(self, residual_action: torch.Tensor | None = None):
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step().")

        if residual_action is None:
            residual_action = torch.zeros(
                (self.num_envs, self.action_dim),
                device=self.vec_env.device,
                dtype=torch.float32,
            )

        with torch.no_grad():
            base_action = self._next_base_action(self._last_obs)

        combined = base_action + residual_action.to(base_action.device)
        if self.action_clip > 0:
            combined = combined.clamp(-self.action_clip, self.action_clip)

        next_obs, reward, terminated, truncated, info = self.vec_env.step(combined)
        info["scaled_action"] = combined

        done = terminated | truncated
        if done.any():
            reset_ids = torch.where(done)[0]
            if hasattr(self.base_policy, "reset"):
                self.base_policy.reset(env_ids=reset_ids)
            self._invalidate_chunks(env_ids=reset_ids)

        self._last_obs = next_obs
        return self._augment(next_obs, base_action), reward, terminated, truncated, info

    def _augment(self, obs: dict[str, torch.Tensor], base_action: torch.Tensor):
        out = dict(obs)
        out["observation.base_action"] = base_action
        return out

    def _invalidate_chunks(self, env_ids: torch.Tensor | list[int] | None = None):
        if env_ids is None:
            ids = range(self.num_envs)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().tolist()
        else:
            ids = env_ids
        for env_id in ids:
            self._cached_chunks[env_id] = None
            self._chunk_idx[env_id] = 0

    def _next_base_action(self, raw_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = torch.zeros((self.num_envs, self.action_dim), device=self.vec_env.device, dtype=torch.float32)
        for env_id in range(self.num_envs):
            chunk = self._cached_chunks[env_id]
            idx = self._chunk_idx[env_id]
            if chunk is None or idx >= chunk.shape[0]:
                env_obs = self._slice_env_obs(raw_obs, env_id)
                pred_action, _info = self.base_policy.infer_action_chunk(env_obs)
                parsed = self._parse_action_chunk(pred_action)
                if parsed is None:
                    parsed = np.zeros((1, self.action_dim), dtype=np.float32)
                self._cached_chunks[env_id] = parsed
                self._chunk_idx[env_id] = 0
                chunk = parsed
                idx = 0

            actions[env_id] = torch.as_tensor(chunk[idx], device=self.vec_env.device, dtype=torch.float32)
            self._chunk_idx[env_id] += 1
        return actions

    @staticmethod
    def _slice_env_obs(raw_obs: dict[str, torch.Tensor], env_id: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in raw_obs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] > env_id:
                out[key] = value[env_id: env_id + 1]
            else:
                out[key] = value
        return out

    def _parse_action_chunk(self, pred_action) -> np.ndarray | None:
        if pred_action is None:
            return None

        if isinstance(pred_action, dict):
            if "action" in pred_action:
                arr = np.asarray(pred_action["action"], dtype=np.float32)
                arr = arr[0] if arr.ndim == 3 else arr
                return arr if arr.ndim == 2 and arr.shape[-1] == self.action_dim else None
            if "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                ee = np.asarray(pred_action["action.ee_delta"], dtype=np.float32)
                gr = np.asarray(pred_action["action.gripper_pos"], dtype=np.float32)
                ee = ee[0] if ee.ndim == 3 else ee
                gr = gr[0] if gr.ndim == 3 else gr
                if ee.ndim != 2:
                    return None
                if gr.ndim == 1:
                    gr = gr[:, None]
                return np.concatenate([ee, gr], axis=-1).astype(np.float32)
            return None

        arr = np.asarray(pred_action, dtype=np.float32)
        arr = arr[0] if arr.ndim == 3 else arr
        if arr.ndim != 2:
            return None
        if arr.shape[-1] > self.action_dim:
            return arr[:, : self.action_dim]
        if arr.shape[-1] < self.action_dim:
            pad = np.zeros((arr.shape[0], self.action_dim - arr.shape[-1]), dtype=np.float32)
            return np.concatenate([arr, pad], axis=-1)
        return arr

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    @property
    def device(self):
        return self.vec_env.device

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)
