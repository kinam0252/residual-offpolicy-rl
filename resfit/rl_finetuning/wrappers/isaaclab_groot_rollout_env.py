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


class IsaacLabGrootRolloutEnv:
    def __init__(self, vec_env, base_policy, action_clip: float = 1.0):
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

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self.base_policy.reset()
        with torch.no_grad():
            base = self.base_policy.select_action(obs)
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
            base_action = self.base_policy.select_action(self._last_obs)

        combined = base_action + residual_action.to(base_action.device)
        if self.action_clip > 0:
            combined = combined.clamp(-self.action_clip, self.action_clip)

        next_obs, reward, terminated, truncated, info = self.vec_env.step(combined)
        info["scaled_action"] = combined

        done = terminated | truncated
        if done.any():
            reset_ids = torch.where(done)[0]
            self.base_policy.reset(env_ids=reset_ids)

        self._last_obs = next_obs
        return self._augment(next_obs, base_action), reward, terminated, truncated, info

    def _augment(self, obs: dict[str, torch.Tensor], base_action: torch.Tensor):
        out = dict(obs)
        out["observation.base_action"] = base_action
        return out

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    @property
    def device(self):
        return self.vec_env.device

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)
