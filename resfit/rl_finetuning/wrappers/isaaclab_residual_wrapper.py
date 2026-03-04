"""
Residual environment wrapper for IsaacLab + GR00T.

Adapted from ``resfit.rl_finetuning.wrappers.residual_env_wrapper.BasePolicyVecEnvWrapper``
but specialised for:

* **IsaacLabVecEnvWrapper** as the underlying vectorised environment (Phase 1).
* **GR00TBasePolicy** as the base policy with action-chunk caching (Phase 2).
* No ``ActionScaler`` / ``StateStandardizer`` — GR00T deltas and residuals
  live in the same ee-delta space; state standardisation is optional.

Action flow
-----------
1. ``reset()`` → query GR00T for first base-action chunk, augment obs.
2. ``step(residual)`` →
   a. ``combined = base_action_token + residual`` (both in ee-delta 7D)
   b. Pass ``combined`` to IsaacLab env (which does IK internally).
   c. Query GR00T for *next* base action (served from cache; server call
      only every 16 steps).
   d. Return augmented obs with ``observation.base_action``.

Usage
-----
::

    from resfit.rl_finetuning.wrappers.isaaclab_residual_wrapper import IsaacLabResidualWrapper
    env = IsaacLabResidualWrapper(isaaclab_env, groot_policy)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class IsaacLabResidualWrapper:
    """Combines GR00T base policy with a residual RL agent on top of IsaacLab.

    This is the top-level environment that the TD3 training loop interacts with.
    The RL agent sees augmented observations (including ``observation.base_action``)
    and outputs a **residual** action that is added to the GR00T base action
    before being sent to the simulator.
    """

    def __init__(
        self,
        vec_env,       # IsaacLabVecEnvWrapper (Phase 1)
        base_policy,   # GR00TBasePolicy (Phase 2)
        state_standardizer=None,  # optional; if None, state is passed raw
        action_clip: float = 1.0,  # clip combined action to [-clip, clip]
    ):
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.state_standardizer = state_standardizer
        self.action_clip = action_clip

        self.num_envs = vec_env.num_envs
        self.action_dim = 7  # ee_delta (3 pos + 3 rot + 1 gripper)

        # Image keys from base policy config
        self.image_keys = list(base_policy.config.image_features.keys())

        # ── Build observation space (augmented with base_action) ──
        self._setup_observation_space()

        # Internal state
        self._last_base_action: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Observation / action spaces
    # ------------------------------------------------------------------

    def _setup_observation_space(self):
        orig = self.vec_env.observation_space

        spaces = {}
        for key, space in orig.spaces.items():
            spaces[key] = space  # keep all original keys

        # Add base_action key
        spaces["observation.base_action"] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

        # Action space = residual (same shape as base action)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset env and base policy, return augmented obs."""
        raw_obs, info = self.vec_env.reset(**kwargs)

        # Reset base policy caches
        self.base_policy.reset()

        # Get first base action
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)

        self._last_base_action = base_action
        augmented_obs = self._augment_obs(raw_obs, base_action)
        return augmented_obs, info

    def step(
        self, residual_action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step with residual action.

        Parameters
        ----------
        residual_action : Tensor (num_envs, 7)
            Residual ee-delta predicted by the TD3 actor.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._last_base_action is None:
            raise RuntimeError("Must call reset() before step()")

        # Combine base + residual
        combined = self._last_base_action + residual_action
        if self.action_clip > 0:
            combined = combined.clamp(-self.action_clip, self.action_clip)

        # Step the IsaacLab env with the combined action
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(combined)

        # Store the combined action for replay buffer
        info["scaled_action"] = combined

        # Get next base action
        with torch.no_grad():
            base_action = self.base_policy.select_action(raw_obs)

        # Reset base policy for terminated envs
        done = terminated | truncated
        if done.any():
            reset_ids = torch.where(done)[0]
            self.base_policy.reset(env_ids=reset_ids)
            # Re-query base action for reset envs (they got a fresh obs from auto-reset)
            with torch.no_grad():
                fresh_base = self.base_policy.select_action(raw_obs)
            # Overwrite only the reset envs
            base_action[reset_ids] = fresh_base[reset_ids]

        self._last_base_action = base_action
        augmented_obs = self._augment_obs(raw_obs, base_action)

        return augmented_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_obs(
        self, raw_obs: dict[str, torch.Tensor], base_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Add ``observation.base_action`` and optionally standardise state."""
        out = dict(raw_obs)  # shallow copy
        out["observation.base_action"] = base_action

        if self.state_standardizer is not None and "observation.state" in out:
            out["observation.state"] = self.state_standardizer.standardize(out["observation.state"])

        return out

    # ------------------------------------------------------------------
    # Pass-through
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    @property
    def fps(self):
        return self.vec_env.fps

    @property
    def env_name(self):
        return getattr(self.vec_env, "env_name", "FrankaPickup")

    @property
    def camera_size(self):
        return getattr(self.vec_env, "camera_size", 84)

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)
