"""
IsaacLab environment wrapper that exposes the same interface as resfit's
VectorizedEnvWrapper (gymnasium Dict obs, torch tensors, batched).

This bridges our FrankaPickupEnv (DirectRLEnv) to the resfit residual-TD3
training loop, which expects:
  - observation_space: gym.spaces.Dict with
      "observation.state"              : (num_envs, state_dim)  float32
      "observation.images.<cam_name>"  : (num_envs, C, H, W)   uint8 [0,255]
  - action_space: gym.spaces.Box(-1, 1, (num_envs, action_dim))
  - reset()  -> (obs_dict, info)
  - step(action) -> (obs_dict, reward, terminated, truncated, info)
  All values are torch tensors on the specified device.

Usage (inside the IsaacLab Python runtime launched via isaaclab.sh):
    from resfit.rl_finetuning.wrappers.isaaclab_env_wrapper import IsaacLabVecEnvWrapper, create_isaaclab_env
    env = create_isaaclab_env(task="Isaac-Franka-Pickup-Direct-v0", num_envs=1,
                              enable_cameras=True, device="cuda:0",
                              image_size=(84, 84))
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class IsaacLabVecEnvWrapper:
    """Wraps an IsaacLab DirectRLEnv to match resfit's VectorizedEnvWrapper interface.

    Key adaptations
    ---------------
    * Splits the monolithic flat obs tensor from DirectRLEnv into a Dict with
      "observation.state" (low-dim) and per-camera image keys.
    * Down-scales camera images to ``image_size`` (default 84×84) so that the
      resfit ViT encoder receives a manageable resolution.
    * Converts IsaacLab's single ``terminated | truncated`` done signal into
      separate ``terminated`` / ``truncated`` booleans.
    * Exposes ``observation_space`` / ``action_space`` as gymnasium spaces so
      that resfit's replay buffer and agent constructors work out of the box.
    """

    # Camera name mapping: IsaacLab sensor key -> resfit obs key
    CAMERA_MAP = {
        "camera_front": "observation.images.front",
        "camera_back": "observation.images.back",
        "camera_wrist": "observation.images.wrist",
    }

    def __init__(
        self,
        env,  # isaaclab DirectRLEnv instance
        image_size: tuple[int, int] = (84, 84),
        device: str = "cuda:0",
    ):
        self.env = env
        self._unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.image_size = image_size  # (H, W) target for downscale
        self.device = device
        self.num_envs = getattr(self._unwrapped, 'num_envs', 1)

        # ── Determine low-dim state dimension ──
        # We'll extract: joint_pos_scaled (9) + joint_vel_scaled (9) + cogact_ref (7)
        #              + contact_obs (3) + ee_xyzrpy (6) = 34
        # But the actual obs is a flat tensor whose dim depends on cfg.observation_space.
        # We split: first (obs_dim - latent_dim) as state, latent is discarded here
        # since resfit uses its own visual encoder.
        # For safety we compute state_dim from known components.
        self._joint_dim = 9  # 7 arm + 2 finger
        self._state_dim = (
            self._joint_dim  # dof_pos_scaled
            + self._joint_dim  # dof_vel_scaled
            + 7  # cogact_reference
            + 3  # contact_obs (left finger binary)
            + 6  # ee_xyzrpy
        )  # = 34

        # ── Build gymnasium spaces ──
        obs_spaces = {
            "observation.state": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_envs, self._state_dim), dtype=np.float32
            ),
        }
        # Add camera spaces
        for isaac_key, resfit_key in self.CAMERA_MAP.items():
            obs_spaces[resfit_key] = gym.spaces.Box(
                low=0, high=255,
                shape=(self.num_envs, 3, image_size[0], image_size[1]),
                dtype=np.uint8,
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action: 7-dim (dp_xyz 3 + drot_rpy 3 + gripper 1), normalised to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_envs, 7),
            dtype=np.float32,
        )

        # Internal bookkeeping
        self._cameras_ready = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset all environments and return augmented obs dict."""
        obs_raw, info = self.env.reset()
        obs_dict = self._build_obs_dict(obs_raw)
        return obs_dict, info if isinstance(info, dict) else {}

    def step(self, actions: torch.Tensor):
        """Step all environments with ``actions`` in [-1, 1] (num_envs, 7).

        Returns (obs_dict, rewards, terminated, truncated, info).
        """
        if actions.device != self._unwrapped.device:
            actions = actions.to(self._unwrapped.device)

        obs_raw, rewards, terminated, truncated, info = self.env.step(actions)
        obs_dict = self._build_obs_dict(obs_raw)

        # Ensure tensors
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.as_tensor(terminated, device=self.device, dtype=torch.bool)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.as_tensor(truncated, device=self.device, dtype=torch.bool)

        return obs_dict, rewards, terminated, truncated, info

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_obs_dict(self, obs_raw) -> dict[str, torch.Tensor]:
        """Convert DirectRLEnv obs into resfit-compatible Dict.

        ``obs_raw`` is either a dict ``{"policy": Tensor(N, obs_dim)}`` or
        a flat tensor ``(N, obs_dim)``.
        """
        if isinstance(obs_raw, dict):
            flat_obs = obs_raw.get("policy", obs_raw.get("obs", None))
            if flat_obs is None:
                flat_obs = next(iter(obs_raw.values()))
        else:
            flat_obs = obs_raw

        # flat_obs: (num_envs, obs_dim) — first self._state_dim columns are state
        state = flat_obs[:, : self._state_dim].to(device=self.device, dtype=torch.float32)

        out: dict[str, torch.Tensor] = {
            "observation.state": state,
        }

        # ── Camera images ──
        self._append_camera_obs(out)

        return out

    def _append_camera_obs(self, out: dict[str, torch.Tensor]):
        """Read cameras from the underlying IsaacLab env and add downscaled
        uint8 images to ``out`` in (N, C, H, W) format."""
        isaac_env = self._unwrapped  # DirectRLEnv

        for isaac_key, resfit_key in self.CAMERA_MAP.items():
            cam = self._find_rgb_camera(preferred_key=isaac_key)

            if cam is not None and hasattr(cam, "data") and hasattr(cam.data, "output"):
                rgb = cam.data.output.get("rgb", None)
                if rgb is not None:
                    # rgb: (N, H, W, C) or (N, H, W, 4) float/uint8
                    img = rgb
                    if img.dtype != torch.uint8:
                        if img.max() <= 1.0:
                            img = (img.clamp(0, 1) * 255).to(torch.uint8)
                        else:
                            img = img.clamp(0, 255).to(torch.uint8)
                    # Drop alpha
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    # (N, H, W, C) -> (N, C, H, W)
                    img = img.permute(0, 3, 1, 2).contiguous()
                    # Downscale
                    if img.shape[2] != self.image_size[0] or img.shape[3] != self.image_size[1]:
                        img_f = img.float()
                        img_f = F.interpolate(
                            img_f,
                            size=self.image_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        img = img_f.to(torch.uint8)
                    out[resfit_key] = img.to(self.device)
                    continue

            # Fallback: zeros if camera not available
            out[resfit_key] = torch.zeros(
                (self.num_envs, 3, self.image_size[0], self.image_size[1]),
                device=self.device, dtype=torch.uint8,
            )

    def _find_rgb_camera(self, preferred_key: str | None = None):
        isaac_env = self._unwrapped

        def has_rgb(candidate) -> bool:
            return (
                candidate is not None
                and hasattr(candidate, "data")
                and hasattr(candidate.data, "output")
                and candidate.data.output.get("rgb", None) is not None
            )

        # 1) Preferred named sensor in scene.sensors
        if hasattr(isaac_env, "scene") and hasattr(isaac_env.scene, "sensors"):
            sensors = isaac_env.scene.sensors
            if preferred_key:
                preferred = sensors.get(preferred_key, None)
                if has_rgb(preferred):
                    return preferred

        # 2) Preferred private/public attributes
        if preferred_key:
            for attr_name in (f"_{preferred_key}", preferred_key):
                cam = getattr(isaac_env, attr_name, None)
                if has_rgb(cam):
                    return cam

        # 3) Any camera-like common names
        for attr_name in ("_camera_front", "camera_front", "_front_camera", "front_camera"):
            cam = getattr(isaac_env, attr_name, None)
            if has_rgb(cam):
                return cam

        # 4) Any scene sensor with rgb output
        if hasattr(isaac_env, "scene") and hasattr(isaac_env.scene, "sensors"):
            sensors = isaac_env.scene.sensors
            for _, sensor in sensors.items():
                if has_rgb(sensor):
                    return sensor

        return None

    # ------------------------------------------------------------------
    # Convenience properties expected by resfit code
    # ------------------------------------------------------------------

    @property
    def fps(self):
        return 20  # matches GR00T control rate

    def render(self) -> np.ndarray:
        """Return (num_envs, H, W, 3) uint8 array for video recording."""
        cam = self._find_rgb_camera(preferred_key="camera_front")
        if cam is not None and hasattr(cam, "data") and hasattr(cam.data, "output"):
            rgb = cam.data.output.get("rgb", None)
            if rgb is not None:
                img = rgb
                if img.dtype != torch.uint8:
                    if img.max() <= 1.0:
                        img = (img.clamp(0, 1) * 255).to(torch.uint8)
                    else:
                        img = img.clamp(0, 255).to(torch.uint8)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                return img.cpu().numpy()
        return np.zeros((self.num_envs, 480, 640, 3), dtype=np.uint8)

    @property
    def env_name(self):
        return "FrankaPickup"

    @property
    def camera_size(self):
        return self.image_size[0]

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying IsaacLab env."""
        return getattr(self.env, name)


# ======================================================================
# Factory
# ======================================================================

def create_isaaclab_env(
    task: str = "Isaac-Franka-Pickup-Direct-v0",
    num_envs: int = 1,
    enable_cameras: bool = True,
    device: str = "cuda:0",
    image_size: tuple[int, int] = (84, 84),
    extra_cfg_overrides: dict | None = None,
) -> IsaacLabVecEnvWrapper:
    """Create an IsaacLab env wrapped for resfit.

    This must be called from within a script launched via ``isaaclab.sh -p``
    so that Isaac Sim is already initialised.

    Parameters
    ----------
    task : str
        Gymnasium task ID registered by isaaclab_tasks (e.g. ``Isaac-Franka-Pickup-Direct-v0``).
    num_envs : int
        Number of parallel environments.
    enable_cameras : bool
        Whether to enable camera sensors (required for image observations).
    device : str
        Torch device for tensors.
    image_size : tuple
        (H, W) target resolution for camera images.
    extra_cfg_overrides : dict | None
        Additional env config overrides (e.g. ``{"use_api_for_pose": True}``).

    Returns
    -------
    IsaacLabVecEnvWrapper
    """
    import gymnasium as gym  # noqa: F811 — needed at call-time after AppLauncher

    t0 = time.time()
    print(
        f"[isaaclab_env_wrapper] create_isaaclab_env start task={task} num_envs={num_envs} device={device}",
        flush=True,
    )

    # isaaclab_tasks must have been imported (which triggers gym.register) before this call.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Parse the env config from the gymnasium registry (handles cfg entry point resolution)
    print("[isaaclab_env_wrapper] parsing env cfg...", flush=True)
    env_cfg = parse_env_cfg(
        task,
        num_envs=num_envs,
        device=device,
        use_fabric=True,
        enable_cameras=enable_cameras,
    )
    print(f"[isaaclab_env_wrapper] env cfg parsed in {time.time() - t0:.2f}s", flush=True)

    # Apply extra config overrides if any
    if extra_cfg_overrides:
        print(f"[isaaclab_env_wrapper] applying overrides: {sorted(extra_cfg_overrides.keys())}", flush=True)
        for k, v in extra_cfg_overrides.items():
            if hasattr(env_cfg, k):
                setattr(env_cfg, k, v)

    # Build the env via gymnasium registry with the resolved config
    t_make = time.time()
    print("[isaaclab_env_wrapper] gym.make start...", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print(f"[isaaclab_env_wrapper] gym.make done in {time.time() - t_make:.2f}s", flush=True)

    # Use the gym-wrapped env (not .unwrapped) so reset/step go through
    # the proper IsaacLab DirectRLEnv lifecycle
    print(f"[isaaclab_env_wrapper] create_isaaclab_env done total={time.time() - t0:.2f}s", flush=True)
    return IsaacLabVecEnvWrapper(env=env, image_size=image_size, device=device)
