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

import os
import time
import traceback

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


_LIFT_REWARD_THRESHOLD_M = 0.01


def _hide_overlapping_ground_prims() -> None:
    """Hide common default ground prims to avoid floor overlap.

    Some task setups instantiate both a terrain plane (e.g. ``/World/ground``)
    and a custom floor mesh/cuboid (e.g. ``/World/Environment/floor``), which
    causes doubled-floor visuals in renders/videos.
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom

        stage = get_current_stage()
        hidden = []
        explicit_roots = {
            "/World/ground",
            "/World/defaultGroundPlane",
            "/World/GroundPlane",
            "/World/terrain",
        }

        def _is_ground_like(path: str) -> bool:
            p = path.lower()
            if p.startswith("/world/environment/floor"):
                return False
            if path in explicit_roots:
                return True
            return any(tok in p for tok in ("ground", "groundplane", "terrain"))

        for prim in stage.TraverseAll():
            try:
                path = str(prim.GetPath())
            except Exception:
                continue
            if not _is_ground_like(path):
                continue
            try:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                    hidden.append(path)
            except Exception:
                continue

        if hidden:
            print(
                f"[isaaclab_env_wrapper] hid overlapping ground prims: {sorted(set(hidden))}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[isaaclab_env_wrapper][warn] failed to hide overlapping ground prims: {exc}",
            flush=True,
        )


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
            obs_spaces[f"{resfit_key}_hires"] = gym.spaces.Box(
                low=0, high=255,
                shape=(self.num_envs, 3, 480, 640),
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
        self._ground_hide_attempted_after_reset = False
        self._initial_cube_z: torch.Tensor | None = None
        self._lift_reward_threshold_m = _LIFT_REWARD_THRESHOLD_M

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset all environments and return augmented obs dict."""
        obs_raw, info = self.env.reset()
        camera_warmup_steps = int(os.environ.get("RESFIT_IFACE_CAMERA_WARMUP_STEPS", "0") or 0)
        settle_steps = int(os.environ.get("RESFIT_RESET_SETTLE_STEPS", "0") or 0)
        total_settle_steps = max(settle_steps, camera_warmup_steps)
        if total_settle_steps > 0:
            zero_action = torch.zeros(
                (self.num_envs, 7),
                device=self._unwrapped.device,
                dtype=torch.float32,
            )
            sim_dt = None
            try:
                sim_dt = float(self._unwrapped.sim.get_physics_dt())
            except Exception:
                sim_dt = None

            for step_idx in range(total_settle_steps):
                if step_idx < camera_warmup_steps:
                    try:
                        if hasattr(self._unwrapped, "_setup_cameras"):
                            self._unwrapped._setup_cameras()
                        for cam_name in ("_camera_front", "_camera_back", "_camera_wrist"):
                            cam = getattr(self._unwrapped, cam_name, None)
                            if cam is not None and hasattr(cam, "update") and sim_dt is not None:
                                cam.update(dt=sim_dt)
                    except Exception:
                        pass
                obs_raw, _, _, _, _ = self.env.step(zero_action)
        self._update_initial_cube_z()
        if not self._ground_hide_attempted_after_reset:
            _hide_overlapping_ground_prims()
            self._ground_hide_attempted_after_reset = True
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

        rewards = self._compute_sparse_lift_reward(default_rewards=rewards)

        done = terminated | truncated
        if done.any():
            self._update_initial_cube_z(done_mask=done)

        return obs_dict, rewards, terminated, truncated, info

    def _read_cube_z(self) -> torch.Tensor | None:
        cube = getattr(self._unwrapped, "cube", None)
        if cube is None and hasattr(self._unwrapped, "scene") and hasattr(self._unwrapped.scene, "rigid_objects"):
            rigid_objects = self._unwrapped.scene.rigid_objects
            if isinstance(rigid_objects, dict):
                cube = rigid_objects.get("cube", None)
        if cube is None:
            return None
        if not hasattr(cube, "data") or not hasattr(cube.data, "root_state_w"):
            return None

        root_state_w = cube.data.root_state_w
        if not isinstance(root_state_w, torch.Tensor) or root_state_w.ndim < 2 or root_state_w.shape[1] < 3:
            return None

        cube_z = root_state_w[:, 2]
        if cube_z.shape[0] != self.num_envs:
            return None
        return cube_z.to(dtype=torch.float32)

    def _update_initial_cube_z(self, done_mask: torch.Tensor | None = None) -> None:
        cube_z = self._read_cube_z()
        if cube_z is None:
            return

        if self._initial_cube_z is None or self._initial_cube_z.shape != cube_z.shape:
            self._initial_cube_z = cube_z.detach().clone()
            return

        if done_mask is None:
            self._initial_cube_z = cube_z.detach().clone()
            return

        mask = done_mask.to(device=cube_z.device, dtype=torch.bool)
        if mask.any():
            self._initial_cube_z[mask] = cube_z[mask].detach()

    def _compute_sparse_lift_reward(self, default_rewards: torch.Tensor) -> torch.Tensor:
        cube_z = self._read_cube_z()
        if cube_z is None:
            return default_rewards

        if self._initial_cube_z is None or self._initial_cube_z.shape != cube_z.shape:
            self._initial_cube_z = cube_z.detach().clone()

        lift_delta = cube_z - self._initial_cube_z.to(device=cube_z.device)
        sparse_reward = (lift_delta >= self._lift_reward_threshold_m).to(dtype=torch.float32)
        return sparse_reward.to(device=default_rewards.device)

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

        try:
            robot = getattr(self._unwrapped, "robot", None)
            if robot is not None and hasattr(robot, "data") and hasattr(robot.data, "joint_pos"):
                joint_pos = robot.data.joint_pos.to(device=self.device, dtype=torch.float32)
                if joint_pos.dim() == 2 and joint_pos.shape[0] == self.num_envs:
                    joint_names = list(getattr(robot.data, "joint_names", []) or [])
                    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
                    arm_joint_ids = [joint_names.index(name) for name in arm_joint_names if name in joint_names]
                    if len(arm_joint_ids) == 7:
                        out["observation.raw_joint_pos"] = joint_pos[:, arm_joint_ids]
                    else:
                        out["observation.raw_joint_pos"] = joint_pos[:, :7]

                    hand_joint_names = sorted([name for name in joint_names if "panda_finger_joint" in name])
                    if len(hand_joint_names) > 0:
                        hand_joint_id = joint_names.index(hand_joint_names[0])
                        grip = torch.clamp(joint_pos[:, hand_joint_id:hand_joint_id + 1] / 0.04, 0.0, 1.0)
                        out["observation.raw_gripper_frac"] = grip
                    elif joint_pos.shape[1] >= 8:
                        grip = torch.clamp(joint_pos[:, 7:8] / 0.04, 0.0, 1.0)
                        out["observation.raw_gripper_frac"] = grip
        except Exception:
            pass

        # ── Camera images ──
        self._append_camera_obs(out)

        return out

    def _append_camera_obs(self, out: dict[str, torch.Tensor]):
        """Read cameras from the underlying IsaacLab env and add downscaled
        uint8 images to ``out`` in (N, C, H, W) format."""
        isaac_env = self._unwrapped  # DirectRLEnv

        try:
            if hasattr(isaac_env, "_setup_cameras"):
                isaac_env._setup_cameras()
            self._apply_iface_camera_views(isaac_env)
            sim_dt = None
            try:
                sim_dt = float(isaac_env.sim.get_physics_dt())
            except Exception:
                sim_dt = None
            if sim_dt is not None:
                for cam_name in ("_camera_front", "_camera_back", "_camera_wrist"):
                    cam = getattr(isaac_env, cam_name, None)
                    if cam is not None and hasattr(cam, "update"):
                        cam.update(dt=sim_dt)
        except Exception:
            pass

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
                    raw_img = img
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
                    out[f"{resfit_key}_hires"] = raw_img.to(self.device)
                    continue

            # Fallback: zeros if camera not available
            out[resfit_key] = torch.zeros(
                (self.num_envs, 3, self.image_size[0], self.image_size[1]),
                device=self.device, dtype=torch.uint8,
            )
            out[f"{resfit_key}_hires"] = torch.zeros(
                (self.num_envs, 3, 480, 640),
                device=self.device, dtype=torch.uint8,
            )

    def _apply_iface_camera_views(self, isaac_env) -> None:
        if str(os.environ.get("RESFIT_FORCE_IFACE_CAMERA_POSE", "1")).lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            scene = getattr(isaac_env, "scene", None)
            env_origins = getattr(scene, "env_origins", None)
            if env_origins is None:
                return

            cam_front = getattr(isaac_env, "_camera_front", None)
            cam_back = getattr(isaac_env, "_camera_back", None)
            if cam_front is None or cam_back is None:
                return

            origins = env_origins.to(device=self.device, dtype=torch.float32)
            eye_front = torch.tensor([0.4, -0.7, 0.8], device=self.device, dtype=torch.float32).unsqueeze(0) + origins
            eye_back = torch.tensor([0.4, 0.7, 0.8], device=self.device, dtype=torch.float32).unsqueeze(0) + origins
            target_pos = torch.tensor([0.25, 0.0, -0.05], device=self.device, dtype=torch.float32).unsqueeze(0) + origins

            if hasattr(cam_front, "set_world_poses_from_view"):
                cam_front.set_world_poses_from_view(eye_front, target_pos)
            if hasattr(cam_back, "set_world_poses_from_view"):
                cam_back.set_world_poses_from_view(eye_back, target_pos)
        except Exception:
            return

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

        # Fallback to gym env render if available
        for candidate in (self.env, self._unwrapped):
            try:
                if hasattr(candidate, "render"):
                    rendered = candidate.render()
                    if rendered is None:
                        continue
                    arr = np.asarray(rendered)
                    if arr.size == 0:
                        continue
                    # Accept (H,W,C) or (N,H,W,C)
                    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        arr = arr[None, ...]
                    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
                        if arr.dtype != np.uint8:
                            if arr.max() <= 1.0:
                                arr = np.clip(arr, 0.0, 1.0) * 255.0
                            else:
                                arr = np.clip(arr, 0.0, 255.0)
                            arr = arr.astype(np.uint8)
                        if arr.shape[-1] == 4:
                            arr = arr[..., :3]
                        return arr
            except Exception:
                continue

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

    use_fabric_env = os.environ.get("RESFIT_PARSE_USE_FABRIC", "1").strip().lower()
    use_fabric = use_fabric_env not in {"0", "false", "no", "off"}
    print(
        f"[isaaclab_env_wrapper] parse config enable_cameras={enable_cameras} use_fabric={use_fabric}",
        flush=True,
    )

    # isaaclab_tasks must have been imported (which triggers gym.register) before this call.
    t_import = time.time()
    print("[isaaclab_env_wrapper] importing isaaclab_tasks...", flush=True)
    import isaaclab_tasks  # noqa: F401
    print(f"[isaaclab_env_wrapper] imported isaaclab_tasks in {time.time() - t_import:.2f}s", flush=True)

    t_utils = time.time()
    print("[isaaclab_env_wrapper] importing parse_env_cfg...", flush=True)
    from isaaclab_tasks.utils import parse_env_cfg
    print(f"[isaaclab_env_wrapper] imported parse_env_cfg in {time.time() - t_utils:.2f}s", flush=True)

    # Parse the env config from the gymnasium registry (handles cfg entry point resolution)
    print("[isaaclab_env_wrapper] parsing env cfg...", flush=True)
    try:
        try:
            env_cfg = parse_env_cfg(
                task,
                num_envs=num_envs,
                device=device,
                use_fabric=use_fabric,
                enable_cameras=enable_cameras,
            )
        except TypeError as sig_exc:
            if "enable_cameras" not in str(sig_exc):
                raise
            print(
                "[isaaclab_env_wrapper] parse_env_cfg signature has no enable_cameras; retrying without it",
                flush=True,
            )
            env_cfg = parse_env_cfg(
                task,
                num_envs=num_envs,
                device=device,
                use_fabric=use_fabric,
            )
            if hasattr(env_cfg, "enable_cameras"):
                setattr(env_cfg, "enable_cameras", enable_cameras)
    except BaseException as exc:
        print(
            f"[isaaclab_env_wrapper][error] parse_env_cfg raised {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    print(f"[isaaclab_env_wrapper] env cfg parsed in {time.time() - t0:.2f}s", flush=True)
    try:
        cfg_attrs = dir(env_cfg)
        cam_attrs = [name for name in cfg_attrs if "camera" in name.lower() or "image" in name.lower()]
        print(f"[isaaclab_env_wrapper] env_cfg camera-like attrs: {cam_attrs[:20]}", flush=True)
        if hasattr(env_cfg, "enable_cameras"):
            print(f"[isaaclab_env_wrapper] env_cfg.enable_cameras={getattr(env_cfg, 'enable_cameras')}", flush=True)
        if hasattr(env_cfg, "scene"):
            scene = getattr(env_cfg, "scene")
            scene_attrs = [name for name in dir(scene) if "camera" in name.lower() or "image" in name.lower()]
            print(f"[isaaclab_env_wrapper] env_cfg.scene camera-like attrs: {scene_attrs[:20]}", flush=True)
    except Exception as cfg_diag_exc:
        print(f"[isaaclab_env_wrapper][warn] env_cfg diagnostics failed: {cfg_diag_exc}", flush=True)

    # Apply extra config overrides if any
    if enable_cameras:
        for cam_flag_name in ("enable_cameras", "use_camera", "use_cameras"):
            if hasattr(env_cfg, cam_flag_name):
                setattr(env_cfg, cam_flag_name, True)
                print(f"[isaaclab_env_wrapper] forced {cam_flag_name}=True", flush=True)

    if extra_cfg_overrides:
        print(f"[isaaclab_env_wrapper] applying overrides: {sorted(extra_cfg_overrides.keys())}", flush=True)
        for k, v in extra_cfg_overrides.items():
            if hasattr(env_cfg, k):
                setattr(env_cfg, k, v)

    # Build the env via gymnasium registry with the resolved config
    t_make = time.time()
    print("[isaaclab_env_wrapper] gym.make start...", flush=True)
    try:
        env = gym.make(task, cfg=env_cfg)
    except BaseException as exc:
        print(
            f"[isaaclab_env_wrapper][error] gym.make raised {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    print(f"[isaaclab_env_wrapper] gym.make done in {time.time() - t_make:.2f}s", flush=True)

    # Hide default terrain/ground prims if a custom floor is also present.
    if task == "Isaac-Franka-Pickup-Direct-v0":
        _hide_overlapping_ground_prims()

    # Use the gym-wrapped env (not .unwrapped) so reset/step go through
    # the proper IsaacLab DirectRLEnv lifecycle
    print(f"[isaaclab_env_wrapper] create_isaaclab_env done total={time.time() - t0:.2f}s", flush=True)
    return IsaacLabVecEnvWrapper(env=env, image_size=image_size, device=device)
