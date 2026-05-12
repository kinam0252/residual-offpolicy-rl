"""
Unified residual environment wrapper for MuJoCo + GR00T.

Supports all tasks (cup, pnp, lift, stack) via configuration parameters.
Key features:
- chunk_sync: batched GR00T inference for all envs (2x speedup)
- Gripper latch: optional (Cup only)
- ActionScaler-compatible obs normalization
- torch.compile for GR00T inference acceleration

Usage:
    wrapper = MuJoCoResidualWrapperUnified(
        vec_env=my_vec_env,
        groot_checkpoint="...",
        task_description="Pick up the cup...",
        chunk_sync=True,
        grip_min=0.0, grip_max=0.04,  # Cup
        use_gripper_latch=True,         # Cup only
    )
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

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

# get_tcp_pose from Mujoco_Franka/src/utils.py
_MUJOCO_FRANKA_SRC = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "src")
if _MUJOCO_FRANKA_SRC not in sys.path:
    sys.path.insert(0, _MUJOCO_FRANKA_SRC)

# Lazy import to avoid EGL crash on login nodes
get_tcp_pose = None

def _lazy_get_tcp_pose():
    global get_tcp_pose
    if get_tcp_pose is None:
        try:
            from utils import get_tcp_pose as _gtp
            get_tcp_pose = _gtp
        except ImportError:
            raise ImportError("get_tcp_pose not available (Mujoco_Franka/src/utils.py)")
    return get_tcp_pose


class _FakeImageFeaturesConfig:
    """Minimal shim for code that reads base_policy.config.image_features."""

    def __init__(self, camera_keys: dict):
        self.image_features = camera_keys


# Default gripper latch thresholds (overridable per-task via constructor)
_DEFAULT_GRIP_CLOSE_LATCH_THRESH = 0.8
_DEFAULT_GRIP_OPEN_LATCH_THRESH = 0.95
_DEFAULT_GRIP_LATCH_OPEN_STEPS = 150


class MuJoCoResidualWrapperUnified:
    """Task-agnostic residual wrapper combining GR00T base policy with RL residual.

    The RL agent sees observation.base_action (7D: pos3+euler3+grip1) and outputs
    a 7D residual delta.
    """

    def __init__(
        self,
        vec_env,
        groot_checkpoint: str,
        embodiment_tag: str = "NEW_EMBODIMENT",
        policy_device: str = "cuda:0",
        task_description: str = "Perform the task.",
        action_horizon: int = 16,
        open_loop_horizon: int = 16,
        residual_pos_scale: float = 0.02,
        residual_rot_scale: float = 0.05,
        residual_grip_scale: float = 0.1,
        action_clip: float = 1.0,
        ema_alpha: float = 0.0,
        torch_compile: bool = False,
        chunk_sync: bool = False,
        async_prefetch: bool = True,
        render_parallel: bool = False,
        # Task-specific
        grip_min: float = 0.0,
        grip_max: float = 1.0,
        use_gripper_latch: bool = False,
        grip_close_latch_thresh: float | None = None,
        grip_open_latch_thresh: float | None = None,
        grip_latch_open_steps: int | None = None,
        camera_keys: dict | None = None,
        # ActionScaler for obs normalization
        action_scaler=None,
        # Object state augmentation (noise + dropout)
        obs_noise_max: float = 0.0,
        obs_dropout_prob: float = 0.0,
        disable_object_state: bool = False,
        # Vision mode
        use_images: bool = False,
        # Realistic rendering: depth-composited images with real backgrounds
        realistic: bool = False,
        realistic_task: str | None = None,
        rl_img_size: int = 84,
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
        self.grip_min = grip_min
        self.grip_max = grip_max
        self.use_gripper_latch = use_gripper_latch
        self.grip_close_latch_thresh = grip_close_latch_thresh if grip_close_latch_thresh is not None else _DEFAULT_GRIP_CLOSE_LATCH_THRESH
        self.grip_open_latch_thresh = grip_open_latch_thresh if grip_open_latch_thresh is not None else _DEFAULT_GRIP_OPEN_LATCH_THRESH
        self.grip_latch_open_steps = grip_latch_open_steps if grip_latch_open_steps is not None else _DEFAULT_GRIP_LATCH_OPEN_STEPS
        self.action_scaler = action_scaler  # for normalizing obs.base_action

        # Object state augmentation
        self.obs_noise_max = obs_noise_max
        self.obs_dropout_prob = obs_dropout_prob
        self.disable_object_state = disable_object_state
        if disable_object_state:
            print("[UnifiedWrapper] Object state DISABLED — observation.object_state will be zeroed")

        # Vision mode
        self.use_images = use_images or realistic
        self.realistic = realistic
        self._realistic_renderer = None
        self._realistic_task = realistic_task
        self._rl_img_size = rl_img_size
        if realistic:
            from resfit.rl_finetuning.rendering.realistic_renderer import RealisticImageRenderer
            assert realistic_task is not None, "realistic_task must be set when realistic=True"
            self._realistic_renderer = RealisticImageRenderer(
                task=realistic_task,
                vecenv=vec_env,
                rl_img_size=rl_img_size,
            )

        # ── Load GR00T policy ──
        self._skip_groot = str(
            os.environ.get("RESFIT_SKIP_GROOT_MODEL_LOAD", "0")
        ).lower() in {"1", "true", "yes"}

        if self._skip_groot or Gr00tPolicy is None:
            self.policy = None
            print("[UnifiedWrapper] GR00T policy SKIPPED")
        else:
            print(f"[UnifiedWrapper] Loading GR00T from {groot_checkpoint}...")
            t0 = time.time()
            tag = getattr(EmbodimentTag, embodiment_tag, EmbodimentTag.NEW_EMBODIMENT)
            self.policy = Gr00tPolicy(
                embodiment_tag=tag,
                model_path=groot_checkpoint,
                device=policy_device,
            )
            print(f"[UnifiedWrapper] GR00T loaded in {time.time() - t0:.1f}s")

            if torch_compile and hasattr(self.policy, 'model'):
                import torch as _torch
                print("[UnifiedWrapper] Applying torch.compile to GR00T model...")
                t1 = time.time()
                _torch._dynamo.config.suppress_errors = True
                self.policy.model = _torch.compile(
                    self.policy.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print(f"[UnifiedWrapper] torch.compile applied in {time.time() - t1:.1f}s")

        # Config shim
        if camera_keys is None:
            camera_keys = {"observation.images.front": None, "observation.images.wrist": None}
        self.config = _FakeImageFeaturesConfig(camera_keys)

        # Per-env chunk cache
        self._cached_chunks: list[dict[str, np.ndarray] | None] = [None] * self.num_envs
        self._chunk_idx: list[int] = [self.open_loop_horizon] * self.num_envs
        self._held_base_action = np.zeros((self.num_envs, 8), dtype=np.float32)
        self.ema_alpha = ema_alpha
        self._ema_pos: list[np.ndarray | None] = [None] * self.num_envs
        self._ema_quat: list[np.ndarray | None] = [None] * self.num_envs

        # Gripper latch state (only active if use_gripper_latch=True)
        self._grip_latched: list[bool] = [False] * self.num_envs
        self._grip_open_count: list[int] = [0] * self.num_envs

        # Chunk sync state
        self._chunk_sync = chunk_sync
        self._global_chunk_step: int = self.open_loop_horizon

        # Async prefetch state
        self._async_prefetch = async_prefetch and chunk_sync
        self._async_groot_future = None
        self._render_pool = ThreadPoolExecutor(max_workers=min(self.num_envs, 8)) if render_parallel else None
        self._async_pool = ThreadPoolExecutor(max_workers=1) if self._async_prefetch else None

        # Observation space
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
        # Drain pending async prefetch (discard result)
        if self._async_groot_future is not None:
            try:
                self._async_groot_future.result()
            except Exception:
                pass
            self._async_groot_future = None
        raw_obs, info = self.vec_env.reset(**kwargs)
        if self.realistic:
            self._inject_realistic_images(raw_obs)
        self._cached_chunks = [None] * self.num_envs
        self._chunk_idx = [self.open_loop_horizon] * self.num_envs
        self._held_base_action[:] = 0.0
        self._ema_pos = [None] * self.num_envs
        self._ema_quat = [None] * self.num_envs
        self._grip_latched = [False] * self.num_envs
        self._grip_open_count = [0] * self.num_envs
        self._global_chunk_step = self.open_loop_horizon
        base_action = self._get_base_actions(force_infer=True)
        augmented_obs = self._augment_obs(raw_obs, base_action)
        augmented_obs = self._apply_object_state_augmentation(augmented_obs)
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
        if self._chunk_sync:
            needs_infer_next = (self._global_chunk_step + 1 >= self.open_loop_horizon)
        else:
            needs_infer_next = any(
                self._cached_chunks[i] is None or self._chunk_idx[i] + 1 >= self.open_loop_horizon
                for i in range(self.num_envs)
            )
        render = "rl_only" if needs_infer_next else "none"
        if self.use_images and not self.realistic:
            render = "full"
        elif self.realistic:
            # Realistic mode: skip standard image rendering, we'll inject our own
            render = "rl_only" if needs_infer_next else "none"
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(combined_t, render_mode=render)

        # Inject realistic images
        if self.realistic:
            self._inject_realistic_images(raw_obs)

        info["scaled_action"] = torch.as_tensor(combined, device=self.device, dtype=torch.float32)

        if self._chunk_sync:
            self._global_chunk_step += 1
        else:
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
            # Drain pending async — done envs invalidate the prefetched result
            if self._async_groot_future is not None:
                try:
                    self._async_groot_future.result()
                except Exception:
                    pass
                self._async_groot_future = None
            self.vec_env.reset_envs(done_ids)
            raw_obs = self.vec_env._build_obs_dict(
                render_mode="full" if (self.use_images and not self.realistic) else "rl_only"
            )
            if self.realistic:
                self._inject_realistic_images(raw_obs)
            if self._chunk_sync:
                self._global_chunk_step = self.open_loop_horizon

        next_base_action = self._get_base_actions()
        if done.any():
            for eid in torch.where(done)[0].tolist():
                next_base_action[eid] = 0.0

        self._last_obs = raw_obs
        augmented_obs = self._augment_obs(raw_obs, next_base_action)
        augmented_obs = self._apply_object_state_augmentation(augmented_obs)

        # Start async prefetch one step BEFORE the boundary so GR00T runs
        # during the training loop (agent.update, agent.act) between step() calls.
        # Renders state from this step — 1 physics step ahead of sync baseline.
        if (self._async_prefetch and self._chunk_sync
                and self._global_chunk_step == self.open_loop_horizon - 1
                and self._async_groot_future is None):
            self._start_async_prefetch()

        return augmented_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Base policy querying
    # ------------------------------------------------------------------

    def _get_base_actions(self, force_infer: bool = False) -> np.ndarray:
        if self._chunk_sync and not force_infer:
            return self._get_base_actions_sync()

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

    def _get_base_actions_sync(self) -> np.ndarray:
        """Chunk-synced mode: all envs share the same chunk boundary."""
        actions = np.zeros((self.num_envs, 8), dtype=np.float32)

        if self._global_chunk_step >= self.open_loop_horizon:
            if self._async_groot_future is not None:
                # Pick up async result and apply atomically on main thread
                prefetched = self._async_groot_future.result()
                self._async_groot_future = None
                if prefetched is not None:
                    for eid, chunk in prefetched.items():
                        self._cached_chunks[eid] = chunk
                        self._chunk_idx[eid] = 0
            else:
                # No async result — run synchronously (first call or async disabled)
                all_ids = list(range(self.num_envs))
                self._query_groot_batch(all_ids)
            self._global_chunk_step = 0

        for i in range(self.num_envs):
            chunk = self._cached_chunks[i]
            idx = self._global_chunk_step
            if chunk is not None and idx < chunk["eef_pos"].shape[0]:
                actions[i, :3] = chunk["eef_pos"][idx]
                actions[i, 3:7] = chunk["eef_quat"][idx]
                actions[i, 7] = chunk["gripper_width"][idx, 0]
            else:
                actions[i] = self._get_current_pose(i)

        self._held_base_action = actions.copy()
        return actions

    def _start_async_prefetch(self) -> None:
        """Start GR00T render+inference async in background thread.
        Uses _query_groot_batch_core which returns results without mutating state.
        """
        if self._async_groot_future is not None:
            return  # already running
        all_ids = list(range(self.num_envs))
        self._async_groot_future = self._async_pool.submit(
            self._query_groot_batch_core, all_ids
        )

    def _get_current_pose(self, env_idx: int) -> np.ndarray:
        """Get current EEF pose as 8D action (hold position)."""
        _gtp = _lazy_get_tcp_pose()
        env = self.vec_env._envs[env_idx]
        model, data, ids = env["model"], env["data"], env["ids"]
        if hasattr(self.vec_env, '_parallel') and self.vec_env._parallel:
            if hasattr(self.vec_env, '_needs_qpos_sync') and self.vec_env._needs_qpos_sync:
                self.vec_env._sync_qpos_all(self.vec_env._envs, self.vec_env.num_envs)
                self.vec_env._needs_qpos_sync = False
        tcp_pos, tcp_R = _gtp(model, data, ids["hand_id"])
        eef_quat_xyzw = Rotation.from_matrix(tcp_R).as_quat()
        finger_id = ids["finger_ids"][0]
        gripper_width = max(0.0, float(data.qpos[model.jnt_qposadr[finger_id]])) if finger_id >= 0 else 0.04
        pose = np.zeros(8, dtype=np.float32)
        pose[:3] = tcp_pos
        pose[3:7] = eef_quat_xyzw
        pose[7] = gripper_width
        return pose

    def _query_groot_batch(self, env_ids: list[int]) -> None:
        """Run GR00T inference and write results directly to cached_chunks."""
        results = self._query_groot_batch_core(env_ids)
        if results is not None:
            for eid, chunk in results.items():
                self._cached_chunks[eid] = chunk
                self._chunk_idx[eid] = 0

    def _query_groot_batch_core(self, env_ids: list[int]) -> dict | None:
        """Core GR00T render+inference. Returns {eid: chunk_dict} without mutating state."""
        if not env_ids:
            return None

        # Pre-sync qpos before parallel rendering to avoid race on _sync_qpos_all
        if (hasattr(self.vec_env, '_parallel') and self.vec_env._parallel
                and hasattr(self.vec_env, '_needs_qpos_sync')
                and self.vec_env._needs_qpos_sync):
            self.vec_env._sync_qpos_all(self.vec_env._envs, self.vec_env.num_envs)
            self.vec_env._needs_qpos_sync = False

        if self.policy is None:
            results = {}
            for eid in env_ids:
                groot_obs = self.vec_env.get_groot_obs(eid, self.task_description)
                pos = groot_obs["state"]["proprio.eef_pos"][0, 0]
                quat = groot_obs["state"]["proprio.eef_quat"][0, 0]
                gw = groot_obs["state"]["proprio.gripper_width"][0, 0]
                results[eid] = {
                    "eef_pos": np.tile(pos, (self.action_horizon, 1)),
                    "eef_quat": np.tile(quat, (self.action_horizon, 1)),
                    "gripper_width": np.tile(gw, (self.action_horizon, 1)),
                }
            return results

        # Render all envs — parallel or sequential
        if self._render_pool is not None and len(env_ids) > 1:
            def _render_one(eid):
                return self.vec_env.get_groot_obs(eid, self.task_description)
            per_env_obs = list(self._render_pool.map(_render_one, env_ids))
        else:
            per_env_obs = [self.vec_env.get_groot_obs(eid, self.task_description) for eid in env_ids]
        batched_obs = {"video": {}, "state": {}, "language": {}}
        for vk in per_env_obs[0]["video"]:
            batched_obs["video"][vk] = np.concatenate([o["video"][vk] for o in per_env_obs], axis=0)
        for sk in per_env_obs[0]["state"]:
            batched_obs["state"][sk] = np.concatenate([o["state"][sk] for o in per_env_obs], axis=0)
        for lk in per_env_obs[0]["language"]:
            batched_obs["language"][lk] = [o["language"][lk][0] for o in per_env_obs]

        try:
            action_result, _info = self.policy.get_action(batched_obs)
        except Exception as e:
            import warnings
            warnings.warn(f"GR00T inference failed: {e}. Using zero actions for {len(env_ids)} envs.")
            horizon = self.open_loop_horizon
            results = {}
            for eid in env_ids:
                results[eid] = {
                    "eef_pos": np.zeros((horizon, 3), dtype=np.float32),
                    "eef_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (horizon, 1)),
                    "gripper_width": np.zeros((horizon, 1), dtype=np.float32),
                }
            return results
        results = {}
        for bi, eid in enumerate(env_ids):
            results[eid] = {
                "eef_pos": np.asarray(action_result["action.eef_pos"][bi], dtype=np.float32),
                "eef_quat": np.asarray(action_result["action.eef_quat"][bi], dtype=np.float32),
                "gripper_width": np.asarray(action_result["action.gripper_width"][bi], dtype=np.float32),
            }
        return results

    # ------------------------------------------------------------------
    # Action combining
    # ------------------------------------------------------------------

    def _combine_actions(self, base_action: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Combine base absolute action with residual delta. Returns (N, 8)."""
        combined = np.zeros_like(base_action)

        # Position: vectorized additive
        combined[:, :3] = base_action[:, :3] + residual[:, :3] * self.residual_pos_scale

        # Rotation: vectorized compose
        base_rots = Rotation.from_quat(base_action[:, 3:7])
        delta_rots = Rotation.from_euler("xyz", residual[:, 3:6] * self.residual_rot_scale)
        combined[:, 3:7] = (base_rots * delta_rots).as_quat()

        # Gripper: clip to range, with optional latch
        raw_grip = np.clip(
            base_action[:, 7] + residual[:, 6] * self.residual_grip_scale,
            self.grip_min, self.grip_max,
        )

        if self.use_gripper_latch:
            for i in range(self.num_envs):
                if self._grip_latched[i]:
                    if raw_grip[i] > self.grip_open_latch_thresh:
                        self._grip_open_count[i] += 1
                        if self._grip_open_count[i] >= self.grip_latch_open_steps:
                            self._grip_latched[i] = False
                            self._grip_open_count[i] = 0
                            combined[i, 7] = raw_grip[i]
                        else:
                            combined[i, 7] = self.grip_min
                    else:
                        self._grip_open_count[i] = 0
                        combined[i, 7] = self.grip_min
                else:
                    if raw_grip[i] < self.grip_close_latch_thresh:
                        self._grip_latched[i] = True
                        self._grip_open_count[i] = 0
                        combined[i, 7] = self.grip_min
                    else:
                        combined[i, 7] = raw_grip[i]
        else:
            combined[:, 7] = raw_grip

        return combined

    # ------------------------------------------------------------------
    # Realistic image injection
    # ------------------------------------------------------------------

    def reinit_realistic_renderer(self) -> None:
        """Re-create the realistic renderer (fresh EGL context).

        Call after heavy CUDA ops (e.g. critic warmup) that may corrupt the
        EGL display/context state.
        """
        if self._realistic_renderer is None:
            return
        self._realistic_renderer.close()
        from resfit.rl_finetuning.rendering.realistic_renderer import RealisticImageRenderer
        self._realistic_renderer = RealisticImageRenderer(
            task=self._realistic_task,
            vecenv=self.vec_env,
            rl_img_size=self._rl_img_size,
        )

    def _inject_realistic_images(self, obs: dict[str, torch.Tensor]) -> None:
        """Replace standard images with realistic-rendered ones in-place."""
        import torch as _torch
        for i in range(self.num_envs):
            imgs = self._realistic_renderer.sync_and_render(self.vec_env, env_idx=i)
            for cam_alias, obs_key in [
                ("back", "observation.images.back"),
                ("wrist", "observation.images.wrist"),
            ]:
                if obs_key in obs and cam_alias in imgs:
                    obs[obs_key][i] = _torch.as_tensor(
                        imgs[cam_alias], device=obs[obs_key].device
                    )

    # ------------------------------------------------------------------
    # Object state augmentation (noise + dropout)
    # ------------------------------------------------------------------

    def _apply_object_state_augmentation(
        self, obs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply noise and/or dropout to observation.object_state in-place.

        Noise:   σ ~ U(0, obs_noise_max), then obs += U(-σ, σ) per dim per env
        Dropout: with prob obs_dropout_prob, zero entire vector per env
        Disable: zero entire object_state always (no_obj ablation)
        """
        key = "observation.object_state"
        if key not in obs:
            return obs
        obj = obs[key]  # (num_envs, D), on device

        if self.disable_object_state:
            obs[key] = torch.zeros_like(obj)
            return obs

        if self.obs_noise_max > 0.0:
            sigma = np.random.uniform(0.0, self.obs_noise_max)
            noise = torch.empty_like(obj).uniform_(-sigma, sigma)
            obj = obj + noise

        if self.obs_dropout_prob > 0.0:
            mask = torch.rand(obj.shape[0], 1, device=obj.device) >= self.obs_dropout_prob
            obj = obj * mask  # zero out entire row with prob p

        obs[key] = obj
        return obs

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_action: np.ndarray) -> dict[str, torch.Tensor]:
        """Add observation.base_action (7D) to the observation dict.
        
        If action_scaler is set, pos+grip are normalized to [-1,1] and euler is zeroed
        (matching offline data normalization for consistent actor input).
        """
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

        if self.action_scaler is not None:
            # Normalize pos+grip to [-1,1], zero euler (matches offline loading)
            ba_t = torch.as_tensor(ba_7d, dtype=torch.float32)
            pg = torch.cat([ba_t[:, :3], ba_t[:, 6:7]], dim=-1)
            pg_norm = self.action_scaler.scale(pg)
            ba_t[:, :3] = pg_norm[:, :3]
            ba_t[:, 3:6] = 0.0
            ba_t[:, 6] = pg_norm[:, 3]
            out["observation.base_action"] = ba_t.to(self.device)
        else:
            out["observation.base_action"] = torch.as_tensor(ba_7d, device=self.device, dtype=torch.float32)
        return out

    # ------------------------------------------------------------------
    # Pass-through / compatibility
    # ------------------------------------------------------------------

    def close(self):
        if self._async_groot_future is not None:
            self._async_groot_future.result()
            self._async_groot_future = None
        if self._async_pool is not None:
            self._async_pool.shutdown(wait=False)
            self._async_pool = None
        if self._render_pool is not None:
            self._render_pool.shutdown(wait=False)
            self._render_pool = None
        self.vec_env.close()

    def reset_envs(self, env_ids: list[int]):
        self.vec_env.reset_envs(env_ids)
        for eid in env_ids:
            self._cached_chunks[eid] = None
            self._chunk_idx[eid] = self.open_loop_horizon

    def render(self) -> np.ndarray:
        return self.vec_env.render()

    @property
    def fps(self) -> int:
        return getattr(self.vec_env, 'fps', 30)

    def get_frame(self, env_id: int, camera: str = "front", size: tuple[int, int] = (128, 160)) -> np.ndarray:
        return self.vec_env.get_frame(env_id, camera, size)

    def set_active_env_ids(self, env_ids: list[int]) -> None:
        pass

    def _build_obs(self) -> dict[str, torch.Tensor]:
        raw_obs = self.vec_env._build_obs_dict()
        base_action = self._held_base_action.copy()
        return self._augment_obs(raw_obs, base_action)

    def __getattr__(self, name: str):
        """Pass through attribute access to vec_env for task-specific shims."""
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.vec_env, name)
