"""Rollout GR00T base policy (residual=0) and save video for visual verification."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import cv2

torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def get_frame_generic(vec_env, env_id, camera="back", size=(360, 640)):
    """Get a rendered frame from any vec env, falling back to available renderers."""
    # Try get_frame first
    if hasattr(vec_env, "get_frame"):
        try:
            return vec_env.get_frame(env_id, camera=camera, size=size)
        except Exception:
            pass

    # Fallback: use renderer_base / renderer_front directly
    import cv2
    env = vec_env._envs[env_id]

    # Sync qpos if parallel
    if getattr(vec_env, "_parallel", False) and getattr(vec_env, "_needs_qpos_sync", False):
        if hasattr(vec_env, "_sync_qpos_all"):
            vec_env._sync_qpos_all(vec_env._envs, vec_env.num_envs)
        elif hasattr(vec_env, "_sync_qpos_from_workers"):
            vec_env._sync_qpos_from_workers()
        vec_env._needs_qpos_sync = False

    # Pick best available renderer
    renderer = None
    cam_id = None
    opt = None
    if camera == "back" and "renderer_base" in env:
        renderer = env["renderer_base"]
        cam_id = env.get("cam_base_id")
        opt = env.get("opt_base")
    elif camera == "front":
        # Use high-res renderer_base with front camera (renderer_front is only 84x84)
        if "renderer_base" in env and "front_cam_id" in env:
            renderer = env["renderer_base"]
            cam_id = env["front_cam_id"]
        elif "renderer_front" in env:
            renderer = env["renderer_front"]
            cam_id = env.get("front_cam_id")
    elif camera == "wrist" and "renderer_wrist" in env:
        renderer = env["renderer_wrist"]
        cam_id = env.get("cam_wrist_id")
        opt = env.get("opt_wrist")

    # Fallback to whatever renderer exists
    if renderer is None:
        for key in ("renderer_base", "renderer_front", "renderer_wrist"):
            if key in env:
                renderer = env[key]
                cam_key = key.replace("renderer_", "cam_") + "_id"
                if cam_key not in env:
                    cam_key = key.replace("renderer_", "") + "_cam_id"
                cam_id = env.get(cam_key, env.get("cam_base_id"))
                opt = env.get("opt_base") if "base" in key else None
                break

    if renderer is None:
        return np.zeros((*size, 3), dtype=np.uint8)

    kw = {"scene_option": opt} if opt else {}
    renderer.update_scene(env["data"], camera=cam_id, **kw)
    frame = renderer.render().copy()
    if size is not None:
        frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return frame


def rollout_base_policy(env, num_episodes, video_dir, camera="back", frame_size=(360, 640)):
    """Run base-policy-only rollout and save per-env videos."""
    num_envs = env.vec_env.num_envs
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_eps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        ep_frames = [[] for _ in range(num_envs)]

        for step_i in range(env.vec_env.max_episode_steps):
            for ei in range(num_envs):
                if not env_done[ei]:
                    frame = get_frame_generic(env.vec_env, ei, camera=camera, size=frame_size)
                    ep_frames[ei].append(frame)

            # Zero residual → base policy only
            action = torch.zeros(num_envs, env.action_dim, device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for ei in range(num_envs):
                if not env_done[ei] and done[ei]:
                    env_done[ei] = True
                    if terminated[ei]:
                        env_success[ei] = True

            if all(env_done):
                break

        # Save videos
        if imageio is not None:
            for ei in range(num_envs):
                if ep_frames[ei]:
                    status = "ok" if env_success[ei] else "fail"
                    vid_path = video_dir / f"env{ei:02d}_ep{ep}_{status}.mp4"
                    imageio.mimwrite(str(vid_path), ep_frames[ei], fps=30)
                    print(f"  Saved {vid_path.name} ({len(ep_frames[ei])} frames)")

        n_succ = sum(env_success)
        total_success += n_succ
        total_eps += num_envs
        print(f"  Episode {ep}: {n_succ}/{num_envs} success")

    print(f"\nTotal: {total_success}/{total_eps} ({total_success/max(total_eps,1):.0%})")


def create_env(task, groot_checkpoint, device, num_envs=2, max_episode_steps=500):
    """Create task env + residual wrapper for base policy rollout."""
    from resfit.rl_finetuning.configs.task_configs import get_task_config
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified

    cfg = get_task_config(task)
    repo_root = Path(__file__).resolve().parents[1]

    # Import vec env class
    import importlib
    mod = importlib.import_module(cfg.vec_env_module)
    VecEnvClass = getattr(mod, cfg.vec_env_class)

    # Task-specific env creation
    if task == "lift":
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            cube_positions=[[0.45, -0.05, 0.02]] * num_envs,
            max_episode_steps=max_episode_steps,
            reward_type="dense",
            device=device,
        )

    elif task == "cup":
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            episode_ids=list(range(num_envs)),
            cup_positions_path=str(repo_root / "configs" / "cup_positions.json"),
            max_episode_steps=max_episode_steps,
            reward_type="dense",
            device=device,
        )

    elif task == "pnp":
        pos_file = str(repo_root / "configs" / "pnp_66ep_positions.json")
        scene_xml = os.path.expanduser(
            "~/Repos/Intern/Mujoco_Franka/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
        )
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            cube_positions=[[0.42, -0.03, 0.02]] * num_envs,
            scene_xml=scene_xml,
            episode_positions_file=pos_file,
            max_episode_steps=max_episode_steps,
            reward_type="dense_v3",
            device=device,
        )

    elif task == "stack":
        with open(repo_root / "configs" / "stack_positions.json") as f:
            pos_data = json.load(f)
        white_pos = [p["white_cube_pos"] for p in pos_data[:num_envs]]
        green_pos = [p["green_cube_pos"] for p in pos_data[:num_envs]]
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            white_cube_positions=white_pos,
            green_cube_positions=green_pos,
            max_episode_steps=max_episode_steps,
            reward_type="dense",
            device=device,
        )

    elif task == "drawer":
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            reward_type="dense",
            device=device,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    # Wrap with GR00T
    env = MuJoCoResidualWrapperUnified(
        vec_env=mujoco_env,
        groot_checkpoint=groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=device,
        task_description=cfg.task_description,
        open_loop_horizon=16,
        chunk_sync=True,
        grip_min=cfg.grip_min,
        grip_max=cfg.grip_max,
        use_gripper_latch=cfg.use_gripper_latch,
        camera_keys=cfg.camera_keys,
        residual_pos_scale=cfg.residual_pos_scale,
        residual_rot_scale=cfg.residual_rot_scale,
        residual_grip_scale=cfg.residual_grip_scale,
        render_parallel=False,  # EGL contexts are thread-affine
        async_prefetch=False,   # async would race with get_frame() rendering
    )

    return env


def main():
    p = argparse.ArgumentParser(description="Rollout GR00T base policy and save video")
    p.add_argument("--task", type=str, required=True, choices=["lift", "cup", "pnp", "stack", "drawer"])
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--num_episodes", type=int, default=2)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--camera", type=str, default=None,
                   help="Camera to use (default: auto per task)")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    # Default camera per task
    camera = args.camera
    if camera is None:
        camera = "back"  # cam_base for all tasks

    print(f"Task: {args.task}, GR00T: {args.groot_checkpoint}")
    print(f"Envs: {args.num_envs}, Episodes: {args.num_episodes}, Camera: {camera}")

    t0 = time.time()
    env = create_env(
        args.task, args.groot_checkpoint, args.device,
        num_envs=args.num_envs, max_episode_steps=args.max_episode_steps,
    )
    print(f"Env created in {time.time()-t0:.1f}s")

    rollout_base_policy(env, args.num_episodes, args.video_dir, camera=camera)
    env.close()
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
