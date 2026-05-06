#!/usr/bin/env python3
"""Video eval for Stack Cube Residual TD3 — records per-env front-camera videos.

Usage:
  python scripts/video_eval_stack.py <checkpoint> --save_dir videos/stack/
  python scripts/video_eval_stack.py <checkpoint> --base_only --save_dir videos/stack/

Records 20 envs × 1 episode, saves per-env videos named with SUCCESS/FAIL.
"""
import sys, os, types, importlib, json
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/nas_main/.cache/huggingface")

if "deepspeed" not in sys.modules:
    _ds = types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__, _ds.__file__ = [], __file__
    _dz = types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    sys.modules["deepspeed"] = _ds
    sys.modules["deepspeed.zero"] = _dz

import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

import numpy as np
import mujoco
import imageio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig

DEFAULT_GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000"
)
DEFAULT_POSITIONS = "configs/stack_cube_positions.json"
VIDEO_FPS = 30


def render_env_front(env_dict, front_cam_id):
    """Render front camera frame from one env using its existing renderer_base (640x360)."""
    renderer = env_dict["renderer_base"]
    renderer.update_scene(env_dict["data"], camera=front_cam_id)
    return renderer.render().copy()


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Video eval for Stack Cube")
    p.add_argument("checkpoint", type=str, help="Path to checkpoint .pt")
    p.add_argument("--base_only", action="store_true", help="Zero residual (base policy only)")
    p.add_argument("--num_envs", type=int, default=20)
    p.add_argument("--save_dir", type=str, default="videos/stack")
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--groot_checkpoint", type=str, default=DEFAULT_GROOT_CKPT)
    p.add_argument("--positions_file", type=str, default=DEFAULT_POSITIONS)
    p.add_argument("--open_loop_horizon", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tag", type=str, default="", help="Tag for output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tag = args.tag or ("base_only" if args.base_only else "residual")

    # Load checkpoint to get training args
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    action_scale = train_args.get("action_scale", 0.1)
    actor_hidden_dim = train_args.get("actor_hidden_dim", 256)
    critic_hidden_dim = train_args.get("critic_hidden_dim", 256)
    reward_type = train_args.get("reward_type", "sparse")
    use_action_scaler = train_args.get("use_action_scaler", True)
    no_action_clamp = train_args.get("no_action_clamp", True)

    print(f"[video_eval] checkpoint={args.checkpoint}", flush=True)
    print(f"[video_eval] base_only={args.base_only} num_envs={args.num_envs} tag={tag}", flush=True)
    print(f"[video_eval] action_scale={action_scale} reward_type={reward_type}", flush=True)

    # Load positions
    with open(args.positions_file) as f:
        pos_list = json.load(f)
    seen = set()
    unique_pos = []
    for pp in pos_list:
        if pp["episode"] not in seen:
            seen.add(pp["episode"])
            unique_pos.append(pp)
    n = min(args.num_envs, len(unique_pos))
    white_positions = [pp["white_cube_pos"] for pp in unique_pos[:n]]
    green_positions = [pp["green_cube_pos"] for pp in unique_pos[:n]]

    # Auto-detect calib
    calib_path = None
    c = str(Path(__file__).resolve().parents[2] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
    if Path(c).exists():
        calib_path = c

    # --- Environment ---
    print(f"[video_eval] Creating {n} envs...", flush=True)
    mujoco_env = MuJoCoVecEnvStack(
        num_envs=n,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        calib_path=calib_path,
        max_episode_steps=args.max_episode_steps,
        reward_type=reward_type,
        device=args.device,
    )

    # Build ActionScaler if training used it
    _action_scaler = None
    if use_action_scaler and not args.base_only:
        # Stats from offline_stack_66ep (18807 samples, pos_x/pos_y/pos_z/grip)
        # Original range: [-0.0658, 0.4639], expanded by 1.5 IQR → [-0.0804, 0.4922]
        _stat_min = [0.4106, -0.0658, 0.0092, 0.0196]
        _stat_max = [0.4639, 0.2246, 0.1185, 0.0398]
        _action_scaler = ActionScaler(
            action_min=torch.tensor(_stat_min, dtype=torch.float32),
            action_max=torch.tensor(_stat_max, dtype=torch.float32),
            action_scale=action_scale,
            no_clamp=no_action_clamp,
        )
        print(f"[video_eval] ActionScaler created (action_scale={action_scale})", flush=True)

    print("[video_eval] Loading GR00T...", flush=True)
    if args.base_only:
        env = MuJoCoResidualWrapperStack(
            vec_env=mujoco_env,
            groot_checkpoint=args.groot_checkpoint,
            embodiment_tag="NEW_EMBODIMENT",
            policy_device=args.device,
            task_description="Pick up the white cube and stack it on the green cube.",
            open_loop_horizon=args.open_loop_horizon,
            residual_pos_scale=0.0,
            residual_rot_scale=0.0,
            residual_grip_scale=0.0,
            ema_alpha=0.0,
            action_scaler=None,
        )
    else:
        env = MuJoCoResidualWrapperStack(
            vec_env=mujoco_env,
            groot_checkpoint=args.groot_checkpoint,
            embodiment_tag="NEW_EMBODIMENT",
            policy_device=args.device,
            task_description="Pick up the white cube and stack it on the green cube.",
            open_loop_horizon=args.open_loop_horizon,
            action_scaler=_action_scaler,
        )

    # --- Agent ---
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    obj_dim = env.observation_space.get("observation.object_state", None)
    obj_dim = obj_dim.shape[1] if obj_dim is not None else 0

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = action_scale
    cfg.agent.actor.hidden_dim = actor_hidden_dim
    cfg.agent.critic.hidden_dim = critic_hidden_dim

    agent = QAgent(
        obs_shape=(3, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=7,
        rl_cameras=[],
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=obj_dim,
        asymmetric_critic=False,
    )

    if args.base_only:
        print("[video_eval] BASE-ONLY mode: zero residual via wrapper", flush=True)
    else:
        agent.load_state_dict(ckpt["model"])
        print("[video_eval] Weights loaded", flush=True)

    agent.eval()

    # --- Get front camera IDs ---
    front_cam_ids = []
    for env_dict in mujoco_env._envs:
        fid = mujoco.mj_name2id(env_dict["model"], mujoco.mjtObj.mjOBJ_CAMERA, "front")
        front_cam_ids.append(fid)

    # --- Run eval with video recording ---
    os.makedirs(args.save_dir, exist_ok=True)
    obs, _ = env.reset()
    env_done = [False] * n
    env_success = [False] * n
    env_steps = [0] * n
    env_returns = [0.0] * n
    frames = [[] for _ in range(n)]

    # Render initial frames
    for i in range(n):
        frames[i].append(render_env_front(mujoco_env._envs[i], front_cam_ids[i]))

    for step in range(args.max_episode_steps):
        with torch.no_grad():
            if args.base_only:
                action = torch.zeros((n, env.action_dim), device=device)
            else:
                action = agent.act(obs, eval_mode=True, stddev=0.0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        for i in range(n):
            if not env_done[i]:
                env_returns[i] += reward[i].item()
                env_steps[i] = step + 1
                frames[i].append(render_env_front(mujoco_env._envs[i], front_cam_ids[i]))
                if done[i]:
                    env_done[i] = True
                    if terminated[i]:
                        env_success[i] = True

        if all(env_done):
            break

        if (step + 1) % 50 == 0:
            done_count = sum(env_done)
            succ_count = sum(env_success)
            print(f"[video_eval] step={step+1}: {done_count}/{n} done, {succ_count} success so far", flush=True)

    # Save per-env videos
    for i in range(n):
        status = "SUCCESS" if env_success[i] else "FAIL"
        video_path = os.path.join(args.save_dir, f"{tag}_env{i:02d}_{status}.mp4")
        if len(frames[i]) > 0:
            writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264",
                                        quality=8, pixelformat="yuv420p")
            for frame in frames[i]:
                writer.append_data(frame)
            writer.close()

    # Summary
    n_success = sum(env_success)
    mean_ret = np.mean(env_returns)
    print(f"\n[video_eval] === SUMMARY ({tag}) ===", flush=True)
    print(f"[video_eval] SR = {n_success}/{n} ({n_success/n:.0%})", flush=True)
    print(f"[video_eval] Mean return = {mean_ret:.3f}", flush=True)
    for i in range(n):
        s = "✓" if env_success[i] else "✗"
        print(f"  env{i:02d}: {s} ret={env_returns[i]:.3f} steps={env_steps[i]}", flush=True)
    print(f"[video_eval] Videos saved to {args.save_dir}/", flush=True)

    env.close()


if __name__ == "__main__":
    main()
