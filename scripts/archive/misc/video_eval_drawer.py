#!/usr/bin/env python3
"""Video eval for Drawer Residual TD3 — records front-camera mp4.

Usage (SLURM or login node):
  python scripts/video_eval_drawer.py <checkpoint> --save_dir videos/
  python scripts/video_eval_drawer.py <checkpoint> --base_only --save_dir videos/

Renders 1 env × N episodes sequentially, saves per-episode + stitched videos.
"""
import argparse, sys, os, types, importlib, time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
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
# Force flash-attention SDPA backend; disable cuDNN SDPA (needs libnvrtc JIT)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco_drawer import ResidualTD3MuJoCoDrawerConfig

DEFAULT_ACTION_MIN = [0.240120, -0.264707, 0.149373, 0.000083]
DEFAULT_ACTION_MAX = [0.524598, 0.178428, 0.346476, 0.000084]
DEFAULT_GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000"
)

VIDEO_FPS = 30


def render_front_camera(renderer, data, cam_id) -> np.ndarray:
    """Render frame from 'front' camera reusing env's existing renderer."""
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()


def save_video(frames: list, path: str, fps: int = VIDEO_FPS):
    """Save list of RGB frames as mp4."""
    import imageio
    path = str(path)
    if not path.endswith(".mp4"):
        path += ".mp4"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"[video] Saved {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Video eval for drawer")
    p.add_argument("checkpoint", type=str, help="Path to checkpoint .pt")
    p.add_argument("--base_only", action="store_true", help="Zero residual (base policy only)")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
    p.add_argument("--save_dir", type=str, default="videos/drawer", help="Output directory")
    p.add_argument("--action_scale", type=float, default=0.05)
    p.add_argument("--open_loop_horizon", type=int, default=16)
    p.add_argument("--active_drawers", type=int, nargs="+", default=[2])
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--groot_checkpoint", type=str, default=DEFAULT_GROOT_CKPT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    p.add_argument("--tag", type=str, default="", help="Tag for output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tag = args.tag or ("base_only" if args.base_only else "residual")

    print(f"[video_eval] checkpoint={args.checkpoint}", flush=True)
    print(f"[video_eval] base_only={args.base_only} episodes={args.episodes} tag={tag}", flush=True)

    # --- Environment (1 env for sequential video recording) ---
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        active_drawers=args.active_drawers,
        max_episode_steps=args.max_episode_steps,
        device=device,
        reward_type="delta",
        contact_z_gate=True,
    )

    scaler = ActionScaler(
        action_min=torch.tensor(DEFAULT_ACTION_MIN, dtype=torch.float32),
        action_max=torch.tensor(DEFAULT_ACTION_MAX, dtype=torch.float32),
        action_scale=args.action_scale,
    )

    print("[video_eval] Loading GR00T...", flush=True)
    env = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="new_embodiment",
        policy_device=str(device),
        task_description="Close the drawer",
        open_loop_horizon=args.open_loop_horizon,
        action_scaler=scaler,
    )

    # --- Agent ---
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    obj_dim = env.observation_space["observation.object_state"].shape[1]
    cfg = ResidualTD3MuJoCoDrawerConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

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
        print("[video_eval] BASE-ONLY mode: residual forced to zero", flush=True)
        def _zero_act(obs, eval_mode=True, stddev=0.0, cpu=False):
            batch = obs["observation.state"].shape[0]
            return torch.zeros(batch, 7, device=device)
        agent.act = _zero_act
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model"])
        print("[video_eval] Weights loaded", flush=True)

    agent.eval()

    # --- Reuse env's existing renderer_base for front camera (avoids EGL context conflicts) ---
    env_dict = mujoco_env._envs[0]
    model, data = env_dict["model"], env_dict["data"]
    front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    renderer = env_dict["renderer_base"]  # reuse existing renderer (640x360)

    # --- Record episodes ---
    all_frames = []
    results = []
    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _info = env.reset()
        frames = []
        ep_return = 0.0
        success = False

        # Render initial frame
        frames.append(render_front_camera(renderer, data, front_cam_id))

        for step in range(args.max_episode_steps):
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            ep_return += reward.sum().item()
            frames.append(render_front_camera(renderer, data, front_cam_id))

            if terminated[0].item():
                success = True
            if done[0].item():
                break

        results.append({"episode": ep, "success": success, "return": ep_return, "steps": step + 1})
        status = "✓" if success else "✗"
        print(f"[video_eval] ep{ep}: {status} return={ep_return:.3f} steps={step+1}", flush=True)

        # Save per-episode video
        ep_path = os.path.join(args.save_dir, f"{tag}_ep{ep}.mp4")
        save_video(frames, ep_path)
        all_frames.extend(frames)

    # Save stitched video of all episodes
    stitched_path = os.path.join(args.save_dir, f"{tag}_all_{args.episodes}ep.mp4")
    save_video(all_frames, stitched_path)

    # Summary
    n_success = sum(r["success"] for r in results)
    mean_ret = np.mean([r["return"] for r in results])
    print(f"\n[video_eval] === SUMMARY ({tag}) ===", flush=True)
    print(f"[video_eval] SR = {n_success}/{args.episodes} ({n_success/args.episodes:.0%})", flush=True)
    print(f"[video_eval] Mean return = {mean_ret:.3f}", flush=True)
    print(f"[video_eval] Videos saved to {args.save_dir}/", flush=True)

    env.close()


if __name__ == "__main__":
    main()
