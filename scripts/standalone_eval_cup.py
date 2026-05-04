#!/usr/bin/env python3
"""Standalone eval for Stand Cup Residual TD3 checkpoints (login node).

Usage:
  python scripts/standalone_eval_cup.py <checkpoint_path> [--episodes 3] [--save_video]
  python scripts/standalone_eval_cup.py --base_only [--episodes 3] [--save_video]

Examples:
  # Eval best checkpoint with video
  python scripts/standalone_eval_cup.py outputs/.../checkpoints/best.pt --save_video --episodes 3

  # Base policy only (no residual)
  python scripts/standalone_eval_cup.py --base_only --save_video --episodes 3
"""
import argparse
import sys
import os
import types
import importlib
from pathlib import Path
from datetime import datetime

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/nas_main/.cache/huggingface")

# Stub deepspeed
if "deepspeed" not in sys.modules:
    _ds = types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__, _ds.__file__ = [], __file__
    _dz = types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _dz.Init = lambda *a, **kw: (lambda f: f)
    _ds.zero = _dz
    sys.modules["deepspeed"] = _ds
    sys.modules["deepspeed.zero"] = _dz

# Patch huggingface local path validation
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate_repo_id(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)
    _hf_val.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass

import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup, load_cup_positions
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig

DEFAULT_GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000"
)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def evaluate_with_video(
    env: MuJoCoResidualWrapperCup,
    agent: QAgent,
    num_episodes: int,
    device: torch.device,
    save_video: bool = False,
    video_dir: Path | None = None,
):
    """Run evaluation with optional per-env video capture."""
    agent.eval()
    num_envs = env.vec_env.num_envs
    successes = 0
    total_return = 0.0
    total_episodes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_returns = [0.0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        # Per-env frame buffers
        frames_per_env = [[] for _ in range(num_envs)] if save_video else None

        for step in range(env.vec_env.max_episode_steps):
            # Capture frame from each env
            if save_video:
                for i in range(num_envs):
                    ei = env.vec_env._envs[i]
                    ei["renderer_base"].update_scene(
                        ei["data"], camera=ei["cam_base_id"],
                        scene_option=ei["opt_base"]
                    )
                    frames_per_env[i].append(ei["renderer_base"].render().copy())

            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(num_envs):
                if not env_done[i]:
                    ep_returns[i] += reward[i].item()
                    if done[i]:
                        env_done[i] = True
                        if terminated[i]:
                            env_success[i] = True

            if all(env_done):
                break

        # Save per-env videos
        if save_video and video_dir is not None:
            for i in range(num_envs):
                if frames_per_env[i]:
                    tag = "success" if env_success[i] else "fail"
                    _save_video(frames_per_env[i],
                                video_dir / f"env{i:02d}_{tag}.mp4")

        for i in range(num_envs):
            if env_success[i]:
                successes += 1
            total_return += ep_returns[i]
            total_episodes += 1

        sr_so_far = successes / total_episodes
        print(f"  [{_ts()}] ep={ep} envs_success={sum(env_success)}/{num_envs} "
              f"cumulative_SR={sr_so_far:.0%}", flush=True)

    agent.train()
    final_sr = successes / max(1, total_episodes)
    mean_ret = total_return / max(1, total_episodes)
    return {"success_rate": final_sr, "mean_return": mean_ret,
            "total_episodes": total_episodes, "successes": successes}


def _save_video(frames: list, path: Path, fps: int = 30):
    """Save frames as mp4 using imageio."""
    try:
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                    output_params=["-crf", "23"])
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  Video saved: {path} ({len(frames)} frames)", flush=True)
    except ImportError:
        print("  [WARN] imageio not installed, skipping video save", flush=True)
    except Exception as e:
        print(f"  [WARN] Video save failed: {e}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Standalone Cup eval (login node)")
    p.add_argument("checkpoint", type=str, nargs="?", default=None,
                   help="Path to checkpoint .pt file (omit for --base_only)")
    p.add_argument("--base_only", action="store_true",
                   help="Zero out residual (eval GR00T base policy only)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Eval episodes (each runs all envs)")
    p.add_argument("--num_envs", type=int, default=20,
                   help="Number of envs (default: 20, same as training eval)")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--action_scale", type=float, default=0.1)
    p.add_argument("--open_loop_horizon", type=int, default=16)
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.004)
    p.add_argument("--groot_checkpoint", type=str, default=DEFAULT_GROOT_CKPT)
    p.add_argument("--cup_positions_file", type=str, default=None)
    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--critic_hidden_dim", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_video", action="store_true", help="Save mp4 videos")
    p.add_argument("--video_dir", type=str, default=None,
                   help="Video output dir (default: ./eval_videos_cup/)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.base_only and args.checkpoint is None:
        print("ERROR: provide checkpoint path or use --base_only", flush=True)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[cup-eval] mode={'BASE_ONLY' if args.base_only else 'RESIDUAL'}", flush=True)
    if not args.base_only:
        print(f"[cup-eval] checkpoint={args.checkpoint}", flush=True)
    print(f"[cup-eval] action_scale={args.action_scale} device={device}", flush=True)

    # --- Cup positions ---
    cup_data = load_cup_positions(args.cup_positions_file)
    num_envs = min(args.num_envs, len(cup_data))
    episode_ids = list(range(num_envs))
    print(f"[cup-eval] {num_envs} cup positions", flush=True)

    # --- Environment ---
    print(f"[cup-eval] Creating MuJoCoVecEnvCup...", flush=True)
    mujoco_env = MuJoCoVecEnvCup(
        num_envs=num_envs,
        episode_ids=episode_ids,
        cup_positions_path=args.cup_positions_file,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        device=args.device,
    )

    print(f"[cup-eval] Loading GR00T from {args.groot_checkpoint}...", flush=True)
    env = MuJoCoResidualWrapperCup(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=str(device),
        task_description="Pick up the cup and stand it upright.",
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=args.residual_pos_scale,
        residual_rot_scale=args.residual_rot_scale,
        residual_grip_scale=args.residual_grip_scale,
        chunk_sync=True,
    )
    print(f"[cup-eval] Environment ready.", flush=True)

    # --- Agent ---
    object_state_dim = env.observation_space["observation.object_state"].shape[1]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(3, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=[],
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=False,
    )

    if args.base_only:
        print("[cup-eval] BASE-ONLY: residual forced to zero", flush=True)
        def _zero_act(obs, eval_mode=True, stddev=0.0, cpu=False):
            batch = obs["observation.state"].shape[0]
            return torch.zeros(batch, action_dim, device=device)
        agent.act = _zero_act
    else:
        print(f"[cup-eval] Loading checkpoint...", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            agent.load_state_dict(ckpt["model"])
        else:
            agent.load_state_dict(ckpt)
        print(f"[cup-eval] Weights loaded.", flush=True)

    # --- Video dir ---
    video_dir = None
    if args.save_video:
        video_dir = Path(args.video_dir) if args.video_dir else Path("eval_videos_cup")
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[cup-eval] Videos → {video_dir}/", flush=True)

    # --- Run ---
    print(f"[cup-eval] Running {args.episodes} episodes × {num_envs} envs...", flush=True)
    metrics = evaluate_with_video(
        env=env, agent=agent,
        num_episodes=args.episodes,
        device=device,
        save_video=args.save_video,
        video_dir=video_dir,
    )

    sr = metrics["success_rate"]
    ret = metrics["mean_return"]
    print(f"\n{'='*50}", flush=True)
    print(f"[RESULT] SR={sr:.0%} ({metrics['successes']}/{metrics['total_episodes']}) "
          f"mean_return={ret:.3f}", flush=True)
    print(f"{'='*50}", flush=True)

    env.close()
    print("[cup-eval] Done.", flush=True)


if __name__ == "__main__":
    main()
