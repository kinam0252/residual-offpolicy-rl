"""
Async eval process for MuJoCo Residual TD3.

Runs as a separate subprocess, polling a checkpoint directory for
eval_request_step*.pt files. When found, loads weights into its own
QAgent, runs eval episodes in its own MuJoCo env, and writes results
as JSON files for the training process to collect.

Usage (launched automatically by training script):
    python eval_async_mujoco.py \
        --checkpoint_dir /path/to/checkpoints \
        --output_dir /path/to/async_eval_results \
        --groot_checkpoint /path/to/groot \
        ...
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types as _types
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Deepspeed mock ──
if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

# ── Patch huggingface_hub repo_id validator ──
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

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

try:
    import wandb
except ImportError:
    wandb = None


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str):
    print(f"[{_ts()}] [async-eval] {msg}", flush=True)


def evaluate(
    env: MuJoCoResidualWrapper,
    agent: QAgent,
    num_episodes: int,
    device: torch.device,
    image_keys: list[str],
    save_video: bool = False,
    video_dir: Path | None = None,
    eval_step: int = 0,
    obs_noise_fn=None,
) -> dict[str, float]:
    """Run evaluation episodes with greedy policy (stddev=0).

    Evaluates ALL envs in the vec_env simultaneously. Each env runs
    independently until done; results are aggregated across all envs.
    """
    agent.eval()
    num_envs = env.vec_env.num_envs
    successes = 0
    total_return = 0.0
    total_steps = 0
    total_episodes = 0
    per_env_sr = [0] * num_envs

    # Run num_episodes rounds; each round evaluates all envs in parallel
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_returns = [0.0] * num_envs
        ep_steps_arr = [0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        frames = [] if save_video else None

        for step in range(env.vec_env.max_episode_steps):
            if save_video:
                frame = env.vec_env.get_frame(0, camera="front", size=(240, 320))
                frames.append(frame)

            # Check success BEFORE step (for envs that terminated last step
            # and got auto-reset — we already captured them). For first step
            # this is a no-op since no env is done yet.

            with torch.no_grad():
                if obs_noise_fn is not None:
                    obs = obs_noise_fn(obs)
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Check cube height BEFORE auto-reset happens (env.step auto-resets)
            # We need to read cube_z for envs that will terminate THIS step.
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated | truncated
            for i in range(num_envs):
                if not env_done[i]:
                    ep_returns[i] += reward[i].item()
                    ep_steps_arr[i] += 1
                    if done[i]:
                        env_done[i] = True
                        # terminated=True means _is_success was True (lift >= 4cm)
                        if terminated[i]:
                            env_success[i] = True

            if all(env_done):
                break

        # Save video for this episode (env 0)
        if save_video and frames and imageio is not None and video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            vid_path = video_dir / f"eval_step{eval_step}_ep{ep}.mp4"
            imageio.mimwrite(str(vid_path), frames, fps=30)
            _log(f"  Video saved: {vid_path}")

        # Count successes (captured at termination time, before auto-reset)
        for i in range(num_envs):
            if env_success[i]:
                successes += 1
                per_env_sr[i] += 1
            total_return += ep_returns[i]
            total_steps += ep_steps_arr[i]
            total_episodes += 1

    agent.train()
    metrics = {
        "eval/success_rate": successes / max(1, total_episodes),
        "eval/mean_return": total_return / max(1, total_episodes),
        "eval/mean_steps": total_steps / max(1, total_episodes),
    }
    # Per-env success rate for debugging
    for i in range(num_envs):
        metrics[f"eval/sr_env{i}"] = per_env_sr[i] / max(1, num_episodes)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Async eval for MuJoCo Residual TD3")
    # Checkpoint polling
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--poll_interval_sec", type=float, default=10.0)
    # Env
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--cube_pos", type=float, nargs=3, default=[0.45, -0.05, 0.02])
    p.add_argument("--perturb_table", type=str, default=None)
    p.add_argument("--random_cube_range", type=str, default=None,
                   help="JSON string: {dx: [lo,hi], dy: [lo,hi], yaw: [lo,hi]} in cm/degrees")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--success_threshold", type=float, default=0.005)
    p.add_argument("--reward_type", type=str, default="dense_clipped")
    p.add_argument("--device", type=str, default="cuda")
    # GR00T
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="gr1")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str, default="pick up the cube")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.5)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--action_scale", type=float, default=0.1)
    # Agent
    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--critic_hidden_dim", type=int, default=256)
    p.add_argument("--asymmetric_critic", action="store_true")
    # Eval
    p.add_argument("--eval_num_episodes", type=int, default=2)
    p.add_argument("--save_video", action="store_true", help="Save eval videos")
    # W&B
    p.add_argument("--wandb_mode", type=str, default="disabled")
    p.add_argument("--wandb_project", type=str, default="mujoco-franka-residual-td3")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="W&B run ID to log eval metrics to (same run as training)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Starting async eval process")
    _log(f"  checkpoint_dir: {checkpoint_dir}")
    _log(f"  output_dir: {output_dir}")
    _log(f"  poll_interval: {args.poll_interval_sec}s")

    # ── Create eval env ──
    _log("Creating eval MuJoCo environment...")
    if args.perturb_table:
        with open(args.perturb_table) as f:
            ptable = json.load(f)
        base_pos = ptable["base_pos"]
        envs_cfg = ptable["envs"]
        cube_positions = [
            [base_pos[0] + e["dx"], base_pos[1] + e["dy"], base_pos[2]]
            for e in envs_cfg
        ]
        args.num_envs = len(cube_positions)
    else:
        cube_positions = [args.cube_pos] * args.num_envs

    # Parse random cube range
    random_cube_range = None
    if hasattr(args, "random_cube_range") and args.random_cube_range:
        rcr = json.loads(args.random_cube_range)
        random_cube_range = {
            "dx": (rcr["dx"][0] / 100.0, rcr["dx"][1] / 100.0),
            "dy": (rcr["dy"][0] / 100.0, rcr["dy"][1] / 100.0),
            "yaw": tuple(rcr.get("yaw", [0, 0])),
        }
        _log(f"Random cube range: dx={rcr['dx']}cm dy={rcr['dy']}cm yaw={rcr.get('yaw',[0,0])}°")

    mujoco_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        cube_positions=cube_positions,
        random_cube_range=random_cube_range,
        max_episode_steps=args.max_episode_steps,
        success_threshold=args.success_threshold,
        reward_type=args.reward_type,
        device=args.device,
        rl_img_size=84,
        scene_xml=args.scene_xml,
        calib_path=args.calib_path,
    )

    policy_device = args.groot_policy_device or args.device
    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=policy_device,
        task_description=args.task_description,
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=args.residual_pos_scale,
        residual_rot_scale=args.residual_rot_scale,
        residual_grip_scale=args.residual_grip_scale,
        ema_alpha=args.ema_alpha,
    )
    _log("Eval environment ready.")

    # ── Create agent (same architecture) ──
    _asymmetric = args.asymmetric_critic
    object_state_dim = 7 if _asymmetric else 0
    if _asymmetric:
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
        img_c, img_h, img_w = 1, 84, 84
    else:
        image_keys = [
            "observation.images.front",
            "observation.images.back",
            "observation.images.wrist",
        ]
        img_c, img_h, img_w = 3, 84, 84

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=_asymmetric,
    )

    # ── W&B (log to same run as training) ──
    _wb = None
    if wandb is not None and args.wandb_mode != "disabled":
        try:
            _wb_kwargs = dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                reinit=True,
            )
            if args.wandb_run_id:
                _wb_kwargs["id"] = args.wandb_run_id
                _wb_kwargs["resume"] = "allow"
                _wb_kwargs["name"] = args.wandb_name
            else:
                _wb_kwargs["name"] = args.wandb_name or "async_eval"
            wandb.init(**_wb_kwargs)
            wandb.define_metric("eval/*", step_metric="eval/step")
            _wb = wandb
            _log(f"W&B initialized (run={wandb.run.id})")
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")

    # ── Main polling loop ──
    _log("Entering polling loop...")
    processed_steps = set()
    best_success = 0.0

    while True:
        # Look for eval_request_step*.pt files
        request_files = sorted(checkpoint_dir.glob("eval_request_step*.pt"))
        new_requests = [
            f for f in request_files
            if int(f.stem.split("step")[1]) not in processed_steps
        ]

        if not new_requests:
            time.sleep(args.poll_interval_sec)
            continue

        # Process latest request only (skip intermediate ones)
        latest = new_requests[-1]
        step = int(latest.stem.split("step")[1])
        _log(f"Found eval request: step={step}")

        try:
            state_dict = torch.load(latest, map_location=device, weights_only=True)
            agent.load_state_dict(state_dict)
            _log(f"  Loaded weights for step {step}")

            eval_start = time.time()
            video_dir = output_dir / "videos" if args.save_video else None
            metrics = evaluate(
                env=env, agent=agent,
                num_episodes=args.eval_num_episodes,
                device=device, image_keys=image_keys,
                save_video=args.save_video,
                video_dir=video_dir,
                eval_step=step,
            )
            eval_time = time.time() - eval_start

            sr = metrics["eval/success_rate"]
            _log(f"  step={step} SR={sr:.2%} return={metrics['eval/mean_return']:.2f} "
                 f"time={eval_time:.1f}s")

            # Track best
            if sr > best_success:
                best_success = sr
                metrics["eval/best_success_rate"] = best_success
                _log(f"  New best SR: {best_success:.2%}")

            metrics["eval/step"] = step
            metrics["eval/eval_time_sec"] = eval_time

            # Write JSON result
            result_path = output_dir / f"eval_step{step}.json"
            with open(result_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Log to W&B (eval/* uses eval/step axis via define_metric)
            if _wb is not None and _wb.run is not None:
                _wb.log(metrics)

            # Mark all skipped + current as processed
            for f in new_requests:
                s = int(f.stem.split("step")[1])
                processed_steps.add(s)

            # Clean up all processed request files immediately (save disk)
            for req_f in new_requests:
                try:
                    req_f.unlink()
                except Exception:
                    pass

        except Exception as e:
            _log(f"  ERROR evaluating step {step}: {e}")
            import traceback
            traceback.print_exc()
            processed_steps.add(step)

    # Cleanup
    if _wb is not None and _wb.run is not None:
        _wb.finish()
    env.close()


if __name__ == "__main__":
    main()
