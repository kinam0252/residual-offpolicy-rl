"""
Async Evaluation Process for Residual TD3 on IsaacLab.

Runs in a SEPARATE Isaac Sim instance from training.
Polls a checkpoint directory for new .pt files, loads them, runs batched
evaluation, and logs results to wandb (same project, separate run or shared).

Usage (inside container):
  isaaclab.sh -p eval_async_isaaclab.py --headless --enable_cameras \
    --checkpoint_dir /path/to/checkpoints \
    --num_envs 10 \
    --groot_model_path /path/to/groot \
    --csv_base_dir /path/to/csv \
    --wandb_mode online --wandb_project MyProject
"""
from __future__ import annotations
import argparse, os, sys, time, glob, json, pprint
import faulthandler

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Async eval for Residual TD3")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with agent_stepN.pt checkpoints")
parser.add_argument("--num_envs", type=int, default=10, help="Number of eval envs")
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.005)
parser.add_argument("--poll_interval_sec", type=int, default=30, help="How often to check for new checkpoints")
parser.add_argument("--eval_all", action="store_true", help="Evaluate ALL checkpoints sequentially, not just newest")
parser.add_argument("--eval_steps", type=str, default=None, help="Comma-separated list of steps to eval, e.g. '0,2500,5000'")
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--cube_perturb_table_path", type=str, default=None, help="Path to JSON cube perturbation table")
parser.add_argument("--custom_env_dir", type=str, default=None, help="Path to custom_env snapshot directory")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--wandb_mode", type=str, default="disabled")
parser.add_argument("--wandb_project", type=str, default="Isaaclab_Franka_Pickup")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_id", type=str, default=None, help="Resume wandb run ID (for shared train/eval logging)")
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import faulthandler as _fh
try:
    _fh.cancel_dump_traceback_later()
except Exception:
    pass

import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import imageio
import cv2

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig
import isaaclab.sim as sim_utils


def _log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [async-eval] {msg}", flush=True)


def make_eval_env(args):
    """Create a standalone IfaceEnvWrapper for eval only."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    _sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=args.csv_base_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args.language_override,
        max_episode_steps=args.max_episode_steps,
        num_envs=args.num_envs,
        success_threshold=args.success_threshold,
        cube_perturb_table_path=getattr(args, 'cube_perturb_table_path', None),
    )
    return env


def make_agent(env, device):
    """Construct a QAgent with the same architecture as training."""
    image_keys = [
        "observation.images.front",
        "observation.images.back",
        "observation.images.wrist",
    ]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    vlm_latent_dim = env.observation_space["observation.vlm_latent"].shape[1] if "observation.vlm_latent" in env.observation_space.spaces else 0

    agent_cfg = QAgentConfig(
        actor_lr=1e-6,
        critic_lr=1e-4,
        critic_target_tau=0.005,
        actor=ActorConfig(
            action_scale=0.05,
            actor_last_layer_init_scale=0.0,
            action_l2_reg_weight=1.0,
        ),
    )

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=agent_cfg,
        residual_actor=True,
        vlm_latent_dim=vlm_latent_dim,
    )
    return agent, image_keys


def load_checkpoint(agent, ckpt_path, device):
    """Load agent weights from a checkpoint file (backward-compatible with prop_dim changes)."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    agent.load_checkpoint_compat(ckpt)
    _log(f"Loaded checkpoint: {ckpt_path}")


def _annotate_frame(
    frame: np.ndarray,
    env_id: int,
    step: int,
    contact_force: float,
    has_contact: float,
    cube_height: float,
    reward: float,
    finger_cube_dist: float,
    success: float,
) -> np.ndarray:
    """Overlay contact force and reward debug info on frame (H, W, 3) uint8."""
    img = frame.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.40, 1
    force_color = (0, 255, 0) if has_contact > 0.5 else (0, 0, 255)
    cv2.putText(img, f"Contact: {contact_force:.2f}N", (4, 14), font, fs, force_color, th)
    cv2.putText(img, f"[{'ON' if has_contact > 0.5 else 'OFF'}]", (160, 14), font, fs, force_color, th)
    h_color = (0, 255, 0) if cube_height > 0.005 else (200, 200, 200)
    cv2.putText(img, f"Height: {cube_height*100:.2f}cm", (4, 30), font, fs, h_color, th)
    cv2.putText(img, f"Dist: {finger_cube_dist*100:.2f}cm", (160, 30), font, fs, (200, 200, 200), th)
    r_color = (0, 255, 0) if reward > 0.01 else (200, 200, 200)
    cv2.putText(img, f"Reward: {reward:.4f}", (4, 46), font, fs, r_color, th)
    if success > 0.5:
        cv2.putText(img, "SUCCESS", (160, 46), font, fs, (0, 255, 0), 2)
    cv2.putText(img, f"Env:{env_id} Step:{step}", (4, 62), font, fs, (255, 255, 255), th)
    bar_max_w = w - 8
    bar_fill = int(min(contact_force / 5.0, 1.0) * bar_max_w)
    bar_y = 70
    cv2.rectangle(img, (4, bar_y), (4 + bar_max_w, bar_y + 8), (60, 60, 60), -1)
    if bar_fill > 0:
        cv2.rectangle(img, (4, bar_y), (4 + bar_fill, bar_y + 8), force_color, -1)
    cv2.putText(img, "Force", (4, bar_y + 20), font, 0.30, (150, 150, 150), 1)
    h_bar_fill = int(min(max(cube_height, 0) / 0.05, 1.0) * bar_max_w)
    h_bar_y = bar_y + 24
    cv2.rectangle(img, (4, h_bar_y), (4 + bar_max_w, h_bar_y + 8), (60, 60, 60), -1)
    if h_bar_fill > 0:
        cv2.rectangle(img, (4, h_bar_y), (4 + h_bar_fill, h_bar_y + 8), h_color, -1)
    cv2.putText(img, "Height", (4, h_bar_y + 20), font, 0.30, (150, 150, 150), 1)
    return img


def run_eval(env, agent, device, image_keys, args, global_step, output_dir):
    """Run batched evaluation and return metrics dict."""
    N = env.num_envs
    max_ep_steps = args.max_episode_steps
    success_threshold = args.success_threshold

    env.reset()

    # Apply custom env snapshot (same cube poses as training)
    if getattr(args, 'custom_env_dir', None):
        _ced = Path(args.custom_env_dir)
        # Find snapshot dir (any hash subdir with env_config.json)
        for sd in sorted(_ced.iterdir()) if _ced.exists() else []:
            if sd.is_dir() and (sd / "env_config.json").exists():
                from resfit.rl_finetuning.utils.custom_env import load_env_snapshot, capture_sanity_frames
                load_env_snapshot(_ced, sd.name, env)
                capture_sanity_frames(env, output_dir, f"eval_step{global_step}")
                break

    obs = env._build_obs()
    eval_env_ids = list(range(N))

    ep_rewards = torch.zeros(N, device=device)
    ep_ever_success = torch.zeros(N, device=device)
    ep_first_success_step = torch.full((N,), float(max_ep_steps), device=device)
    ep_steps = 0

    frames_per_env = [[] for _ in range(N)] if args.save_video else []

    while ep_steps < max_ep_steps:
        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

        next_obs, reward, terminated, truncated, info = env.step(residual_action)

        ep_rewards += reward[:N].to(device)

        cube_z = env.cube.data.root_state_w[:N, 2].to(device)
        cube_init_z = env._initial_cube_z[:N].to(device)
        cube_lifted = ((cube_z - cube_init_z) >= success_threshold).float()
        just_succeeded = (cube_lifted > 0) & (ep_ever_success == 0)
        ep_first_success_step[just_succeeded] = ep_steps
        ep_ever_success = torch.max(ep_ever_success, cube_lifted)
        ep_steps += 1

        if args.save_video and ep_steps % 10 == 0 and hasattr(env, "get_all_frames_batch"):
            debug = getattr(env, "_last_step_debug", None)
            all_frames = env.get_all_frames_batch(camera="front", size=(256, 320))
            for i in range(N):
                frame = all_frames[i]
                if debug is not None:
                    frame = _annotate_frame(
                        frame, i, ep_steps,
                        contact_force=debug["contact_force"][i].item(),
                        has_contact=debug["has_contact"][i].item(),
                        cube_height=debug["cube_height"][i].item(),
                        reward=debug["reward"][i].item(),
                        finger_cube_dist=debug["finger_cube_dist"][i].item(),
                        success=debug["success"][i].item(),
                    )
                frames_per_env[i].append(frame)

        obs = next_obs
        done = terminated | truncated
        if done[:N].all():
            break

    successes = ep_ever_success.cpu().numpy()
    returns = ep_rewards.cpu().numpy()

    succ_mask = successes > 0
    mean_succ_len = float(np.mean(ep_first_success_step.cpu().numpy()[succ_mask])) if succ_mask.any() else 0.0

    metrics = {
        "eval/success_rate": float(np.mean(successes)),
        "eval/mean_return": float(np.mean(returns)),
        "eval/mean_successful_episode_length": mean_succ_len,
        "eval/num_envs": N,
        "eval/episode_steps": ep_steps,
    }

    # Save video — per-env videos in step-based folder + grid
    video_path = None
    if args.save_video and frames_per_env:
        step_vid_dir = output_dir / "eval_videos" / f"step_{global_step}"
        step_vid_dir.mkdir(parents=True, exist_ok=True)

        # Get perturbation names for filenames
        _perturb_names = {}
        _pt_path = getattr(args, 'cube_perturb_table_path', None)
        if _pt_path and Path(_pt_path).exists():
            import json as _jj
            _pt = _jj.loads(Path(_pt_path).read_text())
            for e in _pt.get('envs', []):
                _perturb_names[e['env_id']] = e.get('name', f"env{e['env_id']}")

        # Save ALL per-env videos
        for i in range(N):
            if len(frames_per_env[i]) > 0:
                succ_tag = "succ" if successes[i] > 0 else "fail"
                env_name = _perturb_names.get(i, f"env{i}")
                p = step_vid_dir / f"env{i}_{env_name}_{succ_tag}.mp4"
                w = imageio.get_writer(str(p), fps=20)
                for fr in frames_per_env[i]:
                    w.append_data(fr)
                w.close()

        # Concat grid video (2 rows x 5 cols)
        valid_frames = [f for f in frames_per_env if len(f) > 0]
        if valid_frames:
            min_len = min(len(f) for f in valid_frames)
            N_vid = len(valid_frames)
            cols = min(N_vid, 5)
            rows = (N_vid + cols - 1) // cols
            concat_path = step_vid_dir / f"grid_{N_vid}envs.mp4"
            w = imageio.get_writer(str(concat_path), fps=20)
            for fi in range(min_len):
                grid_rows = []
                for r in range(rows):
                    row_frames = []
                    for c in range(cols):
                        idx = r * cols + c
                        if idx < N_vid:
                            row_frames.append(valid_frames[idx][fi])
                        else:
                            row_frames.append(np.zeros_like(valid_frames[0][fi]))
                    grid_rows.append(np.concatenate(row_frames, axis=1))
                grid = np.concatenate(grid_rows, axis=0)
                w.append_data(grid)
            w.close()
            video_path = concat_path
            _log(f"Eval video: {step_vid_dir} ({N_vid} env videos + grid, {min_len} frames)")

    return metrics, video_path


def extract_step_from_path(p: Path) -> int:
    """Extract step number from agent_stepNNN.pt filename."""
    name = p.stem  # e.g. 'agent_step500'
    try:
        return int(name.replace("agent_step", ""))
    except ValueError:
        return -1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    checkpoint_dir = Path(args_cli.checkpoint_dir)
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else checkpoint_dir.parent / "async_eval_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Creating eval environment...")
    env = make_eval_env(args_cli)
    _log(f"Eval env ready: {env.num_envs} envs")

    _log("Creating agent...")
    agent, image_keys = make_agent(env, device)
    agent.to(device)
    _log("Agent ready")

    # Optional wandb
    _wb = None
    if args_cli.wandb_mode != "disabled":
        try:
            import wandb
            wandb.init(
                project=args_cli.wandb_project,
                entity=args_cli.wandb_entity,
                name=args_cli.wandb_name or f"async_eval_{datetime.now().strftime('%H%M%S')}",
                id=args_cli.wandb_run_id,
                resume="allow" if args_cli.wandb_run_id else None,
                mode=args_cli.wandb_mode,
                tags=["async_eval"],
            )
            # Define eval metrics with their own step so they don't conflict with training steps
            wandb.define_metric("eval/*", step_metric="eval/step")
            _wb = wandb
            _log(f"wandb initialized: {wandb.run.url} (run_id={wandb.run.id})")
        except Exception as e:
            _log(f"wandb init failed: {e}")

    evaluated_steps: set[int] = set()
    _log(f"Polling {checkpoint_dir} every {args_cli.poll_interval_sec}s for new checkpoints...")

    try:
        while True:
            # Find all checkpoints
            ckpt_files = sorted(checkpoint_dir.glob("agent_step*.pt"))
            new_ckpts = []
            for p in ckpt_files:
                step = extract_step_from_path(p)
                if step >= 0 and step not in evaluated_steps:
                    new_ckpts.append((step, p))

            if not new_ckpts:
                # Check if training is done (final_agent.pt exists)
                if (checkpoint_dir / "final_agent.pt").exists() and len(evaluated_steps) > 0:
                    _log("Found final_agent.pt — training seems done. Evaluating final and exiting.")
                    final_step = max(evaluated_steps) + 1 if evaluated_steps else 0
                    load_checkpoint(agent, checkpoint_dir / "final_agent.pt", device)
                    metrics, video_path = run_eval(env, agent, device, image_keys, args_cli, final_step, output_dir)
                    _log(f"Final eval: success={metrics['eval/success_rate']:.3f} return={metrics['eval/mean_return']:.3f}")
                    if _wb is not None and _wb.run is not None:
                        log_dict = {k: v for k, v in metrics.items()}
                        if video_path:
                            log_dict["eval/video"] = _wb.Video(str(video_path), format="mp4")
                        _wb.log(log_dict, step=final_step)
                    break

                time.sleep(args_cli.poll_interval_sec)
                continue

            # Evaluate: all checkpoints or newest only
            new_ckpts.sort(key=lambda x: x[0], reverse=True)  # newest first

            # Filter to specific steps if requested
            if args_cli.eval_steps:
                target_steps = set(int(s) for s in args_cli.eval_steps.split(","))
                new_ckpts = [(s, p) for s, p in new_ckpts if s in target_steps]
                if not new_ckpts:
                    time.sleep(args_cli.poll_interval_sec)
                    continue

            ckpts_to_eval = new_ckpts

            for step, ckpt_path in ckpts_to_eval:
                # Skip if already evaluated (result file exists)
                result_json = output_dir / f"eval_step{step}.json"
                result_done = output_dir / f"eval_step{step}.done"
                if result_json.exists() or result_done.exists():
                    _log(f"Skipping step {step}: already evaluated")
                    evaluated_steps.add(step)
                    continue

                evaluated_steps.add(step)

                _log(f"New checkpoint at step {step}: {ckpt_path}")
                load_checkpoint(agent, ckpt_path, device)

                t0 = time.time()
                try:
                    metrics, video_path = run_eval(env, agent, device, image_keys, args_cli, step, output_dir)
                except Exception as e:
                    import traceback
                    _log(f"ERROR in run_eval at step {step}: {e}")
                    _log(traceback.format_exc())
                    # Write .done so we don't retry
                    (output_dir / f"eval_step{step}.done").write_text(f"FAILED: {e}")
                    continue
                eval_time = time.time() - t0

                _log(f"Step {step}: success={metrics['eval/success_rate']:.3f} "
                     f"return={metrics['eval/mean_return']:.3f} "
                     f"succ_len={metrics['eval/mean_successful_episode_length']:.0f} "
                     f"({eval_time:.1f}s)")

                if _wb is not None and _wb.run is not None:
                    log_dict = {k: v for k, v in metrics.items()}
                    log_dict["eval/eval_time_sec"] = eval_time
                    log_dict["eval/step"] = step  # custom x-axis for eval metrics
                    if video_path:
                        log_dict["eval/video"] = _wb.Video(str(video_path), format="mp4")
                    _wb.log(log_dict)

                # Save metrics to JSON
                metrics_path = output_dir / f"eval_step{step}.json"
                with open(metrics_path, "w") as f:
                    json.dump({"step": step, **metrics, "eval_time_sec": eval_time}, f, indent=2)

            # After eval_all with specific steps, exit
            if args_cli.eval_steps:
                _log("All requested steps evaluated. Exiting.")
                break

    except KeyboardInterrupt:
        _log("Interrupted by user")
    finally:
        if _wb is not None and _wb.run is not None:
            _wb.finish()
        env.close()
        _log("Async eval process finished")
        os._exit(0)


if __name__ == "__main__":
    main()
