"""Batched eval: N parallel envs with batched GR00T inference.

Runs one episode per env, reports per-env success and success rate,
saves per-env mp4 + side-by-side concat mp4.

Usage:
  isaaclab.sh -p resfit/rl_finetuning/scripts/run_batched_eval.py \
    --headless --num_envs 10 \
    --csv_dir ... --groot_model_path ... \
    --output_dir /tmp/batched_eval
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _bootstrap_te():
    try:
        ws = Path(__file__).resolve().parents[3] / "workspace"
        d = Path(os.environ.get("ISO_DEPS_DIR", str(ws / ".pydeps_groot_iso"))).resolve()
        f = d / "typing_extensions.py"
        if not f.exists():
            return
        s = str(d)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
        sp = importlib.util.spec_from_file_location("typing_extensions", str(f))
        if sp and sp.loader:
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            if hasattr(m, "NoExtraItems"):
                sys.modules["typing_extensions"] = m
    except Exception:
        pass


_bootstrap_te()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batched eval with N parallel envs")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--csv_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default=None)
parser.add_argument("--max_episode_steps", type=int, default=1000,
                    help="Max sim steps per episode (default 1000)")
parser.add_argument("--output_dir", type=str, default="/tmp/batched_eval")
parser.add_argument("--save_video", action="store_true", default=True)
parser.add_argument("--video_camera", type=str, default="front",
                    choices=["front", "back", "wrist"])
parser.add_argument("--video_size", type=str, default="120x160",
                    help="Video frame size HxW (default 120x160)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import imageio

import isaaclab.sim as sim_utils
from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper


def main():
    t0 = time.time()
    N = args.num_envs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_h, vid_w = [int(x) for x in args.video_size.split("x")]

    print(f"[EVAL] Creating {N}-env IfaceEnvWrapper...")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=sim,
        csv_dir=args.csv_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.embodiment_tag,
        policy_device=args.policy_device,
        policy_strict=True,
        task_description="pick up mushroom",
        language_override=args.language_override,
        max_episode_steps=args.max_episode_steps,
        num_envs=N,
    )

    print(f"[EVAL] Env created in {time.time() - t0:.1f}s. Resetting...")

    t_reset = time.time()
    obs, _ = env.reset()
    print(f"[EVAL] Reset done in {time.time() - t_reset:.1f}s")

    # ── Rollout (zero residual = pure base policy) ──
    device = torch.device(args.device)
    max_steps = args.max_episode_steps
    ep_rewards = torch.zeros(N, device=device)
    ep_ever_success = torch.zeros(N, device=device)
    frames_per_env: list[list[np.ndarray]] = [[] for _ in range(N)]
    steps_per_action = env.steps_per_action

    print(f"[EVAL] Running {max_steps} steps with residual=0 (pure GR00T)...")
    t_run = time.time()

    for step in range(max_steps):
        # Zero residual (pure base policy eval)
        residual = torch.zeros((N, 7), device=device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(residual)

        ep_rewards += reward.to(device)
        ep_ever_success = torch.max(ep_ever_success, reward.to(device))

        # Capture frames at control-tick intervals
        if args.save_video and (step % steps_per_action == 0):
            frame_list = env.get_all_frames(camera=args.video_camera, size=(vid_h, vid_w))
            for eid in range(N):
                frames_per_env[eid].append(frame_list[eid])

        # Progress
        if step > 0 and step % 200 == 0:
            elapsed = time.time() - t_run
            sps = step / elapsed
            succ_so_far = ep_ever_success.sum().item()
            print(f"[EVAL] step={step}/{max_steps} "
                  f"SPS={sps:.1f} "
                  f"successes_so_far={int(succ_so_far)}/{N}")

        done = terminated | truncated
        if done.all():
            print(f"[EVAL] All envs done at step {step}")
            break

    elapsed = time.time() - t_run
    print(f"[EVAL] Done: {max_steps} steps in {elapsed:.1f}s ({max_steps / elapsed:.1f} SPS)")

    # ── Results ──
    successes = ep_ever_success.cpu().numpy()
    returns = ep_rewards.cpu().numpy()
    success_rate = float(np.mean(successes))

    print(f"\n{'=' * 60}")
    print(f"  BATCHED EVAL RESULTS  ({N} envs, residual=0)")
    print(f"{'=' * 60}")
    print(f"  Success rate: {success_rate * 100:.1f}%  ({int(successes.sum())}/{N})")
    print(f"  Mean return:  {float(np.mean(returns)):.2f}")
    print(f"  Per-env success: {successes.astype(int).tolist()}")
    print(f"  Per-env return:  {[f'{r:.1f}' for r in returns]}")
    print(f"{'=' * 60}\n")

    # ── Save videos ──
    if args.save_video and any(frames_per_env):
        print(f"[EVAL] Saving videos to {out_dir}...")

        # Per-env videos
        for eid in range(N):
            if frames_per_env[eid]:
                p = out_dir / f"env{eid}.mp4"
                w = imageio.get_writer(str(p), fps=20)
                for fr in frames_per_env[eid]:
                    w.append_data(fr)
                w.close()

        # Concat video (side-by-side, 2 rows of 5 if N=10)
        min_len = min(len(f) for f in frames_per_env) if frames_per_env else 0
        if min_len > 0:
            # Arrange in grid: up to 5 per row
            cols = min(N, 5)
            rows = (N + cols - 1) // cols

            concat_path = out_dir / f"eval_concat_{N}envs.mp4"
            w = imageio.get_writer(str(concat_path), fps=20)
            for i in range(min_len):
                grid_rows = []
                for r in range(rows):
                    row_frames = []
                    for c in range(cols):
                        eid = r * cols + c
                        if eid < N:
                            row_frames.append(frames_per_env[eid][i])
                        else:
                            # Pad with black frame
                            row_frames.append(np.zeros_like(frames_per_env[0][i]))
                    grid_rows.append(np.concatenate(row_frames, axis=1))
                grid = np.concatenate(grid_rows, axis=0)
                w.append_data(grid)
            w.close()
            print(f"[EVAL] Concat video: {concat_path} ({min_len} frames, {rows}x{cols} grid)")

        # Summary
        n_vids = sum(1 for f in frames_per_env if f)
        print(f"[EVAL] Saved {n_vids} per-env videos + 1 concat to {out_dir}")

    # Write results text
    results_path = out_dir / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"num_envs: {N}\n")
        f.write(f"max_episode_steps: {max_steps}\n")
        f.write(f"success_rate: {success_rate:.4f}\n")
        f.write(f"mean_return: {float(np.mean(returns)):.4f}\n")
        f.write(f"per_env_success: {successes.astype(int).tolist()}\n")
        f.write(f"per_env_return: {returns.tolist()}\n")
        f.write(f"elapsed_sec: {elapsed:.1f}\n")
        f.write(f"sps: {max_steps / elapsed:.1f}\n")
    print(f"[EVAL] Results saved to {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n{'=' * 60}\nFATAL ERROR:\n{'=' * 60}", flush=True)
        traceback.print_exc()
        print(f"{'=' * 60}\n", flush=True)
    finally:
        simulation_app.close()
