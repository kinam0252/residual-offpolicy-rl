#!/usr/bin/env python3
"""
Debug: Run critic inference on one offline episode.
No simulator needed — loads parquet + video from offline dataset.
Renders annotated video with GT reward vs Critic Q-value per frame.

Usage:
  python debug_critic_offline.py \
    --checkpoint /path/to/agent_step*.pt \
    --offline_data_dir /path/to/lerobot_dataset \
    --episode_idx 0 \
    --output_dir /path/to/debug_output
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.rlpd import QAgentConfig, ActorConfig
from resfit.rl_finetuning.utils.offline_buffer_csv import _read_video_frames


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# Map offline video dirs to observation keys
VIDEO_MAP = {
    "right_image": "observation.images.front",
    "image": "observation.images.back",
    "wrist_image": "observation.images.wrist",
}


def annotate_frame(
    frame: np.ndarray,
    step: int,
    gt_reward: float,
    q_value: float,
    cum_reward: float,
    cum_discounted: float,
    contact_force: float,
    has_contact: float,
    cube_height: float,
) -> np.ndarray:
    """Overlay GT reward, Q-value, and other info on frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.42, 1

    # Step
    cv2.putText(img, f"Step: {step}", (4, 14), font, fs, (255, 255, 255), th)

    # GT Reward
    r_color = (0, 255, 0) if gt_reward > 0.01 else (200, 200, 200)
    cv2.putText(img, f"GT R: {gt_reward:.4f}", (4, 30), font, fs, r_color, th)

    # Q-value
    q_color = (0, 200, 255)
    cv2.putText(img, f"Q: {q_value:.4f}", (160, 30), font, fs, q_color, th)

    # Q - GT gap
    gap = q_value - gt_reward
    gap_color = (0, 255, 0) if abs(gap) < 0.1 else (0, 0, 255)
    cv2.putText(img, f"Gap: {gap:+.4f}", (4, 46), font, fs, gap_color, th)

    # Cumulative reward
    cv2.putText(img, f"CumR: {cum_reward:.2f}", (160, 46), font, fs, (200, 200, 200), th)

    # Discounted return (from this step onward)
    cv2.putText(img, f"G_t: {cum_discounted:.4f}", (4, 62), font, fs, (255, 200, 100), th)

    # Contact
    force_color = (0, 255, 0) if has_contact > 0.5 else (0, 0, 255)
    cv2.putText(img, f"Force: {contact_force:.2f}N", (4, 78), font, fs, force_color, th)
    cv2.putText(img, f"[{'ON' if has_contact > 0.5 else 'OFF'}]", (160, 78), font, fs, force_color, th)

    # Cube height
    h_color = (0, 255, 0) if cube_height > 0.005 else (200, 200, 200)
    cv2.putText(img, f"Height: {cube_height*100:.2f}cm", (4, 94), font, fs, h_color, th)

    # Q-value bar (scale: 0-10)
    bar_y = 104
    bar_max_w = w - 8
    q_fill = int(min(max(q_value, 0) / 10.0, 1.0) * bar_max_w)
    r_fill = int(min(max(gt_reward, 0) / 1.0, 1.0) * bar_max_w)
    # Q bar (orange)
    cv2.rectangle(img, (4, bar_y), (4 + bar_max_w, bar_y + 5), (40, 40, 40), -1)
    if q_fill > 0:
        cv2.rectangle(img, (4, bar_y), (4 + q_fill, bar_y + 5), (0, 200, 255), -1)
    cv2.putText(img, "Q", (4, bar_y + 16), font, 0.28, (0, 200, 255), 1)
    # R bar (green)
    bar_y2 = bar_y + 12
    cv2.rectangle(img, (4, bar_y2), (4 + bar_max_w, bar_y2 + 5), (40, 40, 40), -1)
    if r_fill > 0:
        cv2.rectangle(img, (4, bar_y2), (4 + r_fill, bar_y2 + 5), (0, 255, 0), -1)
    cv2.putText(img, "R", (4, bar_y2 + 16), font, 0.28, (0, 255, 0), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Debug critic on offline episode")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--offline_data_dir", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--image_size_h", type=int, default=84)
    parser.add_argument("--image_size_w", type=int, default=84)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    offline_root = Path(args.offline_data_dir)
    ckpt_path = Path(args.checkpoint)

    if args.output_dir is None:
        args.output_dir = str(ckpt_path.parent.parent / "debug_critic")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load episode data ──
    ep_name = f"episode_{args.episode_idx:06d}"
    parquet_path = offline_root / "data" / "chunk-000" / f"{ep_name}.parquet"
    if not parquet_path.exists():
        _log(f"ERROR: Parquet not found: {parquet_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    T = len(df)
    _log(f"Loaded episode {args.episode_idx}: {T} steps from {parquet_path}")

    states = np.stack(df["state"].to_numpy()).astype(np.float32)
    actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
    rewards = df["reward_total"].to_numpy().astype(np.float32) if "reward_total" in df.columns else df["reward"].to_numpy().astype(np.float32)
    contact_forces = df["contact_force"].to_numpy().astype(np.float32) if "contact_force" in df.columns else np.zeros(T, dtype=np.float32)
    has_contacts = df["has_contact"].to_numpy().astype(np.float32) if "has_contact" in df.columns else np.zeros(T, dtype=np.float32)
    cube_heights = df["cube_height"].to_numpy().astype(np.float32) if "cube_height" in df.columns else np.zeros(T, dtype=np.float32)

    # Compute discounted returns G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}
    gamma = args.gamma
    G = np.zeros(T, dtype=np.float32)
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    cum_rewards = np.cumsum(rewards)

    _log(f"  Rewards: min={rewards.min():.4f} max={rewards.max():.4f} sum={rewards.sum():.2f}")
    _log(f"  Contact: {(has_contacts > 0.5).sum()}/{T} steps with contact")
    _log(f"  Discounted return G_0={G[0]:.4f}")

    # ── Load video frames ──
    image_keys = list(VIDEO_MAP.values())
    image_size = (args.image_size_h, args.image_size_w)
    video_chunk = offline_root / "videos" / "chunk-000"

    view_frames = {}
    for view_dir, obs_key in VIDEO_MAP.items():
        vpath = video_chunk / view_dir / f"{ep_name}.mp4"
        if vpath.exists():
            view_frames[obs_key] = _read_video_frames(vpath, max_frames=T, target_size=image_size)
            _log(f"  Loaded {obs_key}: {view_frames[obs_key].shape}")
        else:
            _log(f"  WARNING: Video not found: {vpath}")
            view_frames[obs_key] = torch.zeros((T, 3, image_size[0], image_size[1]), dtype=torch.uint8)

    # Also load front camera at higher res for annotation video
    front_vpath = video_chunk / "right_image" / f"{ep_name}.mp4"
    front_frames_hr = _read_video_frames(front_vpath, max_frames=T, target_size=(256, 320)) if front_vpath.exists() else None

    T = min(T, *(len(v) for v in view_frames.values()))
    if front_frames_hr is not None:
        T = min(T, len(front_frames_hr))
    _log(f"  Using T={T} steps (min across parquet and videos)")

    # ── Build agent and load checkpoint ──
    img_c, img_h, img_w = 3, image_size[0], image_size[1]
    # Build 10D agent (backward compat handles old 9D checkpoints)
    lowdim_dim = 10
    action_dim = min(actions.shape[1], 7)

    agent_cfg = QAgentConfig(
        actor_lr=3e-7, critic_lr=1e-4, critic_target_tau=0.005,
        clip_q_target_to_reward_range=True,
        actor=ActorConfig(action_scale=0.1, actor_last_layer_init_scale=0.0, action_l2_reg_weight=0.0),
    )
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w), prop_shape=(lowdim_dim,),
        action_dim=action_dim, rl_cameras=image_keys, cfg=agent_cfg, residual_actor=True,
    )
    agent.to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    agent.load_checkpoint_compat(ckpt)
    agent.eval()
    _log(f"Loaded checkpoint: {ckpt_path}")

    # ── Run critic inference ──
    q_values = []
    log_lines = []
    annotated_frames = []

    header = f"{'step':>5} | {'GT_R':>8} | {'Q_val':>8} | {'G_t':>8} | {'Gap':>8} | {'Contact':>8} | {'CubeH':>8}"
    _log(header)
    _log("-" * len(header))
    log_lines.append(header)
    log_lines.append("-" * len(header))

    with torch.no_grad():
        for t in range(T - 1):
            # Build observation — 10D state: EEF(3) + quat(4) + grip(2) + contact_force(1)
            state_10d = np.zeros(10, dtype=np.float32)
            n = min(len(states[t]), 9)
            state_10d[:n] = states[t][:n]
            state_10d[9] = float(contact_forces[t])  # contact force magnitude

            action_7d = actions[t][:7].astype(np.float32)

            obs = {
                "observation.state": torch.tensor(state_10d, dtype=torch.float32, device=device).unsqueeze(0),
                "observation.base_action": torch.tensor(action_7d, dtype=torch.float32, device=device).unsqueeze(0),
            }
            for obs_key in image_keys:
                obs[obs_key] = view_frames[obs_key][t].unsqueeze(0).to(device)

            # Encode images
            obs["feat"] = agent._encode(obs, augment=False)

            # Get Q-value from critic (all heads → mean)
            action_tensor = torch.tensor(action_7d, dtype=torch.float32, device=device).unsqueeze(0)
            q_all = agent.critic(obs["feat"], obs["observation.state"], action_tensor)  # [K, 1, 1]
            q_mean = q_all.squeeze(-1).mean(dim=0).item()  # mean over heads
            q_min = q_all.squeeze(-1).min(dim=0).values.item()  # min over heads

            q_values.append(q_mean)

            gt_r = float(rewards[t])
            g_t = float(G[t])
            cf = float(contact_forces[t])
            hc = float(has_contacts[t])
            ch = float(cube_heights[t])
            gap = q_mean - gt_r

            line = f"{t:5d} | {gt_r:8.4f} | {q_mean:8.4f} | {g_t:8.4f} | {gap:+8.4f} | {cf:8.3f}N {'ON' if hc > 0.5 else '  '} | {ch*100:7.2f}cm"
            if t < 20 or t % 20 == 0 or t >= T - 5:
                _log(line)
            log_lines.append(line)

            # Annotate high-res frame
            if front_frames_hr is not None:
                hr = front_frames_hr[t].permute(1, 2, 0).numpy()  # (H, W, 3)
                ann = annotate_frame(
                    hr, t, gt_r, q_mean, float(cum_rewards[t]), g_t, cf, hc, ch
                )
                annotated_frames.append(ann)

    q_values = np.array(q_values)
    _log(f"\n=== SUMMARY ===")
    _log(f"  Q-value: mean={q_values.mean():.4f} min={q_values.min():.4f} max={q_values.max():.4f}")
    _log(f"  GT reward: mean={rewards[:T-1].mean():.4f} sum={rewards[:T-1].sum():.2f}")
    _log(f"  G_0 (discounted return): {G[0]:.4f}")
    _log(f"  Q-G correlation: {np.corrcoef(q_values, G[:T-1])[0,1]:.4f}")
    _log(f"  Mean |Q - G_t|: {np.mean(np.abs(q_values - G[:T-1])):.4f}")

    log_lines.append("")
    log_lines.append("=== SUMMARY ===")
    log_lines.append(f"  Q-value: mean={q_values.mean():.4f} min={q_values.min():.4f} max={q_values.max():.4f}")
    log_lines.append(f"  GT reward: mean={rewards[:T-1].mean():.4f} sum={rewards[:T-1].sum():.2f}")
    log_lines.append(f"  G_0: {G[0]:.4f}")
    log_lines.append(f"  Corr(Q, G_t): {np.corrcoef(q_values, G[:T-1])[0,1]:.4f}")
    log_lines.append(f"  Mean |Q - G_t|: {np.mean(np.abs(q_values - G[:T-1])):.4f}")

    # ── Save log ──
    log_path = output_dir / f"critic_debug_ep{args.episode_idx}.log"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    _log(f"Log saved: {log_path}")

    # ── Save annotated video ──
    if annotated_frames:
        vid_path = output_dir / f"critic_debug_ep{args.episode_idx}.mp4"
        w = imageio.get_writer(str(vid_path), fps=20)
        for fr in annotated_frames:
            w.append_data(fr)
        w.close()
        _log(f"Video saved: {vid_path} ({len(annotated_frames)} frames)")

    # ── Save Q-value plot data as CSV ──
    csv_path = output_dir / f"critic_debug_ep{args.episode_idx}.csv"
    plot_df = pd.DataFrame({
        "step": np.arange(T - 1),
        "gt_reward": rewards[:T-1],
        "q_value": q_values,
        "discounted_return": G[:T-1],
        "cum_reward": cum_rewards[:T-1],
        "contact_force": contact_forces[:T-1],
        "has_contact": has_contacts[:T-1],
        "cube_height": cube_heights[:T-1],
    })
    plot_df.to_csv(csv_path, index=False)
    _log(f"CSV saved: {csv_path}")

    _log("Done!")


if __name__ == "__main__":
    main()
