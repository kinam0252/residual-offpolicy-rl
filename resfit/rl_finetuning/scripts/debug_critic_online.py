"""
Debug: Run online rollout in Isaac Sim with critic Q-value overlay.
For each step, shows GT reward vs critic Q(s,a) on every frame.
Uses 10 envs with cube perturbations.

Usage (inside container):
  isaaclab.sh -p debug_critic_online.py --headless --enable_cameras \
    --checkpoint /path/to/agent_step*.pt \
    --csv_base_dir /path/to/csv \
    --groot_model_path /path/to/groot \
    --output_dir /path/to/debug_output
"""
from __future__ import annotations
import argparse, os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Online critic debug with cube perturbations")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--output_dir", type=str, default="/tmp/debug_critic_online")
parser.add_argument("--gamma", type=float, default=0.99)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import faulthandler
try: faulthandler.cancel_dump_traceback_later()
except: pass

from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import imageio
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig
import isaaclab.sim as sim_utils


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# 10 cube offsets
CUBE_OFFSETS = [
    (0.0, 0.0, "center"),
    (0.05, 0.0, "right_5cm"),
    (-0.05, 0.0, "left_5cm"),
    (0.0, 0.05, "forward_5cm"),
    (0.0, -0.05, "back_5cm"),
    (0.04, 0.04, "diag_FR_5.6cm"),
    (-0.04, 0.04, "diag_FL_5.6cm"),
    (0.04, -0.04, "diag_BR_5.6cm"),
    (-0.04, -0.04, "diag_BL_5.6cm"),
    (0.08, 0.0, "right_8cm"),
]


def annotate_frame_critic(
    frame: np.ndarray, env_id: int, step: int,
    gt_reward: float, q_value: float, cum_reward: float,
    contact_force: float, has_contact: float, cube_height: float,
    success: float,
) -> np.ndarray:
    """Overlay GT reward, Q-value, contact force on frame."""
    img = frame.copy()
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.40, 1

    # Step & env
    cv2.putText(img, f"Env:{env_id} Step:{step}", (4, 14), font, fs, (255, 255, 255), th)

    # GT Reward
    r_color = (0, 255, 0) if gt_reward > 0.01 else (200, 200, 200)
    cv2.putText(img, f"GT R: {gt_reward:.4f}", (4, 30), font, fs, r_color, th)

    # Q-value
    cv2.putText(img, f"Q: {q_value:.3f}", (170, 30), font, fs, (0, 200, 255), th)

    # Q - R gap
    gap = q_value - gt_reward
    gap_color = (0, 255, 0) if abs(gap) < 0.5 else (0, 100, 255)
    cv2.putText(img, f"Gap: {gap:+.3f}", (4, 46), font, fs, gap_color, th)

    # Cumulative reward
    cv2.putText(img, f"CumR: {cum_reward:.2f}", (170, 46), font, fs, (200, 200, 200), th)

    # Contact force
    force_color = (0, 255, 0) if has_contact > 0.5 else (0, 0, 255)
    cv2.putText(img, f"Force: {contact_force:.2f}N", (4, 62), font, fs, force_color, th)
    cv2.putText(img, f"[{'ON' if has_contact > 0.5 else 'OFF'}]", (170, 62), font, fs, force_color, th)

    # Cube height
    h_color = (0, 255, 0) if cube_height > 0.005 else (200, 200, 200)
    cv2.putText(img, f"H: {cube_height*100:.2f}cm", (4, 78), font, fs, h_color, th)

    if success > 0.5:
        cv2.putText(img, "SUCCESS", (170, 78), font, fs, (0, 255, 0), 2)

    # Q-value bar (0-10)
    bar_y = 88
    bar_max_w = w - 8
    q_fill = int(min(max(q_value, 0) / 10.0, 1.0) * bar_max_w)
    cv2.rectangle(img, (4, bar_y), (4 + bar_max_w, bar_y + 6), (40, 40, 40), -1)
    if q_fill > 0:
        cv2.rectangle(img, (4, bar_y), (4 + q_fill, bar_y + 6), (0, 200, 255), -1)
    cv2.putText(img, "Q", (4, bar_y + 18), font, 0.28, (0, 200, 255), 1)

    # Reward bar (0-1)
    bar_y2 = bar_y + 14
    r_fill = int(min(max(gt_reward, 0) / 1.0, 1.0) * bar_max_w)
    cv2.rectangle(img, (4, bar_y2), (4 + bar_max_w, bar_y2 + 6), (40, 40, 40), -1)
    if r_fill > 0:
        cv2.rectangle(img, (4, bar_y2), (4 + r_fill, bar_y2 + 6), (0, 255, 0), -1)
    cv2.putText(img, "R", (4, bar_y2 + 18), font, 0.28, (0, 255, 0), 1)

    return img


def main():
    N = len(CUBE_OFFSETS)
    device = torch.device("cuda:0")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Create env ──
    _log(f"Creating env with {N} envs...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    _sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=args_cli.csv_base_dir,
        groot_model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        policy_device=args_cli.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args_cli.language_override,
        max_episode_steps=args_cli.max_episode_steps,
        num_envs=N,
        success_threshold=args_cli.success_threshold,
    )
    _log("Env ready")

    # ── Build agent ──
    image_keys = ["observation.images.front", "observation.images.back", "observation.images.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]

    agent_cfg = QAgentConfig(
        actor_lr=3e-7, critic_lr=1e-4, critic_target_tau=0.005,
        clip_q_target_to_reward_range=True,
        actor=ActorConfig(action_scale=0.1, actor_last_layer_init_scale=0.0, action_l2_reg_weight=10.0),
    )
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w), prop_shape=(lowdim_dim,),
        action_dim=action_dim, rl_cameras=image_keys, cfg=agent_cfg, residual_actor=True,
    )
    agent.to(device)

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    agent.load_checkpoint_compat(ckpt)
    _log(f"Loaded checkpoint: {args_cli.checkpoint}")

    # ── Reset + cube perturbations ──
    env.reset()
    sim_dt = env.sim_dt
    for eid, (dx, dy, name) in enumerate(CUBE_OFFSETS):
        cube_pos = env.cube.data.root_state_w[eid, :3].clone()
        cube_pos[0] += dx
        cube_pos[1] += dy
        cube_quat = env.cube.data.root_state_w[eid, 3:7].clone()
        new_pose = torch.cat([cube_pos, cube_quat]).unsqueeze(0)
        eid_tensor = torch.tensor([eid], device=device, dtype=torch.long)
        env.cube.write_root_pose_to_sim(new_pose, env_ids=eid_tensor)
        env.cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=eid_tensor)

    for _ in range(50):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)
    env._initial_cube_z = env.cube.data.root_state_w[:N, 2].clone()
    obs = env._build_obs()
    _log("Cube perturbations applied, starting rollout...")

    # ── Rollout ──
    max_steps = args_cli.max_episode_steps
    ep_rewards = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, device=device)
    frames_per_env = [[] for _ in range(N)]
    q_per_env = [[] for _ in range(N)]  # Q-values per step per env
    r_per_env = [[] for _ in range(N)]  # rewards per step per env

    t0 = time.time()
    _log(f"Running for up to {max_steps} steps...")

    for step in range(max_steps):
        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Compute Q(s, a_combined) for each env
            obs_copy = {k: v.clone() for k, v in obs.items()}
            obs_copy["feat"] = agent._encode(obs_copy, augment=False)
            combined_action = torch.clamp(
                obs["observation.base_action"] + residual_action, -1.0, 1.0
            )
            q_all = agent.critic(
                obs_copy["feat"], obs_copy["observation.state"], combined_action
            )  # [K, N, 1]
            q_mean = q_all.squeeze(-1).mean(dim=0)  # [N]

        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        ep_rewards += reward[:N].to(device)

        # Success check
        cube_z = env.cube.data.root_state_w[:N, 2].to(device)
        cube_init_z = env._initial_cube_z[:N].to(device)
        cube_lifted = ((cube_z - cube_init_z) >= args_cli.success_threshold).float()
        ep_success = torch.max(ep_success, cube_lifted)

        # Store per-env Q and reward
        for eid in range(N):
            q_per_env[eid].append(q_mean[eid].item())
            r_per_env[eid].append(reward[eid].item())

        # Capture annotated frame every 5 steps
        if step % 5 == 0 and hasattr(env, "get_all_frames_batch"):
            debug = getattr(env, "_last_step_debug", None)
            all_frames = env.get_all_frames_batch(camera="front", size=(256, 320))
            for eid in range(N):
                frame = all_frames[eid]
                cf = debug["contact_force"][eid].item() if debug else 0.0
                hc = debug["has_contact"][eid].item() if debug else 0.0
                ch = debug["cube_height"][eid].item() if debug else 0.0
                succ = debug["success"][eid].item() if debug else 0.0
                frame = annotate_frame_critic(
                    frame, eid, step,
                    gt_reward=reward[eid].item(),
                    q_value=q_mean[eid].item(),
                    cum_reward=ep_rewards[eid].item(),
                    contact_force=cf,
                    has_contact=hc,
                    cube_height=ch,
                    success=succ,
                )
                frames_per_env[eid].append(frame)

        obs = next_obs
        done = terminated | truncated
        if done[:N].all():
            break

    eval_time = time.time() - t0
    _log(f"Done in {eval_time:.1f}s, {step+1} steps")

    # ── Per-env results + log ──
    vid_dir = output_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args_cli.checkpoint).stem

    all_log_lines = []
    all_log_lines.append(f"Checkpoint: {args_cli.checkpoint}")
    all_log_lines.append(f"Steps: {step+1}, Time: {eval_time:.1f}s")
    all_log_lines.append("")

    for eid in range(N):
        dx, dy, name = CUBE_OFFSETS[eid]
        succ = bool(ep_success[eid].item() > 0)
        ret = float(ep_rewards[eid].item())
        q_arr = np.array(q_per_env[eid])
        r_arr = np.array(r_per_env[eid])
        cum_r = float(r_arr.sum())

        # Compute discounted return G_t
        G = np.zeros(len(r_arr))
        if len(r_arr) > 0:
            G[-1] = r_arr[-1]
            for t in range(len(r_arr) - 2, -1, -1):
                G[t] = r_arr[t] + args_cli.gamma * G[t + 1]

        q_G_corr = float(np.corrcoef(q_arr, G)[0, 1]) if len(q_arr) > 1 else 0.0
        mean_q = float(q_arr.mean()) if len(q_arr) > 0 else 0.0
        mean_gap = float(np.mean(np.abs(q_arr - G))) if len(q_arr) > 0 else 0.0

        line = (
            f"Env {eid} [{name:12s}] succ={succ} R={cum_r:.2f} "
            f"Q_mean={mean_q:.3f} G_0={G[0]:.3f} "
            f"Corr(Q,G)={q_G_corr:.3f} |Q-G|={mean_gap:.3f}"
        )
        _log(line)
        all_log_lines.append(line)

        # Save per-env video
        if frames_per_env[eid]:
            succ_tag = "succ" if succ else "fail"
            vpath = vid_dir / f"{ckpt_name}_env{eid}_{name}_{succ_tag}.mp4"
            w = imageio.get_writer(str(vpath), fps=20)
            for fr in frames_per_env[eid]:
                w.append_data(fr)
            w.close()

    # Grid video
    valid = [f for f in frames_per_env if len(f) > 0]
    if valid:
        min_len = min(len(f) for f in valid)
        cols, rows = 5, 2
        concat_path = vid_dir / f"{ckpt_name}_grid_10envs.mp4"
        w = imageio.get_writer(str(concat_path), fps=20)
        for fi in range(min_len):
            grid_rows = []
            for r in range(rows):
                row_frames = []
                for c in range(cols):
                    idx = r * cols + c
                    if idx < len(valid):
                        row_frames.append(valid[idx][fi])
                    else:
                        row_frames.append(np.zeros_like(valid[0][fi]))
                grid_rows.append(np.concatenate(row_frames, axis=1))
            grid = np.concatenate(grid_rows, axis=0)
            w.append_data(grid)
        w.close()
        _log(f"Grid video: {concat_path}")

    # Summary
    succ_total = int(ep_success.sum().item())
    succ_rate = succ_total / N
    all_q = np.concatenate([np.array(q) for q in q_per_env])
    all_r = np.concatenate([np.array(r) for r in r_per_env])

    all_log_lines.append("")
    all_log_lines.append(f"=== SUMMARY ({ckpt_name}) ===")
    all_log_lines.append(f"  Success: {succ_total}/{N} ({succ_rate*100:.0f}%)")
    all_log_lines.append(f"  Q-value: mean={all_q.mean():.3f} min={all_q.min():.3f} max={all_q.max():.3f}")
    all_log_lines.append(f"  Reward:  mean={all_r.mean():.4f} total={all_r.sum():.2f}")
    _log(f"SUMMARY: {succ_total}/{N} success, Q_mean={all_q.mean():.3f}")

    # Save log
    log_path = output_dir / f"{ckpt_name}_critic_online.log"
    with open(log_path, "w") as f:
        f.write("\n".join(all_log_lines))
    _log(f"Log: {log_path}")

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
