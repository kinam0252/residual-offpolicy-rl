"""
Debug eval: Step 0 (base policy) with 10 cube position perturbations.
Runs once, saves 10 per-env videos, then exits.

Usage (inside container):
  isaaclab.sh -p eval_debug_cube_perturb.py --headless --enable_cameras \
    --checkpoint /path/to/agent_step0.pt \
    --csv_base_dir /path/to/csv \
    --groot_model_path /path/to/groot
"""
from __future__ import annotations
import argparse, os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug eval with cube perturbations")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.005)
parser.add_argument("--output_dir", type=str, default="/tmp/debug_cube_perturb")
parser.add_argument("--perturb_range", type=float, default=0.02, help="Max offset in meters (default 2cm)")
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

    # Semi-transparent dark background for text readability
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40
    thickness = 1

    # Contact force: green if active (>0.1N), red if not
    force_color = (0, 255, 0) if has_contact > 0.5 else (0, 0, 255)
    cv2.putText(img, f"Contact: {contact_force:.2f}N", (4, 14), font, font_scale, force_color, thickness)

    # Has contact indicator
    indicator = "ON" if has_contact > 0.5 else "OFF"
    cv2.putText(img, f"[{indicator}]", (160, 14), font, font_scale, force_color, thickness)

    # Cube height
    h_color = (0, 255, 0) if cube_height > 0.005 else (200, 200, 200)
    cv2.putText(img, f"Height: {cube_height*100:.2f}cm", (4, 30), font, font_scale, h_color, thickness)

    # Finger-cube distance
    cv2.putText(img, f"Dist: {finger_cube_dist*100:.2f}cm", (160, 30), font, font_scale, (200, 200, 200), thickness)

    # Reward
    r_color = (0, 255, 0) if reward > 0.01 else (200, 200, 200)
    cv2.putText(img, f"Reward: {reward:.4f}", (4, 46), font, font_scale, r_color, thickness)

    # Success flag
    if success > 0.5:
        cv2.putText(img, "SUCCESS", (160, 46), font, font_scale, (0, 255, 0), 2)

    # Step & env
    cv2.putText(img, f"Env:{env_id} Step:{step}", (4, 62), font, font_scale, (255, 255, 255), thickness)

    # Contact force bar (visual indicator)
    bar_max_w = w - 8
    bar_fill = int(min(contact_force / 5.0, 1.0) * bar_max_w)  # scale: 5N = full bar
    bar_y = 70
    cv2.rectangle(img, (4, bar_y), (4 + bar_max_w, bar_y + 8), (60, 60, 60), -1)
    if bar_fill > 0:
        cv2.rectangle(img, (4, bar_y), (4 + bar_fill, bar_y + 8), force_color, -1)
    cv2.putText(img, "Force", (4, bar_y + 20), font, 0.30, (150, 150, 150), 1)

    # Cube height bar
    h_bar_fill = int(min(max(cube_height, 0) / 0.05, 1.0) * bar_max_w)  # scale: 5cm = full
    h_bar_y = bar_y + 24
    cv2.rectangle(img, (4, h_bar_y), (4 + bar_max_w, h_bar_y + 8), (60, 60, 60), -1)
    if h_bar_fill > 0:
        cv2.rectangle(img, (4, h_bar_y), (4 + h_bar_fill, h_bar_y + 8), h_color, -1)
    cv2.putText(img, "Height", (4, h_bar_y + 20), font, 0.30, (150, 150, 150), 1)

    return img


# 10 cube offsets: center + 8 around + 1 further
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


def main():
    N = len(CUBE_OFFSETS)  # 10 envs
    device = torch.device("cuda:0")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Build agent
    image_keys = ["observation.images.front", "observation.images.back", "observation.images.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]

    agent_cfg = QAgentConfig(
        actor_lr=3e-7, critic_lr=1e-4, critic_target_tau=0.005,
        actor=ActorConfig(action_scale=0.05, actor_last_layer_init_scale=0.0, action_l2_reg_weight=10.0),
    )
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w), prop_shape=(lowdim_dim,),
        action_dim=action_dim, rl_cameras=image_keys, cfg=agent_cfg, residual_actor=True,
    )
    agent.to(device)

    # Load checkpoint (backward-compatible with prop_dim changes)
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    agent.load_checkpoint_compat(ckpt)
    _log(f"Loaded checkpoint: {args_cli.checkpoint}")

    # Reset env (standard positions)
    env.reset()
    _log("Env reset done, applying cube perturbations...")

    # Apply per-env cube offsets
    sim_dt = env.sim_dt
    for eid, (dx, dy, name) in enumerate(CUBE_OFFSETS):
        # Read current cube position and shift it
        cube_pos = env.cube.data.root_state_w[eid, :3].clone()
        cube_pos[0] += dx  # X offset
        cube_pos[1] += dy  # Y offset
        cube_quat = env.cube.data.root_state_w[eid, 3:7].clone()
        new_pose = torch.cat([cube_pos, cube_quat]).unsqueeze(0)
        eid_tensor = torch.tensor([eid], device=device, dtype=torch.long)
        env.cube.write_root_pose_to_sim(new_pose, env_ids=eid_tensor)
        env.cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=eid_tensor)
        _log(f"  Env {eid}: {name} (dx={dx:.3f}, dy={dy:.3f})")

    # Settle after perturbation
    for _ in range(50):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)

    # Record initial cube Z for success check
    env._initial_cube_z = env.cube.data.root_state_w[:N, 2].clone()

    obs = env._build_obs()

    # Run eval
    max_steps = args_cli.max_episode_steps
    success_threshold = args_cli.success_threshold
    ep_rewards = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, device=device)
    ep_first_succ_step = torch.full((N,), float(max_steps), device=device)
    frames_per_env = [[] for _ in range(N)]

    _log(f"Running eval for up to {max_steps} steps...")
    t0 = time.time()

    for step in range(max_steps):
        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        ep_rewards += reward[:N].to(device)

        cube_z = env.cube.data.root_state_w[:N, 2].to(device)
        cube_init_z = env._initial_cube_z[:N].to(device)
        cube_lifted = ((cube_z - cube_init_z) >= success_threshold).float()
        just_succ = (cube_lifted > 0) & (ep_success == 0)
        ep_first_succ_step[just_succ] = step
        ep_success = torch.max(ep_success, cube_lifted)

        # Capture frame every 5 steps with contact force overlay (GPU-batched)
        if step % 5 == 0 and hasattr(env, "get_all_frames_batch"):
            debug = getattr(env, "_last_step_debug", None)
            all_frames = env.get_all_frames_batch(camera="front", size=(256, 320))
            for eid in range(N):
                frame = all_frames[eid]
                if debug is not None:
                    frame = _annotate_frame(
                        frame, eid, step,
                        contact_force=debug["contact_force"][eid].item(),
                        has_contact=debug["has_contact"][eid].item(),
                        cube_height=debug["cube_height"][eid].item(),
                        reward=debug["reward"][eid].item(),
                        finger_cube_dist=debug["finger_cube_dist"][eid].item(),
                        success=debug["success"][eid].item(),
                    )
                frames_per_env[eid].append(frame)

        obs = next_obs
        done = terminated | truncated
        if done[:N].all():
            break

    eval_time = time.time() - t0
    _log(f"Eval done in {eval_time:.1f}s, {step+1} steps")

    # Results per env
    vid_dir = output_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for eid in range(N):
        dx, dy, name = CUBE_OFFSETS[eid]
        succ = bool(ep_success[eid].item() > 0)
        ret = float(ep_rewards[eid].item())
        succ_step = int(ep_first_succ_step[eid].item()) if succ else -1
        results.append({
            "env": eid, "name": name, "dx": dx, "dy": dy,
            "success": succ, "return": round(ret, 3), "success_step": succ_step,
        })
        _log(f"  Env {eid} [{name:12s}] dx={dx:+.3f} dy={dy:+.3f} => success={succ} R={ret:.1f} step={succ_step}")

        # Save per-env video
        if frames_per_env[eid]:
            vpath = vid_dir / f"env{eid}_{name}.mp4"
            w = imageio.get_writer(str(vpath), fps=20)
            for fr in frames_per_env[eid]:
                w.append_data(fr)
            w.close()

    # Save concat grid video (2 rows x 5 cols)
    valid = [f for f in frames_per_env if len(f) > 0]
    if valid:
        min_len = min(len(f) for f in valid)
        cols, rows = 5, 2
        concat_path = vid_dir / "grid_all_10envs.mp4"
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
        _log(f"Grid video: {concat_path} ({min_len} frames)")

    # Save JSON
    summary = {
        "checkpoint": args_cli.checkpoint,
        "csv_base_dir": args_cli.csv_base_dir,
        "success_threshold": success_threshold,
        "perturb_range": args_cli.perturb_range,
        "total_success": sum(1 for r in results if r["success"]),
        "total_envs": N,
        "success_rate": sum(1 for r in results if r["success"]) / N,
        "eval_time_sec": round(eval_time, 1),
        "per_env": results,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"Results: {json_path}")
    _log(f"SUMMARY: {summary['total_success']}/{N} success ({summary['success_rate']*100:.0f}%)")

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
