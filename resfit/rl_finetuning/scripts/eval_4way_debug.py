"""
Debug: 4-way eval (step0/step2500 × train/unseen envs).
Info bar is ABOVE the frame (concatenated), not overlaid.

Usage (inside container):
  isaaclab.sh -p eval_4way_debug.py --headless --enable_cameras \
    --ckpt_dir /path/to/checkpoints \
    --csv_base_dir /path/to/csv \
    --groot_model_path /path/to/groot \
    --output_dir /path/to/output
"""
from __future__ import annotations
import argparse, os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="4-way eval debug")
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--output_dir", type=str, default="/tmp/eval_4way_debug")
parser.add_argument("--cube_perturb_table_path", type=str, default=None)
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


# Train env offsets (from cube_perturb_table.json)
TRAIN_OFFSETS = [
    (0.000, 0.000, "center"),
    (0.050, 0.000, "right_5cm"),
    (-0.050, 0.000, "left_5cm"),
    (0.000, 0.050, "forward_5cm"),
    (0.000, -0.050, "back_5cm"),
    (0.040, 0.040, "diag_FR_5.6cm"),
    (-0.040, 0.040, "diag_FL_5.6cm"),
    (0.040, -0.040, "diag_BR_5.6cm"),
    (-0.040, -0.040, "diag_BL_5.6cm"),
    (0.080, 0.000, "right_8cm"),
]

# Unseen env offsets: shifted from train positions by ~2-3cm
UNSEEN_OFFSETS = [
    (0.025, 0.020, "center+shift"),
    (0.070, 0.025, "right_5cm+shift"),
    (-0.070, -0.020, "left_5cm+shift"),
    (0.025, 0.070, "fwd_5cm+shift"),
    (-0.020, -0.070, "back_5cm+shift"),
    (0.060, 0.060, "diag_FR+shift"),
    (-0.060, 0.060, "diag_FL+shift"),
    (0.060, -0.060, "diag_BR+shift"),
    (-0.060, -0.060, "diag_BL+shift"),
    (0.100, 0.025, "right_10cm+shift"),
]


def make_info_bar(
    width: int,
    env_id: int,
    env_name: str,
    step: int,
    contact_force: float,
    has_contact: float,
    cube_height: float,
    reward: float,
    success: bool,
    ckpt_name: str,
    env_type: str,
) -> np.ndarray:
    """Create an info bar image (H_bar, W, 3) to be placed ABOVE the frame."""
    bar_h = 80
    bar = np.zeros((bar_h, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)  # dark gray background

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.38, 1

    # Row 1: env info + checkpoint
    cv2.putText(bar, f"E{env_id}:{env_name}", (4, 13), font, fs, (255, 255, 255), th)
    cv2.putText(bar, f"{ckpt_name} [{env_type}]", (width - 160, 13), font, fs, (200, 200, 200), th)

    # Row 2: step + contact
    force_color = (0, 255, 0) if has_contact > 0.5 else (100, 100, 100)
    cv2.putText(bar, f"Step:{step}", (4, 28), font, fs, (200, 200, 200), th)
    cv2.putText(bar, f"Force:{contact_force:.1f}N", (90, 28), font, fs, force_color, th)
    cv2.putText(bar, f"[{'ON' if has_contact > 0.5 else 'OFF'}]", (210, 28), font, fs, force_color, th)

    # Row 3: height + reward
    h_color = (0, 255, 0) if cube_height > 0.005 else (150, 150, 150)
    cv2.putText(bar, f"H:{cube_height*100:.1f}cm", (4, 43), font, fs, h_color, th)
    r_color = (0, 255, 0) if reward > 0.01 else (150, 150, 150)
    cv2.putText(bar, f"R:{reward:.4f}", (100, 43), font, fs, r_color, th)
    if success:
        cv2.putText(bar, "SUCCESS", (210, 43), font, fs, (0, 255, 0), 2)

    # Row 4: force bar + height bar
    bar_max_w = width - 8
    # Force bar
    f_fill = int(min(contact_force / 5.0, 1.0) * bar_max_w)
    cv2.rectangle(bar, (4, 50), (4 + bar_max_w, 55), (50, 50, 50), -1)
    if f_fill > 0:
        cv2.rectangle(bar, (4, 50), (4 + f_fill, 55), force_color, -1)
    # Height bar
    h_fill = int(min(max(cube_height, 0) / 0.05, 1.0) * bar_max_w)
    cv2.rectangle(bar, (4, 58), (4 + bar_max_w, 63), (50, 50, 50), -1)
    if h_fill > 0:
        cv2.rectangle(bar, (4, 58), (4 + h_fill, 63), h_color, -1)

    # Labels
    cv2.putText(bar, "F", (4, 73), font, 0.28, (100, 100, 100), 1)
    cv2.putText(bar, "H", (4 + bar_max_w // 2, 73), font, 0.28, (100, 100, 100), 1)

    return bar


def run_one_eval(env, agent, device, offsets, ckpt_path, ckpt_name, env_type, output_dir, image_keys):
    """Run eval with given offsets, save per-env + grid videos."""
    N = len(offsets)
    max_steps = args_cli.max_episode_steps
    success_threshold = args_cli.success_threshold

    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device)
    agent.load_checkpoint_compat(ckpt)
    agent.eval()
    _log(f"Loaded {ckpt_name}")

    # Reset and apply perturbations
    env.reset()
    sim_dt = env.sim_dt
    for eid, (dx, dy, name) in enumerate(offsets):
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

    # Run eval
    ep_rewards = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, device=device)
    frames_per_env = [[] for _ in range(N)]
    FRAME_SIZE = (192, 256)  # H, W for video frame

    _log(f"Running {ckpt_name} x {env_type} ({N} envs, {max_steps} steps)...")
    t0 = time.time()

    for step in range(max_steps):
        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        ep_rewards += reward[:N].to(device)

        cube_z = env.cube.data.root_state_w[:N, 2].to(device)
        cube_init_z = env._initial_cube_z[:N].to(device)
        cube_lifted = ((cube_z - cube_init_z) >= success_threshold).float()
        ep_success = torch.max(ep_success, cube_lifted)

        # Capture frame every 5 steps
        if step % 5 == 0 and hasattr(env, "get_all_frames_batch"):
            debug = getattr(env, "_last_step_debug", None)
            all_frames = env.get_all_frames_batch(camera="front", size=FRAME_SIZE)
            for eid in range(N):
                frame = all_frames[eid]  # (H, W, 3) uint8
                # Create info bar
                cf = debug["contact_force"][eid].item() if debug else 0.0
                hc = debug["has_contact"][eid].item() if debug else 0.0
                ch = debug["cube_height"][eid].item() if debug else 0.0
                rw = debug["reward"][eid].item() if debug else 0.0
                succ = bool(ep_success[eid].item() > 0)
                info_bar = make_info_bar(
                    FRAME_SIZE[1], eid, offsets[eid][2], step,
                    cf, hc, ch, rw, succ, ckpt_name, env_type,
                )
                # Concat: info_bar on top of frame
                combined = np.concatenate([info_bar, frame], axis=0)
                frames_per_env[eid].append(combined)

        obs = next_obs
        done = terminated | truncated
        if done[:N].all():
            break

    eval_time = time.time() - t0
    _log(f"Done in {eval_time:.0f}s, {step+1} steps")

    # Save results
    run_name = f"{ckpt_name}_{env_type}"
    vid_dir = output_dir / run_name
    vid_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for eid in range(N):
        dx, dy, name = offsets[eid]
        succ = bool(ep_success[eid].item() > 0)
        ret = float(ep_rewards[eid].item())
        results.append({"env": eid, "name": name, "dx": dx, "dy": dy, "success": succ, "return": round(ret, 2)})
        _log(f"  E{eid} [{name:16s}] succ={succ} R={ret:.1f}")

        # Per-env video
        if frames_per_env[eid]:
            vpath = vid_dir / f"env{eid}_{name}.mp4"
            w = imageio.get_writer(str(vpath), fps=20)
            for fr in frames_per_env[eid]:
                w.append_data(fr)
            w.close()

    # Grid video (2 rows x 5 cols)
    valid = [f for f in frames_per_env if len(f) > 0]
    if valid:
        min_len = min(len(f) for f in valid)
        cols, rows = 5, 2
        concat_path = vid_dir / f"grid_{run_name}.mp4"
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
        _log(f"Grid: {concat_path}")

    succ_rate = sum(1 for r in results if r["success"]) / N
    summary = {"ckpt": ckpt_name, "env_type": env_type, "success_rate": succ_rate, "per_env": results}
    with open(vid_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  => {run_name}: {succ_rate*100:.0f}% success")
    return summary


def main():
    N = 10
    device = torch.device("cuda:0")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args_cli.ckpt_dir)

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

    # 4 eval runs
    all_results = []
    for ckpt_name, ckpt_file in [("step0", "agent_step0.pt"), ("step2500", "agent_step2500.pt")]:
        ckpt_path = ckpt_dir / ckpt_file
        if not ckpt_path.exists():
            _log(f"SKIP: {ckpt_path} not found")
            continue
        for env_type, offsets in [("train", TRAIN_OFFSETS), ("unseen", UNSEEN_OFFSETS)]:
            r = run_one_eval(env, agent, device, offsets, ckpt_path, ckpt_name, env_type, output_dir, image_keys)
            all_results.append(r)

    # Summary
    _log("\n" + "=" * 60)
    _log("SUMMARY:")
    for r in all_results:
        _log(f"  {r['ckpt']:8s} x {r['env_type']:6s} => {r['success_rate']*100:.0f}%")
    _log("=" * 60)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
