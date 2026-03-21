"""
Replay ALL CSV episodes in IsaacLab, compute shaped rewards, save as LeRobot dataset.

Usage (from isaaclab docker):
  cd /workspace/isaaclab
  ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_all_csv_with_reward.py \
    --headless --enable_cameras --num_envs 1 \
    --csv_base_dir /home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom \
    --output_dir /home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_w_reward
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay all CSVs with shaped reward")
parser.add_argument("--csv_base_dir", type=str, required=True, help="Parent dir containing pickMushroom_* folders")
parser.add_argument("--output_dir", type=str, required=True, help="Output LeRobot dataset directory")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes (for testing)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Cancel faulthandler
import faulthandler
try:
    faulthandler.cancel_dump_traceback_later()
except Exception:
    pass

import json
import time
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import (
    _make_scene_cfg,
    _quat_mul_wxyz,
    _yaw_to_quat_wxyz,
)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene


# ── Reward parameters (documented) ──
REWARD_CONFIG = {
    "distance_reward_std": 0.1,
    "distance_reward_weight": 1.0,
    "grasp_finger_threshold": 0.05,     # meters, both fingers within this of cube
    "grasp_gripper_threshold": 0.03,    # meters, finger joint < this = closing
    "grasp_reward_weight": 2.0,
    "height_minimal": 0.005,            # meters, min height for height reward
    "height_reward_std": 0.1,
    "height_reward_weight": 100.0,
    "success_height_threshold": 0.005,  # meters (5mm — matches online training)
    "success_reward_weight": 100.0,
    "early_termination": True,
    "early_termination_threshold": 0.005,  # same as success
}


def compute_shaped_reward(
    finger_pos, cube_pos, left_fp, right_fp, finger_joint_pos, cube_z, initial_cube_z,
):
    """Compute 4-stage shaped reward. Returns (reward_total, reward_dict, is_success)."""
    cfg = REWARD_CONFIG

    # Stage 1: Distance
    finger_cube_dist = torch.norm(finger_pos - cube_pos, dim=-1)
    distance_reward = (1.0 - torch.tanh(finger_cube_dist / cfg["distance_reward_std"])) * cfg["distance_reward_weight"]

    # Stage 2: Grasp (proximity)
    left_dist = torch.norm(left_fp - cube_pos, dim=-1)
    right_dist = torch.norm(right_fp - cube_pos, dim=-1)
    both_close = ((left_dist < cfg["grasp_finger_threshold"]) & (right_dist < cfg["grasp_finger_threshold"])).float()
    gripper_closing = (finger_joint_pos < cfg["grasp_gripper_threshold"]).float()
    grasp_gate = both_close * gripper_closing
    grasp_reward = grasp_gate * cfg["grasp_reward_weight"]

    # Stage 3: Height
    cube_height = cube_z - initial_cube_z
    height_reward = (
        (cube_height > cfg["height_minimal"]).float()
        * torch.tanh(cube_height / cfg["height_reward_std"])
        * cfg["height_reward_weight"]
        * grasp_gate
    )

    # Stage 4: Success
    success_reward = (
        (cube_height >= cfg["success_height_threshold"]).float()
        * cfg["success_reward_weight"]
        * grasp_gate
    )

    total = distance_reward + grasp_reward + height_reward + success_reward
    is_success = (cube_height >= cfg["success_height_threshold"]).item()

    return total.item(), {
        "distance": distance_reward.item(),
        "grasp": grasp_reward.item(),
        "height": height_reward.item(),
        "success": success_reward.item(),
        "cube_height": cube_height.item(),
        "finger_cube_dist": finger_cube_dist.item(),
        "grasp_gate": grasp_gate.item(),
    }, is_success


def reset_scene(scene, sim, csv_dir, all_joint_ids, hand_joint_ids, sim_dt):
    """Reset robot + cube from CSV, settle, return initial_cube_z."""
    N = 1
    robot = scene["robot"]
    cube = scene["cube"]

    js = pd.read_csv(csv_dir / "franka_joint_states.csv", header=0).values.astype(np.float32)
    gs = pd.read_csv(csv_dir / "gripper_joint_states.csv", header=0).values.astype(np.float32)
    js = js[1:, 10:17]
    gs = gs[1:, 4:5]

    arm_init = js[0, :7]
    grip_width = float(np.clip(gs[0, 0], 0.0, 1.0))
    finger_pos_val = grip_width * 0.04
    q_init = np.concatenate([arm_init, [finger_pos_val] * len(hand_joint_ids)]).astype(np.float32)
    q_init_t = torch.tensor(q_init, device="cuda:0").unsqueeze(0)
    qd_init_t = torch.zeros_like(q_init_t)

    robot.write_joint_state_to_sim(q_init_t, qd_init_t, joint_ids=all_joint_ids)
    robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    # Cube init
    object_pos_csv = csv_dir / "object_pos.csv"
    if object_pos_csv.exists():
        od = pd.read_csv(object_pos_csv, header=0).values.astype(np.float32)
        obj = od[0, 3:7].copy()
    else:
        obj = np.array([0.2, 0.0, 0.06, 0.0], dtype=np.float32)
    env_origins = scene.env_origins
    obj_pos_w = env_origins + torch.tensor(obj[:3], device="cuda:0").unsqueeze(0)
    default_rot = cube.data.default_root_state[0, 3:7].cpu().numpy()
    yaw_quat = _yaw_to_quat_wxyz(float(obj[3]))
    base_r = Rotation.from_quat([default_rot[1], default_rot[2], default_rot[3], default_rot[0]])
    yaw_r = Rotation.from_quat([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]])
    cq = (yaw_r * base_r).as_quat()
    obj_quat_w = torch.tensor([[cq[3], cq[0], cq[1], cq[2]]], device="cuda:0", dtype=torch.float32)
    cube.write_root_pose_to_sim(torch.cat([obj_pos_w, obj_quat_w], dim=-1))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device="cuda:0"))

    # Settle
    for _ in range(200):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    initial_cube_z = cube.data.root_state_w[0, 2].clone()
    return initial_cube_z, js, gs


def main():
    csv_base = Path(args_cli.csv_base_dir)
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = 1
    sim_dt = 0.01

    # ── Create sim + scene ──
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = _make_scene_cfg(num_envs=N)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Camera warmup + pose setup
    env_origins = scene.env_origins
    eye_front = env_origins + torch.tensor([[0.4, -0.7, 0.8]], device="cuda:0")
    eye_back = env_origins + torch.tensor([[0.4, 0.7, 0.8]], device="cuda:0")
    target_pos = env_origins + torch.tensor([[0.25, 0.0, -0.05]], device="cuda:0")
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        for cn in ["camera_front", "camera_back", "camera_wrist"]:
            scene[cn].update(dt=sim_dt)
    scene["camera_front"].set_world_poses_from_view(eye_front, target_pos)
    scene["camera_back"].set_world_poses_from_view(eye_back, target_pos)
    scene.update(sim_dt)

    robot = scene["robot"]
    cube = scene["cube"]

    # Finger body indices
    left_finger_idx = robot.find_bodies("panda_leftfinger")[0][0]
    right_finger_idx = robot.find_bodies("panda_rightfinger")[0][0]

    # Joint IDs
    arm_joint_names = sorted([n for n in robot.data.joint_names if "panda_joint" in n and "finger" not in n])
    arm_joint_ids = [robot.data.joint_names.index(n) for n in arm_joint_names]
    hand_joint_names = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
    hand_joint_ids = [robot.data.joint_names.index(n) for n in hand_joint_names]
    all_joint_ids = arm_joint_ids + hand_joint_ids

    # Friction boost
    mat = robot.root_physx_view.get_material_properties().to("cpu")
    mat[..., 0] = 5e3; mat[..., 1] = 5e3
    robot.root_physx_view.set_material_properties(mat, torch.arange(N, device="cpu", dtype=torch.long))

    cameras = {"front": scene["camera_front"], "back": scene["camera_back"], "wrist": scene["camera_wrist"]}
    steps_per_action = 5

    # ── Find all CSV episodes ──
    csv_dirs = sorted(csv_base.glob("pickMushroom_*"))
    csv_dirs = [d for d in csv_dirs if d.is_dir() and (d / "franka_joint_states.csv").exists()]
    if args_cli.max_episodes:
        csv_dirs = csv_dirs[:args_cli.max_episodes]
    print(f"Found {len(csv_dirs)} episodes to replay")

    # ── Output dirs ──
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_base = output_dir / "videos" / "chunk-000"
    for view in ["right_image", "image", "wrist_image"]:
        (video_base / view).mkdir(parents=True, exist_ok=True)

    # ── Stats ──
    episode_stats = []
    global_index = 0
    total_start = time.time()

    for ep_idx, csv_dir in enumerate(csv_dirs):
        ep_start = time.time()

        # Read CSV
        try:
            delta_csv = pd.read_csv(csv_dir / "isaac_deltaEEF_gripper.csv", header=0).values.astype(np.float32)
        except Exception:
            delta_csv = np.zeros((200, 7), dtype=np.float32)

        # Reset scene
        initial_cube_z, js, gs = reset_scene(scene, sim, csv_dir, all_joint_ids, hand_joint_ids, sim_dt)
        T = min(len(js), len(gs))

        # Update cameras after reset
        for cam in cameras.values():
            cam.update(dt=sim_dt)
        scene["camera_front"].set_world_poses_from_view(eye_front, target_pos)
        scene["camera_back"].set_world_poses_from_view(eye_back, target_pos)

        states = []
        actions = []
        rewards_list = []
        reward_components = []
        timestamps = []
        vid_front, vid_back, vid_wrist = [], [], []
        total_reward = 0.0
        success_step = -1

        for t in range(T):
            arm_pos = js[t, :7]
            grip_w = float(np.clip(gs[t, 0], 0.0, 1.0))
            finger_val = grip_w * 0.04
            q_target = np.concatenate([arm_pos, [finger_val] * len(hand_joint_ids)]).astype(np.float32)
            q_target_t = torch.tensor(q_target, device="cuda:0").unsqueeze(0)

            robot.set_joint_position_target(q_target_t, joint_ids=all_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            for cam in cameras.values():
                cam.update(dt=sim_dt)

            # Reward
            cube_pos = cube.data.root_state_w[:N, :3]
            cube_z = cube_pos[:, 2]
            left_fp = robot.data.body_pos_w[:N, left_finger_idx]
            right_fp = robot.data.body_pos_w[:N, right_finger_idx]
            finger_pos = (left_fp + right_fp) / 2.0
            fj = robot.data.joint_pos[:N][:, hand_joint_ids[0]]

            reward, rd, is_success = compute_shaped_reward(
                finger_pos, cube_pos, left_fp, right_fp, fj, cube_z, initial_cube_z
            )
            total_reward += reward
            if is_success and success_step < 0:
                success_step = t

            state = np.concatenate([arm_pos, [grip_w]]).astype(np.float32)
            action = delta_csv[t, :7].astype(np.float32) if t < len(delta_csv) else np.zeros(7, dtype=np.float32)
            states.append(state)
            actions.append(action)
            rewards_list.append(reward)
            reward_components.append({
                "reward_total": reward,
                "reward_distance": rd["distance"],
                "reward_grasp": rd["grasp"],
                "reward_height": rd["height"],
                "reward_success": rd["success"],
                "cube_height": rd["cube_height"],
                "finger_cube_dist": rd["finger_cube_dist"],
                "grasp_gate": rd["grasp_gate"],
                "finger_joint": float(fj.item()),
            })
            timestamps.append(t * sim_dt)

            if t % steps_per_action == 0:
                for cn, frames_list in [("front", vid_front), ("back", vid_back), ("wrist", vid_wrist)]:
                    img = cameras[cn].data.output["rgb"][0].cpu().numpy()
                    frames_list.append(img)

            if is_success:
                break

        # ── Save parquet ──
        ep_name = f"episode_{ep_idx:06d}"
        records = []
        for i in range(len(states)):
            rc = reward_components[i]
            records.append({
                "state": states[i].tolist(),
                "actions": actions[i].tolist(),
                "reward": rewards_list[i],
                "reward_total": rc["reward_total"],
                "reward_distance": rc["reward_distance"],
                "reward_grasp": rc["reward_grasp"],
                "reward_height": rc["reward_height"],
                "reward_success": rc["reward_success"],
                "cube_height": rc["cube_height"],
                "finger_cube_dist": rc["finger_cube_dist"],
                "grasp_gate": rc["grasp_gate"],
                "finger_joint": rc["finger_joint"],
                "timestamp": timestamps[i],
                "frame_index": i,
                "episode_index": ep_idx,
                "index": global_index + i,
                "task_index": 0,
            })
        global_index += len(states)
        df = pd.DataFrame(records)
        df.to_parquet(data_dir / f"{ep_name}.parquet")

        # ── Save videos ──
        view_map = [("right_image", vid_front), ("image", vid_back), ("wrist_image", vid_wrist)]
        for view_name, frames in view_map:
            if not frames:
                continue
            vpath = video_base / view_name / f"{ep_name}.mp4"
            w = imageio.get_writer(str(vpath), fps=20)
            for fr in frames:
                if fr.dtype != np.uint8:
                    fr = (fr * 255).clip(0, 255).astype(np.uint8) if fr.max() <= 1.0 else fr.astype(np.uint8)
                w.append_data(fr)
            w.close()

        ep_time = time.time() - ep_start
        stat = {
            "episode": ep_idx,
            "csv_dir": csv_dir.name,
            "steps": len(states),
            "total_reward": round(total_reward, 2),
            "success": success_step >= 0,
            "success_step": success_step,
            "time_sec": round(ep_time, 1),
        }
        episode_stats.append(stat)

        n_success = sum(1 for s in episode_stats if s["success"])
        if (ep_idx + 1) % 5 == 0 or ep_idx == 0:
            print(f"[{ep_idx+1}/{len(csv_dirs)}] {csv_dir.name}: {len(states)} steps, "
                  f"R={total_reward:.1f}, success={success_step>=0} ({ep_time:.1f}s) "
                  f"| cumulative success: {n_success}/{ep_idx+1}")

    # ── Save metadata ──
    total_time = time.time() - total_start
    n_success = sum(1 for s in episode_stats if s["success"])
    meta = {
        "description": "Offline LeRobot dataset with shaped rewards from CSV replay in IsaacLab",
        "reward_config": REWARD_CONFIG,
        "reward_formula": (
            "reward = distance_reward + grasp_reward + height_reward + success_reward\n"
            "  distance_reward = (1 - tanh(finger_cube_dist / 0.1)) * 1.0  [always active]\n"
            "  grasp_reward = (both_fingers < 5cm) * (finger_joint < 3cm) * 2.0\n"
            "  height_reward = (cube_h > 0.5cm) * tanh(cube_h / 0.1) * 100.0 * grasp_gate\n"
            "  success_reward = (cube_h >= 3cm) * 100.0 * grasp_gate\n"
            "  early_termination: episode ends when cube_h >= 3cm"
        ),
        "total_episodes": len(episode_stats),
        "successful_episodes": n_success,
        "success_rate": round(n_success / max(1, len(episode_stats)), 4),
        "total_transitions": global_index,
        "generation_time_sec": round(total_time, 1),
        "episode_stats": episode_stats,
    }
    meta_path = output_dir / "reward_config_and_stats.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DONE: {len(episode_stats)} episodes, {global_index} transitions")
    print(f"Success: {n_success}/{len(episode_stats)} ({100*n_success/max(1,len(episode_stats)):.1f}%)")
    print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Output: {output_dir}")
    print(f"Reward config: {meta_path}")
    print(f"{'='*60}")

    os._exit(0)


if __name__ == "__main__":
    main()
