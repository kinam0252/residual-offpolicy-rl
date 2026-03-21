"""
Replay CSV joint trajectories in IsaacLab and compute shaped rewards.
Saves as LeRobot-style parquet + video files for offline RL data.

Usage (from isaaclab docker):
  cd /workspace/isaaclab
  ./isaaclab.sh -p /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/replay_csv_with_reward.py \
    --headless --num_envs 1 \
    --csv_dir /home/t-kinamkim/Repos/VLA_RL/Data/pickMushroom/pickMushroom_20251209_081744_174 \
    --output_dir /home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/offline_replay_output
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay CSV in IsaacLab with shaped reward")
parser.add_argument("--csv_dir", type=str, required=True, help="Path to a single CSV episode directory")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LeRobot data")
parser.add_argument("--num_envs", type=int, default=1)

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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation

# ── Load the scene config from iface_env_wrapper ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import (
    _make_scene_cfg,
    _quat_mul_wxyz,
    _yaw_to_quat_wxyz,
)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.math import subtract_frame_transforms


def main():
    csv_dir = Path(args_cli.csv_dir)
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = 1  # single env for replay
    sim_dt = 0.01

    # ── Create sim + scene ──
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.5, 0.0, 0.3])

    scene_cfg = _make_scene_cfg(num_envs=N)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Initialize camera poses (same as IfaceEnvWrapper.reset())
    env_origins = scene.env_origins
    eye_front = env_origins + torch.tensor([[0.4, -0.7, 0.8]], device="cuda:0")
    eye_back = env_origins + torch.tensor([[0.4, 0.7, 0.8]], device="cuda:0")
    target_pos = env_origins + torch.tensor([[0.25, 0.0, -0.05]], device="cuda:0")
    # Camera warmup
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        for cam_name in ["camera_front", "camera_back", "camera_wrist"]:
            scene[cam_name].update(dt=sim_dt)
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
    robot_materials = robot.root_physx_view.get_material_properties().to("cpu")
    robot_materials[..., 0] = 5e3
    robot_materials[..., 1] = 5e3
    robot.root_physx_view.set_material_properties(robot_materials, torch.arange(N, device="cpu", dtype=torch.long))

    # ── Read CSV ──
    js = pd.read_csv(csv_dir / "franka_joint_states.csv", header=0).values.astype(np.float32)
    gs = pd.read_csv(csv_dir / "gripper_joint_states.csv", header=0).values.astype(np.float32)
    delta_csv = pd.read_csv(csv_dir / "isaac_deltaEEF_gripper.csv", header=0).values.astype(np.float32)

    js = js[1:, 10:17]  # skip header row, take 7 arm joints
    gs = gs[1:, 4:5]    # gripper width

    T = min(len(js), len(gs), len(delta_csv))
    print(f"CSV: {T} timesteps from {csv_dir.name}")

    # ── Init scene from CSV row 0 ──
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

    # Settle
    for _ in range(200):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    initial_cube_z = cube.data.root_state_w[0, 2].clone()
    print(f"Initial cube z: {initial_cube_z.item():.4f}")

    # ── Replay loop ──
    states = []
    actions = []
    rewards_list = []
    reward_components = []
    timestamps = []
    video_frames_front = []
    video_frames_back = []
    video_frames_wrist = []

    cameras = {
        "front": scene["camera_front"],
        "back": scene["camera_back"],
        "wrist": scene["camera_wrist"],
    }

    steps_per_action = 5
    total_reward = 0.0
    success_step = -1

    for t in range(T):
        # Set joint target from CSV
        arm_pos = js[t, :7]
        grip_w = float(np.clip(gs[t, 0], 0.0, 1.0))
        finger_val = grip_w * 0.04
        q_target = np.concatenate([arm_pos, [finger_val] * len(hand_joint_ids)]).astype(np.float32)
        q_target_t = torch.tensor(q_target, device="cuda:0").unsqueeze(0)

        robot.set_joint_position_target(q_target_t, joint_ids=all_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Update cameras
        for cam in cameras.values():
            cam.update(dt=sim_dt)

        # ── Compute shaped reward (same as iface_env_wrapper) ──
        cube_pos = cube.data.root_state_w[:N, :3]
        cube_z = cube_pos[:, 2]

        left_fp = robot.data.body_pos_w[:N, left_finger_idx]
        right_fp = robot.data.body_pos_w[:N, right_finger_idx]
        finger_pos = (left_fp + right_fp) / 2.0

        # Stage 1: Distance
        finger_cube_dist = torch.norm(finger_pos - cube_pos, dim=-1)
        distance_reward = (1.0 - torch.tanh(finger_cube_dist / 0.1)) * 1.0

        # Stage 2: Grasp (proximity)
        left_dist = torch.norm(left_fp - cube_pos, dim=-1)
        right_dist = torch.norm(right_fp - cube_pos, dim=-1)
        both_close = ((left_dist < 0.05) & (right_dist < 0.05)).float()
        fj = robot.data.joint_pos[:N][:, hand_joint_ids[0]]
        gripper_closing = (fj < 0.03).float()
        grasp_gate = both_close * gripper_closing
        grasp_reward = grasp_gate * 2.0

        # Stage 3: Height
        cube_height = cube_z - initial_cube_z
        height_reward = (
            (cube_height > 0.005).float()
            * torch.tanh(cube_height / 0.1)
            * 100.0
            * grasp_gate
        )

        # Stage 4: Success (0.5cm threshold — matches online training)
        success_reward = (cube_height >= 0.005).float() * 100.0 * grasp_gate
        is_success = (cube_height >= 0.005).item()

        reward = (distance_reward + grasp_reward + height_reward + success_reward).item()
        total_reward += reward

        if is_success and success_step < 0:
            success_step = t

        # ── Record data ──
        state = np.concatenate([arm_pos, [grip_w]]).astype(np.float32)
        action = delta_csv[t, :7].astype(np.float32) if t < len(delta_csv) else np.zeros(7, dtype=np.float32)

        states.append(state)
        actions.append(action)
        rewards_list.append(reward)
        reward_components.append({
            "reward_total": reward,
            "reward_distance": distance_reward.item(),
            "reward_grasp": grasp_reward.item(),
            "reward_height": height_reward.item(),
            "reward_success": success_reward.item(),
            "cube_height": cube_height.item(),
            "finger_cube_dist": finger_cube_dist.item(),
            "grasp_gate": grasp_gate.item(),
            "finger_joint": fj.item(),
        })
        timestamps.append(t * sim_dt)

        # Save video frames every steps_per_action steps
        if t % steps_per_action == 0:
            for cam_name, cam in cameras.items():
                img = cam.data.output["rgb"][0].cpu().numpy()
                if cam_name == "front":
                    video_frames_front.append(img)
                elif cam_name == "back":
                    video_frames_back.append(img)
                elif cam_name == "wrist":
                    video_frames_wrist.append(img)

        if t % 100 == 0:
            print(f"  step={t}/{T} reward={reward:.3f} cube_h={cube_height.item():.4f} "
                  f"dist={finger_cube_dist.item():.3f} grasp={grasp_gate.item():.0f} "
                  f"fj={fj.item():.4f} total_R={total_reward:.1f}")

        # Early termination on success
        if is_success:
            print(f"  SUCCESS at step {t}! cube_height={cube_height.item():.4f}")
            # Record remaining as done
            break

    print(f"\nReplay done: {len(states)} steps, total_reward={total_reward:.1f}, success_step={success_step}")

    # ── Save as LeRobot parquet ──
    ep_name = csv_dir.name
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

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
            "episode_index": 0,
            "index": i,
            "task_index": 0,
        })

    df = pd.DataFrame(records)
    parquet_path = data_dir / "episode_000000.parquet"
    df.to_parquet(parquet_path)
    print(f"Saved parquet: {parquet_path} ({len(df)} rows)")

    # ── Save videos ──
    video_dir_base = output_dir / "videos" / "chunk-000"
    for view_name, frames in [("right_image", video_frames_front), ("image", video_frames_back), ("wrist_image", video_frames_wrist)]:
        if not frames:
            continue
        vdir = video_dir_base / view_name
        vdir.mkdir(parents=True, exist_ok=True)
        vpath = vdir / "episode_000000.mp4"
        w = imageio.get_writer(str(vpath), fps=20)
        for fr in frames:
            if fr.dtype != np.uint8:
                fr = (fr * 255).clip(0, 255).astype(np.uint8) if fr.max() <= 1.0 else fr.astype(np.uint8)
            w.append_data(fr)
        w.close()
        print(f"Saved video: {vpath} ({len(frames)} frames)")

    # ── Save metadata ──
    meta = {
        "csv_dir": str(csv_dir),
        "total_steps": len(states),
        "total_reward": total_reward,
        "success_step": success_step,
        "is_success": success_step >= 0,
        "initial_cube_z": float(initial_cube_z.item()),
    }
    with open(output_dir / "replay_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {output_dir / 'replay_meta.json'}")

    print("\nDone!")
    os._exit(0)


if __name__ == "__main__":
    main()
