"""
Worker: Replay a range of CSV episodes in IsaacLab, compute shaped rewards, save as LeRobot.
Each worker runs its own Isaac Sim instance and processes a shard of episodes.

Usage:
  isaaclab.sh -p replay_all_worker.py --headless --enable_cameras --num_envs 1 \
    --csv_base_dir /path/to/pickMushroom \
    --output_dir /path/to/output \
    --worker_id 0 --num_workers 8
"""
from __future__ import annotations
import argparse, os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--worker_id", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--success_threshold", type=float, default=0.005)
parser.add_argument("--retry_episodes_json", type=str, default=None, help="JSON file with list of episode indices to retry")
parser.add_argument("--friction", type=float, default=50000.0, help="Gripper+cube friction")
parser.add_argument("--reward_type", type=str, default="dense_clipped", choices=["dense", "sparse", "dense_clipped"], help="Reward type")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import faulthandler
try: faulthandler.cancel_dump_traceback_later()
except: pass

from pathlib import Path
import numpy as np, pandas as pd, torch, imageio
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import _make_scene_cfg, _yaw_to_quat_wxyz
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

SUCCESS_THRESH = args_cli.success_threshold
REWARD_TYPE = args_cli.reward_type

def compute_reward(robot, cube, left_finger_idx, right_finger_idx, hand_joint_ids, initial_cube_z, scene, N=1):
    cube_pos = cube.data.root_state_w[:N, :3]
    cube_z = cube_pos[:, 2]
    left_fp = robot.data.body_pos_w[:N, left_finger_idx]
    right_fp = robot.data.body_pos_w[:N, right_finger_idx]
    finger_pos = (left_fp + right_fp) / 2.0

    finger_cube_dist = torch.norm(finger_pos - cube_pos, dim=-1)
    cube_height = cube_z - initial_cube_z
    is_success = bool((cube_height >= SUCCESS_THRESH).any().item())

    # Contact force from ContactSensor
    contact_force_val = 0.0
    has_contact_val = 0.0
    try:
        contact_sensor = scene["contact_forces"]
        forces = contact_sensor.data.net_forces_w[:N]  # (N, num_bodies, 3)
        if not hasattr(compute_reward, '_finger_sensor_ids'):
            body_names = contact_sensor.body_names
            compute_reward._finger_sensor_ids = [i for i, n in enumerate(body_names) if 'finger' in n.lower()]
            if not compute_reward._finger_sensor_ids:
                compute_reward._finger_sensor_ids = list(range(max(0, len(body_names)-2), len(body_names)))
            print(f"[compute_reward] Contact sensor finger IDs: {compute_reward._finger_sensor_ids} from {body_names}")
        if compute_reward._finger_sensor_ids:
            finger_forces = forces[:, compute_reward._finger_sensor_ids, :]  # (N, n_fingers, 3)
            contact_force_val = float(finger_forces.sum(dim=1).norm(dim=-1)[0].item())
            has_contact_val = 1.0 if contact_force_val > 0.1 else 0.0
    except Exception:
        pass

    has_contact = torch.tensor([has_contact_val], device=cube_pos.device)

    if REWARD_TYPE == "sparse":
        reward = 1.0 if is_success else 0.0
        return reward, is_success, {
            "reward_total": reward,
            "reward_distance": 0.0,
            "reward_grasp": 0.0,
            "reward_height": 0.0,
            "reward_success": reward,
            "cube_h": cube_height.item(),
            "dist": finger_cube_dist.item(),
            "grasp": 0.0,
            "fj": 0.0,
            "contact_force": contact_force_val,
            "has_contact": has_contact_val,
        }

    # Dense shaped reward
    distance_reward = (1.0 - torch.tanh(finger_cube_dist / 0.1)) * 1.0

    left_dist = torch.norm(left_fp - cube_pos, dim=-1)
    right_dist = torch.norm(right_fp - cube_pos, dim=-1)
    both_close = ((left_dist < 0.05) & (right_dist < 0.05)).float()
    fj = robot.data.joint_pos[:N][:, hand_joint_ids[0]]
    gripper_closing = (fj < 0.03).float()
    grasp_gate = both_close * gripper_closing * has_contact

    if REWARD_TYPE == "dense":
        grasp_reward = grasp_gate * 2.0
        height_reward = (cube_height > 0.005).float() * torch.tanh(cube_height / 0.1) * 100.0 * grasp_gate
        success_reward = (cube_height >= SUCCESS_THRESH).float() * 100.0 * grasp_gate
        reward = (distance_reward + grasp_reward + height_reward + success_reward).item()
    elif REWARD_TYPE == "dense_clipped":
        distance_reward = (1.0 - torch.tanh(finger_cube_dist / 0.1)) * 0.1
        contact_reward = grasp_gate * 0.2
        height_reward = (cube_height > 0.005).float() * torch.tanh(cube_height / 0.1) * 0.5 * grasp_gate
        success_reward = (cube_height >= SUCCESS_THRESH).float() * 1.0 * grasp_gate
        reward = torch.clamp(distance_reward + contact_reward + height_reward + success_reward, 0.0, 1.0).item()
    else:
        raise ValueError(f"Unknown reward_type: {REWARD_TYPE}")

    return reward, is_success, {
        "reward_total": reward,
        "reward_distance": distance_reward.item(),
        "reward_grasp": (grasp_gate * 2.0).item() if REWARD_TYPE == "dense" else (grasp_gate * 0.2).item(),
        "reward_height": height_reward.item(),
        "reward_success": success_reward.item(),
        "cube_h": cube_height.item(),
        "dist": finger_cube_dist.item(),
        "grasp": grasp_gate.item(),
        "fj": fj.item(),
        "contact_force": contact_force_val,
        "has_contact": has_contact_val,
    }


def reset_scene(robot, cube, scene, sim, sim_dt, csv_dir, all_joint_ids, hand_joint_ids, N=1):
    js = pd.read_csv(csv_dir / "franka_joint_states.csv", header=0).values.astype(np.float32)
    gs = pd.read_csv(csv_dir / "gripper_joint_states.csv", header=0).values.astype(np.float32)
    js_data = js[1:, 10:17]
    gs_data = gs[1:, 4:5]

    arm_init = js_data[0, :7]
    grip_w = float(np.clip(gs_data[0, 0], 0.0, 1.0))
    finger_val = grip_w * 0.04
    q_init = np.concatenate([arm_init, [finger_val] * len(hand_joint_ids)]).astype(np.float32)
    q_init_t = torch.tensor(q_init, device="cuda:0").unsqueeze(0)
    qd_init_t = torch.zeros_like(q_init_t)

    robot.write_joint_state_to_sim(q_init_t, qd_init_t, joint_ids=all_joint_ids)
    robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    # Cube
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
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device="cuda:0"))

    for _ in range(200):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    initial_cube_z = cube.data.root_state_w[0, 2].clone()
    return js_data, gs_data, initial_cube_z


def main():
    wid = args_cli.worker_id
    nw = args_cli.num_workers
    csv_base = Path(args_cli.csv_base_dir)
    output_dir = Path(args_cli.output_dir)
    N = 1
    sim_dt = 0.01

    # Discover all CSV dirs
    all_csv_dirs = sorted([d for d in csv_base.parent.iterdir() if d.is_dir() and d.name.startswith("pickMushroom_")])
    if not all_csv_dirs:
        all_csv_dirs = sorted(csv_base.parent.glob("pickMushroom_*"))
    total = len(all_csv_dirs)

    # Filter to retry episodes only (if specified)
    retry_set = None
    if args_cli.retry_episodes_json:
        with open(args_cli.retry_episodes_json) as f:
            retry_set = set(json.load(f))
        print(f"[Worker {wid}] Retry mode: {len(retry_set)} episodes to retry")

    # Shard
    shard_size = (total + nw - 1) // nw
    start = wid * shard_size
    end = min(start + shard_size, total)
    my_dirs = all_csv_dirs[start:end]
    print(f"[Worker {wid}] Processing episodes {start}-{end-1} ({len(my_dirs)} episodes out of {total})")

    # Create sim + scene
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = _make_scene_cfg(num_envs=N)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Camera warmup
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
    left_finger_idx = robot.find_bodies("panda_leftfinger")[0][0]
    right_finger_idx = robot.find_bodies("panda_rightfinger")[0][0]
    arm_jn = sorted([n for n in robot.data.joint_names if "panda_joint" in n and "finger" not in n])
    arm_ji = [robot.data.joint_names.index(n) for n in arm_jn]
    hand_jn = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
    hand_ji = [robot.data.joint_names.index(n) for n in hand_jn]
    all_ji = arm_ji + hand_ji

    # Friction boost (very high for sticky grip)
    friction_val = args_cli.friction
    rm = robot.root_physx_view.get_material_properties().to("cpu")
    rm[..., 0] = friction_val; rm[..., 1] = friction_val
    robot.root_physx_view.set_material_properties(rm, torch.arange(N, device="cpu", dtype=torch.long))
    # Also boost cube friction
    cm = cube.root_physx_view.get_material_properties().to("cpu")
    cm[..., 0] = friction_val; cm[..., 1] = friction_val
    cube.root_physx_view.set_material_properties(cm, torch.arange(N, device="cpu", dtype=torch.long))
    print(f"[Worker {wid}] Friction set to {friction_val} (robot + cube)")

    cameras = {
        "right_image": scene["camera_front"],
        "image": scene["camera_back"],
        "wrist_image": scene["camera_wrist"],
    }
    steps_per_action = 5

    # Output dirs
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_base = output_dir / "videos" / "chunk-000"
    for vn in ["right_image", "image", "wrist_image"]:
        (video_base / vn).mkdir(parents=True, exist_ok=True)

    summary = []
    t0 = time.time()

    for ep_local, csv_dir in enumerate(my_dirs):
        ep_global = start + ep_local
        ep_name = f"episode_{ep_global:06d}"

        # Skip if not in retry set
        if retry_set is not None and ep_global not in retry_set:
            continue

        # Read CSV
        delta_csv_path = csv_dir / "isaac_deltaEEF_gripper.csv"
        if not delta_csv_path.exists():
            print(f"[W{wid}] Skip {csv_dir.name}: no delta CSV")
            continue

        js_data, gs_data, initial_cube_z = reset_scene(
            robot, cube, scene, sim, sim_dt, csv_dir, all_ji, hand_ji)
        delta_csv = pd.read_csv(delta_csv_path, header=0).values.astype(np.float32)
        T = min(len(js_data), len(gs_data), len(delta_csv))

        states, actions, rewards_list, reward_components, timestamps = [], [], [], [], []
        vid_frames = {k: [] for k in cameras}
        total_reward = 0.0
        success_step = -1

        # Stable motion replay (matching franka_motion_from_pose_csv.py):
        # - 5 sim steps per CSV row (20Hz playback at 0.01s sim_dt)
        # - max_joint_step = 0.05 clamp per sim step
        # - gripper width clamped [0.3, 1.0]
        steps_per_csv_row = 5
        max_joint_step = 0.05
        sim_step = 0

        for t in range(T):
            arm_pos = js_data[t, :7]
            grip_w = float(np.clip(gs_data[t, 0], 0.3, 1.0))  # match attached code
            finger_val = grip_w * 0.04
            q_target = np.concatenate([arm_pos, [finger_val] * len(hand_ji)]).astype(np.float32)
            q_target_t = torch.tensor(q_target, device="cuda:0").unsqueeze(0)

            # Run multiple sim steps per CSV row (stable 20Hz playback)
            for sub_step in range(steps_per_csv_row):
                # Clamp joint delta for smooth motion
                prev_target = robot.data.joint_pos_target[:1, all_ji]
                delta = q_target_t - prev_target
                delta_clamped = torch.clamp(delta, -max_joint_step, max_joint_step)
                smoothed_target = prev_target + delta_clamped

                robot.set_joint_position_target(smoothed_target, joint_ids=all_ji)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                sim_step += 1

            # Update cameras after all sub-steps
            for cam in cameras.values():
                cam.update(dt=sim_dt)

            reward, is_success, dbg = compute_reward(
                robot, cube, left_finger_idx, right_finger_idx, hand_ji, initial_cube_z, scene)
            total_reward += reward
            if is_success and success_step < 0:
                success_step = t

            # Build state: EEF pos(3) + quat_xyzw(4) + gripper_qpos(2) = 9D
            # (matches online IfaceEnvWrapper._build_obs)
            hand_body_idx = robot.find_bodies("panda_hand")[0][0]
            ee_pos = robot.data.body_pos_w[0, hand_body_idx].detach().cpu().numpy()  # (3,)
            ee_quat_wxyz = robot.data.body_quat_w[0, hand_body_idx].detach().cpu().numpy()  # (4,) wxyz
            ee_quat_xyzw = np.array([ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]])  # xyzw
            gripper_qpos = robot.data.joint_pos[0, hand_ji].detach().cpu().numpy()  # (2,)
            state_9d = np.concatenate([ee_pos, ee_quat_xyzw, gripper_qpos]).astype(np.float32)  # (9,)
            states.append(state_9d)

            # Build normalized base action: dp/0.02, drot/0.2, grip*2-1
            raw_action = delta_csv[t, :7].astype(np.float32) if t < len(delta_csv) else np.zeros(7, dtype=np.float32)
            base_action_norm = np.concatenate([
                raw_action[:3] / 0.02,
                raw_action[3:6] / 0.2,
                [raw_action[6] * 2.0 - 1.0],
            ]).astype(np.float32)
            actions.append(base_action_norm)
            rewards_list.append(reward)
            reward_components.append(dbg)
            timestamps.append(t * steps_per_csv_row * sim_dt)

            # Capture video frame (1 per CSV row = 20 fps)
            for vn, cam in cameras.items():
                img = cam.data.output["rgb"][0].cpu().numpy()
                vid_frames[vn].append(img)

            if is_success:
                break

        # Save parquet
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
                "cube_height": rc["cube_h"],
                "finger_cube_dist": rc["dist"],
                "grasp_gate": rc["grasp"],
                "finger_joint": rc["fj"],
                "contact_force": rc.get("contact_force", 0.0),
                "has_contact": rc.get("has_contact", 0.0),
                "timestamp": timestamps[i],
                "frame_index": i,
                "episode_index": ep_global,
                "index": i,
                "task_index": 0,
            })
        df = pd.DataFrame(records)
        df.to_parquet(data_dir / f"{ep_name}.parquet")

        # Save videos
        for vn, frames in vid_frames.items():
            if frames:
                vpath = video_base / vn / f"{ep_name}.mp4"
                w = imageio.get_writer(str(vpath), fps=20)
                for fr in frames:
                    if fr.dtype != np.uint8:
                        fr = (fr * 255).clip(0, 255).astype(np.uint8) if fr.max() <= 1.0 else fr.astype(np.uint8)
                    w.append_data(fr)
                w.close()

        ep_info = {
            "episode_global": ep_global,
            "csv_dir": csv_dir.name,
            "steps": len(states),
            "total_reward": total_reward,
            "success_step": success_step,
            "is_success": success_step >= 0,
        }
        summary.append(ep_info)

        elapsed = time.time() - t0
        eta = elapsed / (ep_local + 1) * (len(my_dirs) - ep_local - 1)
        print(f"[W{wid}] {ep_local+1}/{len(my_dirs)} ep={ep_global} "
              f"steps={len(states)} R={total_reward:.1f} succ={success_step >= 0} "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    # Save worker summary
    summary_path = output_dir / f"worker_{wid}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    n_succ = sum(1 for s in summary if s["is_success"])
    print(f"[Worker {wid}] Done: {len(summary)} episodes, {n_succ} successes ({100*n_succ/max(1,len(summary)):.1f}%)")
    os._exit(0)


if __name__ == "__main__":
    main()
