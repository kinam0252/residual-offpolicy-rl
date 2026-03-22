"""
Replay CSV episodes in IsaacLab to capture depth images.

Based on offline_data_generation/replay_all_worker_with_groot.py.
Replays the SAME joint trajectories from CSV data, with depth-enabled cameras,
and saves depth frames as compressed .npz alongside the existing lerobot data.

Output (saved INTO the existing lerobot data dir):
  {output_dir}/depth_front/episode_XXXXXX.npz   -> 'frames': (T, 84, 84) float32 raw meters
  {output_dir}/depth_wrist/episode_XXXXXX.npz

Depth is stored RAW (meters). Normalization happens at training time.
"""
from __future__ import annotations
import argparse, os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True, help="Existing lerobot data dir to add depth to")
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--friction", type=float, default=50000.0)
parser.add_argument("--max_episodes", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import faulthandler
try: faulthandler.cancel_dump_traceback_later()
except: pass

from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import _make_scene_cfg, _yaw_to_quat_wxyz
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene


def _rgb_to_uint8_np(rgb_tensor, env_id=0):
    rgb = rgb_tensor[env_id]
    if rgb.is_cuda:
        rgb = rgb.cpu()
    arr = rgb.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def capture_depth(cam, env_id=0, target_size=(84, 84)):
    """Capture depth from camera, resize to target_size, return raw meters (H, W) float32."""
    d = cam.data.output["depth"][env_id]  # (H, W, 1) or (H, W)
    if d.is_cuda:
        d = d.cpu()
    d = d.numpy().squeeze()  # (H, W)
    d = np.nan_to_num(d, nan=10.0, posinf=10.0, neginf=0)
    # Resize
    d_t = torch.tensor(d).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    d_resized = F.interpolate(d_t, size=target_size, mode="bilinear", align_corners=False)
    return d_resized[0, 0].numpy().astype(np.float32)  # (84, 84)


def reset_scene(robot, cube, scene, sim, sim_dt, csv_dir, all_joint_ids, hand_joint_ids, N=1):
    """Reset to CSV initial state (same as original replay worker)."""
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

    return js_data, gs_data


def main():
    wid = args_cli.worker_id
    nw = args_cli.num_workers
    csv_base = Path(args_cli.csv_base_dir)
    output_dir = Path(args_cli.output_dir)
    N = 1
    sim_dt = 0.01

    # Discover CSV dirs (same logic as original)
    all_csv_dirs = sorted([d for d in csv_base.parent.iterdir()
                           if d.is_dir() and d.name.startswith("pickMushroom_")])
    if not all_csv_dirs:
        all_csv_dirs = sorted(csv_base.parent.glob("pickMushroom_*"))
    total = len(all_csv_dirs)
    print(f"[Worker {wid}] Found {total} CSV dirs")

    # Create output dirs FIRST
    depth_front_dir = output_dir / "depth_front"
    depth_wrist_dir = output_dir / "depth_wrist"
    depth_front_dir.mkdir(parents=True, exist_ok=True)
    depth_wrist_dir.mkdir(parents=True, exist_ok=True)

    def _claim_next_batch(batch_size=3):
        """Scan for unclaimed episodes and claim a small batch via lock files."""
        claimed = []
        for ep_idx, csv_dir in enumerate(all_csv_dirs):
            if len(claimed) >= batch_size:
                break
            ep_name = f"episode_{ep_idx:06d}"
            front_out = depth_front_dir / f"{ep_name}.npz"
            wrist_out = depth_wrist_dir / f"{ep_name}.npz"
            lock_file = depth_front_dir / f"{ep_name}.lock"
            if front_out.exists() and wrist_out.exists():
                continue  # done
            if lock_file.exists():
                # Check if lock is stale (>30 min)
                try:
                    age = time.time() - lock_file.stat().st_mtime
                    if age < 1800:  # 30 min
                        continue  # another worker is on it
                except:
                    continue
            # Try to claim with lock
            try:
                lock_file.write_text(f"worker_{wid}_{os.getpid()}_{time.time():.0f}")
                claimed.append((ep_idx, csv_dir, lock_file))
            except:
                continue
        return claimed

    # Create sim + scene WITH depth cameras
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = _make_scene_cfg(num_envs=N)
    # Patch cameras for depth
    scene_cfg.camera_front.data_types = ["rgb", "depth"]
    scene_cfg.camera_back.data_types = ["rgb", "depth"]
    scene_cfg.camera_wrist.data_types = ["rgb", "depth"]
    print("[DEPTH] Cameras patched for RGB + Depth output")

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Camera warmup
    env_origins = scene.env_origins
    eye_front = env_origins + torch.tensor([[0.4, -0.7, 0.8]], device="cuda:0")
    eye_back = env_origins + torch.tensor([[0.4, 0.7, 0.8]], device="cuda:0")
    target_pos = env_origins + torch.tensor([[0.25, 0.0, -0.05]], device="cuda:0")
    for _ in range(10):
        scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
        for cn in ["camera_front", "camera_back", "camera_wrist"]:
            scene[cn].update(dt=sim_dt)
    scene["camera_front"].set_world_poses_from_view(eye_front, target_pos)
    scene["camera_back"].set_world_poses_from_view(eye_back, target_pos)
    scene.update(sim_dt)

    robot = scene["robot"]
    cube = scene["cube"]
    arm_jn = sorted([n for n in robot.data.joint_names if "panda_joint" in n and "finger" not in n])
    arm_ji = [robot.data.joint_names.index(n) for n in arm_jn]
    hand_jn = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
    hand_ji = [robot.data.joint_names.index(n) for n in hand_jn]
    all_ji = arm_ji + hand_ji

    # Friction (match original)
    friction_val = args_cli.friction
    rm = robot.root_physx_view.get_material_properties().to("cpu")
    rm[..., 0] = friction_val; rm[..., 1] = friction_val
    robot.root_physx_view.set_material_properties(rm, torch.arange(N, device="cpu", dtype=torch.long))
    cm = cube.root_physx_view.get_material_properties().to("cpu")
    cm[..., 0] = friction_val; cm[..., 1] = friction_val
    cube.root_physx_view.set_material_properties(cm, torch.arange(N, device="cpu", dtype=torch.long))

    steps_per_csv_row = 5
    max_joint_step = 0.05

    # Depth stats tracking
    all_front_mins, all_front_maxs = [], []
    all_wrist_mins, all_wrist_maxs = [], []

    t0 = time.time()
    total_frames = 0
    total_done = 0
    skipped = 0

    while True:
        batch = _claim_next_batch(batch_size=3)
        if not batch:
            print(f"[W{wid}] No more episodes to process. Done.")
            break
        
        for ep_global, csv_dir, lock_file in batch:
            ep_name = f"episode_{ep_global:06d}"

            # Double-check not done (race condition guard)
            front_out = depth_front_dir / f"{ep_name}.npz"
            wrist_out = depth_wrist_dir / f"{ep_name}.npz"
            if front_out.exists() and wrist_out.exists():
                skipped += 1
                try: lock_file.unlink(missing_ok=True)
                except: pass
                continue

        # Check CSV exists
        if not (csv_dir / "franka_joint_states.csv").exists():
            print(f"[W{wid}] Skip {csv_dir.name}: no joint states CSV")
            continue

        # Reset scene to CSV initial state
        js_data, gs_data = reset_scene(robot, cube, scene, sim, sim_dt, csv_dir, all_ji, hand_ji)
        T = min(len(js_data), len(gs_data))

        depth_front_frames = []
        depth_wrist_frames = []

        for t in range(T):
            # Replay joint trajectory (same as original worker)
            arm_pos = js_data[t, :7]
            grip_w = float(np.clip(gs_data[t, 0], 0.3, 1.0))
            finger_val = grip_w * 0.04
            q_target = np.concatenate([arm_pos, [finger_val] * len(hand_ji)]).astype(np.float32)
            q_target_t = torch.tensor(q_target, device="cuda:0").unsqueeze(0)

            for sub_step in range(steps_per_csv_row):
                prev_target = robot.data.joint_pos_target[:1, all_ji]
                delta = q_target_t - prev_target
                delta_clamped = torch.clamp(delta, -max_joint_step, max_joint_step)
                smoothed_target = prev_target + delta_clamped
                robot.set_joint_position_target(smoothed_target, joint_ids=all_ji)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)

            # Update cameras
            for cn in ["camera_front", "camera_wrist"]:
                scene[cn].update(dt=sim_dt)

            # Capture depth
            df = capture_depth(scene["camera_front"], 0, (84, 84))
            dw = capture_depth(scene["camera_wrist"], 0, (84, 84))
            depth_front_frames.append(df)
            depth_wrist_frames.append(dw)

        # Stack and save
        depth_front_arr = np.stack(depth_front_frames).astype(np.float32)  # (T, 84, 84)
        depth_wrist_arr = np.stack(depth_wrist_frames).astype(np.float32)

        np.savez_compressed(str(front_out), frames=depth_front_arr)
        np.savez_compressed(str(wrist_out), frames=depth_wrist_arr)

        # Remove lock file
        try:
            lock_file.unlink(missing_ok=True)
        except:
            pass

        # Track stats
        all_front_mins.append(depth_front_arr[depth_front_arr > 0.01].min() if (depth_front_arr > 0.01).any() else 0.1)
        all_front_maxs.append(depth_front_arr[depth_front_arr < 9.0].max() if (depth_front_arr < 9.0).any() else 2.0)
        all_wrist_mins.append(depth_wrist_arr[depth_wrist_arr > 0.01].min() if (depth_wrist_arr > 0.01).any() else 0.03)
        all_wrist_maxs.append(depth_wrist_arr[depth_wrist_arr < 4.0].max() if (depth_wrist_arr < 4.0).any() else 0.5)

        total_frames += T
        total_done += 1
        elapsed = time.time() - t0
        fps = total_frames / max(elapsed, 1)

        if total_done % 3 == 0 or total_done == 1:
            print(f"[W{wid}] done={total_done} ep={ep_global} steps={T} "
                  f"front=[{depth_front_arr.min():.3f}, {depth_front_arr.max():.3f}] "
                  f"wrist=[{depth_wrist_arr.min():.3f}, {depth_wrist_arr.max():.3f}] "
                  f"({fps:.1f} fps, {elapsed:.0f}s)")

    # Save depth stats
    elapsed = time.time() - t0
    stats = {
        "worker_id": wid,
        "episodes": total_done,
        "skipped": skipped,
        "total_frames": total_frames,
        "time_sec": elapsed,
        "front_depth": {
            "min": float(np.percentile(all_front_mins, 2)) if all_front_mins else 0.3,
            "max": float(np.percentile(all_front_maxs, 98)) if all_front_maxs else 1.8,
        },
        "wrist_depth": {
            "min": float(np.percentile(all_wrist_mins, 2)) if all_wrist_mins else 0.03,
            "max": float(np.percentile(all_wrist_maxs, 98)) if all_wrist_maxs else 0.4,
        },
    }
    stats_path = output_dir / f"depth_worker_{wid}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== DEPTH REPLAY DONE (Worker {wid}) ===")
    print(f"Episodes: {total_done} ({skipped} skipped)")
    print(f"Frames: {total_frames} in {elapsed:.0f}s ({total_frames/max(elapsed,1):.0f} fps)")
    print(f"Front depth range: [{stats['front_depth']['min']:.4f}, {stats['front_depth']['max']:.4f}]")
    print(f"Wrist depth range: [{stats['wrist_depth']['min']:.4f}, {stats['wrist_depth']['max']:.4f}]")
    print(f"Output: {output_dir}")

    os._exit(0)


if __name__ == "__main__":
    main()
