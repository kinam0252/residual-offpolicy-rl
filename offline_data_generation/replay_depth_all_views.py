"""
Isaac Sim replay: capture RGB + GT Depth from ALL 3 camera views for one episode.
Then run Depth-Anything-V2 comparison and save 3 depth comparison videos.

Usage (inside Isaac Sim container via apptainer):
  isaac_python replay_depth_all_views.py --headless --enable_cameras \
    --csv_base_dir /path/to/pickMushroom_FIRST \
    --episode_idx 0 \
    --output_dir /path/to/output
"""
from __future__ import annotations
import argparse, os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--friction", type=float, default=50000.0)
parser.add_argument("--da_encoder", type=str, default="vitl", choices=["vits", "vitl"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path
import numpy as np, pandas as pd, torch, cv2, imageio
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import _make_scene_cfg, _yaw_to_quat_wxyz
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# ── Depth-Anything-V2 import ──
# parents[0] = offline_data_generation, parents[1] = residual-offpolicy-rl, parents[2] = Intern
DA_DIR = Path(__file__).resolve().parents[2] / "Depth-Anything-V2"
sys.path.insert(0, str(DA_DIR))
from depth_anything_v2.dpt import DepthAnythingV2


def _rgb_np(tensor, env_id=0):
    arr = tensor[env_id].cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _depth_np(tensor, env_id=0, size=(480, 640)):
    d = tensor[env_id].cpu().numpy().squeeze()
    d = np.nan_to_num(d, nan=10.0, posinf=10.0, neginf=0)
    return d  # raw meters, native resolution


def align_depth(pred, gt):
    valid = gt > 0.01
    if valid.sum() < 10:
        return pred.copy()
    p = pred[valid].flatten().astype(np.float64)
    g = gt[valid].flatten().astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    return np.clip(s * pred + t, 0, 20).astype(np.float32)


def depth_to_colormap(d, vmin, vmax):
    d = np.nan_to_num(d, nan=0, posinf=0, neginf=0)
    d_clip = np.clip(d, vmin, vmax)
    d_norm = (d_clip - vmin) / max(vmax - vmin, 1e-6)
    cmap = matplotlib.colormaps.get_cmap("turbo")
    colored = (cmap(1.0 - d_norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def put_text(img, text, org, fs=0.4, color=(255,255,255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
    x, y = org
    cv2.rectangle(img, (x-2, y-th-4), (x+tw+2, y+4), (0,0,0), -1)
    cv2.putText(img, text, (x, y), font, fs, color, 1, cv2.LINE_AA)


def reset_scene(robot, cube, scene, sim, sim_dt, csv_dir, all_ji, hand_ji, N=1):
    js = pd.read_csv(csv_dir / "franka_joint_states.csv", header=0).values.astype(np.float32)
    gs = pd.read_csv(csv_dir / "gripper_joint_states.csv", header=0).values.astype(np.float32)
    js_data = js[1:, 10:17]
    gs_data = gs[1:, 4:5]

    arm_init = js_data[0, :7]
    grip_w = float(np.clip(gs_data[0, 0], 0.0, 1.0))
    finger_val = grip_w * 0.04
    q_init = np.concatenate([arm_init, [finger_val] * len(hand_ji)]).astype(np.float32)
    q_init_t = torch.tensor(q_init, device="cuda:0").unsqueeze(0)
    qd_init_t = torch.zeros_like(q_init_t)

    robot.write_joint_state_to_sim(q_init_t, qd_init_t, joint_ids=all_ji)
    robot.set_joint_position_target(q_init_t, joint_ids=all_ji)

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
        robot.set_joint_position_target(q_init_t, joint_ids=all_ji)

    return js_data, gs_data


def main():
    csv_base = Path(args_cli.csv_base_dir)
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N = 1
    sim_dt = 0.01
    ep_idx = args_cli.episode_idx

    # Discover CSV dirs
    all_csv_dirs = sorted([d for d in csv_base.parent.iterdir()
                           if d.is_dir() and d.name.startswith("pickMushroom_")])
    if ep_idx >= len(all_csv_dirs):
        print(f"Episode {ep_idx} out of range ({len(all_csv_dirs)} episodes)")
        os._exit(1)
    csv_dir = all_csv_dirs[ep_idx]
    print(f"Episode {ep_idx}: {csv_dir.name}")

    # Create sim + scene WITH depth on all cameras
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = _make_scene_cfg(num_envs=N)
    scene_cfg.camera_front.data_types = ["rgb", "depth"]
    scene_cfg.camera_back.data_types = ["rgb", "depth"]
    scene_cfg.camera_wrist.data_types = ["rgb", "depth"]
    print("[OK] Cameras patched for RGB + Depth")

    scene = InteractiveScene(scene_cfg)
    sim.reset()

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

    # Friction
    friction_val = args_cli.friction
    rm = robot.root_physx_view.get_material_properties().to("cpu")
    rm[..., 0] = friction_val; rm[..., 1] = friction_val
    robot.root_physx_view.set_material_properties(rm, torch.arange(N, device="cpu", dtype=torch.long))
    cm_mat = cube.root_physx_view.get_material_properties().to("cpu")
    cm_mat[..., 0] = friction_val; cm_mat[..., 1] = friction_val
    cube.root_physx_view.set_material_properties(cm_mat, torch.arange(N, device="cpu", dtype=torch.long))

    # Camera mapping: video_key → scene camera name
    cam_map = {
        "front": "camera_front",    # right_image
        "back": "camera_back",      # image
        "wrist": "camera_wrist",    # wrist_image
    }

    # ── Load Depth-Anything-V2 ──
    da_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    da_model = DepthAnythingV2(**da_configs[args_cli.da_encoder])
    da_ckpt = DA_DIR / "checkpoints" / f"depth_anything_v2_{args_cli.da_encoder}.pth"
    da_model.load_state_dict(torch.load(str(da_ckpt), map_location="cpu"))
    da_model = da_model.to("cuda:0").eval()
    print(f"[OK] Depth-Anything-V2 ({args_cli.da_encoder}) loaded")

    # ── Reset & replay ──
    js_data, gs_data = reset_scene(robot, cube, scene, sim, sim_dt, csv_dir, all_ji, hand_ji)
    T = min(len(js_data), len(gs_data))
    print(f"Replaying {T} steps...")

    steps_per_csv_row = 5
    max_joint_step = 0.05

    # Buffers per view
    all_rgb = {k: [] for k in cam_map}
    all_gt_depth = {k: [] for k in cam_map}
    all_pred_depth = {k: [] for k in cam_map}

    t0 = time.time()
    for t in range(T):
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
        for cn in cam_map.values():
            scene[cn].update(dt=sim_dt)

        # Capture RGB + GT depth from each camera
        for view_name, cam_name in cam_map.items():
            cam = scene[cam_name]
            rgb = _rgb_np(cam.data.output["rgb"])         # (H, W, 3) uint8
            gt_d = _depth_np(cam.data.output["depth"])    # (H, W) float32
            all_rgb[view_name].append(rgb)
            all_gt_depth[view_name].append(gt_d)

            # DA-V2 prediction
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pred_rel = da_model.infer_image(bgr, input_size=518)
            all_pred_depth[view_name].append(pred_rel)

        if t % 20 == 0 or t == T - 1:
            elapsed = time.time() - t0
            print(f"  [{t+1}/{T}] ({(t+1)/max(elapsed,0.1):.1f} fps)")

    print(f"\nReplay done. Building depth comparison videos...")

    # ── Build comparison videos per view ──
    for view_name in cam_map:
        rgbs = all_rgb[view_name]
        gt_ds = all_gt_depth[view_name]
        pred_ds = all_pred_depth[view_name]

        # Align each predicted depth to GT
        aligned_ds = []
        metrics = []
        for i in range(T):
            gt_d = gt_ds[i]
            pred_aligned = align_depth(pred_ds[i], gt_d)
            aligned_ds.append(pred_aligned)

            valid = gt_d > 0.01
            if valid.sum() > 0:
                abs_err = np.abs(pred_aligned - gt_d)
                rmse = float(np.sqrt((abs_err[valid] ** 2).mean()))
                mae = float(abs_err[valid].mean())
            else:
                rmse = mae = 0.0
            metrics.append({"rmse": rmse, "mae": mae})

        # Unified colormap range
        all_vals = np.concatenate([np.stack(gt_ds).ravel(), np.stack(aligned_ds).ravel()])
        valid_vals = all_vals[(all_vals > 0.01) & (all_vals < 9.0)]
        if len(valid_vals) > 0:
            vmin = float(np.percentile(valid_vals, 1))
            vmax = float(np.percentile(valid_vals, 99))
        else:
            vmin, vmax = 0.1, 2.0

        # Build video frames: [RGB | GT Depth | Pred Depth]
        pw, ph = 320, 240
        vid_path = output_dir / f"episode_{ep_idx:06d}_depth_{view_name}.mp4"
        w = imageio.get_writer(str(vid_path), fps=20, format="FFMPEG",
                               codec="libx264", quality=8, pixelformat="yuv420p")
        for i in range(T):
            rgb_bgr = cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR)
            p1 = cv2.resize(rgb_bgr, (pw, ph))
            put_text(p1, f"RGB {view_name} (f{i})", (4, 16))

            gt_vis = depth_to_colormap(gt_ds[i], vmin, vmax)
            gt_vis = cv2.resize(gt_vis, (pw, ph))
            put_text(gt_vis, "GT Depth", (4, 16))

            pred_vis = depth_to_colormap(aligned_ds[i], vmin, vmax)
            pred_vis = cv2.resize(pred_vis, (pw, ph))
            put_text(pred_vis, f"Pred (DA-V2)", (4, 16))

            row = np.concatenate([p1, gt_vis, pred_vis], axis=1)
            bar = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
            m = metrics[i]
            cv2.putText(bar, f"RMSE={m['rmse']:.4f}m  MAE={m['mae']:.4f}m  |  [{vmin:.2f}, {vmax:.2f}]m",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1, cv2.LINE_AA)
            frame = np.concatenate([row, bar], axis=0)
            w.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w.close()

        mean_rmse = np.mean([m["rmse"] for m in metrics])
        mean_mae = np.mean([m["mae"] for m in metrics])
        print(f"  {view_name}: {vid_path.name}  RMSE={mean_rmse:.4f}m  MAE={mean_mae:.4f}m")

    print(f"\nAll done! Output: {output_dir}")
    os._exit(0)


if __name__ == "__main__":
    main()
