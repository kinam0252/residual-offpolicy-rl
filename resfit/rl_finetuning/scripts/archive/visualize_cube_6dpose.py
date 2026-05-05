"""
Visualize cube 6D pose from Isaac Sim during a GR00T base-policy rollout.

Runs a single episode, extracts cube position (x,y,z) and orientation (quat wxyz)
every step, overlays it on the front-camera RGB frame, and saves a video.

Usage (inside container):
  isaaclab.sh -p visualize_cube_6dpose.py --headless --enable_cameras \
    --groot_model_path /path/to/groot/checkpoint \
    --csv_base_dir /path/to/csv \
    --output_path /path/to/output.mp4
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize cube 6D pose")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (only env 0 is visualized)")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--cube_perturb_table_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None, help="Output video path (default: outputs/cube_6dpose_vis.mp4)")
parser.add_argument("--fps", type=int, default=20, help="Video FPS")
parser.add_argument("--frame_size", type=str, default="480x640", help="HxW frame size for video")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import cv2
import imageio
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
import isaaclab.sim as sim_utils


def quat_wxyz_to_euler_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to euler angles (roll, pitch, yaw) in degrees."""
    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r = Rotation.from_quat(xyzw)
    return r.as_euler('xyz', degrees=True)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return Rotation.from_quat(xyzw).as_matrix()


def project_world_to_pixel(point_world: np.ndarray,
                           cam_pos: np.ndarray, cam_R: np.ndarray,
                           K: np.ndarray,
                           native_wh: tuple[int, int],
                           render_wh: tuple[int, int]) -> tuple[int, int] | None:
    """Project a 3D world point to 2D pixel coordinates.

    Camera convention: IsaacLab "world" (forward=+X, left=+Y, up=+Z).
    To convert to OpenCV/pinhole (right=+X, down=+Y, forward=+Z):
        cv_x = -cam_y,  cv_y = -cam_z,  cv_z = cam_x
    """
    # World → camera frame
    p_cam = cam_R.T @ (point_world - cam_pos)

    # "world" convention → OpenCV pinhole convention
    p_cv = np.array([-p_cam[1], -p_cam[2], p_cam[0]])

    if p_cv[2] <= 0.01:  # behind camera
        return None

    # Pinhole projection
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * p_cv[0] / p_cv[2] + cx
    v = fy * p_cv[1] / p_cv[2] + cy

    # Scale from native resolution to render resolution
    u_scaled = int(u * render_wh[0] / native_wh[0])
    v_scaled = int(v * render_wh[1] / native_wh[1])
    return (u_scaled, v_scaled)


def draw_3d_axes_on_image(frame: np.ndarray,
                          origin_world: np.ndarray, obj_R_world: np.ndarray,
                          cam_pos: np.ndarray, cam_R: np.ndarray,
                          K: np.ndarray,
                          native_wh: tuple[int, int],
                          render_wh: tuple[int, int],
                          axis_length: float = 0.05,
                          thickness: int = 2,
                          label: str = "") -> np.ndarray:
    """Draw 3D orientation axes (X=Red, Y=Green, Z=Blue) at the projected
    world position on the camera image.
    """
    # Project origin
    origin_px = project_world_to_pixel(origin_world, cam_pos, cam_R, K, native_wh, render_wh)
    if origin_px is None:
        return frame

    h, w = frame.shape[:2]
    ox, oy = origin_px
    # Skip if way outside frame
    if ox < -100 or ox > w + 100 or oy < -100 or oy > h + 100:
        return frame

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=Red, Y=Green, Z=Blue (BGR)
    axis_labels = ["X", "Y", "Z"]

    for i in range(3):
        # Axis tip in world frame
        axis_dir = obj_R_world[:, i]  # rotated unit axis
        tip_world = origin_world + axis_dir * axis_length
        tip_px = project_world_to_pixel(tip_world, cam_pos, cam_R, K, native_wh, render_wh)
        if tip_px is None:
            continue

        cv2.arrowedLine(frame, (ox, oy), tip_px, colors[i], thickness, tipLength=0.25)
        cv2.putText(frame, axis_labels[i], (tip_px[0] + 3, tip_px[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    if label:
        cv2.putText(frame, label, (ox - 15, oy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def draw_pose_overlay(frame: np.ndarray, step: int,
                      cube_pos: np.ndarray, cube_quat_wxyz: np.ndarray,
                      ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray,
                      cube_height: float, finger_dist: float,
                      has_contact: bool, reward: float,
                      cam_pos: np.ndarray, cam_R: np.ndarray,
                      K: np.ndarray, native_wh: tuple[int, int]) -> np.ndarray:
    """Draw 6D pose information overlay on frame with projected 3D axes."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    render_wh = (w, h)

    # Semi-transparent dark background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (w - 5, 220), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Convert quaternions to euler
    cube_euler = quat_wxyz_to_euler_deg(cube_quat_wxyz)
    ee_euler = quat_wxyz_to_euler_deg(ee_quat_wxyz)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    color_white = (255, 255, 255)
    color_green = (0, 255, 0)
    color_cyan = (0, 255, 255)
    color_yellow = (0, 255, 255)
    color_red = (0, 0, 255)

    y = 22
    dy = 18

    # Step
    cv2.putText(frame, f"Step: {step}", (12, y), font, font_scale, color_white, thickness)
    y += dy

    # Cube position
    cv2.putText(frame, f"Cube Pos: ({cube_pos[0]:+.4f}, {cube_pos[1]:+.4f}, {cube_pos[2]:+.4f})",
                (12, y), font, font_scale, color_green, thickness)
    y += dy

    # Cube quaternion (wxyz)
    cv2.putText(frame, f"Cube Quat(wxyz): ({cube_quat_wxyz[0]:+.4f}, {cube_quat_wxyz[1]:+.4f}, "
                f"{cube_quat_wxyz[2]:+.4f}, {cube_quat_wxyz[3]:+.4f})",
                (12, y), font, font_scale, color_green, thickness)
    y += dy

    # Cube euler
    cv2.putText(frame, f"Cube Euler(RPY): ({cube_euler[0]:+.1f}, {cube_euler[1]:+.1f}, {cube_euler[2]:+.1f}) deg",
                (12, y), font, font_scale, color_green, thickness)
    y += dy

    # EE position
    cv2.putText(frame, f"EE Pos:   ({ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f})",
                (12, y), font, font_scale, color_cyan, thickness)
    y += dy

    # EE quaternion (wxyz)
    cv2.putText(frame, f"EE Quat(wxyz): ({ee_quat_wxyz[0]:+.4f}, {ee_quat_wxyz[1]:+.4f}, "
                f"{ee_quat_wxyz[2]:+.4f}, {ee_quat_wxyz[3]:+.4f})",
                (12, y), font, font_scale, color_cyan, thickness)
    y += dy

    # EE euler
    cv2.putText(frame, f"EE Euler(RPY): ({ee_euler[0]:+.1f}, {ee_euler[1]:+.1f}, {ee_euler[2]:+.1f}) deg",
                (12, y), font, font_scale, color_cyan, thickness)
    y += dy

    # Relative distance and height
    rel_dist = np.linalg.norm(cube_pos - ee_pos)
    cv2.putText(frame, f"Cube-EE dist: {rel_dist:.4f}m  |  Cube height: {cube_height:.4f}m",
                (12, y), font, font_scale, color_yellow, thickness)
    y += dy

    # Contact and finger distance
    contact_str = "YES" if has_contact else "NO"
    contact_color = color_green if has_contact else color_red
    cv2.putText(frame, f"Contact: {contact_str}  |  Finger dist: {finger_dist:.4f}m  |  Reward: {reward:.3f}",
                (12, y), font, font_scale, contact_color, thickness)
    y += dy

    # Coordinate axes legend (bottom-left)
    legend_y = h - 15
    cv2.putText(frame, "Axes: X=Red Y=Green Z=Blue (projected 3D)", (12, legend_y), font, 0.35, color_white, 1)

    # --- Draw 3D orientation axes at actual projected positions ---
    cube_R = quat_wxyz_to_matrix(cube_quat_wxyz)
    frame = draw_3d_axes_on_image(
        frame, cube_pos, cube_R,
        cam_pos, cam_R, K, native_wh, render_wh,
        axis_length=0.05, thickness=2, label="Cube",
    )

    ee_R = quat_wxyz_to_matrix(ee_quat_wxyz)
    frame = draw_3d_axes_on_image(
        frame, ee_pos, ee_R,
        cam_pos, cam_R, K, native_wh, render_wh,
        axis_length=0.05, thickness=2, label="EE",
    )

    # Draw small dot at projected positions for visibility
    cube_px = project_world_to_pixel(cube_pos, cam_pos, cam_R, K, native_wh, render_wh)
    if cube_px is not None:
        cv2.circle(frame, cube_px, 4, (0, 255, 0), -1)  # green dot
    ee_px = project_world_to_pixel(ee_pos, cam_pos, cam_R, K, native_wh, render_wh)
    if ee_px is not None:
        cv2.circle(frame, ee_px, 4, (0, 255, 255), -1)  # cyan dot

    return frame


def main():
    print("[viz] Starting cube 6D pose visualization...")

    # Default output path
    output_path = args_cli.output_path
    if output_path is None:
        output_dir = Path(__file__).resolve().parents[3] / "outputs" / "cube_6dpose_viz"
        output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"cube_6dpose_{ts}.mp4")

    # Parse frame size
    frame_h, frame_w = [int(x) for x in args_cli.frame_size.split("x")]

    # Create sim + env
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=sim,
        csv_dir=args_cli.csv_base_dir,
        groot_model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        policy_device=args_cli.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args_cli.language_override,
        max_episode_steps=args_cli.max_steps,
        num_envs=args_cli.num_envs,
        success_threshold=0.03,
        cube_perturb_table_path=args_cli.cube_perturb_table_path,
        reward_type="dense_clipped",
    )

    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # Reset env
    print("[viz] Resetting environment...")
    obs = env.reset()
    N = env.num_envs

    # --- Get camera intrinsics & extrinsics for front camera ---
    cam = env.camera_front
    K = cam.data.intrinsic_matrices[0].detach().cpu().numpy()  # (3, 3)
    # Camera position and orientation in world frame ("world" convention)
    cam_pos = cam.data.pos_w[0].detach().cpu().numpy()          # (3,)
    cam_quat_wxyz = cam.data.quat_w_world[0].detach().cpu().numpy()  # (4,) wxyz
    cam_R = quat_wxyz_to_matrix(cam_quat_wxyz)                  # (3, 3)
    native_wh = (cam.image_shape[1], cam.image_shape[0])         # (W, H)
    print(f"[viz] Front camera: pos={cam_pos}, quat_w_world={cam_quat_wxyz}")
    print(f"[viz] Intrinsic K:\n{K}")
    print(f"[viz] Native resolution: {native_wh}")

    # Buffers for the trajectory (env 0)
    frames = []
    pose_log = []  # list of dicts for CSV/json dump

    # Get hand body index for EE pose
    hand_body_idx = env.robot.find_bodies("panda_hand")[0][0]

    print(f"[viz] Running rollout for {args_cli.max_steps} steps (recording env 0)...")
    for step_i in range(args_cli.max_steps):
        # Step with zero residual action
        zero_action = torch.zeros(N, 7, device=env.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(zero_action)

        # Extract cube 6D pose (world frame)
        cube_state = env.cube.data.root_state_w[:N]  # (N, 13): pos(3), quat_wxyz(4), vel(3), angvel(3)
        cube_pos = cube_state[0, :3].detach().cpu().numpy()
        cube_quat_wxyz = cube_state[0, 3:7].detach().cpu().numpy()

        # Extract EE 6D pose (world frame)
        ee_pos = env.robot.data.body_pos_w[0, hand_body_idx].detach().cpu().numpy()
        ee_quat_wxyz = env.robot.data.body_quat_w[0, hand_body_idx].detach().cpu().numpy()

        # Extract metrics
        initial_z = env._initial_cube_z[0].item()
        cube_height = cube_pos[2] - initial_z

        # Finger distance
        finger_pos = env.finger_pos[0].detach().cpu().numpy()
        finger_dist = float(np.linalg.norm(finger_pos - cube_pos))

        # Contact
        finger_force = env.left_finger_force[0].norm().item()
        has_contact = finger_force > 0.1

        r = reward[0].item() if isinstance(reward, torch.Tensor) else reward

        # Log pose data
        pose_log.append({
            "step": step_i,
            "cube_pos_x": float(cube_pos[0]),
            "cube_pos_y": float(cube_pos[1]),
            "cube_pos_z": float(cube_pos[2]),
            "cube_quat_w": float(cube_quat_wxyz[0]),
            "cube_quat_x": float(cube_quat_wxyz[1]),
            "cube_quat_y": float(cube_quat_wxyz[2]),
            "cube_quat_z": float(cube_quat_wxyz[3]),
            "ee_pos_x": float(ee_pos[0]),
            "ee_pos_y": float(ee_pos[1]),
            "ee_pos_z": float(ee_pos[2]),
            "ee_quat_w": float(ee_quat_wxyz[0]),
            "ee_quat_x": float(ee_quat_wxyz[1]),
            "ee_quat_y": float(ee_quat_wxyz[2]),
            "ee_quat_z": float(ee_quat_wxyz[3]),
            "cube_height": float(cube_height),
            "finger_dist": float(finger_dist),
            "has_contact": bool(has_contact),
            "reward": float(r),
        })

        # Get RGB frame from front camera (higher res for visualization)
        frame = env.get_frame(env_id=0, camera="front", size=(frame_h, frame_w))

        # Draw overlay (with projected 3D axes at actual positions)
        frame = draw_pose_overlay(
            frame, step_i, cube_pos, cube_quat_wxyz,
            ee_pos, ee_quat_wxyz,
            cube_height, finger_dist, has_contact, r,
            cam_pos, cam_R, K, native_wh,
        )
        frames.append(frame)

        if step_i % 50 == 0:
            print(f"  step {step_i}: cube=({cube_pos[0]:+.3f}, {cube_pos[1]:+.3f}, {cube_pos[2]:+.3f}) "
                  f"height={cube_height:.4f} contact={'Y' if has_contact else 'N'} r={r:.3f}")

        # Check if terminated
        if terminated[0].item():
            print(f"  Episode terminated at step {step_i}")
            break

    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=args_cli.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[viz] Video saved: {output_path} ({len(frames)} frames)")

    # Save pose log as JSON
    import json
    log_path = output_path.replace(".mp4", "_poses.json")
    with open(log_path, "w") as f:
        json.dump(pose_log, f, indent=2)
    print(f"[viz] Pose log saved: {log_path} ({len(pose_log)} entries)")

    # Print summary statistics
    print(f"\n=== Pose Summary ===")
    positions = np.array([[p["cube_pos_x"], p["cube_pos_y"], p["cube_pos_z"]] for p in pose_log])
    print(f"Cube X range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
    print(f"Cube Y range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
    print(f"Cube Z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")
    heights = np.array([p["cube_height"] for p in pose_log])
    print(f"Cube height range: [{heights.min():.4f}, {heights.max():.4f}]")
    max_height_step = np.argmax(heights)
    print(f"Max height: {heights.max():.4f} at step {max_height_step}")
    contacts = np.array([p["has_contact"] for p in pose_log])
    print(f"Contact frames: {contacts.sum()}/{len(contacts)} ({contacts.mean()*100:.1f}%)")

    simulation_app.close()
    print("[viz] Done!")


if __name__ == "__main__":
    main()
