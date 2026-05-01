"""
Debug eval: visualize dense_v3 reward components per frame with video + CSV logging.
Temporary debugging script — not for production.

Usage:
    python scripts/debug_reward_v3.py --n-envs 5 --levels medium
"""
import sys, os, json, time, importlib, types as _types, argparse, csv
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_BUILD_OPS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__ = "0.0.0"
    m.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None); m.__path__ = []; m.__file__ = "fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    z.Init = lambda *a, **kw: (lambda f: f); m.zero = z
    sys.modules["deepspeed"] = m; sys.modules["deepspeed.zero"] = z

import numpy as np
import torch
import cv2

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)
MAX_STEPS = 500
CUBE_BASE = np.array([0.42, -0.03, 0.02])
BOWL_BASE = np.array([0.42, 0.03, 0.0])

LEVELS = {
    "zero":     {"cube_dx": 0.00, "cube_dy": 0.00, "bowl_dx": 0.00, "bowl_dy": 0.00, "yaw": 0},
    "easy":     {"cube_dx": 0.02, "cube_dy": 0.02, "bowl_dx": 0.02, "bowl_dy": 0.02, "yaw": 5},
    "medium":   {"cube_dx": 0.05, "cube_dy": 0.05, "bowl_dx": 0.05, "bowl_dy": 0.05, "yaw": 15},
    "hard":     {"cube_dx": 0.08, "cube_dy": 0.08, "bowl_dx": 0.08, "bowl_dy": 0.08, "yaw": 30},
    "extreme":  {"cube_dx": 0.12, "cube_dy": 0.10, "bowl_dx": 0.10, "bowl_dy": 0.10, "yaw": 50},
}


def generate_positions(n_envs, level_name, rng):
    lvl = LEVELS[level_name]
    positions = []
    for i in range(n_envs):
        cube_pos = CUBE_BASE.copy()
        cube_pos[0] += rng.uniform(-lvl["cube_dx"], lvl["cube_dx"])
        cube_pos[1] += rng.uniform(-lvl["cube_dy"], lvl["cube_dy"])
        bowl_pos = BOWL_BASE.copy()
        bowl_pos[0] += rng.uniform(-lvl["bowl_dx"], lvl["bowl_dx"])
        bowl_pos[1] += rng.uniform(-lvl["bowl_dy"], lvl["bowl_dy"])
        yaw = rng.uniform(-lvl["yaw"], lvl["yaw"])
        dist = np.linalg.norm(cube_pos[:2] - bowl_pos[:2])
        positions.append({
            "episode": i, "cube_pos": cube_pos.tolist(),
            "bowl_pos": bowl_pos.tolist(), "cube_yaw_deg": float(yaw),
            "distance": float(dist),
        })
    return positions


def compute_reward_components(vec_env, env_idx):
    """Compute all dense_v3 reward components individually for debugging."""
    from scipy.spatial.transform import Rotation
    _MUJOCO_FRANKA_SRC = str(Path(__file__).resolve().parents[1] / "Mujoco_Franka" / "src")
    if _MUJOCO_FRANKA_SRC not in sys.path:
        sys.path.insert(0, _MUJOCO_FRANKA_SRC)
    from utils import get_tcp_pose

    env = vec_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    cube_qposadr = env["cube_qposadr"]
    bowl_body_id = env["bowl_body_id"]

    cube_pos = data.qpos[cube_qposadr:cube_qposadr + 3].copy()
    tcp_pos, tcp_R = get_tcp_pose(model, data, ids["hand_id"])
    grasped = env["grasp_state"]["grasped"]

    if bowl_body_id >= 0:
        bowl_pos = model.body_pos[bowl_body_id].copy()
    else:
        bowl_pos = np.array([0.42, 0.03, 0.0])

    tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
    cube_bowl_xy = float(np.linalg.norm(cube_pos[:2] - bowl_pos[:2]))

    # Cube quaternion
    cube_quat_wxyz = data.qpos[cube_qposadr + 3:cube_qposadr + 7]
    cube_R = Rotation.from_quat([
        cube_quat_wxyz[1], cube_quat_wxyz[2],
        cube_quat_wxyz[3], cube_quat_wxyz[0]
    ]).as_matrix()

    # Alignment (fixed: grip y-axis vs cube y-axis, short 4cm side)
    grip_dir = tcp_R[:2, 1]
    cube_dir = cube_R[:2, 1]
    grip_dir_n = grip_dir / (np.linalg.norm(grip_dir) + 1e-8)
    cube_dir_n = cube_dir / (np.linalg.norm(cube_dir) + 1e-8)
    cos_align = abs(float(np.dot(grip_dir_n, cube_dir_n)))

    # Components
    approach = (1.0 - np.tanh(tcp_cube_dist / 0.1)) * 0.15
    proximity = max(0.0, 1.0 - tcp_cube_dist / 0.15)
    alignment = cos_align * proximity * 0.10
    grasp = 0.10 if grasped else 0.0

    lift_delta = cube_pos[2] - vec_env._initial_cube_z[env_idx]
    lift = 0.0
    if grasped and lift_delta > 0.01:
        lift = np.tanh(lift_delta / 0.1) * 0.10

    transport = 0.0
    if grasped and lift_delta > 0.01:
        transport = (1.0 - np.tanh(cube_bowl_xy / 0.1)) * 0.30

    is_success = vec_env._is_success(env_idx)
    success = 0.25 if is_success else 0.0

    total = approach + alignment + grasp + lift + transport + success
    total = float(np.clip(total, 0.0, 1.0))

    return {
        "tcp_cube_dist": round(tcp_cube_dist, 4),
        "cube_bowl_xy": round(cube_bowl_xy, 4),
        "cos_align": round(cos_align, 3),
        "grasped": int(grasped),
        "lift_delta": round(float(lift_delta), 4),
        "approach": round(float(approach), 4),
        "alignment": round(float(alignment), 4),
        "grasp": round(float(grasp), 4),
        "lift": round(float(lift), 4),
        "transport": round(float(transport), 4),
        "success_rwd": round(float(success), 4),
        "total": round(total, 4),
        "is_success": int(is_success),
        # 3D data for arrow visualization (use x-axes for intuitive display:
        # hand_x = between-fingers direction, cube_x = long 8cm edge)
        "tcp_pos": tcp_pos.copy(),
        "cube_pos": cube_pos.copy(),
        "grip_dir_3d": tcp_R[:, 0].copy(),   # hand x-axis (between fingers, outward)
        "cube_dir_3d": cube_R[:, 0].copy(),   # cube x-axis (long 8cm side)
    }


def overlay_reward_text(frame, components, step):
    """Draw reward component overlay on video frame."""
    h, w = frame.shape[:2]
    # Semi-transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (320, 230), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.40
    c_white = (255, 255, 255)
    c_green = (0, 255, 0)
    c_yellow = (0, 255, 255)
    c_red = (0, 0, 255)

    cv2.putText(frame, f"Step {step}  Total: {components['total']:.3f}", (10, y), font, 0.50, c_green, 1)
    y += 18

    bars = [
        ("approach",  0.15, c_white),
        ("alignment", 0.10, c_yellow),
        ("grasp",     0.10, c_white),
        ("lift",      0.10, c_white),
        ("transport", 0.30, c_green),
        ("success_rwd", 0.25, c_green),
    ]
    for name, max_val, color in bars:
        val = components[name]
        pct = val / max_val if max_val > 0 else 0
        bar_w = int(pct * 150)
        cv2.putText(frame, f"{name:>10s} {val:.3f}/{max_val:.2f}", (10, y), font, fs, color, 1)
        cv2.rectangle(frame, (200, y - 8), (200 + bar_w, y), color, -1)
        cv2.rectangle(frame, (200, y - 8), (200 + 150, y), (100, 100, 100), 1)
        y += 16

    # State info
    y += 4
    cv2.putText(frame, f"tcp_cube={components['tcp_cube_dist']:.3f}m  cube_bowl={components['cube_bowl_xy']:.3f}m",
                (10, y), font, fs, c_white, 1)
    y += 14
    cv2.putText(frame, f"cos_align={components['cos_align']:.2f}  grasped={components['grasped']}  lift={components['lift_delta']:.3f}m",
                (10, y), font, fs, c_white, 1)

    return frame


def project_3d_to_pixel(point_3d, model, data, cam_id, img_w, img_h):
    """Project a 3D world point to 2D pixel coordinates using MuJoCo camera."""
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    # MuJoCo cam axes: x=right, y=up, z=-forward (into screen)
    p_cam = cam_mat.T @ (point_3d - cam_pos)
    if p_cam[2] > -1e-4:
        return None  # behind camera
    fovy = model.cam_fovy[cam_id]
    aspect = img_w / img_h
    f = img_h / (2.0 * np.tan(np.radians(fovy) / 2.0))
    px = int(img_w / 2 + f * p_cam[0] / (-p_cam[2]))
    py = int(img_h / 2 - f * p_cam[1] / (-p_cam[2]))
    return (px, py)


def draw_direction_arrows(frame, comp, model, data, cam_id, render_w, render_h, display_w, display_h):
    """Draw gripper x-axis (cyan) and cube x-axis (magenta) arrows on frame.
    When aligned for grasping, both arrows point in the same direction."""
    tcp_pos = comp["tcp_pos"]
    cube_pos = comp["cube_pos"]
    grip_dir = comp["grip_dir_3d"]
    cube_dir = comp["cube_dir_3d"]

    # Flip cube arrow if anti-parallel to gripper (so arrows visually match)
    if np.dot(grip_dir, cube_dir) < 0:
        cube_dir = -cube_dir

    ARROW_LEN = 0.08  # 8cm arrow

    for origin, direction, color, label in [
        (tcp_pos,  grip_dir, (255, 255, 0), "grip"),   # cyan (BGR)
        (cube_pos, cube_dir, (255, 0, 255), "cube"),   # magenta (BGR)
    ]:
        p0 = project_3d_to_pixel(origin, model, data, cam_id, render_w, render_h)
        p1 = project_3d_to_pixel(origin + direction * ARROW_LEN, model, data, cam_id, render_w, render_h)
        if p0 is None or p1 is None:
            continue
        # Scale from render resolution to display resolution
        sx, sy = display_w / render_w, display_h / render_h
        p0 = (int(p0[0] * sx), int(p0[1] * sy))
        p1 = (int(p1[0] * sx), int(p1[1] * sy))
        cv2.arrowedLine(frame, p0, p1, color, 2, tipLength=0.3)
        cv2.putText(frame, label, (p1[0] + 3, p1[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return frame


def run_debug_eval(positions, groot_ckpt, output_dir, device="cuda"):
    """Run base policy eval with per-frame reward debug logging + video."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP
    from scipy.spatial.transform import Rotation
    import mujoco

    os.makedirs(output_dir, exist_ok=True)
    num_envs = len(positions)
    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]
    cube_yaws = [p["cube_yaw_deg"] for p in positions]

    vec_env = MuJoCoVecEnvPnP(
        num_envs=num_envs,
        cube_positions=cube_positions,
        bowl_positions=bowl_positions,
        max_episode_steps=MAX_STEPS,
        reward_type="dense_v3",
        scene_xml=SCENE_XML,
    )

    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=groot_ckpt,
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=0.0,
    )

    obs, _ = wrapper.reset()

    # Override positions
    for ei in range(num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            data.qpos[cqa:cqa+3] = cube_positions[ei]
            q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
            data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_positions[ei]
        env_data["cube_pos_init"] = np.array(cube_positions[ei])
        env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    # Re-reset + re-override
    obs, _ = wrapper.reset()
    for ei in range(num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            data.qpos[cqa:cqa+3] = cube_positions[ei]
            q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
            data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_positions[ei]
        env_data["cube_pos_init"] = np.array(cube_positions[ei])
        env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    # Setup video writers + CSV loggers
    writers = []
    csv_files = []
    csv_writers = []
    fieldnames = ["step", "tcp_cube_dist", "cube_bowl_xy", "cos_align", "grasped",
                  "lift_delta", "approach", "alignment", "grasp", "lift",
                  "transport", "success_rwd", "total"]

    for ei in range(num_envs):
        vid_path = os.path.join(output_dir, f"env{ei:02d}_yaw{cube_yaws[ei]:+.0f}_d{positions[ei]['distance']:.2f}.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
        writers.append(writer)

        csv_path = os.path.join(output_dir, f"env{ei:02d}_rewards.csv")
        cf = open(csv_path, "w", newline="")
        cw = csv.DictWriter(cf, fieldnames=fieldnames)
        cw.writeheader()
        csv_files.append(cf)
        csv_writers.append(cw)

    # Step loop
    env_done = [False] * num_envs
    env_success = [False] * num_envs
    RENDER_EVERY = 2

    for step in range(MAX_STEPS):
        action = np.zeros((num_envs, wrapper.action_dim))
        obs, rew, term, trunc, info = wrapper.step(action)
        done = term | trunc if hasattr(term, '__or__') else np.array(term) | np.array(trunc)

        for ei in range(num_envs):
            if env_done[ei]:
                continue

            # Compute reward components
            comp = compute_reward_components(vec_env, ei)
            comp["step"] = step
            csv_writers[ei].writerow({k: comp[k] for k in fieldnames})

            # Render video frame
            if step % RENDER_EVERY == 0:
                env_data = vec_env._envs[ei]
                mujoco.mj_forward(env_data["model"], env_data["data"])
                renderer = env_data["renderer_base"]
                renderer.update_scene(env_data["data"], camera=env_data["cam_base_id"])
                frame = renderer.render().copy()

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Resize to 640x480 for overlay readability
                render_h, render_w = frame.shape[:2]
                frame_bgr = cv2.resize(frame_bgr, (640, 480))
                frame_bgr = draw_direction_arrows(
                    frame_bgr, comp, env_data["model"], env_data["data"],
                    env_data["cam_base_id"], render_w, render_h, 640, 480)
                frame_bgr = overlay_reward_text(frame_bgr, comp, step)
                writers[ei].write(frame_bgr)

            # Check success
            if comp["is_success"]:
                env_success[ei] = True

            d_val = done[ei] if hasattr(done, '__getitem__') else done
            if d_val:
                env_done[ei] = True
                t_val = term[ei] if hasattr(term, '__getitem__') else term
                if t_val:
                    env_success[ei] = True

        if all(env_done):
            break

        if step % 50 == 0:
            print(f"  step {step}/{MAX_STEPS}", flush=True)

    # Cleanup
    for w in writers:
        w.release()
    for cf in csv_files:
        cf.close()

    del wrapper, vec_env
    torch.cuda.empty_cache()

    return env_success


def main():
    parser = argparse.ArgumentParser(description="Debug dense_v3 reward visualization")
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--levels", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groot-checkpoint", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"))
    parser.add_argument("--output-dir", type=str, default="outputs/debug_reward_v3")
    parser.add_argument("--positions-file", type=str, default=None,
                        help="JSON file with fixed positions (overrides --levels/--seed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.positions_file:
        with open(args.positions_file) as f:
            all_pos = json.load(f)
        positions = all_pos[:args.n_envs]
        for i, p in enumerate(positions):
            p["episode"] = i
            if "distance" not in p:
                p["distance"] = float(np.linalg.norm(
                    np.array(p["cube_pos"][:2]) - np.array(p["bowl_pos"][:2])))
    else:
        rng = np.random.RandomState(args.seed)
        positions = generate_positions(args.n_envs, args.levels, rng)

    print(f"=== Debug dense_v3 reward ===")
    print(f"Level={args.levels}, N={args.n_envs}, Seed={args.seed}")
    for p in positions:
        print(f"  env{p['episode']}: cube=({p['cube_pos'][0]:.3f},{p['cube_pos'][1]:.3f}) "
              f"bowl=({p['bowl_pos'][0]:.3f},{p['bowl_pos'][1]:.3f}) "
              f"yaw={p['cube_yaw_deg']:+.1f}° dist={p['distance']:.3f}m")
    print(flush=True)

    t0 = time.time()
    env_success = run_debug_eval(positions, args.groot_checkpoint, args.output_dir)
    elapsed = time.time() - t0

    sr = sum(env_success) / len(env_success)
    print(f"\n=== Results ===")
    print(f"SR: {sr*100:.1f}% ({sum(env_success)}/{len(env_success)}) [{elapsed:.0f}s]")
    for i, (s, p) in enumerate(zip(env_success, positions)):
        print(f"  env{i}: {'✓' if s else '✗'} yaw={p['cube_yaw_deg']:+.1f}° dist={p['distance']:.3f}m")

    # Save summary
    summary = {
        "level": args.levels, "n_envs": args.n_envs, "seed": args.seed,
        "success_rate": sr, "per_env": [int(s) for s in env_success],
        "positions": positions, "time_sec": elapsed,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
