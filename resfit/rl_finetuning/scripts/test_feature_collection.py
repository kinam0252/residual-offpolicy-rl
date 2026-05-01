"""Test feature collection for Stack Cube offline data.

Runs 2 episodes (2 positions × 1 ep each) with GR00T base policy,
saves raw reward features + cam_base RGB frames, and generates
a visual debug HTML page.

Usage (login node with GPU):
    cd ~/Repos/Intern/residual-offpolicy-rl
    LD_LIBRARY_PATH=~/.local/lib/gl python resfit/rl_finetuning/scripts/test_feature_collection.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)
import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__="0.0.0"
    _ds.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType("deepspeed.zero"); _dz.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    sys.modules["deepspeed"]=_ds; sys.modules["deepspeed.zero"]=_dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

import numpy as np
import mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import (
    MuJoCoVecEnvStack, CUBE_HALF, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

# ── Config ──
NUM_POSITIONS = 2        # test with 2 positions only
MAX_STEPS = 500
FRAME_INTERVAL = 25      # save cam_base frame every N steps
GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000"
)
POSITIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
    'configs', 'stack_cube_positions.json'
)
OUT_DIR = Path("outputs/test_feature_collection")


def extract_features(mujoco_env, env_idx: int = 0) -> dict:
    """Extract raw reward features from current env state."""
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    qa = env["white_qposadr"]

    white_pos = data.qpos[qa:qa + 3].copy() if qa is not None else np.zeros(3)
    tcp_pos, _ = get_tcp_pose(model, data, ids["hand_id"])
    grasped = env["grasp_state"]["grasped"]

    green_body_id = env["green_body_id"]
    green_pos = model.body_pos[green_body_id].copy() if green_body_id >= 0 else np.zeros(3)

    green_top = green_pos.copy()
    green_top[2] += CUBE_HALF * 2

    tcp_white_dist = float(np.linalg.norm(tcp_pos - white_pos))
    white_to_green_top_3d = float(np.linalg.norm(white_pos - green_top))
    white_green_xy_dist = float(np.linalg.norm(white_pos[:2] - green_pos[:2]))

    # White-green contact
    white_gid = env["white_geom_id"]
    green_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "green_cube_geom")
    white_green_contact = False
    for ci in range(data.ncon):
        c = data.contact[ci]
        if (c.geom1 == white_gid and c.geom2 == green_gid) or \
           (c.geom1 == green_gid and c.geom2 == white_gid):
            white_green_contact = True
            break

    return {
        "tcp_pos": tcp_pos.astype(np.float32),
        "white_pos": white_pos.astype(np.float32),
        "green_pos": green_pos.astype(np.float32),
        "tcp_white_dist": np.float32(tcp_white_dist),
        "white_to_green_top_3d": np.float32(white_to_green_top_3d),
        "white_green_xy_dist": np.float32(white_green_xy_dist),
        "white_z": np.float32(white_pos[2]),
        "green_z": np.float32(green_pos[2]),
        "grasped": np.bool_(grasped),
        "white_green_contact": np.bool_(white_green_contact),
    }


def render_cam_base(mujoco_env, env_idx: int = 0) -> np.ndarray:
    """Render cam_base RGB frame (640x360)."""
    env = mujoco_env._envs[env_idx]
    env["renderer_base"].update_scene(
        env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
    )
    return env["renderer_base"].render().copy()  # H,W,3 uint8


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load positions
    with open(POSITIONS_FILE) as f:
        all_positions = json.load(f)
    positions = all_positions[:NUM_POSITIONS]
    print(f"Testing with {len(positions)} positions, max {MAX_STEPS} steps each")

    # Create env + wrapper (1 env only)
    print("Creating env...")
    mujoco_env = MuJoCoVecEnvStack(
        num_envs=1, reward_type="dense", max_episode_steps=MAX_STEPS
    )
    print("Creating wrapper + loading GR00T...")
    wrapper = MuJoCoResidualWrapperStack(
        vec_env=mujoco_env,
        groot_checkpoint=GROOT_CKPT,
        policy_device="cuda:0",
    )
    print("Ready.\n")

    for pos_idx, pos_info in enumerate(positions):
        print(f"=== Position {pos_idx} ===")
        ep_features = []
        ep_frames = []
        frame_steps = []

        # Reset
        obs, _ = wrapper.reset()
        env_data = mujoco_env._envs[0]
        data = env_data["data"]
        model = env_data["model"]
        wqa = env_data["white_qposadr"]
        green_body_id = env_data["green_body_id"]

        # Set robot initial pose
        for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
        for fid in env_data["ids"]["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        # Set cubes
        white_pos = np.array(pos_info["white_cube_pos"], dtype=np.float64)
        green_pos = np.array(pos_info["green_cube_pos"], dtype=np.float64)
        white_yaw = pos_info.get("white_yaw_deg", 0.0)
        green_yaw = pos_info.get("green_yaw_deg", 0.0)

        if wqa is not None:
            data.qpos[wqa:wqa+3] = white_pos
            q = Rotation.from_euler("z", np.radians(white_yaw)).as_quat()
            data.qpos[wqa+3:wqa+7] = [q[3], q[0], q[1], q[2]]
        if green_body_id >= 0:
            model.body_pos[green_body_id] = green_pos

        env_data["grasp_state"] = {"grasped": False, "contact_count": 0}
        if "weld_eq_idx" in env_data and env_data["weld_eq_idx"] is not None:
            model.eq_active[env_data["weld_eq_idx"]] = 0

        mujoco.mj_forward(model, data)
        mujoco_env._step_counts = np.zeros(1, dtype=int)
        mujoco_env._initial_white_z[0] = data.qpos[wqa + 2] if wqa is not None else 0.0

        # Rollout
        t0 = time.time()
        for step in range(MAX_STEPS):
            # Extract features BEFORE step
            feat = extract_features(mujoco_env, 0)
            feat["step"] = np.int32(step)

            # Save cam_base frame at intervals
            if step % FRAME_INTERVAL == 0:
                frame = render_cam_base(mujoco_env, 0)
                ep_frames.append(frame)
                frame_steps.append(step)

            ep_features.append(feat)

            # Step with zero residual
            residual = torch.zeros((1, 7), dtype=torch.float32)
            next_obs, reward, terminated, truncated, info = wrapper.step(residual)
            obs = next_obs

            if terminated[0] or truncated[0]:
                # Capture final frame (before auto-reset)
                frame = render_cam_base(mujoco_env, 0)
                ep_frames.append(frame)
                frame_steps.append(step)
                break

        elapsed = time.time() - t0
        n_steps = len(ep_features)
        success = bool(terminated[0])
        print(f"  Steps: {n_steps}, Success: {success}, Time: {elapsed:.1f}s")
        print(f"  Final features (last valid step):")
        last = ep_features[-1]
        print(f"    tcp_white_dist={last['tcp_white_dist']:.4f}")
        print(f"    white_to_green_top_3d={last['white_to_green_top_3d']:.4f}")
        print(f"    white_green_xy_dist={last['white_green_xy_dist']:.4f}")
        print(f"    grasped={last['grasped']}, contact={last['white_green_contact']}")
        print(f"    white_z={last['white_z']:.4f}, green_z={last['green_z']:.4f}")

        # Save features as npz
        feat_path = OUT_DIR / f"pos{pos_idx}_features.npz"
        feat_arrays = {}
        for key in ep_features[0].keys():
            feat_arrays[key] = np.array([f[key] for f in ep_features])
        np.savez_compressed(feat_path, **feat_arrays)
        print(f"  Saved features: {feat_path} ({len(ep_features)} steps)")

        # Save frames as npz
        frames_path = OUT_DIR / f"pos{pos_idx}_frames.npz"
        np.savez_compressed(
            frames_path,
            frames=np.array(ep_frames),           # (N, H, W, 3)
            frame_steps=np.array(frame_steps),     # (N,)
        )
        print(f"  Saved {len(ep_frames)} cam_base frames: {frames_path}")

    # Generate HTML debug page
    _generate_html(OUT_DIR, NUM_POSITIONS)
    print(f"\nDone! View debug page: {OUT_DIR / 'debug.html'}")


def _generate_html(out_dir: Path, num_positions: int):
    """Generate HTML page with feature plots and cam_base frames."""
    import base64
    from io import BytesIO

    html_parts = [
        "<html><head><title>Stack Cube Feature Collection Test</title>",
        "<style>body{font-family:monospace;background:#1a1a1a;color:#eee;padding:20px}",
        "img{border:1px solid #444;margin:4px} .grid{display:flex;flex-wrap:wrap}",
        "table{border-collapse:collapse;margin:10px 0} td,th{border:1px solid #555;padding:4px 8px;text-align:right}",
        "th{background:#333} .succ{color:#4f4} .fail{color:#f44}",
        "canvas{background:#222;border:1px solid #444;margin:10px}",
        "</style></head><body>",
        "<h1>Stack Cube Feature Collection Test</h1>",
    ]

    for pos_idx in range(num_positions):
        feat_path = out_dir / f"pos{pos_idx}_features.npz"
        frames_path = out_dir / f"pos{pos_idx}_frames.npz"
        if not feat_path.exists():
            continue

        feat = dict(np.load(feat_path))
        frames_data = dict(np.load(frames_path))
        frames = frames_data["frames"]
        frame_steps = frames_data["frame_steps"]

        n = len(feat["step"])
        success = bool(feat["white_green_contact"][-1]) and feat["white_z"][-1] > feat["green_z"][-1]
        status = '<span class="succ">SUCCESS</span>' if success else '<span class="fail">FAIL</span>'

        html_parts.append(f"<h2>Position {pos_idx} — {n} steps — {status}</h2>")

        # Feature table (sampled every 50 steps)
        html_parts.append("<table><tr><th>step</th><th>tcp→white</th><th>white→green_top</th>"
                         "<th>white↔green_xy</th><th>grasped</th><th>contact</th>"
                         "<th>white_z</th><th>green_z</th></tr>")
        sample_indices = list(range(0, n, 50))
        if (n - 1) not in sample_indices:
            sample_indices.append(n - 1)
        for i in sample_indices:
            g = "✓" if feat["grasped"][i] else ""
            c = "✓" if feat["white_green_contact"][i] else ""
            html_parts.append(
                f"<tr><td>{feat['step'][i]}</td>"
                f"<td>{feat['tcp_white_dist'][i]:.4f}</td>"
                f"<td>{feat['white_to_green_top_3d'][i]:.4f}</td>"
                f"<td>{feat['white_green_xy_dist'][i]:.4f}</td>"
                f"<td>{g}</td><td>{c}</td>"
                f"<td>{feat['white_z'][i]:.4f}</td>"
                f"<td>{feat['green_z'][i]:.4f}</td></tr>"
            )
        html_parts.append("</table>")

        # Feature plot as SVG
        html_parts.append(_svg_plot(feat, pos_idx))

        # Cam base frames
        html_parts.append("<h3>cam_base frames</h3><div class='grid'>")
        for fi, (frame, fstep) in enumerate(zip(frames, frame_steps)):
            # Encode as JPEG for smaller size
            try:
                from PIL import Image
                img = Image.fromarray(frame)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=70)
                b64 = base64.b64encode(buf.getvalue()).decode()
                html_parts.append(
                    f"<div><img src='data:image/jpeg;base64,{b64}' width='320' height='180'>"
                    f"<br>step {fstep}</div>"
                )
            except ImportError:
                # Fallback: skip images if PIL not available
                html_parts.append(f"<div>step {fstep} (PIL not available)</div>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    html_path = out_dir / "debug.html"
    html_path.write_text("\n".join(html_parts))


def _svg_plot(feat: dict, pos_idx: int) -> str:
    """Simple SVG line plot for key features over time."""
    steps = feat["step"]
    n = len(steps)
    if n < 2:
        return ""

    W, H = 800, 250
    margin = 50

    series = [
        ("tcp→white", feat["tcp_white_dist"], "#ff6666"),
        ("white→green_top", feat["white_to_green_top_3d"], "#66ff66"),
        ("white↔green_xy", feat["white_green_xy_dist"], "#6688ff"),
    ]

    # Find global y range
    all_vals = np.concatenate([s[1] for s in series])
    y_min = 0.0
    y_max = max(float(all_vals.max()) * 1.1, 0.01)

    def tx(i):
        return margin + (W - 2 * margin) * i / max(n - 1, 1)
    def ty(v):
        return H - margin - (H - 2 * margin) * (v - y_min) / max(y_max - y_min, 1e-6)

    svg = [f'<svg width="{W}" height="{H}" style="background:#222;border:1px solid #444;margin:10px">']
    # Axes
    svg.append(f'<line x1="{margin}" y1="{H-margin}" x2="{W-margin}" y2="{H-margin}" stroke="#666"/>')
    svg.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{H-margin}" stroke="#666"/>')
    # Y labels
    for v in np.linspace(y_min, y_max, 5):
        y = ty(v)
        svg.append(f'<text x="{margin-5}" y="{y+4}" text-anchor="end" fill="#888" font-size="10">{v:.3f}</text>')
    # X labels
    svg.append(f'<text x="{W/2}" y="{H-5}" text-anchor="middle" fill="#888" font-size="11">step</text>')

    # Grasped region (light background)
    grasped = feat["grasped"]
    in_grasp = False
    grasp_start = 0
    for i in range(n):
        if grasped[i] and not in_grasp:
            grasp_start = i
            in_grasp = True
        elif not grasped[i] and in_grasp:
            svg.append(f'<rect x="{tx(grasp_start)}" y="{margin}" '
                       f'width="{tx(i)-tx(grasp_start)}" height="{H-2*margin}" '
                       f'fill="rgba(255,255,0,0.08)"/>')
            in_grasp = False
    if in_grasp:
        svg.append(f'<rect x="{tx(grasp_start)}" y="{margin}" '
                   f'width="{tx(n-1)-tx(grasp_start)}" height="{H-2*margin}" '
                   f'fill="rgba(255,255,0,0.08)"/>')

    # Lines
    for label, vals, color in series:
        points = " ".join(f"{tx(i)},{ty(vals[i])}" for i in range(n))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')

    # Legend
    for li, (label, _, color) in enumerate(series):
        lx = margin + 10 + li * 180
        svg.append(f'<rect x="{lx}" y="8" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{lx+16}" y="18" fill="#ccc" font-size="11">{label}</text>')
    svg.append(f'<rect x="{margin+10+len(series)*180}" y="8" width="12" height="12" fill="rgba(255,255,0,0.3)"/>')
    svg.append(f'<text x="{margin+10+len(series)*180+16}" y="18" fill="#ccc" font-size="11">grasped</text>')

    svg.append(f'<text x="{W/2}" y="{margin-8}" text-anchor="middle" fill="#eee" font-size="13">Position {pos_idx}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


if __name__ == "__main__":
    main()
