#!/usr/bin/env python3
"""Replay a stand-cup episode from LeRobot parquet in MuJoCo and save video.

Uses v8 pure-physics approach (matching batch_replay_stand.py):
  - Full mj_step throughout (no kinematic phases)
  - Small timestep (0.0002s), stiff contacts (solref=-500000)
  - Per-joint Franka gains with torque limits
  - Hollow cup collision mesh, bilateral grasp check
  - 30° cup upright correction at grasp moment

Usage (login node, no GR00T needed):
    source ~/.venvs/groot/bin/activate
    PYOPENGL_PLATFORM=egl MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 \\
    LD_LIBRARY_PATH="..." \\
    python3 scripts/replay_cup_episode.py --episode 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# ── Import from MSRA physics modules ──
_REPO_ROOT = Path(__file__).resolve().parents[1]  # residual-offpolicy-rl/
_PHYSICS_DIR = str(_REPO_ROOT.parents[0] / "MSRA" / "stand_cup" / "physics")
if _PHYSICS_DIR not in sys.path:
    sys.path.insert(0, _PHYSICS_DIR)

from utils_stand_cup import (
    load_calib, load_calib_wrist, make_model_with_cup,
    get_model_ids, solve_ik, HOME_QPOS, CAM_W, CAM_H,
    WRIST_CAM_INTRINSICS, TCP_OFFSET, INTRINSICS,
    CUP_TOP_RADIUS, CUP_BOTTOM_RADIUS, CUP_HEIGHT, CUP_MASS,
    CUP_WALL_THICKNESS, N_COLLISION_SLICES,
)
from batch_replay_stand import (
    _compute_cup_on_side, _setup_physics, _collect_geom_ids,
    _check_bilateral_grasp,
    SKIP_FRAMES, FPS, DATA_DT, N_SETTLE_FRAMES, TABLE_Z_OFFSET,
    FINGER_TIP_Z_OFFSET, DT, CORRECTION_DEG, GRASP_GRACE_FRAMES,
)


def get_tcp_pose(model, data, hand_id):
    """Get TCP position and quaternion (xyzw) from hand body."""
    hand_pos = data.xpos[hand_id].copy()
    hand_mat = data.xmat[hand_id].reshape(3, 3)
    tcp_pos = hand_pos + hand_mat @ TCP_OFFSET
    tcp_quat_xyzw = Rotation.from_matrix(hand_mat).as_quat()
    return tcp_pos, tcp_quat_xyzw

# ── Rendering ──

def add_overlay(frame, text_lines, y_start=20):
    for i, line in enumerate(text_lines):
        y = y_start + i * 18
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def write_video(frames, path, fps=FPS):
    if not frames:
        return
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"  Saved: {path} ({len(frames)} frames)")


def replay_episode(ep_idx, dataset, output_dir, calib, quiet=False):
    """Replay a single episode. Returns (ep_idx, tag, uprightness)."""
    output = os.path.join(output_dir, f"ep{ep_idx:03d}.mp4")
    _print = (lambda *a, **kw: None) if quiet else print

    # ── Load trajectory from parquet ──
    pq_path = os.path.join(dataset, "data", "chunk-000",
                           f"episode_{ep_idx:06d}.parquet")
    _print(f"[ep{ep_idx:02d}] Loading {pq_path}")
    table = pq.read_table(pq_path)
    states_all = np.array(table['observation.state'].to_pylist(), dtype=np.float64)
    n_total = len(states_all)

    # ── Gripper events (from state gripper_width) ──
    gw = states_all[:, 7]
    close_idx = None
    open_idx = None
    open_thresh, close_thresh = 0.035, 0.005
    for i in range(1, len(gw)):
        if close_idx is None and gw[i-1] > open_thresh and gw[i] < open_thresh:
            close_idx = i
        elif close_idx is not None and open_idx is None and gw[i-1] < close_thresh and gw[i] > close_thresh:
            open_idx = i
            break
    if close_idx is None:
        close_idx = n_total // 3
    if open_idx is None:
        open_idx = n_total - 10

    # ── Cup placement from state at close ──
    close_pos = states_all[close_idx, :3]
    close_quat_xyzw = states_all[close_idx, 3:7]
    close_gw = float(states_all[close_idx, 7])

    _, cup_table_center, R_cup, cup_quat_wxyz = _compute_cup_on_side(
        close_pos, close_quat_xyzw, close_gw)
    cup_quat_wxyz_arr = np.array(cup_quat_wxyz)

    # ── Camera calibration ──
    T_base_cam = load_calib(calib)
    load_calib_wrist(calib)

    # ── Build model (hollow collision cup) ──
    model = make_model_with_cup(T_base_cam, cup_table_center, cup_quat_wxyz,
                                table_z_offset=TABLE_Z_OFFSET, hollow_collision=True)
    data = mujoco.MjData(model)
    ids = get_model_ids(model)

    # Cameras
    cam_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam_base')
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'cam_wrist')

    # Hide hand from wrist cam
    hide_bodies = []
    for bname in ['hand', 'fr3_link6', 'fr3_link7']:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            hide_bodies.append(bid)
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] in hide_bodies and model.geom_group[gi] == 2:
            model.geom_group[gi] = 4

    opt_base = mujoco.MjvOption()
    opt_base.geomgroup[3] = 0; opt_base.geomgroup[4] = 1
    opt_wrist = mujoco.MjvOption()
    opt_wrist.geomgroup[3] = 0; opt_wrist.geomgroup[4] = 0

    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    wrist_h, wrist_w = WRIST_CAM_INTRINSICS['height'], WRIST_CAM_INTRINSICS['width']
    renderer_wrist = mujoco.Renderer(model, height=wrist_h, width=wrist_w)

    cup_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cup_joint')
    cup_qpa = model.jnt_qposadr[cup_jnt_id]
    cup_dof = model.jnt_dofadr[cup_jnt_id]
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')

    # Set timestep and compute substeps
    model.opt.timestep = DT
    substeps = int(round(DATA_DT / DT))

    # Setup physics (collision bits, gains, torque limits, contact params)
    _setup_physics(model, ids)

    # Collect geom IDs for grasp contact detection
    cup_geom_ids, left_finger_gids, right_finger_gids = _collect_geom_ids(model)

    # ── Init robot + cup ──
    q_ref = HOME_QPOS.copy()
    mujoco.mj_resetData(model, data)
    for i, jid in enumerate(ids['jnt_ids']):
        data.qpos[model.jnt_qposadr[jid]] = q_ref[i]
    data.qpos[cup_qpa:cup_qpa+3] = cup_table_center
    data.qpos[cup_qpa+3:cup_qpa+7] = cup_quat_wxyz_arr
    mujoco.mj_forward(model, data)

    # IK to first frame
    pos0 = states_all[0, :3]
    quat0 = states_all[0, 3:7]
    solve_ik(model, data, ids['hand_id'], ids['jnt_ids'], pos0, quat0,
             max_iter=1000, q_ref=q_ref)
    mujoco.mj_forward(model, data)

    # Set initial ctrl
    for ai in range(7):
        jid_a = model.actuator_trnid[ai, 0]
        data.ctrl[ai] = data.qpos[model.jnt_qposadr[jid_a]]
    fp0 = np.clip(float(states_all[0, 7]), 0.0, 0.04)
    data.ctrl[7] = fp0; data.ctrl[8] = fp0
    data.qvel[:] = 0.0

    # Pre-settle: stabilize arm + cup
    arm_qpos_init = [data.qpos[model.jnt_qposadr[jid]] for jid in ids['jnt_ids']]
    for _ in range(30):
        for _ in range(substeps):
            for i, jid in enumerate(ids['jnt_ids']):
                data.qpos[model.jnt_qposadr[jid]] = arm_qpos_init[i]
                data.qvel[model.jnt_dofadr[jid]] = 0.0
            mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # ── Rendering helpers ──
    frames = []
    h_target = 360

    def _get_uprightness():
        cm = data.xmat[cup_bid].reshape(3, 3)
        return float(cm[:, 2][2])

    def _render_combined(fi, phase_str=""):
        renderer.update_scene(data, camera=cam_base_id, scene_option=opt_base)
        base_rgb = renderer.render().copy()
        renderer_wrist.update_scene(data, camera=cam_wrist_id, scene_option=opt_wrist)
        wrist_rgb = renderer_wrist.render().copy()

        base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

        w_base = int(base_bgr.shape[1] * h_target / base_bgr.shape[0])
        w_wrist = int(wrist_bgr.shape[1] * h_target / wrist_bgr.shape[0])
        base_r = cv2.resize(base_bgr, (w_base, h_target))
        wrist_r = cv2.resize(wrist_bgr, (w_wrist, h_target))

        tcp_pos, _ = get_tcp_pose(model, data, ids['hand_id'])
        cup_z = data.qpos[cup_qpa+2]
        uprightness = _get_uprightness()
        fi_safe = min(fi, n_total - 1)

        overlay = [
            f"Frame {fi:03d}/{n_total}  {phase_str}",
            f"TCP: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]",
            f"Grip: {states_all[fi_safe, 7]:.4f}",
            f"Cup Z: {cup_z:.4f}  Upright: {uprightness:.3f}",
        ]
        add_overlay(base_r, overlay)
        add_overlay(wrist_r, [f"WRIST {fi:03d}"])

        combined = np.hstack([base_r, wrist_r])
        frames.append(combined)

    # ── Main simulation loop (pure physics, matching batch_replay v8) ──
    correction_applied = False
    grasp_confirmed = False

    for fi in range(n_total):
        pos = states_all[fi, :3]
        quat_xyzw = states_all[fi, 3:7]
        gw_val = float(states_all[fi, 7])

        # IK solve for control targets (save/restore state)
        save_qpos = data.qpos.copy()
        save_qvel = data.qvel.copy()
        solve_ik(model, data, ids['hand_id'], ids['jnt_ids'], pos, quat_xyzw,
                 max_iter=50, q_ref=q_ref)
        ik_targets = [data.qpos[model.jnt_qposadr[jid]] for jid in ids['jnt_ids']]
        data.qpos[:] = save_qpos
        data.qvel[:] = save_qvel

        # Set ctrl targets
        for ai in range(7):
            data.ctrl[ai] = ik_targets[ai]
        fp = np.clip(gw_val, 0.0, 0.04)
        data.ctrl[7] = fp; data.ctrl[8] = fp

        # Apply 30° upright correction at close frame
        if fi == close_idx and not correction_applied and CORRECTION_DEG > 0:
            cur_quat_wxyz = data.qpos[cup_qpa+3:cup_qpa+7].copy()
            cur_R = Rotation.from_quat([cur_quat_wxyz[1], cur_quat_wxyz[2],
                                        cur_quat_wxyz[3], cur_quat_wxyz[0]])
            cup_z_world = cur_R.as_matrix()[:, 2]
            rot_axis = np.cross(cup_z_world, [0, 0, 1])
            if np.linalg.norm(rot_axis) > 1e-6:
                rot_axis /= np.linalg.norm(rot_axis)
                correction_R = Rotation.from_rotvec(rot_axis * np.radians(CORRECTION_DEG))
                new_R = correction_R * cur_R
                new_q = new_R.as_quat()  # xyzw
                data.qpos[cup_qpa+3] = new_q[3]  # w
                data.qpos[cup_qpa+4:cup_qpa+7] = new_q[:3]  # xyz
            correction_applied = True

        # Step physics
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # Bilateral grasp check
        if correction_applied and not grasp_confirmed:
            if fi <= close_idx + GRASP_GRACE_FRAMES:
                grasp_confirmed = _check_bilateral_grasp(
                    model, data, cup_geom_ids, left_finger_gids, right_finger_gids)

        # Check explosion
        cup_z = data.qpos[cup_qpa+2]
        if cup_z > 0.5 or cup_z < -0.1:
            _print(f"  [ep{ep_idx:02d}] Cup exploded at f={fi}, z={cup_z:.3f}")
            break

        # Render
        phase = "grasp" if fi >= close_idx and fi < open_idx else \
                "release" if fi >= open_idx else "approach"
        _render_combined(fi, phase)

    # ── Settle phase ──
    for si in range(N_SETTLE_FRAMES):
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if si % 10 == 0:
            _render_combined(n_total + si, "settle")

    # ── Success check ──
    cup_mat = data.xmat[cup_bid].reshape(3, 3)
    uprightness = float(cup_mat[:, 2][2])
    cup_vel = np.linalg.norm(data.qvel[cup_dof:cup_dof+6])
    cup_z_pos = data.qpos[cup_qpa+2]
    success = (uprightness > 0.85 and cup_vel < 0.5
               and cup_z_pos > 0.0 and cup_z_pos < CUP_HEIGHT * 1.5)

    tag = "SUCCESS" if success else "FAIL"

    # ── Write video ──
    write_video(frames, output)

    # Copy to .VIEW/
    view_dir = os.path.expanduser("~/Repos/Intern/.VIEW")
    os.makedirs(view_dir, exist_ok=True)
    import shutil
    dst = os.path.join(view_dir, f"cup_replay_ep{ep_idx:03d}_{tag}.mp4")
    shutil.copy2(output, dst)

    renderer.close()
    renderer_wrist.close()

    return (ep_idx, tag, uprightness)


def _worker_init():
    """Per-worker init for multiprocessing (set OMP threads to 1)."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def _worker_fn(args):
    """Wrapper for multiprocessing Pool."""
    ep_idx, dataset, output_dir, calib = args
    try:
        return replay_episode(ep_idx, dataset, output_dir, calib, quiet=True)
    except Exception as e:
        return (ep_idx, f"ERROR: {e}", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Replay stand-cup episode (v8 pure physics)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Single episode index")
    parser.add_argument("--episode-range", type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help="Replay episodes START..END (inclusive)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--dataset", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/datasets/StandCup_sim_27ep"))
    parser.add_argument("--output-dir", type=str, default="videos/cup_replay")
    parser.add_argument("--calib", type=str,
                        default=os.path.expanduser("~/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml"))
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Determine episodes to replay
    if args.episode_range is not None:
        episodes = list(range(args.episode_range[0], args.episode_range[1] + 1))
    elif args.episode is not None:
        episodes = [args.episode]
    else:
        episodes = [0]

    import time
    t0 = time.time()

    if args.parallel > 1 and len(episodes) > 1:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        work = [(ep, args.dataset, args.output_dir, args.calib) for ep in episodes]
        print(f"[batch] Replaying {len(episodes)} episodes with {args.parallel} workers...")
        with mp.Pool(args.parallel, initializer=_worker_init) as pool:
            results = pool.map(_worker_fn, work)
    else:
        # Sequential
        results = []
        for ep in episodes:
            r = replay_episode(ep, args.dataset, args.output_dir, args.calib,
                               quiet=(len(episodes) > 1))
            results.append(r)

    elapsed = time.time() - t0

    # Summary
    n_success = sum(1 for _, tag, _ in results if tag == "SUCCESS")
    n_fail = sum(1 for _, tag, _ in results if tag == "FAIL")
    n_error = sum(1 for _, tag, _ in results if "ERROR" in str(tag))
    print(f"\n{'='*50}")
    print(f"[SUMMARY] {len(results)} episodes in {elapsed:.1f}s "
          f"({elapsed/len(results):.1f}s/ep)")
    print(f"  SUCCESS: {n_success}/{len(results)}  "
          f"FAIL: {n_fail}/{len(results)}  ERROR: {n_error}/{len(results)}")
    print(f"  SR: {n_success/len(results)*100:.1f}%")
    for ep_idx, tag, up in results:
        print(f"  ep{ep_idx:02d}: {tag} (uprightness={up:.3f})")


if __name__ == '__main__':
    main()
