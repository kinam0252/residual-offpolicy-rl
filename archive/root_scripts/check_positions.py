"""Visualize difficulty levels using eval-style rendering (MuJoCoVecEnvPnP).

Grid: Row 1=Easy, Rows 2-4=Normal, Rows 5-7=Hard-extra
Uses vec_env.get_frame(camera='back') same as eval script.
"""
import sys, os, json, math
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
import cv2
from scipy.spatial.transform import Rotation

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

def setup_env(vec_env, cube_pos, bowl_pos, cube_yaw_deg):
    """Set env state exactly like eval script (after wrapper.reset())."""
    env_data = vec_env._envs[0]
    data = env_data["data"]
    model = env_data["model"]
    cqa = env_data["cube_qposadr"]
    bowl_body_id = env_data["bowl_body_id"]

    # Robot joints
    for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
    for fid in env_data["ids"]["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04

    # Cube
    if cqa is not None:
        data.qpos[cqa:cqa+3] = cube_pos
        q_xyzw = Rotation.from_euler("z", np.radians(cube_yaw_deg)).as_quat()
        data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    # Bowl
    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_pos

    mujoco.mj_forward(model, data)

def main():
    scene_xml = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")
    positions = json.load(open("configs/pnp_common5_positions.json"))[:5]

    vec_env = MuJoCoVecEnvPnP(
        num_envs=1,
        scene_xml=scene_xml,
        max_episode_steps=500,
        reward_type="dense_v2",
    )
    vec_env.reset()

    np.random.seed(42)
    N_SAMPLES = 3
    FRAME_SIZE = (360, 640)

    rows = []
    labels = []
    colors = {"EASY": (0,255,0), "NORMAL": (0,200,255), "HARD": (0,100,255)}

    # Row 1: Easy
    row_frames = []
    for p in positions:
        setup_env(vec_env, p["cube_pos"], p["bowl_pos"], p["cube_yaw_deg"])
        frame = vec_env.get_frame(0, camera='back', size=FRAME_SIZE)
        cv2.putText(frame, f"ep{p['episode']} (fixed)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["EASY"], 2)
        row_frames.append(frame)
    rows.append(np.concatenate(row_frames, axis=1))
    labels.append("EASY")

    # Rows 2-4: Normal
    for si in range(N_SAMPLES):
        row_frames = []
        for p in positions:
            dx = np.random.uniform(-0.03, 0.03)
            dy = np.random.uniform(-0.03, 0.03)
            dyaw = np.random.uniform(-15, 15)
            cp = [p["cube_pos"][0]+dx, p["cube_pos"][1]+dy, p["cube_pos"][2]]
            yaw = p["cube_yaw_deg"] + dyaw
            setup_env(vec_env, cp, p["bowl_pos"], yaw)
            frame = vec_env.get_frame(0, camera='back', size=FRAME_SIZE)
            cv2.putText(frame, f"({dx*100:+.1f},{dy*100:+.1f})cm y={dyaw:+.0f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["NORMAL"], 2)
            row_frames.append(frame)
        rows.append(np.concatenate(row_frames, axis=1))
        labels.append(f"NORMAL {si+1}")

    # Rows 5-7: Hard-extra
    for si in range(N_SAMPLES):
        row_frames = []
        for p in positions:
            dx = np.random.uniform(-0.06, 0.06)
            dy = np.random.uniform(-0.06, 0.06)
            dyaw = np.random.uniform(-45, 45)
            cp = [p["cube_pos"][0]+dx, p["cube_pos"][1]+dy, p["cube_pos"][2]]
            yaw = p["cube_yaw_deg"] + dyaw
            setup_env(vec_env, cp, p["bowl_pos"], yaw)
            frame = vec_env.get_frame(0, camera='back', size=FRAME_SIZE)
            cv2.putText(frame, f"({dx*100:+.1f},{dy*100:+.1f})cm y={dyaw:+.0f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["HARD"], 2)
            row_frames.append(frame)
        rows.append(np.concatenate(row_frames, axis=1))
        labels.append(f"HARD {si+1}")

    # Add row labels
    H = FRAME_SIZE[0]
    LABEL_W = 130
    final_rows = []
    for row_img, label in zip(rows, labels):
        label_img = np.zeros((H, LABEL_W, 3), dtype=np.uint8)
        lvl = label.split()[0]
        col = colors.get(lvl, (255,255,255))
        cv2.putText(label_img, label, (5, H//2+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        final_rows.append(np.concatenate([label_img, row_img], axis=1))

    grid = np.concatenate(final_rows, axis=0)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    out_path = "outputs/difficulty_visualization.png"
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(out_path, grid_bgr)
    print(f"Saved: {out_path} ({grid.shape[1]}x{grid.shape[0]})")

if __name__ == "__main__":
    main()
