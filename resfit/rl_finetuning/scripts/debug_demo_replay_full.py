"""Replay ALL 33 demo episodes through MuJoCo drawer env (no GR00T needed).

Fixes active_drawer switching bug (must update env dict + z-range).
Computes per-drawer:
  - success rate, closed fraction
  - ndist statistics (p25/p50/p75/p90, % within threshold)
  - demo_mean (y_norm, z_norm) from actual demo contact points
Saves results JSON for reference.

Usage (login node OK — no GR00T/GPU required):
    python debug_demo_replay_full.py
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
_gl_path = os.path.expanduser("~/.local/lib/gl")
if os.path.isdir(_gl_path):
    import ctypes
    ctypes.CDLL(os.path.join(_gl_path, "libGLdispatch.so.0"))
    ctypes.CDLL(os.path.join(_gl_path, "libOpenGL.so.0"))
    ctypes.CDLL(os.path.join(_gl_path, "libEGL.so.1"))

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, load_episode_mapping, FLOOR_TO_DRAWER,
    DEMO_CONTACT_MEAN, DRAWER_SLIDE, DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
    WALL_THICK,
)

DATASET_DIR = Path(os.path.expanduser(
    "~/DATA/INTERN/datasets/CloseDrawer_sim_33ep/data/chunk-000"))
OUT_DIR = Path(os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl/outputs/demo_replay"))


def _set_active_drawer(env, env_idx, drawer_idx):
    """Properly switch active drawer: update env dict + z-range."""
    e = env._envs[env_idx]
    e["active_drawer"] = drawer_idx
    env._active_drawers[env_idx] = drawer_idx
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + drawer_idx * (slot_h + WALL_THICK)
    e["drawer_z_center"] = dz
    e["drawer_z_min"] = dz - slot_h / 2 - 0.02
    e["drawer_z_max"] = dz + slot_h / 2 + 0.02


def compute_contact_info(env, env_idx):
    """Compute TCP contact info: ndist, y_norm, z_norm, y_ok, z_ok."""
    e = env._envs[env_idx]
    model, data, ids = e["model"], e["data"], e["ids"]
    active_drawer = e["active_drawer"]

    hand_pos = data.xpos[ids["hand_id"]]
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp = hand_pos + hand_mat @ TCP_OFFSET
    cab_pos_w = data.xpos[e["cab_body_id"]]
    cab_mat = data.xmat[e["cab_body_id"]].reshape(3, 3)
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)

    y_ok = abs(tcp_local[1]) < env._face_hy + 0.03
    z_ok = (tcp_local[2] > e["drawer_z_min"]) and (tcp_local[2] < e["drawer_z_max"])

    dm = DEMO_CONTACT_MEAN.get(active_drawer)
    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz_center = e["drawer_z_min"] + face_hz + DRAWER_GAP
    y_norm = tcp_local[1] / env._face_hy if env._face_hy > 0 else 0.0
    z_norm = (tcp_local[2] - dz_center) / face_hz if face_hz > 0 else 0.0

    if not (y_ok and z_ok) or dm is None:
        return float("inf"), y_norm, z_norm, y_ok, z_ok

    ndist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
    return ndist, y_norm, z_norm, y_ok, z_ok


def main():
    eps = load_episode_mapping()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for threshold_val, threshold_label in [(float("inf"), "no_gate"), (1.0, "gate_1.0")]:
        print(f"\n{'='*70}")
        print(f"  Replay with contact_threshold={threshold_label}")
        print(f"{'='*70}")

        env = MuJoCoVecEnvDrawer(
            num_envs=1, reward_type="dense", max_episode_steps=1000,
            contact_threshold=threshold_val,
        )

        drawer_stats = {d: {
            "closed_fracs": [], "ndists": [], "y_norms": [], "z_norms": [],
            "success": 0, "total": 0, "episodes": [],
        } for d in [2, 3, 4]}

        for ei, ep_info in enumerate(eps):
            d = FLOOR_TO_DRAWER.get(ep_info["floor"])
            if d is None:
                continue

            pq_path = DATASET_DIR / f"episode_{ei:06d}.parquet"
            if not pq_path.exists():
                print(f"  ep{ei}: parquet not found, skip")
                continue
            df = pd.read_parquet(pq_path)
            actions = np.array(df["action"].tolist(), dtype=np.float32)

            _set_active_drawer(env, 0, d)
            env.reset()

            ep_ndists = []
            ep_y_norms = []
            ep_z_norms = []
            max_closed = 0.0
            contact_steps = 0

            for t in range(len(actions)):
                action_t = torch.tensor(actions[t:t+1], dtype=torch.float32)
                obs, reward, terminated, truncated, info = env.step(action_t)

                ndist, yn, zn, y_ok, z_ok = compute_contact_info(env, 0)
                if y_ok and z_ok:
                    ep_ndists.append(ndist)
                    ep_y_norms.append(yn)
                    ep_z_norms.append(zn)
                    contact_steps += 1

                closed_frac = 1.0 - (env._drawer_qpos[0] / DRAWER_SLIDE)
                max_closed = max(max_closed, closed_frac)

                if terminated[0]:
                    break

            success = bool(env._is_success(0))
            ds = drawer_stats[d]
            ds["total"] += 1
            ds["closed_fracs"].append(max_closed)
            if ep_ndists:
                ds["ndists"].extend(ep_ndists)
                ds["y_norms"].extend(ep_y_norms)
                ds["z_norms"].extend(ep_z_norms)
            if success:
                ds["success"] += 1

            ds["episodes"].append({
                "ep_idx": ei, "success": success,
                "max_closed": round(max_closed, 4),
                "contact_steps": contact_steps,
                "total_steps": len(actions),
                "ndist_mean": round(float(np.mean(ep_ndists)), 4) if ep_ndists else None,
            })

            tag = "SUCC" if success else "FAIL"
            nd_str = f"{np.mean(ep_ndists):.2f}" if ep_ndists else "inf"
            print(f"  ep{ei:2d} d{d} {tag}  closed={max_closed*100:5.1f}%  "
                  f"contact={contact_steps:3d}/{len(actions)}  ndist_mean={nd_str}")

        # Summary
        print(f"\n--- Summary ({threshold_label}) ---")
        result = {"threshold": threshold_label, "drawers": {}}
        total_s, total_n = 0, 0

        for d in [2, 3, 4]:
            ds = drawer_stats[d]
            if ds["total"] == 0:
                continue
            sr = ds["success"] / ds["total"]
            cf_mean = float(np.mean(ds["closed_fracs"]))
            total_s += ds["success"]
            total_n += ds["total"]

            drawer_result = {
                "success_rate": round(sr, 4),
                "success_count": ds["success"],
                "total_count": ds["total"],
                "closed_frac_mean": round(cf_mean, 4),
                "episodes": ds["episodes"],
            }

            if ds["ndists"]:
                nd_arr = np.array(ds["ndists"])
                pcts = np.percentile(nd_arr, [25, 50, 75, 90])
                in_1 = float((nd_arr <= 1.0).sum()) / len(nd_arr)

                # Compute demo_mean from actual contact points
                demo_y_mean = float(np.mean(ds["y_norms"]))
                demo_z_mean = float(np.mean(ds["z_norms"]))
                demo_y_std = float(np.std(ds["y_norms"]))
                demo_z_std = float(np.std(ds["z_norms"]))

                drawer_result.update({
                    "ndist_p25": round(pcts[0], 4),
                    "ndist_p50": round(pcts[1], 4),
                    "ndist_p75": round(pcts[2], 4),
                    "ndist_p90": round(pcts[3], 4),
                    "ndist_leq_1.0": round(in_1, 4),
                    "demo_mean_y_norm": round(demo_y_mean, 4),
                    "demo_mean_z_norm": round(demo_z_mean, 4),
                    "demo_std_y_norm": round(demo_y_std, 4),
                    "demo_std_z_norm": round(demo_z_std, 4),
                    "contact_steps_total": len(ds["ndists"]),
                })

                print(f"  D{d}: SR={ds['success']}/{ds['total']}={sr*100:.0f}%  "
                      f"closed={cf_mean*100:.1f}%  "
                      f"ndist p25={pcts[0]:.2f} p50={pcts[1]:.2f} p75={pcts[2]:.2f} p90={pcts[3]:.2f}  "
                      f"≤1.0: {in_1*100:.1f}%")
                print(f"       demo_mean: y_norm={demo_y_mean:.4f}±{demo_y_std:.4f}  "
                      f"z_norm={demo_z_mean:.4f}±{demo_z_std:.4f}")

                # Compare with existing DEMO_CONTACT_MEAN
                old_dm = DEMO_CONTACT_MEAN.get(d)
                if old_dm:
                    print(f"       OLD mean:  y_norm={old_dm['y_norm']:.4f}  "
                          f"z_norm={old_dm['z_norm']:.4f}  "
                          f"Δy={demo_y_mean - old_dm['y_norm']:.4f}  "
                          f"Δz={demo_z_mean - old_dm['z_norm']:.4f}")
            else:
                print(f"  D{d}: SR={ds['success']}/{ds['total']}={sr*100:.0f}%  "
                      f"closed={cf_mean*100:.1f}%  no contact steps")

            result["drawers"][str(d)] = drawer_result

        total_sr = total_s / total_n if total_n > 0 else 0
        result["total_sr"] = round(total_sr, 4)
        result["total_success"] = total_s
        result["total_count"] = total_n
        print(f"  Total: {total_s}/{total_n}={total_sr*100:.0f}%")

        all_results[threshold_label] = result
        del env

    # Save
    out_path = OUT_DIR / f"demo_replay_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
