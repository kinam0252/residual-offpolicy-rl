"""Replay demo parquet data through MuJoCo drawer env (no GR00T needed).

For each of 33 episodes, feeds parquet action (TCP pose) directly to
MuJoCoVecEnvDrawer.step(), measures:
  - closed fraction (how far drawer closes)
  - contact norm_dist from demo_mean at each step
  - per-step reward breakdown (staged dense reward)

Usage (login node OK — no GR00T/GPU required):
    python debug_demo_replay.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, load_episode_mapping, FLOOR_TO_DRAWER,
    DEMO_CONTACT_MEAN, DRAWER_SLIDE, DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
)

DATASET_DIR = Path(os.path.expanduser(
    "~/DATA/INTERN/datasets/CloseDrawer_sim_33ep/data/chunk-000"))


def compute_ndist(env, env_idx):
    """Compute norm_dist of TCP from demo_mean. Returns (ndist, y_norm, z_norm, y_ok, z_ok)."""
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
    if not (y_ok and z_ok) or dm is None:
        return float("inf"), 0.0, 0.0, y_ok, z_ok

    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz_center = e["drawer_z_min"] + face_hz + DRAWER_GAP
    y_norm = tcp_local[1] / env._face_hy if env._face_hy > 0 else 0.0
    z_norm = (tcp_local[2] - dz_center) / face_hz if face_hz > 0 else 0.0
    ndist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
    return ndist, y_norm, z_norm, y_ok, z_ok


def main():
    eps = load_episode_mapping()

    # Test both with and without contact gate
    for threshold_val, threshold_label in [(float("inf"), "no_gate"), (1.0, "gate_1.0")]:
        print(f"\n{'='*70}")
        print(f"  Replay with contact_threshold={threshold_label}")
        print(f"{'='*70}")

        env = MuJoCoVecEnvDrawer(
            num_envs=1,
            reward_type="dense",
            max_episode_steps=1000,  # large enough to not truncate
            contact_threshold=threshold_val,
        )

        drawer_stats = {d: {"closed_fracs": [], "ndists": [], "success": 0, "total": 0}
                        for d in [2, 3, 4]}

        for ei, ep_info in enumerate(eps):
            d = FLOOR_TO_DRAWER.get(ep_info["floor"])
            if d is None:
                continue

            # Load parquet
            pq_path = DATASET_DIR / f"episode_{ei:06d}.parquet"
            if not pq_path.exists():
                print(f"  ep{ei}: parquet not found, skip")
                continue
            df = pd.read_parquet(pq_path)
            actions = np.array(df["action"].tolist(), dtype=np.float32)  # (T, 8)

            # Reset env with correct drawer
            env._active_drawers[0] = d
            env.reset()

            ep_ndists = []
            max_closed = 0.0

            for t in range(len(actions)):
                action_t = torch.tensor(actions[t:t+1], dtype=torch.float32)  # (1, 8)
                obs, reward, terminated, truncated, info = env.step(action_t)

                ndist, yn, zn, y_ok, z_ok = compute_ndist(env, 0)
                if y_ok and z_ok:
                    ep_ndists.append(ndist)

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
            if success:
                ds["success"] += 1

            tag = "SUCC" if success else "FAIL"
            nd_mean = np.mean(ep_ndists) if ep_ndists else float("inf")
            nd_str = f"{nd_mean:.2f}" if nd_mean < 100 else "inf"
            print(f"  ep{ei:2d} d{d} {tag}  closed={max_closed*100:5.1f}%  "
                  f"contact_steps={len(ep_ndists):3d}/{len(actions)}  "
                  f"ndist_mean={nd_str}")

        # Summary
        print(f"\n--- Summary ({threshold_label}) ---")
        total_s, total_n = 0, 0
        for d in [2, 3, 4]:
            ds = drawer_stats[d]
            if ds["total"] == 0:
                continue
            sr = ds["success"] / ds["total"] * 100
            cf_mean = np.mean(ds["closed_fracs"]) * 100
            total_s += ds["success"]
            total_n += ds["total"]

            if ds["ndists"]:
                nd_arr = np.array(ds["ndists"])
                pcts = np.percentile(nd_arr, [25, 50, 75, 90])
                in_1 = (nd_arr <= 1.0).sum() / len(nd_arr) * 100
                print(f"  D{d}: SR={ds['success']}/{ds['total']}={sr:.0f}%  "
                      f"closed_mean={cf_mean:.1f}%  "
                      f"ndist p25={pcts[0]:.2f} p50={pcts[1]:.2f} p75={pcts[2]:.2f} p90={pcts[3]:.2f}  "
                      f"≤1.0: {in_1:.1f}%")
            else:
                print(f"  D{d}: SR={ds['success']}/{ds['total']}={sr:.0f}%  "
                      f"closed_mean={cf_mean:.1f}%  no contact steps")

        total_sr = total_s / total_n * 100 if total_n > 0 else 0
        print(f"  Total: {total_s}/{total_n}={total_sr:.0f}%")

        del env


if __name__ == "__main__":
    main()
