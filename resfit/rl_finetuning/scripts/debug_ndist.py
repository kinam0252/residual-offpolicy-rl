"""Measure norm_dist distribution across all 33 episodes."""
import sys, os, importlib, types as _types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__="0.0.0"
    _ds.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType("deepspeed.zero"); _dz.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    _dc = _types.ModuleType("deepspeed.comm"); _dc.__spec__=importlib.machinery.ModuleSpec("deepspeed.comm",None)
    _ds.comm=_dc
    sys.modules["deepspeed"]=_ds; sys.modules["deepspeed.zero"]=_dz; sys.modules["deepspeed.comm"]=_dc
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, load_episode_mapping, FLOOR_TO_DRAWER,
    DEMO_CONTACT_MEAN, DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer


def main():
    ckpt = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000")
    eps = load_episode_mapping()

    print("Creating env (no contact gate)...")
    env = MuJoCoVecEnvDrawer(
        num_envs=1, reward_type="sparse", max_episode_steps=500,
        contact_threshold=float("inf"),
    )
    print("Loading GR00T...")
    wrapper = MuJoCoResidualWrapperDrawer(
        vec_env=env, groot_checkpoint=ckpt, policy_device="cuda:0",
    )
    print("Ready.\n")

    all_ndist = {2: [], 3: [], 4: []}

    for ei, ep_info in enumerate(eps):
        d = FLOOR_TO_DRAWER.get(ep_info["floor"])
        if d is None:
            continue
        env._active_drawers[0] = d
        obs, _ = wrapper.reset()

        for step in range(500):
            e = env._envs[0]
            model, data, ids = e["model"], e["data"], e["ids"]
            hand_pos = data.xpos[ids["hand_id"]]
            hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
            tcp = hand_pos + hand_mat @ TCP_OFFSET
            cab_pos_w = data.xpos[e["cab_body_id"]]
            cab_mat = data.xmat[e["cab_body_id"]].reshape(3, 3)
            tcp_local = cab_mat.T @ (tcp - cab_pos_w)

            y_ok = abs(tcp_local[1]) < env._face_hy + 0.03
            z_ok = (tcp_local[2] > e["drawer_z_min"]) and (tcp_local[2] < e["drawer_z_max"])
            if y_ok and z_ok:
                dm = DEMO_CONTACT_MEAN.get(d)
                face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
                dz_c = e["drawer_z_min"] + face_hz + DRAWER_GAP
                yn = tcp_local[1] / env._face_hy if env._face_hy > 0 else 0
                zn = (tcp_local[2] - dz_c) / face_hz if face_hz > 0 else 0
                nd = np.sqrt((yn - dm["y_norm"])**2 + (zn - dm["z_norm"])**2)
                all_ndist[d].append(nd)

            residual = torch.zeros((1, 7), dtype=torch.float32)
            obs, r, term, trunc, info = wrapper.step(residual)
            if term[0] or trunc[0]:
                break

        print(f"ep{ei} d{d} done ({step+1} steps)", flush=True)

    print(f"\n{'='*70}")
    print("Norm-dist distribution (steps where y_ok & z_ok, i.e. face region)")
    print(f"{'='*70}")
    for d in [2, 3, 4]:
        arr = np.array(all_ndist[d])
        if len(arr) == 0:
            print(f"D{d}: no contact steps")
            continue
        pcts = np.percentile(arr, [10, 25, 50, 75, 90])
        in_05 = (arr <= 0.5).sum() / len(arr) * 100
        in_08 = (arr <= 0.8).sum() / len(arr) * 100
        in_10 = (arr <= 1.0).sum() / len(arr) * 100
        in_15 = (arr <= 1.5).sum() / len(arr) * 100
        print(f"D{d}: n={len(arr):5d}  mean={arr.mean():.3f} std={arr.std():.3f}")
        print(f"     p10={pcts[0]:.2f} p25={pcts[1]:.2f} p50={pcts[2]:.2f} "
              f"p75={pcts[3]:.2f} p90={pcts[4]:.2f}")
        print(f"     ≤0.5: {in_05:.1f}%  ≤0.8: {in_08:.1f}%  "
              f"≤1.0: {in_10:.1f}%  ≤1.5: {in_15:.1f}%")


if __name__ == "__main__":
    main()
