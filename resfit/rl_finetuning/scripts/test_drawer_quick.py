#!/usr/bin/env python3
"""Quick 5-episode drawer eval test on checkpoint 100k."""
import os, sys, json, time

os.environ['MUJOCO_GL'] = 'egl'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

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

print("=== Quick Drawer Eval: 5 episodes, ckpt 100k ===", flush=True)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer, DRAWER_SLIDE, FLOOR_TO_DRAWER, load_episode_mapping,
    WALL_THICK, DRAWER_SLOT_HEIGHT,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer

print("Imports OK", flush=True)

env = MuJoCoVecEnvDrawer(num_envs=1, max_episode_steps=500)
print(f"Env created. Active drawer: {env._active_drawers}", flush=True)

obs, info = env.reset()
print(f"Reset OK. State: {obs['observation.state'].shape}, Drawer qpos: {env._drawer_qpos[0]:.4f}", flush=True)

ckpt = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000")
print(f"Loading GR00T from {ckpt}...", flush=True)
wrapper = MuJoCoResidualWrapperDrawer(
    vec_env=env, groot_checkpoint=ckpt, policy_device="cuda:0"
)
print("Wrapper ready.", flush=True)

eps = load_episode_mapping()
print(f"Total episodes: {len(eps)}", flush=True)

test_eps = [1, 2, 0]  # drawer 2 (floor3), drawer 3 (floor4), drawer 4 (floor5)
results = []
for ei in test_eps:
    ep = eps[ei]
    floor = ep["floor"]
    active_drawer = FLOOR_TO_DRAWER[floor]

    env._active_drawers[0] = active_drawer
    env_data = env._envs[0]
    env_data["active_drawer"] = active_drawer
    slot_h = DRAWER_SLOT_HEIGHT
    dz = WALL_THICK + slot_h / 2 + active_drawer * (slot_h + WALL_THICK)
    env_data["drawer_z_center"] = dz
    env_data["drawer_z_min"] = dz - slot_h / 2 - 0.02
    env_data["drawer_z_max"] = dz + slot_h / 2 + 0.02

    obs, _ = wrapper.reset()

    t0 = time.time()
    success = False
    final_step = 0
    min_slide = DRAWER_SLIDE  # track minimum slide during episode
    import mujoco as _mj
    import numpy as np
    TCP_OFF = np.array([0, 0, 0.1034])
    for step in range(500):
        # Record slide BEFORE step (pre auto-reset)
        cur_slide = env._drawer_qpos[0]
        min_slide = min(min_slide, cur_slide)
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, rew, term, trunc, info = wrapper.step(residual)
        # Also record after env step but before wrapper auto-reset
        # Note: wrapper.step() already reset if term, so read from within env step
        post_slide = env._drawer_qpos[0]
        min_slide = min(min_slide, post_slide)
        if step % 50 == 0 or term[0] or trunc[0]:
            # Debug: TCP position relative to drawer face
            e = env._envs[0]
            d = e["data"]; ids_ = e["ids"]
            h_pos = d.xpos[ids_["hand_id"]]
            h_mat = d.xmat[ids_["hand_id"]].reshape(3,3)
            tcp_w = h_pos + h_mat @ TCP_OFF
            cab_p = d.xpos[e["cab_body_id"]]
            cab_m = d.xmat[e["cab_body_id"]].reshape(3,3)
            tcp_l = cab_m.T @ (tcp_w - cab_p)
            face_x_now = env._face_x_closed + cur_slide
            print(f"    step={step:3d} slide={cur_slide:.4f} tcp_local=[{tcp_l[0]:.4f},{tcp_l[1]:.4f},{tcp_l[2]:.4f}] "
                  f"face_x_now={face_x_now:.4f} dist={tcp_l[0]-face_x_now:.4f} "
                  f"y_ok={abs(tcp_l[1])<env._face_hy+0.03} z_ok={tcp_l[2]>e['drawer_z_min'] and tcp_l[2]<e['drawer_z_max']}", flush=True)
        if term[0]:
            success = True
            final_step = step + 1
            break
        if trunc[0]:
            final_step = step + 1
            break
    else:
        final_step = 500

    dt = time.time() - t0
    closed_pct = (1 - min_slide / DRAWER_SLIDE) * 100
    tag = "SUCC" if success else "FAIL"
    print(f"  ep{ei} floor={floor} drawer={active_drawer} => {tag} steps={final_step} "
          f"closed={closed_pct:.0f}% min_slide={min_slide:.4f} t={dt:.1f}s", flush=True)
    results.append({"ep": ei, "floor": floor, "success": success, "closed_pct": closed_pct})

print(flush=True)
sr = sum(1 for r in results if r["success"]) / len(results) * 100
avg_closed = sum(r["closed_pct"] for r in results) / len(results)
print(f"=== Results: SR={sr:.0f}% ({sum(1 for r in results if r['success'])}/{len(results)}) "
      f"Avg Closed={avg_closed:.0f}% ===", flush=True)
