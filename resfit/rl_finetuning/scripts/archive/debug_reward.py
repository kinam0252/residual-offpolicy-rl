"""Quick reward debug: 3 episodes (D2/D3/D4), print per-step reward breakdown."""
import sys, os, importlib, types as _types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

# Deepspeed mock
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
    DEMO_CONTACT_MEAN, DRAWER_SLIDE, DRAWER_SLOT_HEIGHT, DRAWER_GAP, TCP_OFFSET,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer


def reward_breakdown(mujoco_env, env_idx):
    """Recompute reward components for debugging (mirrors _compute_reward dense)."""
    env = mujoco_env._envs[env_idx]
    model, data, ids = env["model"], env["data"], env["ids"]
    active_drawer = env["active_drawer"]

    hand_pos = data.xpos[ids["hand_id"]]
    hand_mat = data.xmat[ids["hand_id"]].reshape(3, 3)
    tcp = hand_pos + hand_mat @ TCP_OFFSET
    cab_pos_w = data.xpos[env["cab_body_id"]]
    cab_mat = data.xmat[env["cab_body_id"]].reshape(3, 3)
    tcp_local = cab_mat.T @ (tcp - cab_pos_w)

    dm = DEMO_CONTACT_MEAN.get(active_drawer)
    face_hz = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP
    dz_center = env["drawer_z_min"] + face_hz + DRAWER_GAP
    target_x = mujoco_env._face_x_closed + mujoco_env._drawer_qpos[env_idx]
    target_y = dm["y_norm"] * mujoco_env._face_hy if dm else 0.0
    target_z = dz_center + dm["z_norm"] * face_hz if dm else dz_center

    dist_3d = float(np.linalg.norm(tcp_local - np.array([target_x, target_y, target_z])))
    approach = (1.0 - np.tanh(dist_3d / 0.1)) * 0.25

    y_ok = abs(tcp_local[1]) < mujoco_env._face_hy + 0.03
    z_ok = (tcp_local[2] > env["drawer_z_min"]) and (tcp_local[2] < env["drawer_z_max"])
    contact = 0.0
    norm_dist = float('inf')
    if y_ok and z_ok and dm is not None:
        y_norm = tcp_local[1] / mujoco_env._face_hy if mujoco_env._face_hy > 0 else 0.0
        z_norm = (tcp_local[2] - dz_center) / face_hz if face_hz > 0 else 0.0
        norm_dist = np.sqrt((y_norm - dm["y_norm"])**2 + (z_norm - dm["z_norm"])**2)
        if norm_dist <= 1.0:
            contact = 0.25

    closed_frac = 1.0 - (mujoco_env._drawer_qpos[env_idx] / DRAWER_SLIDE)
    push = float(np.clip(closed_frac, 0.0, 1.0)) * 0.25
    success = 0.25 if mujoco_env._is_success(env_idx) else 0.0

    return {
        "approach": round(approach, 4),
        "contact": round(contact, 4),
        "push": round(push, 4),
        "success": round(success, 4),
        "total": round(approach + contact + push + success, 4),
        "dist_3d": round(dist_3d, 4),
        "norm_dist": round(norm_dist, 4) if norm_dist < 100 else "inf",
        "closed_frac": round(float(np.clip(closed_frac, 0, 1)), 4),
        "y_ok": y_ok, "z_ok": z_ok,
    }


def main():
    ckpt = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000")
    eps = load_episode_mapping()

    print("Creating env (dense reward, contact_threshold=1.0)...")
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        reward_type="dense",
        max_episode_steps=500,
        contact_threshold=1.0,
    )
    print("Loading GR00T...")
    wrapper = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=ckpt,
        policy_device="cuda:0",
    )
    print("Ready.\n")

    # Pick one episode per drawer
    picked = {}
    for ei, ep in enumerate(eps):
        d = FLOOR_TO_DRAWER.get(ep["floor"])
        if d and d not in picked:
            picked[d] = (ei, ep)
        if len(picked) == 3:
            break

    for drawer_idx in sorted(picked):
        ei, ep = picked[drawer_idx]
        print(f"{'='*70}")
        print(f"  Drawer {drawer_idx}  (ep{ei})")
        print(f"{'='*70}")

        mujoco_env._active_drawers[0] = drawer_idx
        obs, _ = wrapper.reset()

        for step in range(500):
            residual = torch.zeros((1, 7), dtype=torch.float32)
            obs, reward, terminated, truncated, info = wrapper.step(residual)

            bd = reward_breakdown(mujoco_env, 0)
            env_reward = float(reward[0])

            # Print every 10 steps, or on key events
            key_event = (step < 5 or
                         bd["contact"] > 0 or
                         bd["success"] > 0 or
                         (step % 25 == 0))
            if key_event:
                print(f"  step={step:3d}  r={env_reward:.4f}  "
                      f"app={bd['approach']:.3f} con={bd['contact']:.3f} "
                      f"push={bd['push']:.3f} suc={bd['success']:.3f}  "
                      f"dist3d={bd['dist_3d']:.3f} ndist={bd['norm_dist']} "
                      f"cf={bd['closed_frac']:.3f} y={bd['y_ok']} z={bd['z_ok']}")

            if terminated[0] or truncated[0]:
                print(f"  >>> DONE at step {step}: terminated={bool(terminated[0])} "
                      f"truncated={bool(truncated[0])}")
                # Print final breakdown
                print(f"  >>> Final: r={env_reward:.4f}  breakdown={bd}")
                break

        print()


if __name__ == "__main__":
    main()
