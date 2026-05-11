#!/usr/bin/env python3
"""Render test: standard MuJoCo rendering for each task.
Saves PNG images to outputs/render_test/
"""
import os, sys, time
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "resfit", "rl_finetuning"))
sys.path.insert(0, os.path.join(BASE, "resfit", "rl_finetuning", "wrappers"))

OUT_DIR = os.path.join(BASE, "outputs", "render_test")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────
def render_env(env, env_idx=0):
    """Render base + wrist cameras from a VecEnv. Returns (base_rgb, wrist_rgb) HWC."""
    e = env._envs[env_idx]
    model, data = e["model"], e["data"]

    e["renderer_base"].update_scene(data, camera=e["cam_base_id"],
                                     scene_option=e["opt_base"])
    img_base = e["renderer_base"].render().copy()

    e["renderer_wrist"].update_scene(data, camera=e["cam_wrist_id"],
                                      scene_option=e["opt_wrist"])
    img_wrist = e["renderer_wrist"].render().copy()

    return img_base, img_wrist


def save_imgs(task_name, img_base, img_wrist, tag="standard"):
    """Save RGB images as PNGs (convert RGB→BGR for cv2)."""
    cv2.imwrite(os.path.join(OUT_DIR, f"{task_name}_{tag}_base.png"),
                cv2.cvtColor(img_base, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT_DIR, f"{task_name}_{tag}_wrist.png"),
                cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR))
    print(f"  ✓ {task_name} ({tag}): base={img_base.shape}, wrist={img_wrist.shape}")


# ── test each task ───────────────────────────────────────────────
def test_pnp():
    from mujoco_vec_env_pnp import MuJoCoVecEnvPnP
    env = MuJoCoVecEnvPnP(num_envs=1, parallel_envs=False)
    env.reset()
    img_b, img_w = render_env(env)
    save_imgs("pnp", img_b, img_w)
    return env


def test_stack():
    from mujoco_vec_env_stack import MuJoCoVecEnvStack
    env = MuJoCoVecEnvStack(num_envs=1, parallel_envs=False)
    env.reset()
    img_b, img_w = render_env(env)
    save_imgs("stack", img_b, img_w)
    return env


def test_drawer():
    from mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
    env = MuJoCoVecEnvDrawer(num_envs=1, parallel_envs=False)
    env.reset()
    img_b, img_w = render_env(env)
    save_imgs("drawer", img_b, img_w)
    return env


def test_cup():
    from mujoco_vec_env_cup import MuJoCoVecEnvCup
    env = MuJoCoVecEnvCup(num_envs=1, parallel_envs=False)
    env.reset()
    img_b, img_w = render_env(env)
    save_imgs("cup", img_b, img_w)
    return env


def test_lift():
    from mujoco_vec_env import MuJoCoVecEnv
    env = MuJoCoVecEnv(num_envs=1, parallel_envs=False)
    env.reset()
    img_b, img_w = render_env(env)
    save_imgs("lift", img_b, img_w)
    return env


# ── realistic rendering test (PnP only) ─────────────────────────
def test_realistic(env, task_name="pnp"):
    """Test RealisticRenderHelper with existing VecEnv's model/data."""
    realistic_src = os.path.expanduser(
        "~/Repos/Intern/Mujoco_Franka/src")
    sys.path.insert(0, realistic_src)

    try:
        from scene_realistic import RealisticRenderHelper, make_model_generic
    except ImportError as exc:
        print(f"  ✗ Realistic import failed: {exc}")
        return

    # Check real_background.png
    bg_path = os.path.join(realistic_src, "output", "real_background.png")
    if not os.path.isfile(bg_path):
        print(f"  ✗ Realistic skipped: real_background.png missing at {bg_path}")
        print(f"    → Generate it first, then re-run")
        return

    e = env._envs[0]
    model, data = e["model"], e["data"]

    # Shadow bodies = object bodies in the scene
    shadow_bodies = []
    for name in ("cube", "green_cube", "white_cube", "bowl"):
        bid = -1
        try:
            import mujoco
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            pass
        if bid >= 0:
            shadow_bodies.append(name)

    try:
        helper = RealisticRenderHelper(
            model, shadow_bodies=shadow_bodies,
            cam_h=360, cam_w=640,
            wrist_h=360, wrist_w=640,
        )
        img_base_bgr = helper.render_base(data)
        img_wrist_bgr = helper.render_wrist(data)

        cv2.imwrite(os.path.join(OUT_DIR, f"{task_name}_realistic_base.png"), img_base_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"{task_name}_realistic_wrist.png"), img_wrist_bgr)
        print(f"  ✓ {task_name} (realistic): base={img_base_bgr.shape}, wrist={img_wrist_bgr.shape}")
        helper.close()
    except Exception as exc:
        print(f"  ✗ Realistic rendering failed: {exc}")
        import traceback; traceback.print_exc()


# ── main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Render test — output: {OUT_DIR}\n")

    tasks = {
        "pnp":    test_pnp,
        "stack":  test_stack,
        "drawer": test_drawer,
        "cup":    test_cup,
        "lift":   test_lift,
    }

    envs = {}
    for name, fn in tasks.items():
        t0 = time.time()
        print(f"[{name}] Initializing...")
        try:
            envs[name] = fn()
            print(f"  ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            import traceback; traceback.print_exc()

    # Realistic test on PnP
    print(f"\n[pnp] Testing realistic renderer...")
    if "pnp" in envs:
        test_realistic(envs["pnp"], "pnp")

    print(f"\n✅ Done! Images saved to {OUT_DIR}")
