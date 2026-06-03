#!/usr/bin/env python3
"""Debug kinematic eval: track drawer_qpos over time to understand why SR=0%.

Prints per-step drawer position for a few envs so we can see how far
the drawer closes and where it stalls.
"""
from __future__ import annotations
import importlib, os, sys, types as _types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DS_BUILD_OPS", "0")

if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []
    sys.modules["deepspeed"] = _ds

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer, DRAWER_SLIDE
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--drawer", type=int, default=2)
    parser.add_argument("--groot_checkpoint", type=str,
                        default=os.path.expanduser("checkpoints/<TASK>/checkpoint"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vec_env = MuJoCoVecEnvDrawer(
        num_envs=args.num_envs,
        active_drawers=[args.drawer] * args.num_envs,
        max_episode_steps=args.max_steps,
        device=device,
        reward_type="delta",
        contact_z_gate=True,
        physics_drawer=False,  # KINEMATIC
    )

    env = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="new_embodiment",
        policy_device=str(device),
        task_description="Close the drawer",
        open_loop_horizon=16,
    )

    obs, info = env.reset()

    print(f"DRAWER_SLIDE = {DRAWER_SLIDE}")
    print(f"Success threshold = {DRAWER_SLIDE * 0.1:.4f}")
    print()

    # Monkey-patch env.step to capture actions sent to vec_env
    _orig_step = env.step
    _last_combined = [None]
    def _patched_step(residual_action):
        # Capture what _combine_actions produces
        import types
        residual_np = residual_action.detach().cpu().numpy() if isinstance(residual_action, torch.Tensor) else np.asarray(residual_action)
        base_action = env._get_base_actions()
        combined = env._combine_actions(base_action, residual_np)
        _last_combined[0] = (base_action.copy(), combined.copy())
        return _orig_step(residual_action)
    env.step = _patched_step

    # Track drawer_qpos over time
    for step in range(args.max_steps):
        action = torch.zeros(args.num_envs, 7, device=device)  # base only
        obs, reward, terminated, truncated, info = env.step(action)

        # Get drawer positions
        qpos_list = []
        for i in range(args.num_envs):
            qpos_list.append(vec_env._drawer_qpos[i] if hasattr(vec_env, '_drawer_qpos') else -1)

        if step % 50 == 0 or step < 10 or any(terminated.cpu().numpy()):
            frac_list = [1.0 - q / DRAWER_SLIDE for q in qpos_list]
            ba, comb = _last_combined[0] if _last_combined[0] else (None, None)
            if ba is not None:
                print(f"Step {step:4d} | reward={reward.mean().item():.4f} | "
                      f"qpos={[f'{q:.4f}' for q in qpos_list]} | "
                      f"closed_frac={[f'{f:.2f}' for f in frac_list]}")
                print(f"  base_action[0]={np.array2string(ba[0], precision=4, suppress_small=True)}")
                print(f"  combined[0]  ={np.array2string(comb[0], precision=4, suppress_small=True)}")
            else:
                print(f"Step {step:4d} | reward={reward.mean().item():.4f} | "
                      f"qpos={[f'{q:.4f}' for q in qpos_list]}")

        if terminated.any():
            print(f"  >>> SUCCESS at step {step}!")
            break

    # Final summary
    print()
    final_qpos = []
    final_frac = []
    for i in range(args.num_envs):
        q = vec_env._drawer_qpos[i] if hasattr(vec_env, '_drawer_qpos') else -1
        final_qpos.append(q)
        final_frac.append(1.0 - q / DRAWER_SLIDE)
    print(f"Final drawer_qpos: {[f'{q:.4f}' for q in final_qpos]}")
    print(f"Final closed_frac: {[f'{f:.2f}' for f in final_frac]}")
    print(f"Success threshold: qpos < {DRAWER_SLIDE * 0.1:.4f} (closed_frac > 0.90)")
    print(f"Min qpos: {min(final_qpos):.4f}, Max closed_frac: {max(final_frac):.2f}")


if __name__ == "__main__":
    main()
