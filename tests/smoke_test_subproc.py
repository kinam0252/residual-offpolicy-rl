#!/usr/bin/env python3
"""Smoke test for SubprocVecEnvMixin across all 5 MuJoCo tasks.

Usage:
    python tests/smoke_test_subproc.py [task]
    
    task: lift, cup, pnp, stack, drawer (default: all)
    
Each test creates 4 envs with 2 workers, runs reset → 5 steps → reset_envs → close.
"""

import sys
import os
import time
import json
import traceback
import numpy as np
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Add Mujoco_Franka/src for utils
MUJOCO_FRANKA_SRC = str(ROOT.parent / "Mujoco_Franka" / "src")
if MUJOCO_FRANKA_SRC not in sys.path:
    sys.path.insert(0, MUJOCO_FRANKA_SRC)

NUM_ENVS = 4
NUM_WORKERS = 2
NUM_STEPS = 5
DEVICE = "cpu"


def test_lift():
    """Smoke test Lift SubprocVecEnv."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

    positions = [[0.45, -0.05, 0.02]] * NUM_ENVS

    env = MuJoCoVecEnv(
        num_envs=NUM_ENVS,
        cube_positions=positions,
        max_episode_steps=50,
        reward_type="dense",
        parallel_envs=True,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    return _run_smoke(env, "lift")


def test_cup():
    """Smoke test Cup SubprocVecEnv."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup

    positions_path = str(ROOT / "configs" / "cup_positions.json")

    env = MuJoCoVecEnvCup(
        num_envs=NUM_ENVS,
        episode_ids=list(range(NUM_ENVS)),
        cup_positions_path=positions_path,
        max_episode_steps=50,
        reward_type="dense",
        parallel_envs=True,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    return _run_smoke(env, "cup")


def test_pnp():
    """Smoke test PnP SubprocVecEnv."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

    positions_file = str(ROOT / "configs" / "pnp_66ep_positions.json")

    scene_xml = os.path.expanduser(
        "~/Repos/Intern/Mujoco_Franka/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
    )

    env = MuJoCoVecEnvPnP(
        num_envs=NUM_ENVS,
        episode_positions_file=positions_file,
        scene_xml=scene_xml,
        max_episode_steps=50,
        reward_type="dense_v3",
        parallel_envs=True,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    return _run_smoke(env, "pnp")


def test_stack():
    """Smoke test Stack SubprocVecEnv."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack

    positions_file = ROOT / "configs" / "stack_positions.json"
    with open(positions_file) as f:
        all_positions = json.load(f)
    positions = all_positions[:NUM_ENVS]

    white_positions = [p["white_cube_pos"] for p in positions]
    green_positions = [p["green_cube_pos"] for p in positions]

    env = MuJoCoVecEnvStack(
        num_envs=NUM_ENVS,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        max_episode_steps=50,
        reward_type="dense",
        parallel_envs=True,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    return _run_smoke(env, "stack")


def test_drawer():
    """Smoke test Drawer SubprocVecEnv."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer

    env = MuJoCoVecEnvDrawer(
        num_envs=NUM_ENVS,
        max_episode_steps=50,
        reward_type="dense",
        parallel_envs=True,
        num_workers=NUM_WORKERS,
        device=DEVICE,
    )
    return _run_smoke(env, "drawer")


def _run_smoke(env, task_name: str) -> bool:
    """Run standard smoke test sequence on an env."""
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {task_name.upper()}")
    print(f"  num_envs={env.num_envs}, parallel={getattr(env, '_parallel', False)}")
    print(f"{'='*60}")

    try:
        # 1. Reset
        print(f"[{task_name}] reset()...", end=" ", flush=True)
        t0 = time.time()
        obs, info = env.reset()
        dt = time.time() - t0
        print(f"OK ({dt:.2f}s)")
        
        _check_obs(obs, env.num_envs, task_name, "reset")

        # 2. Step with random actions
        for step_i in range(NUM_STEPS):
            actions = torch.randn(env.num_envs, 8, device=DEVICE)
            actions[:, :3] = actions[:, :3] * 0.01 + 0.45
            actions[:, 3:7] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
            actions[:, 7] = 1.0

            t0 = time.time()
            obs, rewards, terminated, truncated, info = env.step(actions, render_mode="none")
            dt = time.time() - t0

            assert rewards.shape == (env.num_envs,), f"rewards shape {rewards.shape}"
            assert terminated.shape == (env.num_envs,), f"terminated shape {terminated.shape}"
            assert truncated.shape == (env.num_envs,), f"truncated shape {truncated.shape}"

            if step_i == 0:
                print(f"[{task_name}] step {step_i}: OK ({dt:.2f}s) "
                      f"reward={rewards.mean():.4f}", flush=True)
        print(f"[{task_name}] {NUM_STEPS} steps: OK", flush=True)

        # 3. Reset specific envs
        print(f"[{task_name}] reset_envs([0,2])...", end=" ", flush=True)
        env.reset_envs([0, 2])
        print("OK")

        # 4. Step again after selective reset
        actions = torch.randn(env.num_envs, 8, device=DEVICE)
        actions[:, :3] = actions[:, :3] * 0.01 + 0.45
        actions[:, 3:7] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        actions[:, 7] = 1.0
        obs, rewards, terminated, truncated, info = env.step(actions, render_mode="none")
        print(f"[{task_name}] step after reset_envs: OK")

        # 5. get_groot_obs (requires qpos sync)
        print(f"[{task_name}] get_groot_obs(0)...", end=" ", flush=True)
        groot_obs = env.get_groot_obs(0)
        assert "video" in groot_obs, "groot_obs missing 'video'"
        assert "state" in groot_obs, "groot_obs missing 'state'"
        print("OK")

        # 6. Close
        print(f"[{task_name}] close()...", end=" ", flush=True)
        env.close()
        print("OK")

        print(f"\n  ✅ {task_name.upper()} PASSED\n")
        return True

    except Exception as e:
        print(f"\n  ❌ {task_name.upper()} FAILED: {e}")
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        return False


def _check_obs(obs: dict, num_envs: int, task_name: str, context: str):
    """Verify obs dict has expected keys and shapes."""
    assert "observation.state" in obs, f"[{task_name}:{context}] missing observation.state"
    state = obs["observation.state"]
    assert state.shape[0] == num_envs, f"[{task_name}:{context}] state batch dim {state.shape[0]} != {num_envs}"
    print(f"[{task_name}] obs keys: {sorted(obs.keys())[:5]}... state={state.shape}")


def main():
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["lift", "cup", "pnp", "stack", "drawer"]
    
    test_fns = {
        "lift": test_lift,
        "cup": test_cup,
        "pnp": test_pnp,
        "stack": test_stack,
        "drawer": test_drawer,
    }

    results = {}
    for task in tasks:
        if task not in test_fns:
            print(f"Unknown task: {task}")
            continue
        results[task] = test_fns[task]()

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for task, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {task:8s}: {status}")
    
    all_passed = all(results.values())
    print(f"\n  {'All tests passed!' if all_passed else 'Some tests FAILED!'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
