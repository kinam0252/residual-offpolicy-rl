"""Scan cube positions to find GR00T success boundary in MuJoCo."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

SCENE_XML = os.path.expanduser('~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml')
CHECKPOINT = os.path.expanduser('~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000')
MAX_STEPS = 300
BASE = [0.45, -0.05, 0.02]

# Scan grid: dx from -8cm to +8cm, dy from -8cm to +8cm, step 2cm
offsets = []
for dx in np.arange(-0.08, 0.085, 0.02):
    for dy in np.arange(-0.08, 0.085, 0.02):
        offsets.append((round(dx, 3), round(dy, 3)))

print(f"Scanning {len(offsets)} positions...")
print(f"Base: {BASE}")

mujoco_env = MuJoCoVecEnv(
    num_envs=1,
    cube_positions=[BASE],
    scene_xml=SCENE_XML,
    device='cpu',
    max_episode_steps=MAX_STEPS,
    reward_type='dense_clipped',
    success_threshold=0.03,
)

env = MuJoCoResidualWrapper(
    vec_env=mujoco_env,
    groot_checkpoint=CHECKPOINT,
    embodiment_tag='NEW_EMBODIMENT',
    policy_device='cuda:0',
    task_description='lift the cube',
    open_loop_horizon=16,
)

results = []
for i, (dx, dy) in enumerate(offsets):
    pos = [BASE[0] + dx, BASE[1] + dy, BASE[2]]
    mujoco_env._envs[0]["cube_pos_init"] = np.array(pos, dtype=np.float64)
    
    obs, _ = env.reset()
    max_lift = 0.0
    grasped_any = False
    
    for step in range(MAX_STEPS):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(residual)
        
        qa = mujoco_env._envs[0]["cube_qposadr"]
        cube_z = mujoco_env._envs[0]["data"].qpos[qa + 2]
        init_z = mujoco_env._initial_cube_z[0]
        lift = cube_z - init_z
        max_lift = max(max_lift, lift)
        if mujoco_env._envs[0]["grasp_state"]["grasped"]:
            grasped_any = True
        
        done = terminated | truncated
        if done[0]:
            break
    
    success = terminated[0].item() if isinstance(terminated, torch.Tensor) else bool(terminated[0])
    lift_cm = max_lift * 100
    tag = "SUCC" if success else ("GRSP" if grasped_any else "FAIL")
    
    results.append({
        "dx": dx, "dy": dy,
        "pos": pos,
        "success": success,
        "grasped": grasped_any,
        "max_lift_cm": round(lift_cm, 2),
        "steps": step + 1,
    })
    
    if (i + 1) % 9 == 0 or i == len(offsets) - 1:
        n_succ = sum(1 for r in results if r["success"])
        n_grsp = sum(1 for r in results if r["grasped"])
        print(f"  [{i+1}/{len(offsets)}] dx={dx:+.3f} dy={dy:+.3f} → {tag} lift={lift_cm:.1f}cm | "
              f"total: {n_succ}/{len(results)} succ, {n_grsp} grasped")

# Summary
print(f"\n{'='*60}")
print(f"Total: {len(results)} positions scanned")
n_succ = sum(1 for r in results if r["success"])
n_grsp = sum(1 for r in results if r["grasped"])
print(f"Success: {n_succ}/{len(results)} = {n_succ/len(results):.0%}")
print(f"Grasped: {n_grsp}/{len(results)}")

# Print grid
print(f"\nSuccess grid (dx across, dy down):")
dxs = sorted(set(r["dx"] for r in results))
dys = sorted(set(r["dy"] for r in results))
header = "dy\\dx"
print(f"{header:>8s}", end="")
for dx in dxs:
    print(f" {dx*100:+5.0f}", end="")
print()
for dy in dys:
    print(f"{dy*100:+7.0f}cm", end="")
    for dx in dxs:
        r = next(r for r in results if r["dx"] == dx and r["dy"] == dy)
        if r["success"]:
            print(f"   ●", end=" ")
        elif r["grasped"]:
            print(f"   △", end=" ")
        else:
            print(f"   ·", end=" ")
    print()

# Save results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 
                        'outputs', 'position_scan', f'scan_{time.strftime("%Y%m%d_%H%M%S")}.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump({"base": BASE, "results": results, "n_success": n_succ, "n_total": len(results)}, f, indent=2)
print(f"\nSaved: {out_path}")

env.close()
