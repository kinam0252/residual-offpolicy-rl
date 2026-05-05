"""Evaluate GR00T base policy (zero residual) across 10 cube positions.

Runs each position for 1 episode, records success/fail, max lift height,
and saves per-env videos side-by-side.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import cv2
import numpy as np
import subprocess
import torch

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

SCENE_XML = os.path.expanduser('~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml')
CHECKPOINT = os.path.expanduser('~/DATA/INTERN/training/gr00t_groot_v2/checkpoint-30000')
PERTURB_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'configs', 'mujoco_cube_perturb_table.json')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'outputs', 'zero_residual_baseline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.04  # 4cm lift (grasped + lifted)
FPS = 15

# Load perturbation table
with open(PERTURB_TABLE) as f:
    perturb = json.load(f)

base_pos = perturb["base_pos"]
envs_config = perturb["envs"]
num_positions = len(envs_config)

print(f"=== Zero-Residual Baseline Evaluation ===")
print(f"Positions: {num_positions}")
print(f"Max steps: {MAX_STEPS}")
print(f"Success threshold: {SUCCESS_THRESHOLD*100:.0f}cm lift (grasped)")
print()

# We evaluate one position at a time (sequential) to reuse the GR00T model
# First, build all cube positions
cube_positions = []
for ec in envs_config:
    pos = [base_pos[0] + ec["dx"], base_pos[1] + ec["dy"], base_pos[2]]
    cube_positions.append(pos)
    print(f"  env {ec['env_id']:2d} ({ec['name']:>16s}): cube=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

print()
print("Creating environment (1 env, sequential eval)...")
mujoco_env = MuJoCoVecEnv(
    num_envs=1,
    cube_positions=[cube_positions[0]],
    scene_xml=SCENE_XML,
    device='cpu',
    max_episode_steps=MAX_STEPS,
    reward_type='dense_clipped',
    success_threshold=0.03,
)

print("Loading GR00T policy...")
env = MuJoCoResidualWrapper(
    vec_env=mujoco_env,
    groot_checkpoint=CHECKPOINT,
    embodiment_tag='NEW_EMBODIMENT',
    policy_device='cuda:0',
    task_description='lift the cube',
    open_loop_horizon=16,
)
print("Ready.\n")

results = []

for idx, ec in enumerate(envs_config):
    name = ec["name"]
    pos = cube_positions[idx]

    # Update cube position for this env
    e = mujoco_env._envs[0]
    e["cube_pos_init"] = np.array(pos, dtype=np.float64)

    # Reset
    obs, _ = env.reset()

    frames = []
    max_lift = 0.0
    max_lift_grasped = 0.0
    ep_reward = 0.0
    success = False
    success_step = -1

    for step in range(MAX_STEPS):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(residual)

        ep_reward += reward[0].item()

        # Track cube lift
        qa = e["cube_qposadr"]
        cube_z = e["data"].qpos[qa + 2]
        init_z = mujoco_env._initial_cube_z[0]
        lift = cube_z - init_z
        max_lift = max(max_lift, lift)
        if e["grasp_state"]["grasped"]:
            max_lift_grasped = max(max_lift_grasped, lift)

        # Record frame (front camera, small)
        if step % 2 == 0:  # every other frame to save memory
            e["renderer_front"].update_scene(e["data"], camera=e["front_cam_id"])
            frame = e["renderer_front"].render().copy()
            frame = cv2.resize(frame, (160, 120))
            # Add text overlay
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"{name}", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"s{step} h={lift*100:.1f}cm", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            if e["grasp_state"]["grasped"]:
                cv2.putText(frame_bgr, "GRASPED", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            frames.append(frame_bgr)

        done = terminated | truncated
        if done[0]:
            if terminated[0]:
                success = True
                success_step = step
            break

    status = "SUCCESS" if success else "FAIL"
    print(f"  [{idx+1:2d}/{num_positions}] {name:>16s}: {status:7s}  "
          f"max_lift={max_lift*100:.2f}cm  max_lift_grasped={max_lift_grasped*100:.2f}cm  "
          f"reward={ep_reward:.2f}  steps={step+1}"
          + (f"  (success@{success_step})" if success else ""))

    results.append({
        "env_id": ec["env_id"],
        "name": name,
        "cube_pos": pos,
        "success": success,
        "success_step": success_step,
        "max_lift_cm": round(max_lift * 100, 2),
        "max_lift_grasped_cm": round(max_lift_grasped * 100, 2),
        "total_reward": round(ep_reward, 2),
        "steps": step + 1,
    })

    # Save per-env video
    if frames:
        vid_path = os.path.join(OUTPUT_DIR, f"env{ec['env_id']:02d}_{name}_{status.lower()}.mp4")
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = 'ffmpeg'
        h, w = frames[0].shape[:2]
        tmp = vid_path + '.tmp.mp4'
        proc = subprocess.Popen([
            ffmpeg_exe, '-y', '-loglevel', 'warning',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
            '-r', str(FPS // 2), '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '25', '-preset', 'fast', tmp,
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0:
            os.rename(tmp, vid_path)
        elif os.path.exists(tmp):
            os.unlink(tmp)

# Summary
print("\n" + "=" * 70)
n_success = sum(1 for r in results if r["success"])
print(f"SUCCESS RATE: {n_success}/{num_positions} = {n_success/num_positions:.0%}")
print()
print(f"{'Name':>16s}  {'Result':>7s}  {'MaxLift':>8s}  {'Grasped':>8s}  {'Reward':>7s}  {'Steps':>5s}")
print("-" * 70)
for r in results:
    s = "SUCC" if r["success"] else "FAIL"
    print(f"{r['name']:>16s}  {s:>7s}  {r['max_lift_cm']:>7.2f}cm  {r['max_lift_grasped_cm']:>7.2f}cm  {r['total_reward']:>7.2f}  {r['steps']:>5d}")

# Save results JSON
result_path = os.path.join(OUTPUT_DIR, f"baseline_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
with open(result_path, 'w') as f:
    json.dump({
        "timestamp": time.strftime('%Y%m%d_%H%M%S'),
        "success_rate": n_success / num_positions,
        "n_success": n_success,
        "n_total": num_positions,
        "max_steps": MAX_STEPS,
        "success_threshold_m": SUCCESS_THRESHOLD,
        "checkpoint": CHECKPOINT,
        "results": results,
    }, f, indent=2)
print(f"\nResults saved: {result_path}")

env.close()
print("Done.")
