"""Run one episode with zero residual and save video."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
os.environ['MUJOCO_GL'] = 'egl'

import cv2
import numpy as np
import torch
import subprocess
import time

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

SCENE_XML = os.path.expanduser('~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml')
CHECKPOINT = os.path.expanduser('~/DATA/INTERN/training/gr00t_groot_v2/checkpoint-30000')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'outputs', 'zero_residual_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'zero_residual_{time.strftime("%Y%m%d_%H%M%S")}.mp4')

MAX_STEPS = 300
FPS = 15

print('Creating MuJoCoVecEnv...')
mujoco_env = MuJoCoVecEnv(
    num_envs=1,
    cube_positions=[[0.45, -0.05, 0.02]],
    scene_xml=SCENE_XML,
    device='cpu',
    max_episode_steps=MAX_STEPS,
)

print('Loading GR00T policy...')
env = MuJoCoResidualWrapper(
    vec_env=mujoco_env,
    groot_checkpoint=CHECKPOINT,
    embodiment_tag='NEW_EMBODIMENT',
    policy_device='cuda:0',
    task_description='lift the cube',
    open_loop_horizon=16,
)

print('Resetting...')
obs, _ = env.reset()

frames_base = []
frames_wrist = []
frames_front = []

print(f'Running {MAX_STEPS} steps with zero residual...')
for step in range(MAX_STEPS):
    residual = torch.zeros((1, 7), dtype=torch.float32)
    obs, reward, terminated, truncated, info = env.step(residual)

    e = mujoco_env._envs[0]
    data = e['data']

    # Render cam_base + cam_wrist
    e['renderer_base'].update_scene(data, camera=e['cam_base_id'])
    fb = e['renderer_base'].render().copy()
    frames_base.append(fb)

    e['renderer_wrist'].update_scene(data, camera=e['cam_wrist_id'])
    fw = e['renderer_wrist'].render().copy()
    frames_wrist.append(fw)

    e['renderer_front'].update_scene(data, camera=e['front_cam_id'])
    ff = e['renderer_front'].render().copy()
    frames_front.append(ff)

    done = terminated | truncated
    if step % 50 == 0 or done[0]:
        from utils import get_tcp_pose
        tcp_pos, _ = get_tcp_pose(e['model'], data, e['ids']['hand_id'])
        cube_z = data.qpos[e['cube_qposadr'] + 2]
        init_z = mujoco_env._initial_cube_z[0]
        grasped = e['grasp_state']['grasped']
        ba = obs['observation.base_action'][0].numpy()
        print(f'  step={step:3d} eef=({tcp_pos[0]:.3f},{tcp_pos[1]:.3f},{tcp_pos[2]:.3f}) '
              f'grip={ba[7]:.3f} cube_lift={((cube_z-init_z)*100):.2f}cm grasped={grasped} r={reward[0].item():.1f}')

    if done[0]:
        reason = "SUCCESS (terminated)" if terminated[0] else "TRUNCATED (max steps)"
        print(f'  Episode ended at step {step}: {reason}')
        break

# Final result
cube_z = mujoco_env._envs[0]['data'].qpos[mujoco_env._envs[0]['cube_qposadr'] + 2]
init_z = mujoco_env._initial_cube_z[0]
lift_cm = max(0, (cube_z - init_z) * 100)
success = terminated[0].item() if isinstance(terminated, torch.Tensor) else terminated[0]
print(f'\nResult: lift={lift_cm:.2f}cm {"SUCCESS" if success else "FAIL"}')

# Write video: cam_base | cam_wrist side-by-side, front below
print(f'\nSaving video to {OUTPUT_PATH}...')
combined = []
for fb, fw, ff in zip(frames_base, frames_wrist, frames_front):
    top = np.concatenate([fb, fw], axis=1)  # (360, 1280, 3)
    # Resize front to match top width
    ff_resized = cv2.resize(ff, (top.shape[1], fb.shape[0]))
    frame = np.concatenate([top, ff_resized], axis=0)  # (720, 1280, 3)
    # Convert RGB→BGR for video
    combined.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

h, w = combined[0].shape[:2]
# Make even dimensions
w = w & ~1
h = h & ~1

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    ffmpeg_exe = 'ffmpeg'

tmp_path = OUTPUT_PATH + '.tmp.mp4'
proc = subprocess.Popen([
    ffmpeg_exe, '-y', '-loglevel', 'warning',
    '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
    '-r', str(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-crf', '23', '-preset', 'fast', tmp_path,
], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

for frame in combined:
    proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
proc.stdin.close()
proc.wait()

if proc.returncode == 0:
    os.rename(tmp_path, OUTPUT_PATH)
    print(f'Video saved: {OUTPUT_PATH} ({len(combined)} frames)')
else:
    stderr = proc.stderr.read()
    print(f'ffmpeg error: {stderr.decode()[:500]}')
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

env.close()
print('Done.')
