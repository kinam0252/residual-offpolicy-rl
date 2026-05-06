import sys, os, numpy as np, torch
torch.backends.cuda.enable_cudnn_sdp(False)
os.chdir('/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl')
sys.path.insert(0, '.')
from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup
import imageio

env = MuJoCoVecEnvCup(num_envs=1, episode_ids=[0], max_episode_steps=250,
    reward_type='dense', device='cuda:0',
    cup_positions_path='configs/cup_positions.json', parallel_envs=False)
wrapper = MuJoCoResidualWrapperCup(
    vec_env=env, groot_checkpoint=os.path.expanduser(
        '~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000'),
    embodiment_tag='NEW_EMBODIMENT', policy_device='cuda:0',
    task_description="Pick up the cup lying on its side and stand it upright",
    ema_alpha=0.3)

obs, info = wrapper.reset()
e = env._envs[0]
model, data = e['model'], e['data']
frames_base, frames_wrist = [], []
total_r = 0.0
zero_res = torch.zeros((1, wrapper.action_dim), device='cuda:0')

for step in range(250):
    obs, reward, terminated, truncated, info = wrapper.step(zero_res)
    total_r += reward.item()
    e['renderer_base'].update_scene(data, camera=e['cam_base_id'], scene_option=e.get('scene_option'))
    frames_base.append(e['renderer_base'].render().copy())
    e['renderer_wrist'].update_scene(data, camera=e['cam_wrist_id'])
    frames_wrist.append(e['renderer_wrist'].render().copy())
    if step % 50 == 0:
        print(f"step={step} rew={reward.item():.3f} total_r={total_r:.1f}")

combined = [np.concatenate([fb, fw], axis=1) for fb, fw in zip(frames_base, frames_wrist)]
out = 'outputs/cup_base_videos/correction30_sidebyside.mp4'
os.makedirs(os.path.dirname(out), exist_ok=True)
imageio.mimwrite(out, combined, fps=15)
print(f"\nSaved {out} ({len(combined)} frames, total_r={total_r:.1f})")
