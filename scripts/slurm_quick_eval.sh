#!/bin/bash
#SBATCH --job-name=qeval
#SBATCH --output=/home/nas_main/kinamkim/slurms/qeval_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/qeval_%j.out
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:00:00

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/src:$PYTHONPATH

python3 << 'PYEOF'
import sys, os, json, time
sys.path.insert(0, '.')
sys.path.insert(0, '../Mujoco_Franka/src')
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)

import importlib, types as _types
if 'deepspeed' not in sys.modules:
    _ds = _types.ModuleType('deepspeed'); _ds.__version__='0.0.0'
    _ds.__spec__=importlib.machinery.ModuleSpec('deepspeed',None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType('deepspeed.zero'); _dz.__spec__=importlib.machinery.ModuleSpec('deepspeed.zero',None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    sys.modules['deepspeed']=_ds; sys.modules['deepspeed.zero']=_dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith('/') or repo_id.startswith('.')): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

import numpy as np
from utils import setup_egl
setup_egl()
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

# Teleop episode 1: cube position from gripper close moment
CUBE_POS = [0.504, -0.042, 0.02]

CHECKPOINTS = {
    '32ep': '/home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_32ep/checkpoint-300000',
    '66ep': '/home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000',
    '100ep': '/home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000',
}
CALIB_66 = '/home/nas_main/kinamkim/Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml'

N_ENVS = 20
N_EPISODES = 50
MAX_STEPS = 500

results = {}
for ckpt_name, ckpt_path in CHECKPOINTS.items():
    use_calib = '66ep' in ckpt_name or '100ep' in ckpt_name
    env_kwargs = dict(
        num_envs=N_ENVS,
        cube_positions=[CUBE_POS] * N_ENVS,
        max_episode_steps=MAX_STEPS,
        reward_type='dense_clipped',
        device='cpu',
    )
    if use_calib:
        env_kwargs['calib_path'] = CALIB_66
        env_kwargs['use_calibrated_wrist'] = True

    print(f'\n=== {ckpt_name} | cube={CUBE_POS} | {N_EPISODES} eps, {MAX_STEPS} steps ===', flush=True)
    env = MuJoCoVecEnv(**env_kwargs)
    wrapper = MuJoCoResidualWrapper(
        vec_env=env, groot_checkpoint=ckpt_path,
        embodiment_tag='NEW_EMBODIMENT', policy_device='cuda:0',
        task_description='lift the cube',
    )

    total_success = 0
    total_episodes = 0
    ep_success = np.zeros(N_ENVS, dtype=bool)
    ep_step = np.zeros(N_ENVS, dtype=int)

    obs, _ = wrapper.reset()
    t0 = time.time()

    while total_episodes < N_EPISODES:
        residual = torch.zeros((N_ENVS, 7), dtype=torch.float32)
        obs, reward, term, trunc, info = wrapper.step(residual)
        ep_step += 1

        for i in range(N_ENVS):
            succ_list = info.get('success', [False]*N_ENVS)
            if (succ_list[i] if hasattr(succ_list, '__getitem__') else succ_list):
                ep_success[i] = True

        done_mask = term | trunc | (ep_step >= MAX_STEPS)
        any_done = False
        for i in range(N_ENVS):
            d = done_mask[i] if hasattr(done_mask, '__getitem__') else bool(done_mask)
            if d and total_episodes < N_EPISODES:
                any_done = True
                total_episodes += 1
                if ep_success[i]:
                    total_success += 1
                ep_success[i] = False
                ep_step[i] = 0

        if any_done:
            obs, _ = wrapper.reset()
            ep_step[:] = 0
            ep_success[:] = False
            rate = total_success / total_episodes * 100
            print(f'  ep={total_episodes}/{N_EPISODES} success={total_success} rate={rate:.1f}% t={time.time()-t0:.0f}s', flush=True)

    rate = total_success / total_episodes * 100
    elapsed = time.time() - t0
    results[ckpt_name] = {'success': total_success, 'total': total_episodes, 'rate': rate, 'time': elapsed}
    print(f'  >>> FINAL {ckpt_name}: {total_success}/{total_episodes} = {rate:.1f}% in {elapsed:.0f}s', flush=True)

    del wrapper, env
    torch.cuda.empty_cache()

print(f'\n=== SUMMARY ===')
for k, v in results.items():
    print(f'  {k}: {v[rate]:.1f}% ({v[success]}/{v[total]})')

out_dir = '/home/nas_main/kinamkim/Repos/Intern/outputs/base_eval_quick'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'result.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {out_dir}/result.json')
PYEOF
