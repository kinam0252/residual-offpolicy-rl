
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

CKPT = os.environ['EVAL_CKPT']
DIFF = os.environ['EVAL_DIFF']
RCR_JSON = os.environ['EVAL_RCR']
OUT_DIR = os.environ['EVAL_OUT']
USE_CALIB = os.environ.get('EVAL_CALIB', '') != ''
CALIB_PATH = os.environ.get('EVAL_CALIB', '')
SCENE_XML = os.environ['EVAL_SCENE']
N_ENVS = 40
N_EPISODES = 200

rcr = json.loads(RCR_JSON)
rcr_parsed = {'dx': (rcr['dx'][0]/100, rcr['dx'][1]/100), 'dy': (rcr['dy'][0]/100, rcr['dy'][1]/100), 'yaw': tuple(rcr.get('yaw', [0,0]))}

env_kwargs = dict(
    num_envs=N_ENVS,
    cube_positions=[[0.45, -0.05, 0.02]] * N_ENVS,
    random_cube_range=rcr_parsed,
    scene_xml=SCENE_XML,
    max_episode_steps=300,
    reward_type='dense_clipped',
    device='cpu',
)
if USE_CALIB:
    env_kwargs['calib_path'] = CALIB_PATH
    env_kwargs['use_calibrated_wrist'] = True

print(f'=== Base Eval: {os.path.basename(CKPT)} / {DIFF} ===')
print(f'  {N_ENVS} envs, {N_EPISODES} episodes, random cube')
env = MuJoCoVecEnv(**env_kwargs)
wrapper = MuJoCoResidualWrapper(
    vec_env=env, groot_checkpoint=CKPT,
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

    done_mask = term | trunc | (ep_step >= 300)

    for i in range(N_ENVS):
        if info.get('success', [False]*N_ENVS)[i]:
            ep_success[i] = True

    n_done = done_mask.sum().item() if hasattr(done_mask, 'sum') else sum(done_mask)
    if n_done > 0:
        for i in range(N_ENVS):
            if done_mask[i] if hasattr(done_mask, '__getitem__') else done_mask:
                total_episodes += 1
                if ep_success[i]:
                    total_success += 1
                ep_success[i] = False
                ep_step[i] = 0
                if total_episodes >= N_EPISODES:
                    break

        if total_episodes < N_EPISODES:
            obs, _ = wrapper.reset()
            ep_step[:] = 0
            ep_success[:] = False

        elapsed = time.time() - t0
        rate = total_success / max(1,total_episodes) * 100
        print(f'  ep={total_episodes}/{N_EPISODES} success={total_success} rate={rate:.1f}% t={elapsed:.0f}s')

elapsed = time.time() - t0
rate = total_success / total_episodes * 100
print(f'\n=== FINAL: {DIFF} ===')
print(f'  Episodes: {total_episodes}')
print(f'  Success: {total_success} ({rate:.1f}%)')
print(f'  Time: {elapsed:.0f}s')

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, 'result.json'), 'w') as f:
    json.dump({'difficulty': DIFF, 'episodes': total_episodes, 'success': total_success, 'rate': rate, 'time': elapsed}, f, indent=2)
