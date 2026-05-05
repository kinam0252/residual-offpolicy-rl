"""Check intra-chunk jitter in GR00T action predictions."""
import sys, os, importlib, types as _types
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _p(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _p
except: pass

_repo = str(Path(__file__).resolve().parents[3])
if _repo not in sys.path: sys.path.insert(0, _repo)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

print("Creating env (1 env)...", flush=True)
mujoco_env = MuJoCoVecEnv(
    num_envs=1,
    scene_xml=os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"),
    max_episode_steps=500,
    reward_type="dense",
    device="cuda:0",
)

print("Creating wrapper with GR00T...", flush=True)
env = MuJoCoResidualWrapper(
    vec_env=mujoco_env,
    groot_checkpoint=os.path.expanduser("~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000"),
    embodiment_tag="NEW_EMBODIMENT",
    policy_device="cuda:0",
    task_description="lift the cube",
    open_loop_horizon=16,
    action_horizon=16,
)

print("Reset...", flush=True)
obs, _ = env.reset()

# Run 3 chunks worth of steps (48 steps), logging every action
print("\n=== Intra-chunk action analysis ===", flush=True)
print(f"{'step':>4} {'chunk':>5} {'idx':>3}  {'pos_x':>8} {'pos_y':>8} {'pos_z':>8}  {'dx':>8} {'dy':>8} {'dz':>8}  {'grip':>6}", flush=True)

prev_pos = None
for step in range(48):
    chunk_num = step // 16
    idx_in_chunk = step % 16
    
    # Get base action (no residual)
    zero_residual = torch.zeros(1, 7, device="cuda:0")
    
    # Peek at cached chunk before stepping
    cached = env._cached_chunks[0]
    cidx = env._chunk_idx[0]
    
    if cached is not None and cidx < cached["eef_pos"].shape[0]:
        pos = cached["eef_pos"][cidx]
        grip = cached["gripper_width"][cidx, 0]
    else:
        pos = np.zeros(3)
        grip = 0.0
    
    dx, dy, dz = (0, 0, 0) if prev_pos is None else (pos - prev_pos)
    
    print(f"{step:4d} {chunk_num:5d} {idx_in_chunk:3d}  {pos[0]:8.5f} {pos[1]:8.5f} {pos[2]:8.5f}  {dx:8.5f} {dy:8.5f} {dz:8.5f}  {grip:6.3f}", flush=True)
    
    prev_pos = pos.copy()
    
    # Step with zero residual
    obs, reward, term, trunc, info = env.step(zero_residual)

# Also print deltas stats per chunk
print("\n=== Per-chunk delta stats ===", flush=True)
