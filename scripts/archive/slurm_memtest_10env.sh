#!/bin/bash
#SBATCH --job-name=mj_memtest
#SBATCH --partition=core
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/mj_memtest_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/mj_memtest_%j.err

# ── Memory profiling: 10-env MuJoCo + GR00T residual RL ──

set -e
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

# Fake CUDA toolkit for transformers
FAKE_CUDA=/tmp/fake_cuda_$$
mkdir -p $FAKE_CUDA/bin $FAKE_CUDA/lib64 $FAKE_CUDA/include
cat > $FAKE_CUDA/bin/nvcc << 'NVCC'
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 12.8, V12.8.93"
NVCC
chmod +x $FAKE_CUDA/bin/nvcc
export CUDA_HOME=$FAKE_CUDA
export PATH=$FAKE_CUDA/bin:$PATH

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$HOME/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

python3 -u << 'PYEOF'
import sys, os, time, json
sys.path.insert(0, '.')
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np

print("=== Phase 1: Baseline GPU memory ===")
torch.cuda.empty_cache()
print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")

import psutil
proc = psutil.Process()
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n=== Phase 2: Create 10-env MuJoCoVecEnv ===")
t0 = time.time()
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

# Load perturbation table
perturb_path = 'configs/mujoco_cube_perturb_table.json'
with open(perturb_path) as f:
    perturb = json.load(f)
base_pos = perturb["base_pos"]
cube_positions = [[base_pos[0]+e["dx"], base_pos[1]+e["dy"], base_pos[2]] for e in perturb["envs"]]

SCENE_XML = os.path.expanduser('~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml')

env = MuJoCoVecEnv(
    num_envs=10,
    cube_positions=cube_positions,
    scene_xml=SCENE_XML,
    device='cuda:0',
    max_episode_steps=300,
    reward_type='dense_clipped',
    success_threshold=0.03,
)
print(f"  Created in {time.time()-t0:.1f}s")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n=== Phase 3: Reset + get obs ===")
obs, _ = env.reset()
print(f"  Obs keys: {sorted(obs.keys())}")
for k, v in sorted(obs.items()):
    mem_mb = v.element_size() * v.nelement() / 1e6
    print(f"    {k}: {v.shape} {v.dtype} ({mem_mb:.2f} MB)")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n=== Phase 4: Load GR00T + Create Residual Wrapper ===")
t0 = time.time()
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
CHECKPOINT = os.path.expanduser('~/DATA/INTERN/training/gr00t_groot_v2/checkpoint-30000')

wrapper = MuJoCoResidualWrapper(
    vec_env=env,
    groot_checkpoint=CHECKPOINT,
    embodiment_tag='NEW_EMBODIMENT',
    policy_device='cuda:0',
    task_description='lift the cube',
    open_loop_horizon=16,
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n=== Phase 5: Create QAgent (asymmetric, depth) ===")
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

cfg = ResidualTD3MuJoCoConfig()
cfg.agent.actor.hidden_dim = 512
cfg.agent.critic.hidden_dim = 1024

image_keys = ['observation.depth.front', 'observation.depth.wrist']
agent = QAgent(
    obs_shape=(1, 84, 84),
    prop_shape=(34,),
    action_dim=7,
    rl_cameras=image_keys,
    cfg=cfg.agent,
    residual_actor=True,
    object_state_dim=7,
    asymmetric_critic=True,
)

print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n=== Phase 6: Reset wrapper + step 10 times ===")
obs, _ = wrapper.reset()
print(f"  Obs keys after wrapper: {sorted(obs.keys())}")

t0 = time.time()
for step in range(10):
    residual = torch.zeros((10, 7), dtype=torch.float32, device='cuda:0')
    obs, reward, term, trunc, info = wrapper.step(residual)
step_time = (time.time() - t0) / 10
print(f"  Avg step time: {step_time:.3f}s ({1/step_time:.1f} steps/s)")

print(f"\n  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n=== Phase 7: Simulate training update (batch=256) ===")
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform

lowdim_keys = ['observation.state', 'observation.base_action', 'observation.object_state']
device = torch.device('cuda:0')

# Fill small buffer with dummy data
rb = TensorDictPrioritizedReplayBuffer(
    storage=LazyTensorStorage(max_size=2000, device='cpu'),
    alpha=0.0, beta=0.0, eps=1e-6,
    priority_key='_priority',
    transform=MultiStepTransform(n_steps=3, gamma=0.99),
    batch_size=256,
)

obs_keys = set(image_keys) | set(lowdim_keys)
print(f"  Filling buffer with 500 dummy transitions...")
for i in range(500):
    curr = {k: obs[k][0].cpu() for k in obs.keys() if k in obs_keys}
    nxt = {k: obs[k][0].cpu() for k in obs.keys() if k in obs_keys}
    td = TensorDict({
        "obs": TensorDict(curr, batch_size=[]),
        "next": TensorDict({
            "obs": TensorDict(nxt, batch_size=[]),
            "done": torch.tensor(False),
            "reward": torch.tensor(0.1),
        }, batch_size=[]),
        "action": torch.randn(7),
        "_priority": torch.tensor(10.0),
    }, batch_size=[]).unsqueeze(0)
    rb.add(td)

batch = rb.sample(256).to(device, non_blocking=True)
print(f"  Batch sampled: {batch.shape}")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Do one update
agent.train()
metrics = agent.update(batch, stddev=0.05, update_actor=True, bc_batch=None, ref_agent=agent)
print(f"  Update done! critic_loss={metrics.get('train/critic_loss', 'N/A')}")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"  CPU RAM: {proc.memory_info().rss / 1e9:.2f} GB")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
print(f"  CPU RAM used: {proc.memory_info().rss / 1e9:.2f} GB")
print(f"  Step throughput: {1/step_time:.1f} steps/s (10 envs)")
total_gpu = torch.cuda.get_device_properties(0).total_mem / 1e9
free_pct = (total_gpu - torch.cuda.memory_reserved()/1e9) / total_gpu * 100
print(f"  GPU total: {total_gpu:.1f} GB, free: {free_pct:.0f}%")
print(f"  VERDICT: {'OK — fits on 1 GPU' if torch.cuda.memory_reserved()/1e9 < total_gpu * 0.9 else 'WARNING — tight fit'}")

wrapper.close()
print("\nDone.")
PYEOF
