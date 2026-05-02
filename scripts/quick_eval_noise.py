"""Quick eval: test stddev=0.0 vs stddev=0.05 on existing checkpoint."""
import sys, os, types, importlib
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/nas_main/.cache/huggingface")

if "deepspeed" not in sys.modules:
    _ds_mock = types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

import torch
import numpy as np

sys.path.insert(0, os.getcwd())

from resfit.rl_finetuning.scripts.eval_async_mujoco_drawer import evaluate_drawer
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler

CKPT_PATH = sys.argv[1]  # path to best.pt
STDDEV = float(sys.argv[2])  # 0.0 or 0.05
NUM_EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 20
GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[quick-eval] ckpt={CKPT_PATH} stddev={STDDEV} episodes={NUM_EPISODES}")

# Load checkpoint to get args
ckpt = torch.load(CKPT_PATH, map_location="cpu")
ckpt_args = ckpt.get("args", {})
action_scale = ckpt_args.get("action_scale", 0.05)
print(f"[quick-eval] action_scale={action_scale}")

# Create env
active_drawers = [2]
mujoco_env = MuJoCoVecEnvDrawer(
    num_envs=len(active_drawers),
    active_drawers=active_drawers,
    max_episode_steps=500,
    device=device,
    reward_type="delta",
)

# Load offline data for scaler
offline_dir = "outputs/offline_drawer_zgate_100k"
offline_data = torch.load(os.path.join(offline_dir, "offline_buffer.pt"), map_location="cpu")
action_scaler = ActionScaler(offline_data["actions"], scale_factor=action_scale)

# Load GR00T
from gr00t.model.policy import Gr00tPolicy
groot = Gr00tPolicy(
    model_path=GROOT_CKPT,
    embodiment_tag="new_embodiment",
    denoising_steps=4,
    device=device,
)

env = MuJoCoResidualWrapperDrawer(
    vec_env=mujoco_env,
    groot_policy=groot,
    action_horizon=16,
    open_loop_horizon=1,
    action_scaler=action_scaler,
    contact_z_gate=True,
    device=device,
)

# Create agent and load weights
obs_sample, _ = env.reset()
obs_shape = {}
for k, v in obs_sample.items():
    if isinstance(v, torch.Tensor):
        obs_shape[k] = v.shape[1:]
action_dim = 7

from dataclasses import dataclass
@dataclass
class AgentCfg:
    obs_shape: dict
    action_shape: tuple = (7,)
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 1024
    discount: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    encoder_lr: float = 1e-4
    critic_tau: float = 0.005
    critic_target_update_every_steps: int = 2
    actor_update_every_steps: int = 2
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 2.0
    critic_grad_clip_norm: float = 1.0
    actor_grad_clip_norm: float = 1.0
    action_scale: float = action_scale

cfg = AgentCfg(obs_shape=obs_shape)
agent = QAgent(cfg=cfg, device=device)
agent.load_state_dict(ckpt["model"])
agent.to(device)

# Run eval
metrics = evaluate_drawer(
    env=env,
    agent=agent,
    num_episodes=NUM_EPISODES,
    device=device,
    active_drawers=active_drawers,
    eval_step=0,
    eval_stddev=STDDEV,
)

print(f"\n[quick-eval] RESULT: stddev={STDDEV} SR={metrics['eval/success_rate']} return={metrics['eval/mean_return']:.3f}")
