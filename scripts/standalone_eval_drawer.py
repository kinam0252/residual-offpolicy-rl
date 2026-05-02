#!/usr/bin/env python3
"""Standalone eval for Drawer Residual TD3 checkpoints.

Usage:
  python scripts/standalone_eval_drawer.py <checkpoint_path> [--stddev 0.0] [--episodes 10] [--action_scale 0.05]

Example:
  python scripts/standalone_eval_drawer.py outputs/.../checkpoints/best.pt --episodes 20
  python scripts/standalone_eval_drawer.py outputs/.../checkpoints/best.pt --stddev 0.05 --episodes 10
"""
import argparse
import sys
import os
import types
import importlib

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/nas_main/.cache/huggingface")

# Stub deepspeed to avoid import errors
if "deepspeed" not in sys.modules:
    _ds = types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__, _ds.__file__ = [], __file__
    _dz = types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    sys.modules["deepspeed"] = _ds
    sys.modules["deepspeed.zero"] = _dz

import torch

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resfit.rl_finetuning.scripts.eval_async_mujoco_drawer import evaluate_drawer
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco_drawer import ResidualTD3MuJoCoDrawerConfig

# Default offline stats for ActionScaler (from drawer sim 33ep dataset)
DEFAULT_ACTION_MIN = [0.240120, -0.264707, 0.149373, 0.000083]
DEFAULT_ACTION_MAX = [0.524598, 0.178428, 0.346476, 0.000084]
DEFAULT_GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000"
)


def parse_args():
    p = argparse.ArgumentParser(description="Standalone drawer eval")
    p.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    p.add_argument("--stddev", type=float, default=0.0, help="Eval noise stddev (default: 0.0)")
    p.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    p.add_argument("--action_scale", type=float, default=0.05, help="Residual action scale")
    p.add_argument("--open_loop_horizon", type=int, default=16, help="GR00T chunk horizon")
    p.add_argument("--active_drawers", type=int, nargs="+", default=[2], help="Active drawer IDs")
    p.add_argument("--max_episode_steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--groot_checkpoint", type=str, default=DEFAULT_GROOT_CKPT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[eval] ckpt={args.checkpoint}", flush=True)
    print(f"[eval] stddev={args.stddev} episodes={args.episodes} device={device}", flush=True)
    print(f"[eval] action_scale={args.action_scale} horizon={args.open_loop_horizon}", flush=True)

    # --- Environment ---
    print("[eval] Creating env...", flush=True)
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        active_drawers=args.active_drawers,
        max_episode_steps=args.max_episode_steps,
        device=device,
        reward_type="delta",
        contact_z_gate=True,
    )

    scaler = ActionScaler(
        action_min=torch.tensor(DEFAULT_ACTION_MIN, dtype=torch.float32),
        action_max=torch.tensor(DEFAULT_ACTION_MAX, dtype=torch.float32),
        action_scale=args.action_scale,
    )

    print("[eval] Loading GR00T...", flush=True)
    env = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="new_embodiment",
        policy_device=str(device),
        task_description="Close the drawer",
        open_loop_horizon=args.open_loop_horizon,
        action_scaler=scaler,
    )

    # --- Agent ---
    print("[eval] Creating agent...", flush=True)
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    obj_dim = env.observation_space["observation.object_state"].shape[1]
    cfg = ResidualTD3MuJoCoDrawerConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(3, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=7,
        rl_cameras=[],
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=obj_dim,
        asymmetric_critic=False,
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["model"])
    print("[eval] Weights loaded, starting eval...", flush=True)

    # --- Run eval ---
    stddevs = [0.0, args.stddev] if args.stddev > 0 else [0.0]
    for sd in stddevs:
        metrics = evaluate_drawer(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            device=device,
            active_drawers=args.active_drawers,
            eval_step=0,
            eval_stddev=sd,
        )
        sr = metrics["eval/success_rate"]
        ret = metrics["eval/mean_return"]
        print(f"[eval] stddev={sd:.2f} → SR={sr:.0%} return={ret:.3f}", flush=True)

    print("[eval] Done.", flush=True)


if __name__ == "__main__":
    main()
