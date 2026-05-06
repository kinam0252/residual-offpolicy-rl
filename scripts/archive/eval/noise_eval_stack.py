#!/usr/bin/env python3
"""Noise robustness eval for Stack Cube — adds Gaussian noise to object_state.

Usage:
  python scripts/noise_eval_stack.py <checkpoint> --noise_std 0.01
  python scripts/noise_eval_stack.py <checkpoint> --noise_std 0.0  # baseline
"""
import sys, os, types, importlib, json
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/nas_main/.cache/huggingface")

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
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig

DEFAULT_GROOT_CKPT = os.path.expanduser(
    "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000"
)
DEFAULT_POSITIONS = "configs/stack_cube_positions.json"


def add_object_state_noise(obs, noise_std, quat_noise_scale=0.1):
    """Add Gaussian noise to observation.object_state.

    object_state is 10D: white_pos(3) + white_quat(4) + green_pos(3)
    - Position dims (0-2, 7-9): noise_std directly
    - Quaternion dims (3-6): noise_std * quat_noise_scale, then renormalize
    """
    if noise_std <= 0:
        return obs

    obj = obs["observation.object_state"].clone()
    device = obj.device
    B = obj.shape[0]

    # Position noise: white_pos (0:3) and green_pos (7:10)
    obj[:, 0:3] += torch.randn(B, 3, device=device) * noise_std
    obj[:, 7:10] += torch.randn(B, 3, device=device) * noise_std

    # Quaternion noise: white_quat (3:7), smaller scale + renormalize
    quat_std = noise_std * quat_noise_scale
    obj[:, 3:7] += torch.randn(B, 4, device=device) * quat_std
    obj[:, 3:7] = obj[:, 3:7] / (obj[:, 3:7].norm(dim=1, keepdim=True) + 1e-8)

    obs["observation.object_state"] = obj
    return obs


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Noise robustness eval for Stack Cube")
    p.add_argument("checkpoint", type=str, help="Path to checkpoint .pt")
    p.add_argument("--noise_std", type=float, required=True, help="Gaussian noise std for positions (meters)")
    p.add_argument("--quat_noise_scale", type=float, default=0.1, help="Quaternion noise = noise_std * this")
    p.add_argument("--num_envs", type=int, default=20)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--groot_checkpoint", type=str, default=DEFAULT_GROOT_CKPT)
    p.add_argument("--positions_file", type=str, default=DEFAULT_POSITIONS)
    p.add_argument("--open_loop_horizon", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    action_scale = train_args.get("action_scale", 0.1)
    actor_hidden_dim = train_args.get("actor_hidden_dim", 256)
    critic_hidden_dim = train_args.get("critic_hidden_dim", 256)
    reward_type = train_args.get("reward_type", "sparse")
    no_action_clamp = train_args.get("no_action_clamp", True)

    print(f"[noise_eval] noise_std={args.noise_std} quat_scale={args.quat_noise_scale}", flush=True)
    print(f"[noise_eval] checkpoint={args.checkpoint}", flush=True)

    # Load positions
    with open(args.positions_file) as f:
        pos_list = json.load(f)
    seen = set()
    unique_pos = []
    for pp in pos_list:
        if pp["episode"] not in seen:
            seen.add(pp["episode"])
            unique_pos.append(pp)
    n = min(args.num_envs, len(unique_pos))
    white_positions = [pp["white_cube_pos"] for pp in unique_pos[:n]]
    green_positions = [pp["green_cube_pos"] for pp in unique_pos[:n]]

    calib_path = None
    c = str(Path(__file__).resolve().parents[2] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
    if Path(c).exists():
        calib_path = c

    mujoco_env = MuJoCoVecEnvStack(
        num_envs=n,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        calib_path=calib_path,
        max_episode_steps=args.max_episode_steps,
        reward_type=reward_type,
        device=args.device,
    )

    # ActionScaler (same stats as training)
    _stat_min = [0.4106, -0.0658, 0.0092, 0.0196]
    _stat_max = [0.4639, 0.2246, 0.1185, 0.0398]
    _action_scaler = ActionScaler(
        action_min=torch.tensor(_stat_min, dtype=torch.float32),
        action_max=torch.tensor(_stat_max, dtype=torch.float32),
        action_scale=action_scale,
        no_clamp=no_action_clamp,
    )

    print("[noise_eval] Loading GR00T...", flush=True)
    env = MuJoCoResidualWrapperStack(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=args.device,
        task_description="Pick up the white cube and stack it on the green cube.",
        open_loop_horizon=args.open_loop_horizon,
        action_scaler=_action_scaler,
    )

    # Agent
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    obj_dim = env.observation_space.get("observation.object_state", None)
    obj_dim = obj_dim.shape[1] if obj_dim is not None else 0

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = action_scale
    cfg.agent.actor.hidden_dim = actor_hidden_dim
    cfg.agent.critic.hidden_dim = critic_hidden_dim

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
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    print("[noise_eval] Weights loaded", flush=True)

    # Run eval
    obs, _ = env.reset()
    env_done = [False] * n
    env_success = [False] * n
    env_returns = [0.0] * n
    env_steps = [0] * n

    for step in range(args.max_episode_steps):
        # Inject noise into object_state
        noisy_obs = add_object_state_noise(obs, args.noise_std, args.quat_noise_scale)

        with torch.no_grad():
            action = agent.act(noisy_obs, eval_mode=True, stddev=0.0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        for i in range(n):
            if not env_done[i]:
                env_returns[i] += reward[i].item()
                env_steps[i] = step + 1
                if done[i]:
                    env_done[i] = True
                    if terminated[i]:
                        env_success[i] = True

        if all(env_done):
            break

    # Summary
    n_success = sum(env_success)
    mean_ret = np.mean(env_returns)
    print(f"\n[noise_eval] === RESULT (noise_std={args.noise_std}) ===", flush=True)
    print(f"[noise_eval] SR = {n_success}/{n} ({n_success/n:.0%})", flush=True)
    print(f"[noise_eval] Mean return = {mean_ret:.3f}", flush=True)
    for i in range(n):
        s = "✓" if env_success[i] else "✗"
        print(f"  env{i:02d}: {s} ret={env_returns[i]:.3f} steps={env_steps[i]}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
