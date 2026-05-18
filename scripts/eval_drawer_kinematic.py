#!/usr/bin/env python3
"""Kinematic D2-only eval: base policy vs residual RL.

Usage:
    python3 scripts/eval_drawer_kinematic.py --mode base
    python3 scripts/eval_drawer_kinematic.py --mode residual --ckpt outputs/.../best.pt
"""
from __future__ import annotations

import importlib
import os
import sys
import types as _types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DS_BUILD_OPS", "0")

if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []
    sys.modules["deepspeed"] = _ds

import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "residual"], required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="Residual checkpoint path")
    parser.add_argument("--groot_checkpoint", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000"))
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--drawer", type=int, default=2)
    parser.add_argument("--serial", action="store_true", help="Disable parallel envs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified

    print(f"[eval] Mode={args.mode}, drawer=D{args.drawer}, envs={args.num_envs}, episodes={args.episodes}")

    use_parallel = not args.serial
    vec_env = MuJoCoVecEnvDrawer(
        num_envs=args.num_envs,
        active_drawers=[args.drawer] * args.num_envs,
        max_episode_steps=args.max_steps,
        device=device,
        reward_type="delta",
        contact_z_gate=True,
        physics_drawer=False,  # KINEMATIC
        parallel_envs=use_parallel,
    )
    print(f"[eval] parallel_envs={use_parallel}")

    env = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="new_embodiment",
        policy_device=str(device),
        task_description="Close the drawer",
        open_loop_horizon=16,
    )

    agent = None
    if args.mode == "residual":
        assert args.ckpt, "--ckpt required for residual mode"
        print(f"[eval] Loading residual checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

        # Reconstruct QAgent matching the training config
        from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
        from resfit.rl_finetuning.config.residual_td3_mujoco_drawer import ResidualTD3MuJoCoDrawerConfig

        train_args = ckpt.get("args", {})
        cfg = ResidualTD3MuJoCoDrawerConfig()
        # Override from saved args
        cfg.agent.actor.action_scale = train_args.get("action_scale", 0.05)
        cfg.agent.actor.action_l2_reg_weight = train_args.get("action_l2_reg", 0.01)
        cfg.agent.actor.hidden_dim = train_args.get("actor_hidden_dim", 512)
        cfg.agent.critic.hidden_dim = train_args.get("critic_hidden_dim", 1024)

        lowdim_dim = env.observation_space["observation.state"].shape[1]  # 8
        object_state_dim = env.observation_space["observation.object_state"].shape[1]  # 5

        agent = QAgent(
            obs_shape=(3, 84, 84),  # dummy, state-only
            prop_shape=(lowdim_dim,),
            action_dim=7,
            rl_cameras=[],  # state-only
            cfg=cfg.agent,
            residual_actor=True,
            object_state_dim=object_state_dim,
        )
        agent.load_state_dict(ckpt["model"])
        agent.to(device)
        agent.eval()
        print(f"[eval] QAgent loaded (lowdim={lowdim_dim}, obj_state={object_state_dim})")
    else:
        print("[eval] Base-only mode (zero residual)")

    total_success = 0
    total_eps = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        ep_rewards = np.zeros(args.num_envs)
        dones = np.zeros(args.num_envs, dtype=bool)
        for step in range(args.max_steps):
            with torch.no_grad():
                if agent is not None:
                    action = agent.act(obs, eval_mode=True, stddev=0.0)
                else:
                    action = torch.zeros(args.num_envs, 7, device=device)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rewards += reward.cpu().numpy()
            dones |= terminated.cpu().numpy().astype(bool)

        successes = int(dones.sum())
        total_success += successes
        total_eps += args.num_envs
        print(f"  Episode {ep}: {successes}/{args.num_envs} success, return={ep_rewards.mean():.3f}")

    sr = total_success / total_eps * 100
    print(f"")
    print(f"[RESULT] Mode={args.mode}, D{args.drawer}, SR={sr:.1f}% ({total_success}/{total_eps})")


if __name__ == "__main__":
    main()
