#!/usr/bin/env python3
"""
Rollout trajectory collector for base vs residual policy analysis.

Runs N episodes in MuJoCo without video saving, records per-step:
  actions(7D), base_actions(7D), residuals(7D), states, rewards, success.
Saves results as a single .npz file.

Usage:
  python rollout_trajectory_analysis.py \
      --task lift --mode residual --output_dir outputs/trajectory_analysis
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -- Deepspeed mock --
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
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate_repo_id(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)
    _hf_val.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


# ── Task-specific configurations ──

TASK_CONFIGS = {
    "lift": dict(
        groot_checkpoint=os.path.expanduser(
            "~/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000"),
        residual_checkpoint=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/"
            "gr00t_sim_100ep/easy_s1L2_s2/checkpoints/best.pt"),
        task_description="lift the cube",
        max_episode_steps=300,
        num_envs=30,
        action_scale=0.2,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        asymmetric_critic=True,
        object_state_dim=7,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.1,
        ema_alpha=0.0,
        open_loop_horizon=16,
        embodiment_tag="NEW_EMBODIMENT",
    ),
    "pnp": dict(
        groot_checkpoint=os.path.expanduser(
            "~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"),
        residual_checkpoint=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/outputs/pnp_train_v3/"
            "v3_normal_as0.1_l21.0_g0.95_s42/checkpoints/best.pt"),
        task_description="Pick up the red cube and place it onto the plate.",
        max_episode_steps=500,
        num_envs=15,
        action_scale=0.1,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        asymmetric_critic=True,
        object_state_dim=10,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.1,
        eval_positions_file=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/configs/pnp_eval_normal.json"),
        ema_alpha=0.0,
        open_loop_horizon=16,
        embodiment_tag="NEW_EMBODIMENT",
    ),
    "stack": dict(
        groot_checkpoint=os.path.expanduser(
            "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000"),
        residual_checkpoint=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/outputs/stack_td3_75k/"
            "hard_l25.0_g095/checkpoints/best.pt"),
        task_description="Pick up the white cube and stack it on the green cube.",
        max_episode_steps=1000,
        num_envs=20,
        action_scale=0.1,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        asymmetric_critic=True,
        object_state_dim=10,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.004,
        eval_positions_file=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/configs/stack_cube_positions.json"),
        ema_alpha=0.0,
        open_loop_horizon=16,
        embodiment_tag="NEW_EMBODIMENT",
    ),
}


def _log(msg: str):
    print(f"[trajectory] {msg}", flush=True)


def _create_env(task: str, cfg: dict, device: str):
    """Create task-specific MuJoCo env + residual wrapper."""
    import json as _json

    # Load fixed eval positions if specified
    eval_pos = None
    pos_file = cfg.get("eval_positions_file")
    if pos_file and os.path.exists(pos_file):
        with open(pos_file) as f:
            pos_list = _json.load(f)
        # Deduplicate by episode id
        seen = set()
        eval_pos = []
        for pp in pos_list:
            eid = pp.get("episode", len(eval_pos))
            if eid not in seen:
                seen.add(eid)
                eval_pos.append(pp)
        n = min(cfg["num_envs"], len(eval_pos))
        eval_pos = eval_pos[:n]
        cfg["num_envs"] = n
        _log(f"Loaded {n} fixed eval positions from {os.path.basename(pos_file)}")

    if task == "lift":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
        from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
        mujoco_env = MuJoCoVecEnv(
            num_envs=cfg["num_envs"],
            max_episode_steps=cfg["max_episode_steps"],
            device=device,
        )
    elif task == "pnp":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv
        from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP as MuJoCoResidualWrapper
        kw = dict(
            num_envs=cfg["num_envs"],
            max_episode_steps=cfg["max_episode_steps"],
            reward_type="dense",
            device=device,
        )
        if eval_pos:
            kw["cube_positions"] = [pp["cube_pos"] for pp in eval_pos]
            kw["bowl_positions"] = [pp["bowl_pos"] for pp in eval_pos]
            kw["cube_yaw_degs"] = [pp.get("cube_yaw_deg", 0.0) for pp in eval_pos]
        else:
            kw["random_cube_range"] = cfg.get("random_cube_range")
            kw["random_bowl_range"] = cfg.get("random_bowl_range")
        mujoco_env = MuJoCoVecEnv(**kw)
    elif task == "stack":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack as MuJoCoVecEnv
        from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack as MuJoCoResidualWrapper
        kw = dict(
            num_envs=cfg["num_envs"],
            max_episode_steps=cfg["max_episode_steps"],
            reward_type="dense",
            device=device,
        )
        if eval_pos:
            kw["white_cube_positions"] = [pp["white_cube_pos"] for pp in eval_pos]
            kw["green_cube_positions"] = [pp["green_cube_pos"] for pp in eval_pos]
        else:
            kw["random_cube_range"] = cfg.get("random_cube_range")
        mujoco_env = MuJoCoVecEnv(**kw)
    else:
        raise ValueError(f"Unknown task: {task}")

    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=cfg["groot_checkpoint"],
        embodiment_tag=cfg["embodiment_tag"],
        policy_device=device,
        task_description=cfg["task_description"],
        open_loop_horizon=cfg["open_loop_horizon"],
        residual_pos_scale=cfg["residual_pos_scale"],
        residual_rot_scale=cfg["residual_rot_scale"],
        residual_grip_scale=cfg["residual_grip_scale"],
        ema_alpha=cfg["ema_alpha"],
    )
    return env


def _create_agent(env, cfg: dict, device: torch.device):
    """Create QAgent matching the task config."""
    image_keys = ["observation.depth.front", "observation.depth.wrist"] \
        if cfg["asymmetric_critic"] else \
        ["observation.images.front", "observation.images.back", "observation.images.wrist"]
    img_c = 1 if cfg["asymmetric_critic"] else 3

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    agent_cfg = ResidualTD3MuJoCoConfig()
    agent_cfg.agent.actor.action_scale = cfg["action_scale"]
    agent_cfg.agent.actor.hidden_dim = cfg["actor_hidden_dim"]
    agent_cfg.agent.critic.hidden_dim = cfg["critic_hidden_dim"]

    agent = QAgent(
        obs_shape=(img_c, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=agent_cfg.agent,
        residual_actor=True,
        object_state_dim=cfg.get("object_state_dim", 0),
        asymmetric_critic=cfg["asymmetric_critic"],
    )
    return agent


def run_rollouts(env, agent, num_episodes: int, max_steps: int,
                 num_envs: int, device: torch.device, mode: str):
    """Run rollout episodes, collecting trajectory data.

    Returns dict of numpy arrays ready for np.savez.
    """
    agent.eval()

    all_actions = []       # (N_ep, T, 7)
    all_base_actions = []  # (N_ep, T, 7)
    all_residuals = []     # (N_ep, T, 7)
    all_states = []        # (N_ep, T, state_dim)
    all_rewards = []       # (N_ep, T)
    all_successes = []     # (N_ep,)
    all_lengths = []       # (N_ep,)

    total_episodes = 0
    total_success = 0

    for ep_round in range(num_episodes):
        obs, _ = env.reset()

        # Per-env buffers
        state_dim = obs["observation.state"].shape[1]
        ep_actions = [[] for _ in range(num_envs)]
        ep_base_actions = [[] for _ in range(num_envs)]
        ep_residuals = [[] for _ in range(num_envs)]
        ep_states = [[] for _ in range(num_envs)]
        ep_rewards = [[] for _ in range(num_envs)]
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(max_steps):
            # Record current state
            state_np = obs["observation.state"].detach().cpu().numpy()  # (N, state_dim)
            base_action_7d = obs["observation.base_action"].detach().cpu().numpy()  # (N, 7)

            with torch.no_grad():
                if mode == "base":
                    # Zero residual — only base policy acts
                    action = torch.zeros(num_envs, env.action_dim, device=device)
                else:
                    action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            residual_np = action.detach().cpu().numpy()  # (N, 7) raw residual in [-1, 1]

            obs, reward, terminated, truncated, info = env.step(action)

            # The combined action (base + scaled residual) is in info
            if "scaled_action" in info:
                combined_8d = info["scaled_action"].detach().cpu().numpy()  # (N, 8)
                # Convert to 7D: pos3 + euler3 + grip1
                from scipy.spatial.transform import Rotation
                combined_7d = np.zeros((num_envs, 7), dtype=np.float32)
                for i in range(num_envs):
                    combined_7d[i, :3] = combined_8d[i, :3]
                    quat = combined_8d[i, 3:7]
                    qn = np.linalg.norm(quat)
                    if qn > 1e-6:
                        combined_7d[i, 3:6] = Rotation.from_quat(quat / qn).as_euler("xyz")
                    combined_7d[i, 6] = combined_8d[i, 7]
            else:
                combined_7d = base_action_7d.copy()

            done = terminated | truncated
            for i in range(num_envs):
                if not env_done[i]:
                    ep_actions[i].append(combined_7d[i].copy())
                    ep_base_actions[i].append(base_action_7d[i].copy())
                    ep_residuals[i].append(residual_np[i].copy())
                    ep_states[i].append(state_np[i].copy())
                    ep_rewards[i].append(reward[i].item())
                    if done[i]:
                        env_done[i] = True
                        if terminated[i]:
                            env_success[i] = True

            if all(env_done):
                break

        # Collect per-env results
        for i in range(num_envs):
            T = len(ep_actions[i])
            if T == 0:
                continue
            all_actions.append(np.array(ep_actions[i]))
            all_base_actions.append(np.array(ep_base_actions[i]))
            all_residuals.append(np.array(ep_residuals[i]))
            all_states.append(np.array(ep_states[i]))
            all_rewards.append(np.array(ep_rewards[i]))
            all_successes.append(env_success[i])
            all_lengths.append(T)
            total_episodes += 1
            if env_success[i]:
                total_success += 1

        sr = sum(env_success) / num_envs
        _log(f"Round {ep_round}: SR={sr:.0%} ({sum(env_success)}/{num_envs})")

    _log(f"Total: SR={total_success}/{total_episodes} = {total_success/max(1,total_episodes):.1%}")

    return {
        "actions": np.array(all_actions, dtype=object),
        "base_actions": np.array(all_base_actions, dtype=object),
        "residuals": np.array(all_residuals, dtype=object),
        "states": np.array(all_states, dtype=object),
        "rewards": np.array(all_rewards, dtype=object),
        "successes": np.array(all_successes, dtype=bool),
        "episode_lengths": np.array(all_lengths, dtype=np.int32),
        "sr": total_success / max(1, total_episodes),
        "total_episodes": total_episodes,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Rollout trajectory collector")
    p.add_argument("--task", type=str, required=True, choices=["lift", "pnp", "stack"])
    p.add_argument("--mode", type=str, required=True, choices=["base", "residual"])
    p.add_argument("--output_dir", type=str, default="outputs/trajectory_analysis")
    p.add_argument("--num_episodes", type=int, default=1,
                   help="Number of eval rounds (each round uses all num_envs)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = TASK_CONFIGS[args.task].copy()
    _log(f"Task={args.task}  Mode={args.mode}  Episodes={args.num_episodes}×{cfg['num_envs']}envs")

    # Create env
    _log("Creating environment...")
    env = _create_env(args.task, cfg, args.device)
    _log("Environment ready.")

    # Create agent
    agent = _create_agent(env, cfg, device)
    agent.to(device)

    # Load residual checkpoint (for both modes — base mode just zeros the action)
    if args.mode == "residual":
        ckpt_path = cfg["residual_checkpoint"]
        _log(f"Loading residual checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        agent.load_state_dict(ckpt, strict=False)
        _log("Residual checkpoint loaded.")
    else:
        _log("Base-only mode — residual will be zeros.")

    # Run rollouts
    _log("Starting rollouts...")
    t0 = time.time()
    results = run_rollouts(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        max_steps=cfg["max_episode_steps"],
        num_envs=cfg["num_envs"],
        device=device,
        mode=args.mode,
    )
    elapsed = time.time() - t0
    _log(f"Rollouts complete in {elapsed:.1f}s")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_{args.mode}.npz"
    np.savez(
        str(out_path),
        actions=results["actions"],
        base_actions=results["base_actions"],
        residuals=results["residuals"],
        states=results["states"],
        rewards=results["rewards"],
        successes=results["successes"],
        episode_lengths=results["episode_lengths"],
        sr=results["sr"],
        total_episodes=results["total_episodes"],
        task=args.task,
        mode=args.mode,
    )
    _log(f"Saved: {out_path} ({results['total_episodes']} episodes, SR={results['sr']:.1%})")


if __name__ == "__main__":
    main()
