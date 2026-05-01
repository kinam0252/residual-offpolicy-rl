#!/usr/bin/env python3
"""
Object State Ablation Eval.

Loads a trained checkpoint and evaluates under 3 conditions:
  1. normal: GT object state
  2. zero:   object state = 0
  3. random: object state randomized each step

Usage:
  python scripts/eval_ablation_object_state.py \
    --checkpoint outputs/mujoco_td3/.../checkpoints/best.pt \
    --num_episodes 5
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import types as _types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Deepspeed mock ──
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

# Patch huggingface_hub
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched(repo_id, *a, **kw):
        try:
            return _orig_validate(repo_id, *a, **kw)
        except Exception:
            pass
    _hf_val.validate_repo_id = _patched
except Exception:
    pass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.utils.normalization import ActionScaler


def make_ablation_fn(mode: str, object_state_dim: int = 7):
    """Return an obs modification function for the given ablation mode."""
    if mode == "normal":
        return None
    elif mode == "zero":
        def _zero(obs):
            obs["observation.object_state"] = torch.zeros_like(
                obs["observation.object_state"]
            )
            return obs
        return _zero
    elif mode == "random":
        def _random(obs):
            obs["observation.object_state"] = torch.randn_like(
                obs["observation.object_state"]
            )
            return obs
        return _random
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")


def evaluate(env, agent, num_episodes, device, image_keys, ablation_fn=None):
    """Run eval episodes with optional object state ablation."""
    agent.eval()
    num_envs = env.vec_env.num_envs
    successes = 0
    total_return = 0.0
    total_episodes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_returns = [0.0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(env.vec_env.max_episode_steps):
            with torch.no_grad():
                if ablation_fn is not None:
                    obs = ablation_fn(obs)
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(num_envs):
                if not env_done[i]:
                    ep_returns[i] += reward[i].item()
                    if done[i]:
                        env_done[i] = True
                        if terminated[i]:
                            env_success[i] = True

            if all(env_done):
                break

        for i in range(num_envs):
            if env_success[i]:
                successes += 1
            total_return += ep_returns[i]
            total_episodes += 1

    agent.train()
    sr = successes / max(1, total_episodes)
    mean_ret = total_return / max(1, total_episodes)
    return sr, mean_ret, total_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or checkpoint file")
    parser.add_argument("--num_episodes", type=int, default=5, help="Eval episodes per env (total = num_episodes * num_envs)")
    parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel envs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Load checkpoint and extract training args
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    if isinstance(train_args, dict):
        train_args = argparse.Namespace(**train_args)

    print(f"[ablation] Checkpoint: {args.checkpoint}")
    print(f"[ablation] action_scale={train_args.action_scale}, use_action_scaler={train_args.use_action_scaler}")
    print(f"[ablation] num_envs={args.num_envs}, num_episodes={args.num_episodes}")

    # Build ActionScaler if needed
    _action_scaler = None
    if getattr(train_args, "use_action_scaler", False):
        from resfit.rl_finetuning.scripts.train_residual_td3_mujoco import _compute_action_stats_from_offline
        stats = _compute_action_stats_from_offline(train_args.offline_data_dir)
        _action_scaler = ActionScaler(
            action_min=torch.tensor(stats["min"], dtype=torch.float32),
            action_max=torch.tensor(stats["max"], dtype=torch.float32),
            action_scale=train_args.action_scale,
        )
        print(f"[ablation] ActionScaler: min={stats['min']}, max={stats['max']}")

    # Determine policy device
    policy_device = args.device

    # Parse random_cube_range and eval_perturb_table
    import json as _json
    _random_cube_range = getattr(train_args, "random_cube_range", None)
    if isinstance(_random_cube_range, str):
        _random_cube_range = _json.loads(_random_cube_range)
    if isinstance(_random_cube_range, dict):
        _random_cube_range = {
            "dx": (_random_cube_range["dx"][0] / 100.0, _random_cube_range["dx"][1] / 100.0),
            "dy": (_random_cube_range["dy"][0] / 100.0, _random_cube_range["dy"][1] / 100.0),
            "yaw": tuple(_random_cube_range.get("yaw", [0, 0])),
        }

    _eval_perturb = getattr(train_args, "eval_perturb_table", None)
    _max_ep_steps = getattr(train_args, "max_episode_steps", 500)
    _use_calib_wrist = getattr(train_args, "use_calibrated_wrist", False)

    # Auto-detect calibration path for 66ep checkpoint
    from pathlib import Path as _Path
    _calib_path = None
    ckpt_lower = train_args.groot_checkpoint.lower()
    if "66ep" in ckpt_lower or "100ep" in ckpt_lower:
        _calib_66ep = str(_Path(__file__).resolve().parents[1] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
        if _Path(_calib_66ep).exists():
            _calib_path = _calib_66ep
            _use_calib_wrist = True

    # Build cube positions from eval_perturb_table
    cube_positions = None
    num_envs = args.num_envs
    if _eval_perturb:
        _perturb_path = _Path(__file__).resolve().parents[1] / _eval_perturb
        if _perturb_path.exists():
            with open(_perturb_path) as f:
                ptable = _json.load(f)
            base_pos = ptable["base_pos"]
            envs_cfg = ptable["envs"]
            cube_positions = [
                [base_pos[0] + e["dx"], base_pos[1] + e["dy"], base_pos[2]]
                for e in envs_cfg
            ]
            num_envs = len(cube_positions)
            print(f"[ablation] Using eval_perturb_table: {num_envs} envs")

    if cube_positions is None:
        cube_positions = [[0.45, -0.05, 0.02]] * num_envs

    # Create MuJoCoVecEnv
    from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
    mujoco_env = MuJoCoVecEnv(
        num_envs=num_envs,
        cube_positions=cube_positions,
        random_cube_range=_random_cube_range,
        max_episode_steps=_max_ep_steps,
        success_threshold=getattr(train_args, "success_threshold", 0.03),
        reward_type=getattr(train_args, "reward_type", "dense_clipped"),
        device=args.device,
        rl_img_size=84,
        calib_path=_calib_path,
        use_calibrated_wrist=_use_calib_wrist,
    )

    # Create wrapper
    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=train_args.groot_checkpoint,
        embodiment_tag=getattr(train_args, "groot_embodiment_tag", "NEW_EMBODIMENT"),
        residual_pos_scale=train_args.residual_pos_scale,
        residual_rot_scale=train_args.residual_rot_scale,
        residual_grip_scale=train_args.residual_grip_scale,
        policy_device=policy_device,
        task_description=getattr(train_args, "task_description", "lift the cube"),
        open_loop_horizon=train_args.open_loop_horizon,
        ema_alpha=getattr(train_args, "ema_alpha", 0.0),
        action_scaler=_action_scaler,
    )
    print(f"[ablation] Environment ready: {num_envs} envs")

    # Create agent
    image_keys = ["observation.depth.front", "observation.depth.wrist"]
    object_state_dim = 7
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = train_args.action_scale
    cfg.agent.actor.hidden_dim = getattr(train_args, "actor_hidden_dim", 256)
    cfg.agent.critic.hidden_dim = getattr(train_args, "critic_hidden_dim", 256)

    agent = QAgent(
        obs_shape=(1, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=True,
    )
    agent.load_state_dict(ckpt["model"])
    agent = agent.to(args.device)
    print(f"[ablation] Agent loaded")

    # Run ablation
    modes = ["normal", "zero", "random"]
    results = {}

    for mode in modes:
        ablation_fn = make_ablation_fn(mode, object_state_dim)
        sr, mean_ret, n_eps = evaluate(
            env, agent, args.num_episodes, args.device, image_keys, ablation_fn
        )
        results[mode] = {"sr": sr, "mean_return": mean_ret, "n_episodes": n_eps}
        print(f"[ablation] {mode:>8}: SR={sr*100:.1f}% ({int(sr*n_eps)}/{n_eps})  return={mean_ret:.3f}")

    # Summary
    print("\n" + "=" * 50)
    print("Object State Ablation Summary")
    print("=" * 50)
    print(f"{'Mode':<10} {'SR':>8} {'Return':>10} {'Episodes':>10}")
    print("-" * 50)
    for mode in modes:
        r = results[mode]
        print(f"{mode:<10} {r['sr']*100:>7.1f}% {r['mean_return']:>10.3f} {r['n_episodes']:>10}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
