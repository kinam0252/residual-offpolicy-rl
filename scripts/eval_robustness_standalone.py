#!/usr/bin/env python3
"""
Standalone robustness evaluation.

Loads a best.pt checkpoint, runs eval under clean and noisy conditions,
and saves results as JSON.

Usage:
    python scripts/eval_robustness_standalone.py \
        --task pnp \
        --checkpoint outputs/pnp_rl_747/checkpoints/best.pt \
        --output_dir outputs/robustness_eval/pnp_clean \
        --eval_num_episodes 3 \
        --noise_max 0.01 --dropout_prob 0.1
"""
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified
from resfit.rl_finetuning.configs.task_configs import get_task_config


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [robustness-eval] {msg}", flush=True)


def _compute_action_stats_from_offline(data_dir: str) -> dict:
    """Compute per-dim min/max for 4D pos+grip actions from offline npz."""
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    all_pg = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        ba = data["obs_base_action"].astype(np.float32)
        pg = np.concatenate([ba[:, :3], ba[:, 6:7]], axis=-1)
        all_pg.append(pg)
    all_pg = np.concatenate(all_pg, axis=0)
    return {
        "min": all_pg.min(axis=0).tolist(),
        "max": all_pg.max(axis=0).tolist(),
    }


def evaluate(env, agent, num_episodes, device):
    """Run evaluation: num_episodes rounds × all envs."""
    agent.eval()
    num_envs = env.vec_env.num_envs
    successes = 0
    total_return = 0.0
    total_steps = 0
    total_episodes = 0
    per_env_sr = [0] * num_envs

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_returns = [0.0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(env.vec_env.max_episode_steps):
            with torch.no_grad():
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
                per_env_sr[i] += 1
            total_return += ep_returns[i]
            total_episodes += 1

    agent.train()
    sr = successes / max(1, total_episodes)
    return {
        "success_rate": sr,
        "mean_return": total_return / max(1, total_episodes),
        "total_episodes": total_episodes,
        "successes": successes,
        "per_env_sr": [per_env_sr[i] / max(1, num_episodes) for i in range(num_envs)],
    }


def create_env(args, ckpt_args, task_cfg, obs_noise_max=0.0, obs_dropout_prob=0.0, action_scaler=None):
    """Create eval environment with optional noise/dropout, reusing eval_async logic."""
    from resfit.rl_finetuning.scripts.eval_async_unified import _create_eval_env

    num_envs = ckpt_args.get("eval_num_envs", 20)

    # Build a namespace that _create_eval_env expects
    eval_args = SimpleNamespace(
        task=args.task,
        max_episode_steps=ckpt_args.get("max_episode_steps", 500),
        reward_type=ckpt_args.get("reward_type", "dense"),
        device=args.device,
        groot_checkpoint=ckpt_args.get("groot_checkpoint", ""),
        scene_xml=ckpt_args.get("scene_xml"),
        episode_positions_file=ckpt_args.get("eval_positions_file") or ckpt_args.get("episode_positions_file"),
        cup_positions_file=ckpt_args.get("cup_positions_file"),
        success_threshold=ckpt_args.get("success_threshold"),
        use_calibrated_wrist=ckpt_args.get("use_calibrated_wrist", False),
        calib_path=ckpt_args.get("calib_path"),
        reward_config=ckpt_args.get("reward_config"),
    )

    mujoco_env, num_envs = _create_eval_env(eval_args, task_cfg, num_envs)

    policy_device = ckpt_args.get("groot_policy_device") or "cuda"
    env = MuJoCoResidualWrapperUnified(
        vec_env=mujoco_env,
        groot_checkpoint=ckpt_args["groot_checkpoint"],
        embodiment_tag=ckpt_args.get("groot_embodiment_tag", "NEW_EMBODIMENT"),
        policy_device=policy_device,
        task_description=ckpt_args.get("task_description") or task_cfg.task_description,
        open_loop_horizon=ckpt_args.get("open_loop_horizon", 16),
        residual_pos_scale=ckpt_args.get("residual_pos_scale", 0.02),
        residual_rot_scale=ckpt_args.get("residual_rot_scale", 0.05),
        residual_grip_scale=ckpt_args.get("residual_grip_scale", 0.004),
        ema_alpha=ckpt_args.get("ema_alpha", 0.0),
        torch_compile=False,
        chunk_sync=ckpt_args.get("chunk_sync", True),
        grip_min=task_cfg.grip_min,
        grip_max=task_cfg.grip_max,
        use_gripper_latch=task_cfg.use_gripper_latch,
        camera_keys=task_cfg.camera_keys,
        action_scaler=action_scaler,
        async_prefetch=False,
        obs_noise_max=obs_noise_max,
        obs_dropout_prob=obs_dropout_prob,
    )
    return env, num_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_num_episodes", type=int, default=3)
    parser.add_argument("--noise_max", type=float, default=0.01)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    _log(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    task_cfg = get_task_config(args.task)

    # Build ActionScaler from offline data
    action_scaler = None
    offline_dir = ckpt_args.get("offline_data_dir")
    if ckpt_args.get("use_action_scaler") and offline_dir:
        from resfit.rl_finetuning.utils.normalization import ActionScaler
        _log(f"Computing ActionScaler from {offline_dir}...")
        stats = _compute_action_stats_from_offline(offline_dir)
        action_scaler = ActionScaler(
            action_min=torch.tensor(stats["min"], dtype=torch.float32),
            action_max=torch.tensor(stats["max"], dtype=torch.float32),
            action_scale=ckpt_args.get("action_scale", 0.1),
            device="cpu",
            no_clamp=ckpt_args.get("no_action_clamp", True),
        )
        _log(f"ActionScaler ready (scale={ckpt_args.get('action_scale')})")

    # Create agent
    object_state_dim = task_cfg.object_state_dim
    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = ckpt_args.get("action_scale", 0.1)
    cfg.agent.actor.hidden_dim = ckpt_args.get("actor_hidden_dim", 256)
    cfg.agent.critic.hidden_dim = ckpt_args.get("critic_hidden_dim", 256)

    # ── Eval 1: Clean ──
    _log("=" * 50)
    _log("EVAL 1: Clean (no augmentation)")
    _log("=" * 50)

    env_clean, num_envs = create_env(
        args, ckpt_args, task_cfg,
        obs_noise_max=0.0, obs_dropout_prob=0.0,
        action_scaler=action_scaler,
    )

    lowdim_dim = env_clean.observation_space["observation.state"].shape[1]
    action_dim = env_clean.action_dim
    image_keys = task_cfg.rl_image_keys if ckpt_args.get("use_images") else []

    agent = QAgent(
        obs_shape=(3, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=False,
    )
    agent.load_state_dict(ckpt["model"])
    agent.to(device)

    t0 = time.time()
    results_clean = evaluate(env_clean, agent, args.eval_num_episodes, device)
    t_clean = time.time() - t0
    _log(f"Clean SR: {results_clean['success_rate']:.2%} ({results_clean['successes']}/{results_clean['total_episodes']}) in {t_clean:.1f}s")

    # Close clean env to free GPU memory
    del env_clean

    # ── Eval 2: Noisy ──
    _log("=" * 50)
    _log(f"EVAL 2: Noisy (noise={args.noise_max}, dropout={args.dropout_prob})")
    _log("=" * 50)

    env_noisy, _ = create_env(
        args, ckpt_args, task_cfg,
        obs_noise_max=args.noise_max,
        obs_dropout_prob=args.dropout_prob,
        action_scaler=action_scaler,
    )

    t0 = time.time()
    results_noisy = evaluate(env_noisy, agent, args.eval_num_episodes, device)
    t_noisy = time.time() - t0
    _log(f"Noisy SR: {results_noisy['success_rate']:.2%} ({results_noisy['successes']}/{results_noisy['total_episodes']}) in {t_noisy:.1f}s")

    del env_noisy

    # ── Save results ──
    results = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "eval_num_episodes": args.eval_num_episodes,
        "num_envs": num_envs,
        "total_episodes_per_condition": results_clean["total_episodes"],
        "noise_max": args.noise_max,
        "dropout_prob": args.dropout_prob,
        "clean": results_clean,
        "noisy": results_noisy,
        "sr_drop": results_clean["success_rate"] - results_noisy["success_rate"],
    }

    out_path = output_dir / "robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"Results saved to {out_path}")

    # Summary
    _log("=" * 50)
    _log(f"SUMMARY: {args.task}")
    _log(f"  Clean SR:  {results_clean['success_rate']:.2%}")
    _log(f"  Noisy SR:  {results_noisy['success_rate']:.2%}")
    _log(f"  SR Drop:   {results['sr_drop']:.2%}")
    _log("=" * 50)


if __name__ == "__main__":
    main()
