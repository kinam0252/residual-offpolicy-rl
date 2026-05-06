#!/usr/bin/env python3
"""
Standalone evaluation: Best RL checkpoint vs Base policy (GR00T only).
Reports per-drawer (D2, D3, D4) success rates for comparison.

Usage:
  python scripts/eval_drawer_compare.py \
    --best_checkpoint outputs/drawer_td3_v5_20260501_192236/checkpoints/best.pt \
    --groot_checkpoint /path/to/groot/checkpoint-100000 \
    --offline_data_dir /path/to/offline_npz \
    --num_episodes 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resfit.rl_finetuning.config.residual_td3_mujoco_drawer import ResidualTD3MuJoCoDrawerConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import ACTIVE_DRAWERS, MuJoCoVecEnvDrawer


def _log(msg):
    print(f"[eval-compare] {msg}", flush=True)


def _compute_action_stats_from_offline(data_dir: str):
    """Compute action statistics from offline data for ActionScaler."""
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    all_pos_grip = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        ba = data["obs_base_action"].astype(np.float32)
        pos_grip = np.concatenate([ba[:, :3], ba[:, 6:7]], axis=-1)
        all_pos_grip.append(pos_grip)
    all_pos_grip = np.concatenate(all_pos_grip, axis=0)
    return {
        "min": all_pos_grip.min(axis=0).tolist(),
        "max": all_pos_grip.max(axis=0).tolist(),
    }


def evaluate_per_drawer(env, agent, num_episodes, active_drawers, mode_label="RL"):
    """Run evaluation and return per-drawer SR."""
    num_envs = env.vec_env.num_envs

    drawer_successes = {d: 0 for d in active_drawers}
    drawer_episodes = {d: 0 for d in active_drawers}

    for ep in range(num_episodes):
        obs, _ = env.reset()
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(env.vec_env.max_episode_steps):
            with torch.no_grad():
                if agent is not None:
                    action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                else:
                    # Base policy only: zero residual
                    action = torch.zeros(num_envs, env.action_dim, device=obs["observation.state"].device)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(num_envs):
                if not env_done[i]:
                    if done[i]:
                        env_done[i] = True
                        if terminated[i]:
                            env_success[i] = True

            if all(env_done):
                break

        for i in range(num_envs):
            d_idx = env.vec_env._envs[i]["active_drawer"]
            drawer_episodes[d_idx] = drawer_episodes.get(d_idx, 0) + 1
            if env_success[i]:
                drawer_successes[d_idx] = drawer_successes.get(d_idx, 0) + 1

    # Print results
    total_ep = sum(drawer_episodes.values())
    total_suc = sum(drawer_successes.values())
    overall_sr = total_suc / max(1, total_ep)
    _log(f"\n{'='*50}")
    _log(f"  [{mode_label}] Overall SR: {overall_sr:.2%} ({total_suc}/{total_ep})")
    _log(f"{'='*50}")
    for d in active_drawers:
        n_ep = drawer_episodes.get(d, 0)
        n_suc = drawer_successes.get(d, 0)
        sr = n_suc / max(1, n_ep)
        _log(f"  D{d}: {sr:.2%} ({n_suc}/{n_ep})")

    return {
        "overall_sr": overall_sr,
        "per_drawer": {d: drawer_successes[d] / max(1, drawer_episodes[d]) for d in active_drawers},
        "drawer_episodes": drawer_episodes,
        "drawer_successes": drawer_successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare best RL checkpoint vs base policy per-drawer")
    parser.add_argument("--best_checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--groot_checkpoint", type=str, required=True)
    parser.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    parser.add_argument("--task_description", type=str, default="Close the drawer")
    parser.add_argument("--open_loop_horizon", type=int, default=16)
    # Env
    parser.add_argument("--active_drawers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--contact_z_gate", action="store_true", default=True)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    # Residual scales (must match training)
    parser.add_argument("--residual_pos_scale", type=float, default=0.02)
    parser.add_argument("--residual_rot_scale", type=float, default=0.05)
    parser.add_argument("--residual_grip_scale", type=float, default=0.004)
    parser.add_argument("--ema_alpha", type=float, default=0.0)
    parser.add_argument("--action_scale", type=float, default=0.1)
    # ActionScaler
    parser.add_argument("--use_action_scaler", action="store_true", default=True)
    parser.add_argument("--offline_data_dir", type=str, default=None)
    # Agent architecture
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    # Eval
    parser.add_argument("--num_episodes", type=int, default=20, help="Episodes per eval round (each round uses all envs)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    active_drawers = args.active_drawers

    _log(f"Device: {device}")
    _log(f"Active drawers: {active_drawers}")
    _log(f"Episodes per round: {args.num_episodes}")
    _log(f"Total episodes: {args.num_episodes * len(active_drawers)}")

    # ── Create env (one env per drawer for balanced eval) ──
    num_eval_envs = len(active_drawers)
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=num_eval_envs,
        active_drawers=active_drawers,
        contact_z_gate=args.contact_z_gate,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        rl_img_size=84,
    )

    # ── ActionScaler ──
    _action_scaler = None
    if args.use_action_scaler and args.offline_data_dir:
        _log("Computing action stats from offline data...")
        stats = _compute_action_stats_from_offline(args.offline_data_dir)
        _action_scaler = ActionScaler(
            action_min=torch.tensor(stats["min"], dtype=torch.float32),
            action_max=torch.tensor(stats["max"], dtype=torch.float32),
            action_scale=args.action_scale,
            device="cpu",
        )
        _log(f"ActionScaler created (action_scale={args.action_scale})")

    # ── GR00T + Residual wrapper ──
    _log("Loading GR00T policy...")
    policy_device = args.device
    env = MuJoCoResidualWrapperDrawer(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=policy_device,
        task_description=args.task_description,
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=args.residual_pos_scale,
        residual_rot_scale=args.residual_rot_scale,
        residual_grip_scale=args.residual_grip_scale,
        ema_alpha=args.ema_alpha,
        action_scaler=_action_scaler,
    )
    _log("Environment ready.")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Evaluate BASE POLICY (zero residual)
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("PHASE 1: Evaluating BASE POLICY (GR00T only, zero residual)")
    _log("=" * 60)
    t0 = time.time()
    base_results = evaluate_per_drawer(env, agent=None, num_episodes=args.num_episodes,
                                       active_drawers=active_drawers, mode_label="BASE")
    _log(f"  Time: {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Load best checkpoint and evaluate RL POLICY
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("PHASE 2: Evaluating RL POLICY (best checkpoint)")
    _log("=" * 60)

    # Create agent
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    object_state_dim = env.observation_space["observation.object_state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoDrawerConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(3, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=[],
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=False,
    )

    # Load checkpoint
    ckpt_path = Path(args.best_checkpoint)
    _log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # best.pt from eval_async saves with key "model"
    state_key = "model" if "model" in ckpt else "agent_state_dict"
    agent.load_state_dict(ckpt[state_key])
    agent.to(device)
    agent.eval()
    _log(f"Checkpoint loaded (key='{state_key}', step={ckpt.get('global_step', '?')})")

    t0 = time.time()
    rl_results = evaluate_per_drawer(env, agent=agent, num_episodes=args.num_episodes,
                                     active_drawers=active_drawers, mode_label="RL")
    _log(f"  Time: {time.time() - t0:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    _log("\n" + "=" * 60)
    _log("COMPARISON SUMMARY")
    _log("=" * 60)
    _log(f"{'Drawer':<10} {'Base SR':<12} {'RL SR':<12} {'Delta':<12}")
    _log("-" * 46)
    for d in active_drawers:
        b_sr = base_results["per_drawer"][d]
        r_sr = rl_results["per_drawer"][d]
        delta = r_sr - b_sr
        sign = "+" if delta >= 0 else ""
        _log(f"  D{d:<7} {b_sr:<12.2%} {r_sr:<12.2%} {sign}{delta:.2%}")
    b_overall = base_results["overall_sr"]
    r_overall = rl_results["overall_sr"]
    d_overall = r_overall - b_overall
    sign = "+" if d_overall >= 0 else ""
    _log("-" * 46)
    _log(f"  {'Overall':<7} {b_overall:<12.2%} {r_overall:<12.2%} {sign}{d_overall:.2%}")
    _log("=" * 60)


if __name__ == "__main__":
    main()
