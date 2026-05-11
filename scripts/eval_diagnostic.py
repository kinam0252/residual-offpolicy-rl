"""
Diagnostic eval script for residual action analysis.

Runs evaluation episodes and saves per-step:
- base_action (8D): GR00T raw output
- residual_action (7D): RL actor raw output in [-1,1]
- scaled_residual: actual deltas applied (pos*scale, rot*scale, grip*scale)
- combined_action (8D): final action sent to env
- cup features: cup_pos, uprightness, grasped, tcp_cup_dist
- reward, done, success

Usage:
    python scripts/eval_diagnostic.py \
        --checkpoint outputs/cup_rl_297/checkpoints/best.pt \
        --num_envs 20 --num_episodes 1 \
        --output_dir outputs/cup_diagnostic_297
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types as _types
from datetime import datetime
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

import numpy as np
import torch

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified
from resfit.rl_finetuning.configs.task_configs import get_task_config


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print(f"[{_ts()}] [diag-eval] {msg}", flush=True)


def run_diagnostic_eval(env, agent, num_episodes, device, output_dir, base_only=False):
    """Run eval episodes, recording base/residual/combined actions per step."""
    agent.eval()
    num_envs = env.vec_env.num_envs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_episode_data = []

    for ep in range(num_episodes):
        _log(f"Episode {ep}/{num_episodes}")
        obs, _ = env.reset()

        # Per-env recorders
        recorders = [{
            "base_actions": [], "residual_actions": [], "scaled_residuals": [],
            "combined_actions": [], "obs_states": [], "rewards": [],
            "dones": [], "cup_features": [],
        } for _ in range(num_envs)]
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(env.vec_env.max_episode_steps):
            # 1. Get base action (before RL step)
            base_action = env._held_base_action.copy()  # (N, 8)

            # 2. Get residual action from RL agent
            with torch.no_grad():
                residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
            if base_only:
                residual_action = torch.zeros_like(residual_action)
            residual_np = residual_action.detach().cpu().numpy()  # (N, 7)

            # 3. Compute scaled residual (what actually gets added)
            scaled_res = np.zeros_like(residual_np)
            scaled_res[:, :3] = residual_np[:, :3] * env.residual_pos_scale
            scaled_res[:, 3:6] = residual_np[:, 3:6] * env.residual_rot_scale
            scaled_res[:, 6] = residual_np[:, 6] * env.residual_grip_scale

            # 4. Step env (combined = base + scaled_residual)
            obs, reward, terminated, truncated, info = env.step(residual_action)
            combined = info.get("scaled_action", None)
            if combined is not None:
                combined_np = combined.detach().cpu().numpy() if torch.is_tensor(combined) else np.asarray(combined)
            else:
                combined_np = np.zeros((num_envs, 8))

            done = terminated | truncated

            # 5. Extract cup features from vec_env
            cup_feats = np.zeros((num_envs, 5))  # tcp_cup_dist, grasped, uprightness, cup_vel, cup_z
            for i in range(num_envs):
                try:
                    e = env.vec_env._envs[i]
                    if hasattr(env.vec_env, '_get_cup_features'):
                        cf = env.vec_env._get_cup_features(i)
                        cup_feats[i] = [cf.get('tcp_cup_dist', 0), cf.get('grasped', 0),
                                        cf.get('uprightness', 0), cf.get('cup_vel', 0), cf.get('cup_z', 0)]
                    elif hasattr(e, 'get') and 'data' in e:
                        # Try direct access
                        data = e['data']
                        model = e['model']
                        ids = e['ids']
                        from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import _get_tcp_pose
                        tcp_pos, _ = _get_tcp_pose(model, data, ids["hand_id"])
                        cup_qpa = e.get("cup_qposadr")
                        if cup_qpa is not None:
                            cup_pos = data.qpos[cup_qpa:cup_qpa + 3]
                            cup_feats[i, 0] = float(np.linalg.norm(tcp_pos - cup_pos))
                            cup_feats[i, 4] = float(cup_pos[2])
                        gs = e.get("grasp_state", {})
                        cup_feats[i, 1] = float(gs.get("grasped", False))
                except Exception:
                    pass

            # 6. Record per-env
            obs_state = obs["observation.state"].detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy() if torch.is_tensor(reward) else np.asarray(reward)
            done_np = done.detach().cpu().numpy() if torch.is_tensor(done) else np.asarray(done)

            for i in range(num_envs):
                if not env_done[i]:
                    recorders[i]["base_actions"].append(base_action[i])
                    recorders[i]["residual_actions"].append(residual_np[i])
                    recorders[i]["scaled_residuals"].append(scaled_res[i])
                    recorders[i]["combined_actions"].append(combined_np[i])
                    recorders[i]["obs_states"].append(obs_state[i])
                    recorders[i]["rewards"].append(reward_np[i])
                    recorders[i]["dones"].append(done_np[i])
                    recorders[i]["cup_features"].append(cup_feats[i])

                    if done_np[i]:
                        env_done[i] = True
                        if terminated[i] if torch.is_tensor(terminated) else bool(terminated[i]):
                            env_success[i] = True

            if all(env_done):
                break

        # Save per-env episode data
        for i in range(num_envs):
            ep_data = {k: np.array(v, dtype=np.float32) for k, v in recorders[i].items()}
            ep_data["success"] = np.array([env_success[i]], dtype=bool)
            ep_data["num_steps"] = np.array([len(recorders[i]["rewards"])], dtype=np.int32)

            fname = output_dir / f"ep{ep}_env{i}.npz"
            np.savez_compressed(fname, **ep_data)
            status = "✓ SUCCESS" if env_success[i] else "✗ FAIL"
            n_steps = len(recorders[i]["rewards"])
            _log(f"  env{i}: {status} ({n_steps} steps) → {fname.name}")

            all_episode_data.append({
                "episode": ep, "env_id": i, "success": bool(env_success[i]),
                "num_steps": n_steps,
                "total_reward": float(sum(recorders[i]["rewards"])),
            })

    # Summary
    n_succ = sum(1 for e in all_episode_data if e["success"])
    n_total = len(all_episode_data)
    _log(f"\nOverall SR: {n_succ}/{n_total} = {n_succ/max(1,n_total):.1%}")

    # Aggregate residual stats
    all_res = []
    succ_res = []
    fail_res = []
    for ep in range(num_episodes):
        for i in range(num_envs):
            fname = output_dir / f"ep{ep}_env{i}.npz"
            d = np.load(fname)
            res = d["residual_actions"]
            all_res.append(res)
            if d["success"][0]:
                succ_res.append(res)
            else:
                fail_res.append(res)

    all_res = np.concatenate(all_res, axis=0)
    stats = {
        "overall_sr": n_succ / max(1, n_total),
        "num_episodes": n_total,
        "residual_stats": {
            "mean": all_res.mean(axis=0).tolist(),
            "std": all_res.std(axis=0).tolist(),
            "abs_mean": np.abs(all_res).mean(axis=0).tolist(),
            "min": all_res.min(axis=0).tolist(),
            "max": all_res.max(axis=0).tolist(),
        },
        "dim_labels": ["pos_x", "pos_y", "pos_z", "euler_x", "euler_y", "euler_z", "grip"],
        "episodes": all_episode_data,
    }

    if succ_res:
        sr = np.concatenate(succ_res, axis=0)
        stats["success_residual_stats"] = {
            "mean": sr.mean(axis=0).tolist(),
            "std": sr.std(axis=0).tolist(),
            "abs_mean": np.abs(sr).mean(axis=0).tolist(),
        }
    if fail_res:
        fr = np.concatenate(fail_res, axis=0)
        stats["failure_residual_stats"] = {
            "mean": fr.mean(axis=0).tolist(),
            "std": fr.std(axis=0).tolist(),
            "abs_mean": np.abs(fr).mean(axis=0).tolist(),
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    _log(f"Summary saved to {summary_path}")

    # Print analysis
    print("\n" + "=" * 70)
    print("RESIDUAL ACTION ANALYSIS")
    print("=" * 70)
    labels = stats["dim_labels"]
    print(f"\n{'Dim':<10} {'Mean':>8} {'Std':>8} {'|Mean|':>8} {'Min':>8} {'Max':>8}")
    print("-" * 54)
    for j, lbl in enumerate(labels):
        print(f"{lbl:<10} {stats['residual_stats']['mean'][j]:>8.4f} "
              f"{stats['residual_stats']['std'][j]:>8.4f} "
              f"{stats['residual_stats']['abs_mean'][j]:>8.4f} "
              f"{stats['residual_stats']['min'][j]:>8.4f} "
              f"{stats['residual_stats']['max'][j]:>8.4f}")

    if "success_residual_stats" in stats and "failure_residual_stats" in stats:
        print(f"\n{'Dim':<10} {'Succ |Mean|':>12} {'Fail |Mean|':>12} {'Ratio':>8}")
        print("-" * 44)
        for j, lbl in enumerate(labels):
            s = stats["success_residual_stats"]["abs_mean"][j]
            f_ = stats["failure_residual_stats"]["abs_mean"][j]
            ratio = s / max(f_, 1e-6)
            print(f"{lbl:<10} {s:>12.4f} {f_:>12.4f} {ratio:>8.2f}x")

    # Temporal analysis: residual magnitude over episode progress
    print(f"\nResidual magnitude over episode (10 bins):")
    all_mag_bins = []
    for ep in range(num_episodes):
        for i in range(num_envs):
            d = np.load(output_dir / f"ep{ep}_env{i}.npz")
            res = d["residual_actions"]
            mag = np.linalg.norm(res, axis=1)
            n = len(mag)
            if n < 10:
                continue
            bins = np.array_split(mag, 10)
            all_mag_bins.append([b.mean() for b in bins])
    if all_mag_bins:
        avg_bins = np.mean(all_mag_bins, axis=0)
        pct_labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
        for lbl, val in zip(pct_labels, avg_bins):
            bar = "█" * int(val * 30)
            print(f"  {lbl:>8}: {val:.4f} {bar}")

    return stats


def main():
    p = argparse.ArgumentParser(description="Diagnostic eval with action recording")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    p.add_argument("--num_envs", type=int, default=20)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="outputs/cup_diagnostic")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base_only", action="store_true",
                   help="Zero out residual to measure base policy SR")
    args = p.parse_args()

    # Load checkpoint and extract training args
    _log(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]

    task = train_args["task"]
    task_cfg = get_task_config(task)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _log(f"Task: {task}, SR: {ckpt.get('success_rate', 'N/A')}, Step: {ckpt.get('eval_step', 'N/A')}")
    _log(f"Action scale: {train_args.get('action_scale', 'N/A')}")

    # Create ActionScaler if training used one
    _action_scaler = None
    if train_args.get("use_action_scaler") or train_args.get("normalize_base_action"):
        scaler_min = train_args.get("action_scaler_min")
        scaler_max = train_args.get("action_scaler_max")
        if scaler_min is not None and scaler_max is not None:
            from resfit.rl_finetuning.utils.normalization import ActionScaler
            _action_scaler = ActionScaler(
                action_min=torch.tensor(scaler_min, dtype=torch.float32),
                action_max=torch.tensor(scaler_max, dtype=torch.float32),
                action_scale=train_args.get("action_scale", 0.1),
                device="cpu",
                no_clamp=train_args.get("no_action_clamp", True),
            )
            _log("ActionScaler loaded")

    # Create environment
    num_envs = args.num_envs
    _log(f"Creating {task} eval env ({num_envs} envs)...")

    mod = importlib.import_module(task_cfg.vec_env_module)
    VecEnvClass = getattr(mod, task_cfg.vec_env_class)

    if task == "cup":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import load_cup_positions
        cup_data = load_cup_positions(train_args.get("cup_positions_file"))
        num_envs = min(num_envs, len(cup_data))
        # Use eval positions if available
        eval_pos_file = train_args.get("eval_positions_file")
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            episode_ids=list(range(num_envs)),
            cup_positions_path=train_args.get("cup_positions_file"),
            max_episode_steps=train_args.get("max_episode_steps", 500),
            reward_type=train_args.get("reward_type", "sparse"),
            device=args.device,
        )
    elif task == "pnp":
        mujoco_env = VecEnvClass(
            num_envs=num_envs,
            cube_positions=[[0.45, -0.05, 0.02]] * num_envs,
            cube_yaw=[0.0] * num_envs,
            bowl_positions=[[0.42, 0.03, 0.0]] * num_envs,
            max_episode_steps=train_args.get("max_episode_steps", 500),
            reward_type=train_args.get("reward_type", "sparse"),
            device=args.device,
        )
    else:
        raise ValueError(f"Task {task} not yet supported in diagnostic eval")

    # Wrap with residual wrapper
    groot_ckpt = train_args.get("groot_checkpoint")
    if not groot_ckpt or not os.path.exists(groot_ckpt):
        home = os.path.expanduser("~")
        if task == "cup":
            groot_ckpt = f"{home}/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000"
        elif task == "pnp":
            groot_ckpt = f"{home}/DATA/INTERN/training/groot_pnp_sim_27ep/checkpoint-100000"
        _log(f"Using default groot checkpoint: {groot_ckpt}")

    env = MuJoCoResidualWrapperUnified(
        vec_env=mujoco_env,
        groot_checkpoint=groot_ckpt,
        embodiment_tag=train_args.get("groot_embodiment_tag", "NEW_EMBODIMENT"),
        policy_device=args.device,
        task_description=task_cfg.task_description,
        open_loop_horizon=train_args.get("open_loop_horizon", 16),
        residual_pos_scale=train_args.get("residual_pos_scale", 0.02),
        residual_rot_scale=train_args.get("residual_rot_scale", 0.05),
        residual_grip_scale=train_args.get("residual_grip_scale", 0.1),
        ema_alpha=train_args.get("ema_alpha", 0.0),
        chunk_sync=train_args.get("chunk_sync", True),
        grip_min=task_cfg.grip_min,
        grip_max=task_cfg.grip_max,
        use_gripper_latch=task_cfg.use_gripper_latch,
        camera_keys=task_cfg.camera_keys,
        action_scaler=_action_scaler,
        async_prefetch=False,
    )
    _log("Environment ready.")

    # Create agent
    object_state_dim = task_cfg.object_state_dim
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = train_args.get("action_scale", 0.1)
    cfg.agent.actor.hidden_dim = train_args.get("actor_hidden_dim", 256)
    cfg.agent.critic.hidden_dim = train_args.get("critic_hidden_dim", 256)

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

    # Load weights
    agent.load_state_dict(ckpt["model"])
    agent = agent.to(device)
    _log(f"Agent loaded ({sum(p.numel() for p in agent.parameters())} params)")

    # Run diagnostic eval
    stats = run_diagnostic_eval(
        env=env, agent=agent,
        num_episodes=args.num_episodes,
        device=device,
        output_dir=args.output_dir,
        base_only=args.base_only,
    )

    _log("Done.")


if __name__ == "__main__":
    main()
