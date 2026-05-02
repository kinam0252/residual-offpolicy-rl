"""
Async eval process for MuJoCo Close Drawer Residual TD3.

Polls checkpoint directory for eval_request_step*.pt files.
Evaluates per-drawer success rate and writes JSON results.
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

from resfit.rl_finetuning.config.residual_td3_mujoco_drawer import ResidualTD3MuJoCoDrawerConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_drawer import MuJoCoResidualWrapperDrawer
from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import MuJoCoVecEnvDrawer, ACTIVE_DRAWERS
from resfit.rl_finetuning.utils.normalization import ActionScaler

try:
    import wandb
except ImportError:
    wandb = None


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str):
    print(f"[{_ts()}] [drawer-eval] {msg}", flush=True)


def evaluate_drawer(
    env: MuJoCoResidualWrapperDrawer,
    agent: QAgent,
    num_episodes: int,
    device: torch.device,
    active_drawers: list[int],
    eval_step: int = 0,
) -> dict[str, float]:
    """Evaluate per-drawer success rate.

    Runs num_episodes rounds. Each round resets all envs (cycling drawers).
    Reports overall SR + per-drawer SR.
    """
    agent.eval()
    num_envs = env.vec_env.num_envs

    # Track per-drawer stats
    drawer_successes = {d: 0 for d in active_drawers}
    drawer_episodes = {d: 0 for d in active_drawers}
    total_return = 0.0
    total_steps = 0
    total_episodes = 0

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

        # Aggregate results
        for i in range(num_envs):
            d_idx = env.vec_env._envs[i]["active_drawer"]
            drawer_episodes[d_idx] = drawer_episodes.get(d_idx, 0) + 1
            if env_success[i]:
                drawer_successes[d_idx] = drawer_successes.get(d_idx, 0) + 1
            total_return += ep_returns[i]
            total_steps += env.vec_env.max_episode_steps  # approximate
            total_episodes += 1

    agent.train()

    # Compute metrics
    total_successes = sum(drawer_successes.values())
    metrics = {
        "eval/step": eval_step,
        "eval/success_rate": total_successes / max(1, total_episodes),
        "eval/mean_return": total_return / max(1, total_episodes),
        "eval/total_episodes": total_episodes,
    }
    for d in active_drawers:
        n_ep = drawer_episodes.get(d, 0)
        n_suc = drawer_successes.get(d, 0)
        metrics[f"eval/sr_D{d}"] = n_suc / max(1, n_ep)
        metrics[f"eval/episodes_D{d}"] = n_ep

    _log(f"  Overall SR: {metrics['eval/success_rate']:.2%} ({total_successes}/{total_episodes})")
    for d in active_drawers:
        _log(f"  D{d}: {metrics[f'eval/sr_D{d}']:.2%} ({drawer_successes[d]}/{drawer_episodes[d]})")

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Async eval for MuJoCo Drawer Residual TD3")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--poll_interval_sec", type=float, default=10.0)
    # Env
    p.add_argument("--active_drawers", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--contact_z_gate", action="store_true", default=True)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    # GR00T
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str, default="Close the drawer")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.004)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--action_scale", type=float, default=0.1)
    # ActionScaler
    p.add_argument("--use_action_scaler", action="store_true")
    p.add_argument("--action_scaler_min", type=float, nargs="+", default=None)
    p.add_argument("--action_scaler_max", type=float, nargs="+", default=None)
    # Agent
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    # Eval
    p.add_argument("--eval_num_episodes", type=int, default=10)
    # W&B
    p.add_argument("--wandb_mode", type=str, default="disabled")
    p.add_argument("--wandb_project", type=str, default="mujoco-drawer-residual-td3")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Starting drawer async eval")
    _log(f"  checkpoint_dir: {checkpoint_dir}")
    _log(f"  active_drawers: {args.active_drawers}")

    # ── Create eval env (one env per drawer for balanced eval) ──
    num_eval_envs = len(args.active_drawers)
    mujoco_env = MuJoCoVecEnvDrawer(
        num_envs=num_eval_envs,
        active_drawers=args.active_drawers,
        contact_z_gate=args.contact_z_gate,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        rl_img_size=84,
    )

    policy_device = args.groot_policy_device or args.device

    # Build ActionScaler
    _action_scaler = None
    if args.use_action_scaler and args.action_scaler_min and args.action_scaler_max:
        _action_scaler = ActionScaler(
            action_min=torch.tensor(args.action_scaler_min, dtype=torch.float32),
            action_max=torch.tensor(args.action_scaler_max, dtype=torch.float32),
            action_scale=args.action_scale,
        )
        _log(f"ActionScaler created (action_scale={args.action_scale})")

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
    _log("Eval environment ready.")

    # ── Create agent (state-only, same architecture as training) ──
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
    _log(f"QAgent created (state-only, action_dim={action_dim})")

    # ── W&B ──
    _wb = None
    if wandb is not None and args.wandb_mode != "disabled":
        try:
            _wb_kwargs = dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                reinit=True,
            )
            if args.wandb_run_id:
                _wb_kwargs["id"] = args.wandb_run_id
                _wb_kwargs["resume"] = "allow"
            wandb.init(**_wb_kwargs)
            wandb.define_metric("eval/*", step_metric="eval/step")
            _wb = wandb
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")

    # ── Polling loop ──
    _log("Entering polling loop...")
    processed_steps = set()
    best_success = 0.0

    while True:
        request_files = sorted(checkpoint_dir.glob("eval_request_step*.pt"))
        new_requests = [
            f for f in request_files
            if int(f.stem.split("step")[1]) not in processed_steps
        ]

        if not new_requests:
            time.sleep(args.poll_interval_sec)
            continue

        # Process latest request only (skip older ones)
        latest = new_requests[-1]
        step = int(latest.stem.split("step")[1])
        _log(f"Processing eval request: step={step}")

        try:
            ckpt = torch.load(str(latest), map_location=device, weights_only=False)
            agent.load_state_dict(ckpt["model"])
            _log(f"  Loaded weights from {latest.name}")

            metrics = evaluate_drawer(
                env=env,
                agent=agent,
                num_episodes=args.eval_num_episodes,
                device=device,
                active_drawers=args.active_drawers,
                eval_step=step,
            )

            # Save JSON result
            result_path = output_dir / f"eval_step{step}.json"
            with open(result_path, "w") as f:
                json.dump(metrics, f, indent=2)
            _log(f"  Results saved: {result_path}")

            # W&B logging
            if _wb is not None and _wb.run is not None:
                _wb.log(metrics)

            # Track best
            sr = metrics["eval/success_rate"]
            if sr > best_success:
                best_success = sr
                best_path = checkpoint_dir / "best.pt"
                torch.save(ckpt, best_path)
                _log(f"  New best SR={sr:.2%}, saved {best_path}")

        except Exception as e:
            _log(f"  ERROR: {e}")

        # Mark all up to this step as processed
        for f in new_requests:
            s = int(f.stem.split("step")[1])
            processed_steps.add(s)

        # Cleanup old request files
        for f in request_files:
            try:
                f.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
