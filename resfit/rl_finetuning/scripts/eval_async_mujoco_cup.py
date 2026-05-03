"""
Async eval process for MuJoCo Stand Cup Residual TD3.

Polls checkpoint_dir for eval_request_step*.pt files, loads weights,
runs eval episodes in its own MuJoCo env, writes JSON results.
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

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup as MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup as MuJoCoVecEnv

try:
    import wandb
except ImportError:
    wandb = None


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str):
    print(f"[{_ts()}] [async-eval-cup] {msg}", flush=True)


def evaluate(
    env: MuJoCoResidualWrapper,
    agent: QAgent,
    num_episodes: int,
    device: torch.device,
    image_keys: list[str],
    save_video: bool = False,
    video_dir: Path | None = None,
    eval_step: int = 0,
) -> dict[str, float]:
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
        ep_steps_arr = [0] * num_envs
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
                    ep_steps_arr[i] += 1
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
            total_steps += ep_steps_arr[i]
            total_episodes += 1

    agent.train()
    metrics = {
        "eval/success_rate": successes / max(1, total_episodes),
        "eval/mean_return": total_return / max(1, total_episodes),
        "eval/mean_steps": total_steps / max(1, total_episodes),
    }
    for i in range(num_envs):
        metrics[f"eval/sr_env{i}"] = per_env_sr[i] / max(1, num_episodes)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Async eval for MuJoCo Stand Cup Residual TD3")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--poll_interval_sec", type=float, default=10.0)
    # Env
    p.add_argument("--cup_positions_file", type=str, default=None)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--reward_type", type=str, default="dense")
    p.add_argument("--device", type=str, default="cuda")
    # GR00T
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str,
                   default="Pick up the cup and stand it upright.")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.004)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--action_scale", type=float, default=0.1)
    # ActionScaler (not used for cup yet, but kept for interface)
    p.add_argument("--use_action_scaler", action="store_true")
    p.add_argument("--action_scaler_min", type=float, nargs="+", default=None)
    p.add_argument("--action_scaler_max", type=float, nargs="+", default=None)
    p.add_argument("--no_action_clamp", action="store_true")
    # Agent
    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--critic_hidden_dim", type=int, default=256)
    p.add_argument("--asymmetric_critic", action="store_true")
    # Eval
    p.add_argument("--eval_num_episodes", type=int, default=1)
    p.add_argument("--max_eval_envs", type=int, default=20)
    p.add_argument("--save_video", action="store_true")
    # W&B
    p.add_argument("--wandb_mode", type=str, default="disabled")
    p.add_argument("--wandb_project", type=str, default="mujoco-franka-cup-residual-td3")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_run_id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Starting async eval process")
    _log(f"  checkpoint_dir: {checkpoint_dir}")
    _log(f"  output_dir: {output_dir}")

    # -- Load cup positions --
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import load_cup_positions
    cup_data = load_cup_positions(args.cup_positions_file)
    num_envs = min(len(cup_data), getattr(args, 'max_eval_envs', len(cup_data)))
    episode_ids = list(range(num_envs))
    _log(f"Eval envs: {num_envs} cup positions")

    _log("Creating eval environment...")
    mujoco_env = MuJoCoVecEnv(
        num_envs=num_envs,
        episode_ids=episode_ids,
        cup_positions_path=args.cup_positions_file,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
    )

    policy_device = args.groot_policy_device or args.device
    env = MuJoCoResidualWrapper(
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
    )
    _log("Eval environment ready.")

    # -- Create agent (state-only) --
    image_keys = []
    object_state_dim = env.observation_space["observation.object_state"].shape[1]  # 8
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
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

    # -- W&B --
    _wb = None
    if wandb is not None and args.wandb_mode != "disabled":
        try:
            _wb_kwargs = dict(
                project=args.wandb_project, entity=args.wandb_entity,
                mode=args.wandb_mode, reinit=True,
            )
            if args.wandb_run_id:
                _wb_kwargs["id"] = args.wandb_run_id
                _wb_kwargs["resume"] = "allow"
                _wb_kwargs["name"] = args.wandb_name
            else:
                _wb_kwargs["name"] = args.wandb_name or "async_eval_cup"
            wandb.init(**_wb_kwargs)
            wandb.define_metric("eval/*", step_metric="eval/step")
            _wb = wandb
            _log(f"W&B initialized (run={wandb.run.id})")
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")

    # -- Polling loop --
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

        latest = new_requests[-1]
        step = int(latest.stem.split("step")[1])
        _log(f"Found eval request: step={step}")

        try:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                agent.load_state_dict(ckpt["model"])
            else:
                agent.load_state_dict(ckpt)
            _log(f"  Loaded weights for step {step}")

            eval_start = time.time()
            metrics = evaluate(
                env=env, agent=agent,
                num_episodes=args.eval_num_episodes,
                device=device, image_keys=image_keys,
                eval_step=step,
            )
            eval_time = time.time() - eval_start

            sr = metrics["eval/success_rate"]
            _log(f"  step={step} SR={sr:.2%} return={metrics['eval/mean_return']:.2f} "
                 f"time={eval_time:.1f}s")

            if sr > best_success:
                best_success = sr
                metrics["eval/best_success_rate"] = best_success
                _log(f"  New best SR: {best_success:.2%}")

            metrics["eval/step"] = step
            metrics["eval/eval_time_sec"] = eval_time

            result_path = output_dir / f"eval_step{step}.json"
            with open(result_path, "w") as f:
                json.dump(metrics, f, indent=2)

            if _wb is not None and _wb.run is not None:
                _wb.log(metrics)

            for f in new_requests:
                s = int(f.stem.split("step")[1])
                processed_steps.add(s)
            for req_f in new_requests:
                try:
                    req_f.unlink()
                except Exception:
                    pass

        except Exception as e:
            _log(f"  ERROR evaluating step {step}: {e}")
            import traceback
            traceback.print_exc()
            processed_steps.add(step)

    if _wb is not None and _wb.run is not None:
        _wb.finish()
    env.close()


if __name__ == "__main__":
    main()
