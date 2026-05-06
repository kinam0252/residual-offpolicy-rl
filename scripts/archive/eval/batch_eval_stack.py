#!/usr/bin/env python3
"""
Batch eval + cleanup for Stack experiments with dead eval subprocess.
Evaluates all eval_request_step*.pt checkpoints, keeps best.pt, cleans up.

Usage:
    python scripts/batch_eval_stack.py --exp_dirs stk_on2 stk_on3 stk_on4
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
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

    def _patched(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)

    _hf_val.validate_repo_id = _patched
except Exception:
    pass

_repo_root = str(Path(__file__).resolve().parents[1])
for p in [_repo_root, str(Path(_repo_root) / "resfit")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from resfit.rl_finetuning.off_policy.config import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnv


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate(env, agent, num_episodes, device):
    """Run eval episodes and return success rate."""
    n_envs = env.num_envs
    total_successes = 0
    total_episodes = 0

    for ep in range(num_episodes):
        obs_dict = env.reset()
        done = np.zeros(n_envs, dtype=bool)
        ep_success = np.zeros(n_envs, dtype=bool)

        for t in range(env.max_episode_steps):
            with torch.no_grad():
                state = torch.tensor(
                    obs_dict["observation.state"], dtype=torch.float32, device=device
                )
                obj_state = torch.tensor(
                    obs_dict["observation.object_state"],
                    dtype=torch.float32,
                    device=device,
                )
                action = agent.get_eval_action(
                    image=None,
                    proprioception=state,
                    object_state=obj_state,
                )
            obs_dict, rewards, dones, infos = env.step(action.cpu().numpy())
            for i in range(n_envs):
                if not done[i] and dones[i]:
                    done[i] = True
                    if infos.get("success", [False] * n_envs)[i]:
                        ep_success[i] = True

            if done.all():
                break

        total_successes += ep_success.sum()
        total_episodes += n_envs

    return total_successes / max(total_episodes, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dirs", nargs="+", required=True, help="Experiment names under outputs/stack_sweep/"
    )
    parser.add_argument(
        "--groot_checkpoint",
        default=os.path.expanduser(
            "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000"
        ),
    )
    parser.add_argument("--eval_positions_file", default="configs/stack_eval_20.json")
    parser.add_argument("--eval_num_envs", type=int, default=20)
    parser.add_argument("--eval_num_episodes", type=int, default=1)
    parser.add_argument("--sweep_dir", default="outputs/stack_sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")
    _log(f"Experiments: {args.exp_dirs}")

    # Config per experiment
    ACTION_SCALES = {
        "stk_on2": 0.1,
        "stk_on3": 0.05,
        "stk_on4": 0.05,
        # Core experiments
        "stk_1": 0.1,
        "stk_2": 0.1,
        "stk_3": 0.1,
        "stk_4": 0.1,
        "stk_5": 0.05,
        "stk_6": 0.05,
        "stk_7": 0.05,
        "stk_8": 0.05,
    }

    # Load eval positions
    with open(args.eval_positions_file) as f:
        pos_list = json.load(f)
    seen = set()
    unique_pos = []
    for pp in pos_list:
        ep = pp.get("episode", len(seen))
        if ep not in seen:
            seen.add(ep)
            unique_pos.append(pp)
    n = min(args.eval_num_envs, len(unique_pos))
    white_positions = [pp["white_cube_pos"] for pp in unique_pos[:n]]
    green_positions = [pp["green_cube_pos"] for pp in unique_pos[:n]]
    _log(f"Loaded {n} eval positions from {args.eval_positions_file}")

    # Auto-detect calib
    calib_path = None
    for c in [
        Path(_repo_root).parent / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml",
        Path.home() / "Repos/Intern/Mujoco_Franka/config/camera_info_66ep.yaml",
    ]:
        if c.exists():
            calib_path = str(c)
            break
    _log(f"Calib: {calib_path}")

    # Create env ONCE (expensive - loads GR00T)
    _log("Creating eval environment (loading GR00T)...")
    mujoco_env = MuJoCoVecEnv(
        num_envs=n,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        calib_path=calib_path,
        max_episode_steps=1000,
        reward_type="dense",
        device="cuda",
    )
    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device="cuda",
        task_description="pick up the white cube and place it on top of the green cube",
        open_loop_horizon=16,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.004,
        ema_alpha=0.0,
    )
    _log("Eval environment ready.")

    # Get dims
    object_state_dim = env.observation_space["observation.object_state"].shape[1]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    # Process each experiment
    for exp_name in args.exp_dirs:
        exp_path = Path(args.sweep_dir) / exp_name
        ckpt_dir = exp_path / "checkpoints"
        results_dir = exp_path / "async_eval_results"

        _log(f"\n{'='*60}")
        _log(f"Processing: {exp_name}")
        _log(f"{'='*60}")

        if not ckpt_dir.exists():
            _log(f"  SKIP: {ckpt_dir} does not exist")
            continue

        # Find all eval_request files
        request_files = sorted(ckpt_dir.glob("eval_request_step*.pt"))
        if not request_files:
            _log(f"  No eval_request files found. Skipping.")
            continue

        _log(f"  Found {len(request_files)} checkpoints to evaluate")

        # Get action_scale for this experiment
        action_scale = ACTION_SCALES.get(exp_name, 0.1)
        _log(f"  action_scale={action_scale}")

        # Create agent
        cfg = ResidualTD3MuJoCoConfig()
        cfg.agent.actor.action_scale = action_scale
        cfg.agent.actor.hidden_dim = 256
        cfg.agent.critic.hidden_dim = 256

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

        best_sr = 0.0
        best_step = -1
        best_ckpt_path = None
        results = []

        for req_file in request_files:
            step = int(req_file.stem.split("step")[1])
            _log(f"  Evaluating step={step}...")

            try:
                ckpt = torch.load(req_file, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    agent.load_state_dict(ckpt["model"])
                else:
                    agent.load_state_dict(ckpt)

                sr = evaluate(env, agent, args.eval_num_episodes, device)
                _log(f"    SR={sr:.1%}")

                results.append({"step": step, "success_rate": float(sr)})

                if sr > best_sr:
                    best_sr = sr
                    best_step = step
                    best_ckpt_path = req_file
                    _log(f"    >>> New best! SR={sr:.1%} at step={step}")

            except Exception as e:
                _log(f"    ERROR: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Save results JSON
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "batch_eval_results.json", "w") as f:
            json.dump(
                {
                    "experiment": exp_name,
                    "action_scale": action_scale,
                    "results": results,
                    "best_step": best_step,
                    "best_success_rate": best_sr,
                },
                f,
                indent=2,
            )

        _log(f"\n  === SUMMARY for {exp_name} ===")
        _log(f"  Best SR: {best_sr:.1%} at step {best_step}")
        for r in results:
            marker = " <<<" if r["step"] == best_step else ""
            _log(f"    step={r['step']:>6d}  SR={r['success_rate']:.1%}{marker}")

        # Save best checkpoint
        if best_ckpt_path is not None:
            best_dst = ckpt_dir / "best.pt"
            shutil.copy2(str(best_ckpt_path), str(best_dst))
            _log(f"  Saved best.pt from step {best_step}")

        # Cleanup eval_request files
        cleaned = 0
        for req_file in request_files:
            try:
                req_file.unlink()
                cleaned += 1
            except Exception:
                pass
        _log(f"  Cleaned up {cleaned} eval_request files")

    env.close()
    _log("\nAll done!")


if __name__ == "__main__":
    main()
