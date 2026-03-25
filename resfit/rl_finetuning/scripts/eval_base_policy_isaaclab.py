"""
Eval base policy only (residual = 0). No RL checkpoint needed.

Runs Isaac Sim with GR00T base policy and zero residual action.
Measures base policy success rate for comparison.

Usage:
  isaac_python eval_base_policy_isaaclab.py --headless --enable_cameras \
    --num_envs 20 --max_episode_steps 1000 ...
"""
from __future__ import annotations
import argparse, os, sys, time, json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval base policy (residual=0)")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--cube_perturb_table_path", type=str, default=None)
parser.add_argument("--reward_type", type=str, default="dense_clipped")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
import isaaclab.sim as sim_utils

_log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [base-eval] {msg}", flush=True)


def main():
    device = torch.device(args.groot_policy_device)
    N = args.num_envs
    
    _log("Creating environment...")
    _sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=args.csv_base_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args.language_override,
        max_episode_steps=args.max_episode_steps,
        num_envs=N,
        success_threshold=args.success_threshold,
        cube_perturb_table_path=args.cube_perturb_table_path,
        reward_type=args.reward_type,
    )
    _log(f"Env ready: {N} envs")

    all_rates = []
    
    for ep in range(args.num_episodes):
        env.reset()
        obs = env._build_obs()
        
        ep_rewards = torch.zeros(N, device=device)
        ep_ever_success = torch.zeros(N, device=device)
        
        zero_action = torch.zeros(N, 7, device=device, dtype=torch.float32)
        
        for step in range(args.max_episode_steps):
            next_obs, reward, terminated, truncated, info = env.step(zero_action)
            ep_rewards += reward[:N].to(device)
            
            cube_z = env.cube.data.root_state_w[:N, 2].to(device)
            cube_init_z = env._initial_cube_z[:N].to(device)
            cube_lifted = ((cube_z - cube_init_z) >= args.success_threshold).float()
            _contact = getattr(env, '_contact_history', None)
            if _contact is not None:
                _gate = _contact.any(dim=1).float().to(device)
                cube_lifted = cube_lifted * _gate
            ep_ever_success = torch.max(ep_ever_success, cube_lifted)
            
            obs = next_obs
            done = terminated | truncated
            if done[:N].all():
                break
        
        rate = ep_ever_success.mean().item()
        ret = ep_rewards.mean().item()
        all_rates.append(rate)
        _log(f"Episode {ep}: success={rate*100:.1f}% return={ret:.1f}")
    
    mean_rate = np.mean(all_rates)
    _log(f"\n{'='*60}")
    _log(f"BASE POLICY RESULT: success_rate={mean_rate:.3f} ({mean_rate*100:.1f}%)")
    _log(f"  episodes={args.num_episodes}, envs={N}")
    _log(f"{'='*60}")
    
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "success_rate": mean_rate,
            "all_rates": all_rates,
            "num_envs": N,
            "num_episodes": args.num_episodes,
            "max_episode_steps": args.max_episode_steps,
        }
        (out / "base_eval_result.json").write_text(json.dumps(result, indent=2))
        _log(f"Saved: {out / 'base_eval_result.json'}")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
