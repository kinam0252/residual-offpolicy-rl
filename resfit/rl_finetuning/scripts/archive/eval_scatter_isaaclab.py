"""
Scatter Eval: independently evaluate unevaluated checkpoints.

Multiple instances can run in parallel — each claims a checkpoint via a .lock file,
evaluates it, writes a .done result, then moves to the next.

Usage:
  isaac_python eval_scatter_isaaclab.py --headless --enable_cameras \
    --checkpoint_dir /path/to/checkpoints \
    --eval_output_dir /path/to/eval_results \
    --num_envs 20 ...
"""
from __future__ import annotations
import argparse, os, sys, time, json, random, signal
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scatter eval for checkpoints")
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--cube_perturb_table_path", type=str, default=None)
parser.add_argument("--action_scale", type=float, default=0.1)
parser.add_argument("--reward_type", type=str, default="dense_clipped")
parser.add_argument("--use_depth", action="store_true")
parser.add_argument("--depth_norm_path", type=str, default=None)
parser.add_argument("--use_state", action="store_true")
parser.add_argument("--object_state_mode", type=str, default="raw")
parser.add_argument("--contact_binary", action="store_true")
parser.add_argument("--asymmetric_critic", action="store_true")
parser.add_argument("--critic_hidden_dim", type=int, default=None)
parser.add_argument("--actor_hidden_dim", type=int, default=None)
parser.add_argument("--wandb_mode", type=str, default="disabled")
parser.add_argument("--max_evals", type=int, default=0, help="Max evals before exiting (0=unlimited)")
parser.add_argument("--poll_interval", type=int, default=60, help="Seconds to wait if no checkpoints available")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

# Reuse existing eval infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from resfit.rl_finetuning.scripts.eval_async_isaaclab import (
    make_env, make_agent, load_checkpoint,
)
from resfit.rl_finetuning.off_policy.rl import utils

_log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [scatter-eval] {msg}", flush=True)


def run_eval_episode_simple(env, agent, device, max_ep_steps, success_threshold, args):
    """Run one eval episode across all envs and return success rate + mean return."""
    N = args.num_envs
    env.reset()
    obs = env._build_obs()
    
    ep_rewards = torch.zeros(N, device=device)
    ep_ever_success = torch.zeros(N, device=device)
    ep_steps = 0
    
    while ep_steps < max_ep_steps:
        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        
        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        ep_rewards += reward[:N].to(device)
        
        cube_z = env.cube.data.root_state_w[:N, 2].to(device)
        cube_init_z = env._initial_cube_z[:N].to(device)
        cube_lifted = ((cube_z - cube_init_z) >= success_threshold).float()
        _contact_smoothed = getattr(env, '_contact_history', None)
        if _contact_smoothed is not None:
            _has_contact_gate = _contact_smoothed.any(dim=1).float().to(device)
            cube_lifted = cube_lifted * _has_contact_gate
        ep_ever_success = torch.max(ep_ever_success, cube_lifted)
        ep_steps += 1
        
        obs = next_obs
        done = terminated | truncated
        if done[:N].all():
            break
    
    success_rate = ep_ever_success.mean().item()
    mean_return = ep_rewards.mean().item()
    succ_mask = ep_ever_success > 0
    mean_succ_len = 0.0
    if succ_mask.any():
        mean_succ_len = max_ep_steps  # simplified
    
    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_successful_episode_length": mean_succ_len,
    }


def find_unevaluated(ckpt_dir: Path, eval_dir: Path) -> list[tuple[int, Path]]:
    """Find checkpoints that don't have .done or .lock files."""
    available = []
    for f in sorted(ckpt_dir.glob("agent_step*.pt")):
        # Extract step number
        name = f.stem  # agent_step12000
        step_str = name.replace("agent_step", "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        
        done_file = eval_dir / f"eval_step{step}.done"
        lock_file = eval_dir / f"eval_step{step}.lock"
        json_file = eval_dir / f"eval_step{step}.json"
        
        if done_file.exists() or lock_file.exists() or json_file.exists():
            continue
        available.append((step, f))
    
    return available


def claim_checkpoint(step: int, eval_dir: Path, job_id: str) -> bool:
    """Try to claim a checkpoint by creating a .lock file atomically."""
    lock_file = eval_dir / f"eval_step{step}.lock"
    try:
        # O_CREAT | O_EXCL ensures atomic creation (fails if exists)
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{job_id}\n{time.time()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(step: int, eval_dir: Path):
    """Remove lock file (on error or cleanup)."""
    lock_file = eval_dir / f"eval_step{step}.lock"
    try:
        lock_file.unlink(missing_ok=True)
    except:
        pass


def save_result(step: int, eval_dir: Path, result: dict):
    """Save eval result as .done file."""
    done_file = eval_dir / f"eval_step{step}.done"
    done_file.write_text(json.dumps(result, indent=2))
    # Remove lock
    release_lock(step, eval_dir)


def main():
    ckpt_dir = Path(args.checkpoint_dir)
    eval_dir = Path(args.output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    job_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    _log(f"Starting scatter eval (job={job_id})")
    _log(f"  ckpt_dir: {ckpt_dir}")
    _log(f"  eval_dir: {eval_dir}")
    
    # Create env and agent once (reuse for all evals)
    _log("Creating eval environment...")
    env = make_env(args)
    _log(f"Eval env ready: {args.num_envs} envs")
    
    _log("Creating agent...")
    agent, image_keys = make_agent(env, args.groot_policy_device, args)
    agent.eval()
    agent.to(args.groot_policy_device)
    _log("Agent ready")
    
    device = torch.device(args.groot_policy_device)
    num_evals_done = 0
    
    # Cleanup locks on exit
    _claimed_step = [None]
    def cleanup_handler(signum, frame):
        if _claimed_step[0] is not None:
            _log(f"Signal {signum}: releasing lock for step {_claimed_step[0]}")
            release_lock(_claimed_step[0], eval_dir)
        sys.exit(1)
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    
    while True:
        # Find unevaluated checkpoints
        available = find_unevaluated(ckpt_dir, eval_dir)
        
        if not available:
            _log(f"No unevaluated checkpoints. Waiting {args.poll_interval}s...")
            time.sleep(args.poll_interval)
            continue
        
        # Shuffle and try to claim one
        random.shuffle(available)
        claimed = False
        
        for step, ckpt_path in available:
            if claim_checkpoint(step, eval_dir, job_id):
                _claimed_step[0] = step
                _log(f"Claimed step {step} ({ckpt_path.name}) [{len(available)} available]")
                claimed = True
                
                try:
                    # Load checkpoint
                    load_checkpoint(agent, ckpt_path, device)
                    agent.eval()
                    
                    # Run eval episode
                    t0 = time.time()
                    result = run_eval_episode_simple(
                        env=env,
                        agent=agent,
                        device=device,
                        max_ep_steps=args.max_episode_steps,
                        success_threshold=args.success_threshold,
                        args=args,
                    )
                    elapsed = time.time() - t0
                    
                    # Save result
                    result_dict = {
                        "step": step,
                        "eval/success_rate": result["success_rate"],
                        "eval/mean_return": result["mean_return"],
                        "eval/mean_successful_episode_length": result.get("mean_successful_episode_length", 0),
                        "eval/num_envs": args.num_envs,
                        "eval/episode_steps": args.max_episode_steps,
                        "eval_time_sec": elapsed,
                        "eval_job_id": job_id,
                    }
                    save_result(step, eval_dir, result_dict)
                    
                    sr = result["success_rate"]
                    _log(f"Step {step}: success={sr:.3f} return={result['mean_return']:.1f} ({elapsed:.1f}s)")
                    
                    num_evals_done += 1
                    _claimed_step[0] = None
                    
                except Exception as e:
                    _log(f"ERROR evaluating step {step}: {e}")
                    release_lock(step, eval_dir)
                    _claimed_step[0] = None
                    import traceback
                    traceback.print_exc()
                
                break  # go back to scan
        
        if not claimed:
            _log("All available checkpoints are locked by other workers. Waiting 30s...")
            time.sleep(30)
        
        if args.max_evals > 0 and num_evals_done >= args.max_evals:
            _log(f"Reached max_evals={args.max_evals}. Exiting.")
            break
    
    _log(f"Scatter eval complete. Total evals: {num_evals_done}")
    simulation_app.close()


if __name__ == "__main__":
    main()
