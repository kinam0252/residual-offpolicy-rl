"""
Scatter 3-episode eval for v27d checkpoints.

Scans for checkpoints with async eval success >= 0.5,
picks one randomly, runs 3-episode eval, saves result.
Multiple instances can run in parallel via .lock files.

Usage:
  isaac_python eval_scatter_3ep.py --headless --enable_cameras \
    --checkpoint_dir /path/to/checkpoints \
    --async_eval_dir /path/to/async_eval_results \
    --output_dir /path/to/scatter_3ep_results \
    --num_envs 20 --min_rate 0.5 ...
"""
from __future__ import annotations
import argparse, os, sys, time, json, random, signal
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scatter 3-episode eval")
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--async_eval_dir", type=str, required=True, help="Dir with eval_step*.done/.json from async eval")
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
parser.add_argument("--depth_norm_path", type=str, default=None)
parser.add_argument("--min_rate", type=float, default=0.5, help="Min async eval rate to consider")
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--max_evals", type=int, default=0, help="Max checkpoints to eval (0=unlimited)")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import glob as _glob

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.rlpd import QAgentConfig, ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
import isaaclab.sim as sim_utils

_log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [scatter-3ep] {msg}", flush=True)


def find_candidates(ckpt_dir, async_dir, output_dir, min_rate):
    """Find checkpoints with async eval >= min_rate that haven't been 3ep-evaluated yet."""
    # Read async eval results
    async_rates = {}
    for f in _glob.glob(f'{async_dir}/eval_step*.done') + _glob.glob(f'{async_dir}/eval_step*.json'):
        try:
            d = json.load(open(f))
            async_rates[d['step']] = d.get('eval/success_rate', 0)
        except:
            pass
    
    candidates = []
    for step, rate in async_rates.items():
        if rate < min_rate:
            continue
        ckpt_path = Path(ckpt_dir) / f"agent_step{step}.pt"
        if not ckpt_path.exists():
            continue
        # Check if already evaluated or locked
        result_file = Path(output_dir) / f"step{step}_3ep.json"
        lock_file = Path(output_dir) / f"step{step}_3ep.lock"
        if result_file.exists() or lock_file.exists():
            continue
        candidates.append((step, rate, str(ckpt_path)))
    
    return candidates


def claim(step, output_dir, job_id):
    lock_file = Path(output_dir) / f"step{step}_3ep.lock"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{job_id}\n{time.time()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(step, output_dir):
    lock_file = Path(output_dir) / f"step{step}_3ep.lock"
    try:
        lock_file.unlink(missing_ok=True)
    except:
        pass


def main():
    device = torch.device(args.groot_policy_device)
    N = args.num_envs
    job_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    _log(f"Starting scatter 3-ep eval (job={job_id})")
    _log(f"  ckpt_dir: {args.checkpoint_dir}")
    _log(f"  async_dir: {args.async_eval_dir}")
    _log(f"  output_dir: {args.output_dir}")
    _log(f"  min_rate: {args.min_rate}")
    
    # Create env once
    _log("Creating environment...")
    _sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    env = IfaceEnvWrapper(
        sim=_sim, csv_dir=args.csv_base_dir,
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
        use_depth=True,
        depth_norm_path=args.depth_norm_path,
    )
    _log(f"Env ready: {N} envs")
    
    # Create agent template
    image_keys = ["observation.depth.front", "observation.depth.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    
    agent_cfg = QAgentConfig(
        actor_lr=1e-6, critic_lr=1e-4, critic_target_tau=0.005,
        actor=ActorConfig(action_scale=args.action_scale, actor_last_layer_init_scale=0.0, action_l2_reg_weight=1.0),
    )
    
    # Cleanup handler
    _claimed = [None]
    def cleanup(signum, frame):
        if _claimed[0] is not None:
            release(_claimed[0], out_dir)
        sys.exit(1)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    
    num_done = 0
    
    while True:
        candidates = find_candidates(args.checkpoint_dir, args.async_eval_dir, str(out_dir), args.min_rate)
        
        if not candidates:
            _log("No more candidates. Exiting.")
            break
        
        random.shuffle(candidates)
        claimed = False
        
        for step, async_rate, ckpt_path in candidates:
            if claim(step, out_dir, job_id):
                _claimed[0] = step
                _log(f"Claimed step {step} (async_rate={async_rate*100:.0f}%, {len(candidates)} remaining)")
                
                try:
                    # Create fresh agent and load checkpoint
                    agent = QAgent(
                        obs_shape=(img_c, img_h, img_w),
                        prop_shape=(lowdim_dim,),
                        action_dim=action_dim,
                        rl_cameras=image_keys,
                        cfg=agent_cfg,
                        residual_actor=True,
                    )
                    ckpt = torch.load(ckpt_path, map_location=device)
                    agent.load_checkpoint_compat(ckpt)
                    agent.eval()
                    agent.to(device)
                    
                    # Run 3 episodes
                    all_rates = []
                    all_returns = []
                    
                    for ep_i in range(args.num_episodes):
                        env.reset()
                        obs = env._build_obs()
                        ep_rewards = torch.zeros(N, device=device)
                        ep_ever_success = torch.zeros(N, device=device)
                        
                        for s in range(args.max_episode_steps):
                            with torch.no_grad(), utils.eval_mode(agent):
                                residual = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
                            next_obs, reward, term, trunc, info = env.step(residual)
                            ep_rewards += reward[:N].to(device)
                            
                            cube_z = env.cube.data.root_state_w[:N, 2].to(device)
                            cube_init_z = env._initial_cube_z[:N].to(device)
                            lifted = ((cube_z - cube_init_z) >= args.success_threshold).float()
                            _ch = getattr(env, '_contact_history', None)
                            if _ch is not None:
                                lifted = lifted * _ch.any(dim=1).float().to(device)
                            ep_ever_success = torch.max(ep_ever_success, lifted)
                            
                            obs = next_obs
                            if (term | trunc)[:N].all():
                                break
                        
                        rate = ep_ever_success.mean().item()
                        ret = ep_rewards.mean().item()
                        all_rates.append(rate)
                        all_returns.append(ret)
                        _log(f"  Step {step} Ep{ep_i+1}: success={rate*100:.1f}% return={ret:.1f}")
                    
                    avg_rate = np.mean(all_rates)
                    avg_ret = np.mean(all_returns)
                    
                    result = {
                        "step": step,
                        "async_rate": async_rate,
                        "success_rate_3ep": avg_rate,
                        "mean_return_3ep": avg_ret,
                        "all_rates": all_rates,
                        "all_returns": all_returns,
                        "num_episodes": args.num_episodes,
                        "num_envs": N,
                        "job_id": job_id,
                    }
                    result_file = out_dir / f"step{step}_3ep.json"
                    result_file.write_text(json.dumps(result, indent=2))
                    release(step, out_dir)
                    _claimed[0] = None
                    
                    _log(f"  Step {step}: 3ep_avg={avg_rate*100:.1f}% (async was {async_rate*100:.0f}%)")
                    num_done += 1
                    
                    del agent
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    _log(f"ERROR step {step}: {e}")
                    release(step, out_dir)
                    _claimed[0] = None
                    import traceback; traceback.print_exc()
                
                claimed = True
                break
        
        if not claimed:
            _log("All candidates locked. Waiting 30s...")
            time.sleep(30)
        
        if args.max_evals > 0 and num_done >= args.max_evals:
            _log(f"Reached max_evals={args.max_evals}. Exiting.")
            break
    
    _log(f"Done. Total evaluated: {num_done}")
    simulation_app.close()


if __name__ == "__main__":
    main()
