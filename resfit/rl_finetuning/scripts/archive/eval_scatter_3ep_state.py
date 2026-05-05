"""
Scatter 3-episode eval for state-based checkpoints (v31a asymmetric, v28-v30 state).

Optimized version: uses sim.step(render=False) on non-GR00T-inference steps,
since the RL actor only needs physics state (no images).
Rendering is auto-enabled when GR00T needs camera data (~2/80 steps).

Expected speedup: ~3-5x vs full-render eval.

Usage:
  isaac_python eval_scatter_3ep_state.py --headless --enable_cameras \
    --checkpoint_dir /path/to/checkpoints \
    --async_eval_dir /path/to/async_eval_results \
    --output_dir /path/to/scatter_3ep_results \
    --num_envs 20 --min_rate 0.5 --asymmetric_critic ...
"""
from __future__ import annotations
import argparse, os, sys, time, json, random, signal
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scatter 3-episode eval (state-mode, render-skip)")
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--async_eval_dir", type=str, required=True)
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
parser.add_argument("--min_rate", type=float, default=0.5)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--max_evals", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
# Mode flags
parser.add_argument("--asymmetric_critic", action="store_true", help="v31a: actor=state-only, critic=depth+state")
parser.add_argument("--use_state", action="store_true", help="v28-v30: state-only mode")
parser.add_argument("--use_depth", action="store_true", help="Use depth obs (for critic in asymmetric)")
parser.add_argument("--object_state_mode", type=str, default="raw", choices=["raw", "relative", "full"])
parser.add_argument("--critic_hidden_dim", type=int, default=None)
parser.add_argument("--actor_hidden_dim", type=int, default=None)
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

_log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [scatter-3ep-state] {msg}", flush=True)


def find_candidates(ckpt_dir, async_dir, output_dir, min_rate):
    """Find checkpoints with async eval >= min_rate that haven't been 3ep-evaluated yet."""
    async_rates = {}
    for f in _glob.glob(f'{async_dir}/eval_step*.done') + _glob.glob(f'{async_dir}/eval_step*.json'):
        try:
            d = json.load(open(f))
            async_rates[d['step']] = d.get('eval/success_rate', 0)
        except Exception:
            pass

    candidates = []
    for step, rate in async_rates.items():
        if rate < min_rate:
            continue
        ckpt_path = Path(ckpt_dir) / f"agent_step{step}.pt"
        if not ckpt_path.exists():
            continue
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
    except Exception:
        pass


def main():
    device = torch.device(args.groot_policy_device)
    N = args.num_envs
    job_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _asymmetric = args.asymmetric_critic
    _use_state = args.use_state or _asymmetric
    _use_depth = args.use_depth or _asymmetric

    _log(f"Starting scatter 3-ep eval [STATE-MODE render-skip] (job={job_id})")
    _log(f"  ckpt_dir: {args.checkpoint_dir}")
    _log(f"  async_dir: {args.async_eval_dir}")
    _log(f"  output_dir: {args.output_dir}")
    _log(f"  min_rate: {args.min_rate}")
    _log(f"  asymmetric={_asymmetric}, use_state={_use_state}, use_depth={_use_depth}")
    _log(f"  render=False optimization ENABLED (render only on GR00T inference steps)")

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
        use_depth=_use_depth,
        depth_norm_path=args.depth_norm_path,
        use_state=_use_state,
        object_state_mode=args.object_state_mode,
    )
    _log(f"Env ready: {N} envs, inference_interval={env.inference_interval}")

    # Create agent (same architecture as training)
    if _asymmetric:
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
    elif _use_state and not _use_depth:
        image_keys = []
    elif _use_depth:
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
    else:
        image_keys = ["observation.images.front", "observation.images.back", "observation.images.wrist"]

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    if image_keys:
        img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    else:
        img_c, img_h, img_w = 3, 84, 84  # dummy

    action_dim = env.action_space.shape[1]
    object_state_dim = 0
    if _use_state and "observation.object_state" in env.observation_space.spaces:
        object_state_dim = env.observation_space["observation.object_state"].shape[1]

    agent_cfg = QAgentConfig(
        actor_lr=1e-6, critic_lr=1e-4, critic_target_tau=0.005,
        actor=ActorConfig(
            action_scale=args.action_scale,
            actor_last_layer_init_scale=0.0,
            action_l2_reg_weight=1.0,
        ),
    )
    if args.critic_hidden_dim is not None:
        agent_cfg.critic.hidden_dim = args.critic_hidden_dim
    if args.actor_hidden_dim is not None:
        agent_cfg.actor.hidden_dim = args.actor_hidden_dim

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
                        object_state_dim=object_state_dim,
                        asymmetric_critic=_asymmetric,
                    )
                    ckpt = torch.load(ckpt_path, map_location=device)
                    agent.load_checkpoint_compat(ckpt)
                    agent.eval()
                    agent.to(device)

                    all_rates = []
                    all_returns = []
                    render_stats = {"rendered": 0, "skipped": 0}

                    for ep_i in range(args.num_episodes):
                        env.reset()
                        obs = env._build_obs()
                        ep_rewards = torch.zeros(N, device=device)
                        ep_ever_success = torch.zeros(N, device=device)

                        for s in range(args.max_episode_steps):
                            with torch.no_grad(), utils.eval_mode(agent):
                                residual = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

                            # Key optimization: render=False for state-only actor
                            # IfaceEnvWrapper.step() auto-renders when GR00T needs inference
                            next_obs, reward, term, trunc, info = env.step(residual, render=False)

                            # Track render stats from env
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
                        "render_skip": True,
                        "mode": "asymmetric" if _asymmetric else "state",
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
