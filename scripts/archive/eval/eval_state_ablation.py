"""State ablation evaluation for Stack Cube Residual TD3.

Tests how much the RL policy depends on observation.state and observation.object_state
by injecting noise or zeroing out these components.

Ablation modes:
  normal       - no modification (control)
  zero_obj     - zero out observation.object_state
  rand_obj     - replace observation.object_state with random values
  noise_obj_S  - add Gaussian noise to object_state (S = pos_std in cm)
  zero_state   - zero out observation.state (proprio)
  rand_state   - replace observation.state with random values
  zero_ba      - zero out observation.base_action
  rand_ba      - replace observation.base_action with random values
"""

from __future__ import annotations
import argparse, importlib, json, os, sys, time, types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
    def _patched_validate(repo_id):
        try: return _orig_validate(repo_id)
        except Exception: pass
    _hf_val.validate_repo_id = _patched_validate
except Exception:
    pass

import copy
import numpy as np
import torch

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack as MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack as MuJoCoResidualWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig


def _log(msg):
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_noise_fn(mode: str):
    """Return a function obs -> obs that applies the specified ablation."""
    if mode == "normal":
        return None

    def _fn(obs):
        obs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

        if mode == "zero_obj":
            if "observation.object_state" in obs:
                obs["observation.object_state"] = torch.zeros_like(obs["observation.object_state"])

        elif mode == "rand_obj":
            if "observation.object_state" in obs:
                obs["observation.object_state"] = torch.randn_like(obs["observation.object_state"])

        elif mode.startswith("noise_obj_"):
            pos_std_cm = float(mode.split("_")[-1])
            pos_std = pos_std_cm / 100.0  # cm -> m
            if "observation.object_state" in obs:
                obj = obs["observation.object_state"]
                # pos noise (first 3 dims of each 5D or 7D or 10D block)
                obj[:, :3] += torch.randn_like(obj[:, :3]) * pos_std
                # rot noise (remaining dims) - scale proportional
                if obj.shape[-1] > 3:
                    obj[:, 3:] += torch.randn_like(obj[:, 3:]) * 0.1
                obs["observation.object_state"] = obj

        elif mode == "zero_state":
            obs["observation.state"] = torch.zeros_like(obs["observation.state"])

        elif mode == "rand_state":
            obs["observation.state"] = torch.randn_like(obs["observation.state"]) * 0.1

        elif mode == "zero_ba":
            obs["observation.base_action"] = torch.zeros_like(obs["observation.base_action"])

        elif mode == "rand_ba":
            obs["observation.base_action"] = torch.randn_like(obs["observation.base_action"]) * 0.1

        return obs
    return _fn


def evaluate(env, agent, num_episodes, device, image_keys, noise_fn=None, object_state_dim=0):
    agent.eval()
    num_envs = env.vec_env.num_envs
    successes = 0
    total_return = 0.0
    total_steps = 0
    total_episodes = 0
    per_env_sr = [0] * num_envs

    def _truncate_obj_state(obs):
        """Truncate object_state to match agent's expected dim."""
        if object_state_dim > 0 and "observation.object_state" in obs:
            obs["observation.object_state"] = obs["observation.object_state"][:, :object_state_dim]
        return obs

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = _truncate_obj_state(obs)
        if noise_fn:
            obs = noise_fn(obs)
        ep_returns = [0.0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs

        for step in range(env.vec_env.max_episode_steps):
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env.step(action)
            obs = _truncate_obj_state(obs)
            if noise_fn:
                obs = noise_fn(obs)

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
        "per_env_sr": [per_env_sr[i] / max(1, num_episodes) for i in range(num_envs)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--eval_positions_file", type=str, default="configs/stack_cube_positions.json")
    p.add_argument("--noise_mode", type=str, required=True,
                   help="normal|zero_obj|rand_obj|noise_obj_5|zero_state|rand_state|zero_ba|rand_ba")
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--max_positions", type=int, default=None,
                   help="Limit number of positions to evaluate (first N)")
    p.add_argument("--difficulty", type=str, default=None,
                   help="Filter positions by difficulty: easy/normal/hard (from training config)")
    # Env
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--reward_type", type=str, default="dense")
    p.add_argument("--device", type=str, default="cuda")
    # GR00T
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--task_description", type=str,
                   default="Pick up the white cube and stack it on the green cube.")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.004)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--action_scale", type=float, default=0.1)
    # Agent
    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--critic_hidden_dim", type=int, default=256)
    p.add_argument("--asymmetric_critic", action="store_true")
    p.add_argument("--object_state_dim", type=int, default=None,
                   help="Override object_state_dim. If None, auto-detect from checkpoint.")
    # Output
    p.add_argument("--output_file", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"State ablation eval: mode={args.noise_mode}")
    _log(f"  checkpoint={args.checkpoint}")
    _log(f"  device={device}")

    # Load positions
    with open(args.eval_positions_file) as f:
        all_positions = json.load(f)

    # Optional difficulty filter
    if args.difficulty:
        # Use predefined difficulty ranges (from training configs)
        difficulty_ranges = {
            "easy": list(range(0, 20)),
            "normal": list(range(20, 46)),
            "hard": list(range(46, 66)),
        }
        pos_indices = difficulty_ranges.get(args.difficulty, list(range(len(all_positions))))
        all_positions = [all_positions[i] for i in pos_indices if i < len(all_positions)]
        _log(f"  difficulty={args.difficulty} -> {len(all_positions)} positions")

    if args.max_positions and len(all_positions) > args.max_positions:
        all_positions = all_positions[:args.max_positions]
        _log(f"  truncated to {len(all_positions)} positions (--max_positions)")

    num_envs = len(all_positions)
    _log(f"  {num_envs} evaluation positions, {args.num_episodes} episodes each")

    white_positions = [p["white_cube_pos"] for p in all_positions]
    green_positions = [p["green_cube_pos"] for p in all_positions]

    # Auto-detect calib
    _calib_path = args.calib_path
    if _calib_path is None:
        c = str(Path(__file__).resolve().parents[2] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
        if Path(c).exists():
            _calib_path = c

    _log("Creating environment...")
    mujoco_env = MuJoCoVecEnv(
        num_envs=num_envs,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        scene_xml=args.scene_xml,
        calib_path=_calib_path,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
    )

    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.device,
        task_description=args.task_description,
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=args.residual_pos_scale,
        residual_rot_scale=args.residual_rot_scale,
        residual_grip_scale=args.residual_grip_scale,
        ema_alpha=args.ema_alpha,
    )
    _log("Environment ready.")

    # Create agent
    _asymmetric = args.asymmetric_critic

    # Auto-detect object_state_dim from checkpoint if not specified
    if args.object_state_dim is not None:
        object_state_dim = args.object_state_dim
    elif args.checkpoint.lower() != "none":
        _log("Auto-detecting object_state_dim from checkpoint...")
        ckpt_tmp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd_tmp = ckpt_tmp["model"] if isinstance(ckpt_tmp, dict) and "model" in ckpt_tmp else ckpt_tmp
        actor_in = sd_tmp["actor.policy.0.weight"].shape[1]
        actor_out = sd_tmp["actor.policy.8.weight"].shape[0]  # action_dim
        lowdim_from_env = env.observation_space["observation.state"].shape[1]
        object_state_dim = actor_in - actor_out - lowdim_from_env
        _log(f"  actor_in={actor_in}, actor_out={actor_out}, lowdim={lowdim_from_env} -> object_state_dim={object_state_dim}")
        del ckpt_tmp, sd_tmp
    else:
        object_state_dim = 10 if _asymmetric else 0
    image_keys = (["observation.depth.front", "observation.depth.wrist"] if _asymmetric
                  else ["observation.images.front", "observation.images.back", "observation.images.wrist"])
    img_c = 1 if _asymmetric else 3

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(img_c, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=_asymmetric,
    )

    # Load checkpoint
    if args.checkpoint.lower() != "none":
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            agent.load_state_dict(ckpt["model"])
        else:
            agent.load_state_dict(ckpt)
        _log(f"Loaded checkpoint: {args.checkpoint}")
    else:
        _log("No checkpoint loaded (base policy only)")

    agent = agent.to(device)

    # Build noise function
    noise_fn = build_noise_fn(args.noise_mode)
    if noise_fn:
        _log(f"Noise injection: {args.noise_mode}")

    # Evaluate
    t0 = time.time()
    metrics = evaluate(env, agent, args.num_episodes, device, image_keys, noise_fn, object_state_dim)
    elapsed = time.time() - t0

    _log(f"Result: SR={metrics['success_rate']:.1%} "
         f"({int(metrics['success_rate'] * metrics['total_episodes'])}/{metrics['total_episodes']}) "
         f"return={metrics['mean_return']:.2f} time={elapsed:.1f}s")

    # Per-env breakdown
    per_env = metrics["per_env_sr"]
    n_perfect = sum(1 for s in per_env if s >= 0.99)
    n_zero = sum(1 for s in per_env if s < 0.01)
    _log(f"  {n_perfect} envs 100% | {n_zero} envs 0% | {len(per_env) - n_perfect - n_zero} envs partial")

    # Save results
    result = {
        "noise_mode": args.noise_mode,
        "checkpoint": args.checkpoint,
        "difficulty": args.difficulty,
        "num_positions": num_envs,
        "num_episodes": args.num_episodes,
        **metrics,
        "elapsed_sec": elapsed,
    }

    out_path = args.output_file
    if out_path is None:
        diff_tag = f"_{args.difficulty}" if args.difficulty else ""
        out_path = f"outputs/state_ablation/{args.noise_mode}{diff_tag}.json"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
