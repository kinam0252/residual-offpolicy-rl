"""
Residual TD3 training on MuJoCo + GR00T.

Standalone script (no Isaac Sim dependency) that uses ``MuJoCoVecEnv`` +
``MuJoCoResidualWrapper`` for environment interaction, and reuses the
same QAgent/TD3, replay buffers, and critic warmup logic from the
IsaacLab version.

Launch:
    source ~/.venvs/groot/bin/activate
    MUJOCO_GL=egl LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-} \\
    python resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \\
        --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2/checkpoint-30000 \\
        --num_envs 1 --total_timesteps 50000
"""

from __future__ import annotations

import argparse
import os
import pprint
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

try:
    import wandb
except ImportError:
    wandb = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _log(msg: str) -> None:
    print(f"[mujoco-td3] {msg}", flush=True)


# ── Transition storage (same as IsaacLab version) ──

def _add_transitions(
    obs, next_obs, actions, reward, done,
    device, image_keys, lowdim_keys, num_envs, online_rb,
):
    obs_keys = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        curr_obs_i = {k: v[i] for k, v in obs.items() if k in obs_keys}
        next_obs_i = {k: v[i] for k, v in next_obs.items() if k in obs_keys}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict({
            "obs": TensorDict(curr_obs_i, batch_size=[]),
            "next": TensorDict({
                "obs": TensorDict(next_obs_i, batch_size=[]),
                "done": done[i],
                "reward": reward[i],
            }, batch_size=[]),
            "action": actions[i],
            "_priority": torch.tensor(10.0, dtype=torch.float32, device=device),
        }, batch_size=[]).unsqueeze(0)
        online_rb.add(td)


# ── Evaluation ──

def evaluate(
    env: MuJoCoResidualWrapper,
    agent: QAgent,
    num_episodes: int,
    device: torch.device,
    image_keys: list[str],
) -> dict[str, float]:
    """Run evaluation episodes with greedy policy (stddev=0)."""
    successes = 0
    total_return = 0.0
    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_steps = 0

        for step in range(env.vec_env.max_episode_steps):
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward[0].item()
            ep_steps += 1

            done = terminated | truncated
            if done[0]:
                break

        # Check success
        cube_z = env.vec_env._envs[0]["data"].qpos[
            env.vec_env._envs[0]["cube_qposadr"] + 2
        ]
        init_z = env.vec_env._initial_cube_z[0]
        if cube_z - init_z >= env.vec_env.success_threshold:
            successes += 1

        total_return += ep_return
        total_steps += ep_steps

    return {
        "eval/success_rate": successes / max(1, num_episodes),
        "eval/mean_return": total_return / max(1, num_episodes),
        "eval/mean_steps": total_steps / max(1, num_episodes),
    }


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="Residual TD3 on MuJoCo + GR00T")
    # Environment
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--cube_pos", type=float, nargs=3, default=[0.45, -0.05, 0.02])
    p.add_argument("--cube_yaw", type=float, default=0.0)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--success_threshold", type=float, default=0.005)
    p.add_argument("--reward_type", type=str, default="sparse", choices=["sparse", "dense"])
    # GR00T
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str, default="lift the cube")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual scales
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.1)
    # Algorithm
    p.add_argument("--total_timesteps", type=int, default=50_000)
    p.add_argument("--learning_starts", type=int, default=1_000)
    p.add_argument("--critic_warmup_steps", type=int, default=1_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--random_action_noise_scale", type=float, default=0.1)
    p.add_argument("--stddev_max", type=float, default=0.05)
    p.add_argument("--stddev_min", type=float, default=0.05)
    p.add_argument("--action_scale", type=float, default=0.1)
    p.add_argument("--actor_lr", type=float, default=3e-7)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="outputs/mujoco_td3")
    p.add_argument("--eval_interval", type=int, default=5_000)
    p.add_argument("--eval_num_episodes", type=int, default=10)
    p.add_argument("--debug_zero_residual", action="store_true")
    p.add_argument("--save_video", action="store_true")
    # W&B
    p.add_argument("--wandb_mode", type=str, default="disabled")
    p.add_argument("--wandb_project", type=str, default="mujoco-franka-residual-td3")
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Output dirs
    outputs_dir = Path(args.output_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outputs_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Seed: {args.seed}")
    _log(f"Device: {device}")
    _log(f"Output: {outputs_dir}")

    # ── Create MuJoCo environment ──
    _log("Creating MuJoCo environment...")
    cube_positions = [args.cube_pos] * args.num_envs
    mujoco_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        cube_positions=cube_positions,
        cube_yaw_deg=args.cube_yaw,
        scene_xml=args.scene_xml,
        calib_path=args.calib_path,
        max_episode_steps=args.max_episode_steps,
        success_threshold=args.success_threshold,
        reward_type=args.reward_type,
        device=args.device,
        rl_img_size=84,
    )
    _log(f"MuJoCo env: {args.num_envs} envs, cube_pos={args.cube_pos}")

    # ── Wrap with residual + GR00T ──
    _log("Creating residual wrapper with GR00T policy...")
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
    )
    _log("Environment ready.")

    # ── Dimensions ──
    num_envs = args.num_envs
    image_keys = [
        "observation.images.front",
        "observation.images.back",
        "observation.images.wrist",
    ]
    lowdim_keys = ["observation.state", "observation.base_action"]
    img_c, img_h, img_w = 3, 84, 84
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim  # 7 (residual)

    _log(f"lowdim_dim={lowdim_dim}, img=({img_c},{img_h},{img_w}), action_dim={action_dim}")

    # ── QAgent ──
    cfg = ResidualTD3MuJoCoConfig()
    # Override from CLI
    cfg.agent.actor_lr = args.actor_lr
    cfg.agent.critic_lr = args.critic_lr
    cfg.agent.actor.action_scale = args.action_scale

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    # ── Replay buffer ──
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        pin_memory=True,
        prefetch=4,
        batch_size=args.batch_size,
    )

    # ── W&B ──
    _wb = wandb
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mujoco_td3_seed{args.seed}"
    if args.wandb_name:
        run_name = f"{args.wandb_name}__{run_name}"
    if _wb is not None:
        try:
            _wb.init(
                project=args.wandb_project,
                name=run_name,
                mode=args.wandb_mode,
                config=vars(args),
                reinit=True,
            )
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")
            _wb = None

    # ══════════════════════════════════════════════════════════════════
    # Warm-up: fill buffer with base-policy + noise exploration
    # ══════════════════════════════════════════════════════════════════
    _log(f"Warm-up: collecting {args.learning_starts} transitions (base policy + noise)...")
    obs, _ = env.reset()
    warmup_transitions = 0
    warmup_ep_count = 0

    while warmup_transitions < args.learning_starts:
        noise = (torch.rand((num_envs, action_dim), device=device) * 2 - 1) * args.random_action_noise_scale
        next_obs, reward, terminated, truncated, info = env.step(noise)
        done = terminated | truncated

        # Store transitions
        combined_action = info.get("scaled_action", noise)
        if combined_action.shape[0] > num_envs:
            combined_action = combined_action[:num_envs]

        _add_transitions(
            obs=obs, next_obs=next_obs, actions=combined_action,
            reward=reward, done=done, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        warmup_transitions += num_envs

        if done.any():
            warmup_ep_count += done.sum().item()

        obs = next_obs

        if warmup_transitions % 200 == 0:
            _log(f"  warmup: {warmup_transitions}/{args.learning_starts} transitions, {warmup_ep_count} episodes")

    _log(f"Warm-up done: {warmup_transitions} transitions, {warmup_ep_count} episodes")

    # ══════════════════════════════════════════════════════════════════
    # Critic warmup (critic-only updates)
    # ══════════════════════════════════════════════════════════════════
    if args.critic_warmup_steps > 0:
        _log(f"Critic warmup: {args.critic_warmup_steps} updates...")
        for i in range(args.critic_warmup_steps):
            batch = online_rb.sample(args.batch_size).to(device, non_blocking=True)
            metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if i % 200 == 0:
                cl = metrics.get("train/critic_loss", 0)
                _log(f"  critic warmup: {i}/{args.critic_warmup_steps} loss={cl:.4f}")
        _log("Critic warmup done.")

    # ══════════════════════════════════════════════════════════════════
    # Main training loop
    # ══════════════════════════════════════════════════════════════════
    _log(f"Training for {args.total_timesteps} steps...")
    obs, _ = env.reset()
    global_step = 0
    episode_count = 0
    best_success = 0.0
    ep_cum_reward = torch.zeros(num_envs, device=device)
    ep_step_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    while global_step <= args.total_timesteps:
        # ── (1) Collect transition ──
        stddev = args.stddev_min  # constant noise (can add schedule later)

        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

        if args.debug_zero_residual:
            residual_action = torch.zeros_like(residual_action)

        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        done = terminated | truncated

        # Episode bookkeeping
        ep_cum_reward += reward
        ep_step_counter += 1
        if done.any():
            done_mask = done.bool()
            n_done = done_mask.sum().item()
            episode_count += n_done

            if _wb is not None and _wb.run is not None:
                ep_return = float(ep_cum_reward[done_mask].mean().item())
                ep_steps = float(ep_step_counter[done_mask].float().mean().item())
                _wb.log({
                    "training/episode_return": ep_return,
                    "training/episode_steps": ep_steps,
                    "training/episode_count": episode_count,
                }, step=global_step)

            ep_cum_reward[done_mask] = 0.0
            ep_step_counter[done_mask] = 0

        # Store transition
        combined_action = info.get("scaled_action", residual_action)
        if combined_action.shape[0] > num_envs:
            combined_action = combined_action[:num_envs]

        _add_transitions(
            obs=obs, next_obs=next_obs, actions=combined_action,
            reward=reward, done=done, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        obs = next_obs

        # ── (2) Update agent ──
        if len(online_rb) >= args.batch_size:
            batch = online_rb.sample(args.batch_size).to(device, non_blocking=True)
            update_actor = global_step >= args.critic_warmup_steps
            metrics = agent.update(
                batch, stddev=stddev,
                update_actor=update_actor,
                bc_batch=None, ref_agent=agent,
            )

            # Log
            if global_step % 100 == 0 and _wb is not None and _wb.run is not None:
                log_dict = {f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                log_dict["training/global_step"] = global_step
                log_dict["training/buffer_size"] = len(online_rb)
                _wb.log(log_dict, step=global_step)

        # ── (3) Periodic logging ──
        if global_step % 1000 == 0:
            ba = obs["observation.base_action"][0].detach().cpu().numpy()
            ra = residual_action[0].detach().cpu().numpy() if residual_action is not None else np.zeros(7)
            rw = reward[0].item()
            _log(f"step={global_step} reward={rw:.4f} base_action={ba[:3]} residual={ra[:3]}")

        # ── (4) Evaluation ──
        if global_step > 0 and global_step % args.eval_interval == 0:
            _log(f"Evaluating ({args.eval_num_episodes} episodes)...")
            eval_metrics = evaluate(
                env=env, agent=agent,
                num_episodes=args.eval_num_episodes,
                device=device, image_keys=image_keys,
            )
            sr = eval_metrics["eval/success_rate"]
            _log(f"  success_rate={sr:.2%}, mean_return={eval_metrics['eval/mean_return']:.2f}")

            if _wb is not None and _wb.run is not None:
                _wb.log(eval_metrics, step=global_step)

            # Save checkpoint
            if sr >= best_success:
                best_success = sr
                ckpt_path = checkpoint_dir / f"best_step{global_step}_sr{sr:.2f}.pt"
                torch.save(agent.state_dict(), ckpt_path)
                _log(f"  Saved best checkpoint: {ckpt_path}")

        global_step += 1

    # ── Final eval ──
    _log("Final evaluation...")
    eval_metrics = evaluate(
        env=env, agent=agent,
        num_episodes=args.eval_num_episodes,
        device=device, image_keys=image_keys,
    )
    _log(f"Final: success_rate={eval_metrics['eval/success_rate']:.2%}")

    # Save final checkpoint
    final_ckpt = checkpoint_dir / f"final_step{global_step}.pt"
    torch.save(agent.state_dict(), final_ckpt)
    _log(f"Final checkpoint: {final_ckpt}")

    if _wb is not None and _wb.run is not None:
        _wb.log(eval_metrics, step=global_step)
        _wb.finish()

    env.close()
    _log("Training complete.")


if __name__ == "__main__":
    main()
