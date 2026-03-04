"""
Residual TD3 training on IsaacLab + GR00T.

Adapted from ``resfit/rl_finetuning/scripts/train_residual_td3.py``.
Replaces robosuite + ACT with IsaacLab + GR00T while reusing:
  - QAgent (TD3 actor/critic)
  - Replay buffers (online + offline with n-step returns)
  - Prioritised experience replay
  - Critic warmup + progressive clipping

Launch (from isaaclab.sh runtime):
    isaaclab.sh -p resfit/rl_finetuning/scripts/train_residual_td3_isaaclab.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")  # RTX 5090 (sm_120)

# ── Ensure resfit's isaaclab_tasks takes priority over Honda_IsaacLab's ──
_resfit_tasks_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "isaaclab", "source", "isaaclab_tasks")
_resfit_tasks_path = os.path.abspath(_resfit_tasks_path)
if os.path.isdir(_resfit_tasks_path) and _resfit_tasks_path not in sys.path:
    sys.path.insert(0, _resfit_tasks_path)

# ── CRITICAL: AppLauncher must run BEFORE any isaaclab imports ──
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Residual TD3 on IsaacLab + GR00T")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--total_timesteps", type=int, default=300_000)
parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
parser.add_argument("--gr00t_port", type=int, default=5555)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb_mode", type=str, default="disabled")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import isaaclab / isaaclab_tasks ──
import hashlib
import json
import logging
import pprint
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import tensordict
import torch
import torchrl
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from resfit.rl_finetuning.config.residual_td3_isaaclab import ResidualTD3IsaacLabConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.offline_buffer_csv import populate_offline_buffer_from_csv
from resfit.rl_finetuning.wrappers.isaaclab_env_wrapper import IsaacLabVecEnvWrapper, create_isaaclab_env
from resfit.rl_finetuning.wrappers.isaaclab_residual_wrapper import IsaacLabResidualWrapper
from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Timing utility (same as original) ──
class TrainingTimer:
    def __init__(self):
        self.times = defaultdict(list)

    @contextmanager
    def time(self, stage_name: str):
        start = time.perf_counter()
        yield
        self.times[stage_name].append(time.perf_counter() - start)

    def get_timing_stats(self) -> dict[str, float]:
        total = sum(sum(t) for t in self.times.values())
        if total == 0:
            return {}
        stats = {}
        for name, tl in self.times.items():
            s = sum(tl)
            stats[f"timing/{name}_percentage"] = (s / total) * 100
            stats[f"timing/{name}_avg_ms"] = (s / len(tl)) * 1000 if tl else 0
        return stats

    def reset(self):
        self.times = defaultdict(list)


# ── Replay buffer helper ──
def _add_transitions(
    obs, next_obs, actions, reward, done, info,
    device, image_keys, lowdim_keys, num_envs, online_rb,
):
    obs_keys = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        if done[i] and "final_obs" in info and info["final_obs"] is not None:
            try:
                next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in info["final_obs"][i].items()}
            except Exception:
                next_obs_i = {k: v[i] for k, v in next_obs.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items() if k in obs_keys}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys}
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
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        }, batch_size=[]).unsqueeze(0)
        online_rb.add(td)


# ── Main ──
def main(cfg: ResidualTD3IsaacLabConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Seed
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    print(f"Seed: {cfg.seed}")

    # ── Create IsaacLab env ──
    ecfg = cfg.isaaclab_env
    extra_overrides = {
        "use_api_for_pose": True,
        "gr00t_host": cfg.groot_policy.host,
        "gr00t_port": cfg.groot_policy.port,
    }
    # Always provide csv_base_dir for reset initialization
    if ecfg.csv_base_dir:
        extra_overrides["csv_base_dir"] = ecfg.csv_base_dir
        extra_overrides["csv_init_row_index"] = ecfg.csv_init_row_index
    else:
        extra_overrides["enforce_csv_reset_init"] = False

    isaac_env = create_isaaclab_env(
        task=ecfg.task,
        num_envs=ecfg.num_envs,
        enable_cameras=ecfg.enable_cameras,
        device=ecfg.device,
        image_size=(ecfg.image_size_h, ecfg.image_size_w),
        extra_cfg_overrides=extra_overrides,
    )

    # ── Create GR00T base policy ──
    gcfg = cfg.groot_policy
    groot = GR00TBasePolicy(
        host=gcfg.host,
        port=gcfg.port,
        num_envs=ecfg.num_envs,
        device=cfg.device,
        task_description=gcfg.task_description,
        language_override=gcfg.language_override,
        action_horizon=gcfg.action_horizon,
    )

    # ── Wrap with residual wrapper ──
    env = IsaacLabResidualWrapper(
        vec_env=isaac_env,
        base_policy=groot,
    )

    # ── Dimensions ──
    image_keys = list(cfg.rl_camera)
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    lowdim_keys = ["observation.state", "observation.base_action"]
    num_envs = ecfg.num_envs

    print(f"lowdim_dim={lowdim_dim}, img=({img_c},{img_h},{img_w}), action_dim={action_dim}")

    # ── QAgent ──
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    # ── Replay buffers ──
    acfg = cfg.algo
    online_batch_size = int(acfg.batch_size * (1 - acfg.offline_fraction))
    offline_batch_size = int(acfg.batch_size * acfg.offline_fraction)

    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=acfg.buffer_size, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=acfg.n_step, gamma=acfg.gamma),
        pin_memory=True,
        batch_size=max(online_batch_size, 1),
    )

    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max(1, acfg.buffer_size), device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=acfg.n_step, gamma=acfg.gamma),
        pin_memory=True,
        batch_size=max(offline_batch_size, 1),
    )

    # ── Populate offline buffer from CSV episodes ──
    if cfg.offline_data is not None and acfg.offline_fraction > 0.0:
        print("Populating offline buffer from CSV episodes...")
        n_offline = populate_offline_buffer_from_csv(
            data_dir=cfg.offline_data.csv_data_dir,
            rb=offline_rb,
            split_file=cfg.offline_data.split_file if cfg.offline_data.split_file else None,
            split=cfg.offline_data.split,
            image_keys=image_keys,
            image_size=(ecfg.image_size_h, ecfg.image_size_w),
            max_episodes=cfg.offline_data.num_episodes,
        )
        print(f"Offline buffer: {n_offline} transitions, size={len(offline_rb)}")
    else:
        print("Skipping offline buffer (offline_fraction=0 or no offline_data config)")

    # ── Warm-up ──
    print(f"Warm-up: filling online buffer with {acfg.learning_starts} random steps...")
    obs, _ = env.reset()
    while len(online_rb) < acfg.learning_starts:
        if acfg.use_base_policy_for_warmup:
            rand_actions = (
                torch.rand((num_envs, action_dim), device=device) * 2 - 1
            ) * acfg.random_action_noise_scale
        else:
            base_action = obs["observation.base_action"]
            pure_random = (
                torch.rand((num_envs, action_dim), device=device) * 2 - 1
            ) * acfg.random_action_noise_scale
            rand_actions = pure_random - base_action

        next_obs, reward, terminated, truncated, info = env.step(rand_actions)
        done = terminated | truncated
        combined_action = info.get("scaled_action", rand_actions)

        _add_transitions(
            obs=obs, next_obs=next_obs, actions=combined_action,
            reward=reward, done=done, info=info, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        obs = next_obs

        if len(online_rb) % 1000 == 0:
            print(f"  warm-up: {len(online_rb)}/{acfg.learning_starts}")

    print(f"Warm-up done. Online buffer: {len(online_rb)}")

    # ── Critic warmup ──
    if acfg.critic_warmup_steps > 0:
        print(f"Critic warmup: {acfg.critic_warmup_steps} steps...")
        for i in range(acfg.critic_warmup_steps):
            batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
            if offline_batch_size > 0 and len(offline_rb) > 0:
                obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                batch = torch.cat([batch, obatch], dim=0)
            agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if i % 500 == 0:
                print(f"  critic warmup: {i}/{acfg.critic_warmup_steps}")
        print("Critic warmup done.")

    # ── W&B ──
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_isaaclab_residual_td3_seed{cfg.seed}"
    if wandb is not None:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, '__dataclass_fields__') else {},
            name=run_name,
            mode=cfg.wandb.mode,
        )

    # ── Main training loop ──
    obs, _ = env.reset()
    global_step = 0
    episode_count = 0
    best_success = 0.0
    timer = TrainingTimer()
    train_start = time.time()
    actor_updates = 0

    print(f"Starting training for {acfg.total_timesteps} steps...")

    while global_step <= acfg.total_timesteps:
        # (1) Collect
        with timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(acfg.stddev_schedule, global_step)
                action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

            if acfg.progressive_clipping_steps > 0:
                clip_factor = min(1.0, global_step / acfg.progressive_clipping_steps)
                action = action * clip_factor

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

        if done.any():
            episode_count += done.float().sum().item()

        combined_action = info.get("scaled_action", action)
        _add_transitions(
            obs=obs, next_obs=next_obs, actions=combined_action,
            reward=reward, done=done, info=info, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        obs = next_obs
        global_step += num_envs

        # (2) Update
        if global_step % acfg.update_every_n_steps == 0 or global_step == num_envs:
            actor_cadence = max(1, acfg.num_updates_per_iteration // acfg.actor_updates_per_iteration)
            for i in range(acfg.num_updates_per_iteration):
                with timer.time("batch_sampling"):
                    batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
                    if offline_batch_size > 0 and len(offline_rb) > 0:
                        obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                        batch = torch.cat([batch, obatch], dim=0)

                update_actor = (i + 1) % actor_cadence == 0
                if update_actor:
                    actor_updates += 1

                with timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

        # (3) Logging
        if global_step % 500 == 0:
            sps = int(global_step / (time.time() - train_start)) if (time.time() - train_start) > 0 else 0
            ts = timer.get_timing_stats()
            env_pct = ts.get("timing/env_step_percentage", 0)
            grad_pct = ts.get("timing/gradient_update_percentage", 0)

            msg = (
                f"[{global_step}] episodes={int(episode_count)} SPS={sps} "
                f"critic_loss={metrics.get('train/critic_loss', 0):.4f} "
                f"| time%: env={env_pct:.1f} grad={grad_pct:.1f}"
            )
            if "train/actor_loss_base" in metrics:
                msg += f" actor_loss={metrics['train/actor_loss_base']:.4f}"
            print(msg)

            if wandb is not None and wandb.run is not None:
                log_dict = {
                    "training/global_step": global_step,
                    "training/SPS": sps,
                    "training/episode_count": episode_count,
                    "buffer/online_size": len(online_rb),
                }
                log_dict.update({k: v for k, v in metrics.items() if not k.startswith("_")})
                log_dict.update(ts)
                wandb.log(log_dict, step=global_step)

    total_time = time.time() - train_start
    print(f"Training finished in {total_time:.1f}s ({global_step} steps, {episode_count} episodes)")

    env.close()
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    cfg = ResidualTD3IsaacLabConfig()
    cfg.isaaclab_env.num_envs = args_cli.num_envs
    cfg.num_envs = args_cli.num_envs
    cfg.algo.total_timesteps = args_cli.total_timesteps
    cfg.groot_policy.host = args_cli.gr00t_host
    cfg.groot_policy.port = args_cli.gr00t_port
    cfg.seed = args_cli.seed
    cfg.device = args_cli.device
    cfg.wandb.mode = args_cli.wandb_mode

    main(cfg)
    simulation_app.close()
