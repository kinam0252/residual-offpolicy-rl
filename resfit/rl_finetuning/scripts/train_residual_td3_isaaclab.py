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
parser.add_argument("--learning_starts", type=int, default=None)
parser.add_argument("--critic_warmup_steps", type=int, default=None)
parser.add_argument("--update_every_n_steps", type=int, default=None)
parser.add_argument("--num_updates_per_iteration", type=int, default=None)
parser.add_argument("--offline_fraction", type=float, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--buffer_size", type=int, default=None)
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--n_step", type=int, default=None)
parser.add_argument("--stddev_max", type=float, default=None)
parser.add_argument("--stddev_min", type=float, default=None)
parser.add_argument("--actor_lr", type=float, default=None)
parser.add_argument("--action_scale", type=float, default=None)
parser.add_argument("--random_action_noise_scale", type=float, default=None)
parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
parser.add_argument("--gr00t_port", type=int, default=5555)
parser.add_argument("--groot_model_path", type=str, default=None)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default=None)
parser.add_argument("--groot_policy_strict", action="store_true")
parser.add_argument("--task_description", type=str, default=None)
parser.add_argument("--language_override", type=str, default=None)
parser.add_argument("--csv_base_dir", type=str, default=None)
parser.add_argument("--csv_init_row_index", type=int, default=None)
parser.add_argument("--max_episode_steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb_mode", type=str, default="disabled")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--wandb_notes", type=str, default=None)
parser.add_argument("--wandb_continue_run_id", type=str, default=None)
parser.add_argument("--wandb_log_every_steps", type=int, default=10)
parser.add_argument("--eval_interval_every_steps", type=int, default=None)
parser.add_argument("--eval_num_episodes", type=int, default=None)
parser.add_argument("--eval_first", action="store_true")
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--heartbeat_interval_sec", type=int, default=20)
parser.add_argument("--stack_dump_interval_sec", type=int, default=180)
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
import threading
import time
import faulthandler
from dataclasses import asdict, is_dataclass
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
from resfit.rl_finetuning.utils.offline_buffer_lerobot_local import populate_offline_buffer_from_lerobot_local
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.iface_eval_runner import run_iface_evaluation
from resfit.rl_finetuning.wrappers.isaaclab_env_wrapper import IsaacLabVecEnvWrapper, create_isaaclab_env
from resfit.rl_finetuning.wrappers.isaaclab_groot_rollout_env import IsaacLabGrootRolloutEnv
from resfit.rl_finetuning.wrappers.isaaclab_residual_wrapper import IsaacLabResidualWrapper
from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_PHASE = "startup"
_PHASE_SINCE = time.time()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


def _set_phase(phase: str):
    global _PHASE, _PHASE_SINCE
    _PHASE = phase
    _PHASE_SINCE = time.time()
    _log(f"PHASE -> {phase}")


def _start_phase_heartbeat(interval_sec: int):
    interval_sec = max(1, int(interval_sec))

    def _worker():
        while True:
            elapsed = time.time() - _PHASE_SINCE
            _log(f"heartbeat phase={_PHASE} elapsed={elapsed:.1f}s")
            time.sleep(interval_sec)

    t = threading.Thread(target=_worker, name="phase-heartbeat", daemon=True)
    t.start()


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
            stats[f"timing/{name}_total_s"] = s
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
    heartbeat_interval_sec = max(1, int(getattr(cfg, "heartbeat_interval_sec", 20)))
    stack_dump_interval_sec = int(getattr(cfg, "stack_dump_interval_sec", 180))

    if stack_dump_interval_sec > 0:
        try:
            faulthandler.enable()
            faulthandler.dump_traceback_later(stack_dump_interval_sec, repeat=True)
            _log(f"Enabled periodic stack dump every {stack_dump_interval_sec}s")
        except Exception as err:
            _log(f"Failed to enable faulthandler: {err}")

    _start_phase_heartbeat(heartbeat_interval_sec)
    _set_phase("setup")

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
    _log(f"Seed: {cfg.seed}")

    # ── Create IsaacLab env ──
    ecfg = cfg.isaaclab_env
    if not cfg.groot_policy.model_path:
        raise ValueError(
            "Local GR00T mode requires --groot_model_path. "
            "Server mode is disabled for this training entrypoint."
        )
    extra_overrides = {
        "use_api_for_pose": False,
        "gr00t_host": cfg.groot_policy.host,
        "gr00t_port": cfg.groot_policy.port,
    }
    # Always provide csv_base_dir for reset initialization
    if ecfg.csv_base_dir:
        extra_overrides["csv_base_dir"] = ecfg.csv_base_dir
        extra_overrides["csv_init_row_index"] = ecfg.csv_init_row_index
    else:
        extra_overrides["enforce_csv_reset_init"] = False

    _set_phase("create_isaaclab_env")
    _log("Creating IsaacLab env...")
    isaac_env = create_isaaclab_env(
        task=ecfg.task,
        num_envs=ecfg.num_envs,
        enable_cameras=ecfg.enable_cameras,
        device=ecfg.device,
        image_size=(ecfg.image_size_h, ecfg.image_size_w),
        extra_cfg_overrides=extra_overrides,
    )
    _set_phase("isaaclab_env_ready")
    _log("IsaacLab env ready.")

    # ── Create GR00T base policy ──
    gcfg = cfg.groot_policy
    groot_policy_device = gcfg.policy_device if gcfg.policy_device else cfg.device
    _set_phase("init_groot_base_policy")
    _log("Initializing GR00T base policy...")
    groot = GR00TBasePolicy(
        host=gcfg.host,
        port=gcfg.port,
        num_envs=ecfg.num_envs,
        device=groot_policy_device,
        task_description=gcfg.task_description,
        language_override=gcfg.language_override,
        action_horizon=gcfg.action_horizon,
        model_path=gcfg.model_path,
        embodiment_tag=gcfg.embodiment_tag,
        strict=gcfg.strict,
    )
    _set_phase("groot_base_policy_ready")
    _log("GR00T base policy ready.")

    # ── Wrap with residual wrapper ──
    _set_phase("build_residual_wrapper")
    _log("Building residual wrapper...")
    env = IsaacLabResidualWrapper(
        vec_env=isaac_env,
        base_policy=groot,
    )
    eval_env = IsaacLabGrootRolloutEnv(
        vec_env=isaac_env,
        base_policy=groot,
    )
    _set_phase("residual_wrapper_ready")
    _log("Residual wrapper ready. Eval uses rollout wrapper parity path.")

    # ── Dimensions ──
    image_keys = list(cfg.rl_camera)
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    lowdim_keys = ["observation.state", "observation.base_action"]
    num_envs = ecfg.num_envs

    _log(f"lowdim_dim={lowdim_dim}, img=({img_c},{img_h},{img_w}), action_dim={action_dim}")

    # ── QAgent ──
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_isaaclab_residual_td3_seed{cfg.seed}"
    if getattr(cfg.wandb, "name", None):
        run_name = f"{cfg.wandb.name}__{run_name}"

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

    def _make_offline_rb(max_size: int):
        return TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=max(1, int(max_size)), device="cpu"),
            alpha=0.0, beta=0.0, eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=acfg.n_step, gamma=acfg.gamma),
            pin_memory=True,
            batch_size=max(offline_batch_size, 1),
        )

    offline_rb = _make_offline_rb(acfg.buffer_size)

    # ── Populate offline buffer from CSV episodes ──
    if cfg.offline_data is not None and acfg.offline_fraction > 0.0:
        offline_root = Path(cfg.offline_data.csv_data_dir)
        is_lerobot_local = (offline_root / "data" / "chunk-000").exists() and (offline_root / "videos" / "chunk-000").exists()

        def _populate_current_offline_rb():
            if is_lerobot_local:
                _log(f"Populating offline buffer from LeRobot local dataset: {offline_root}")
                return populate_offline_buffer_from_lerobot_local(
                    data_dir=offline_root,
                    rb=offline_rb,
                    image_keys=image_keys,
                    image_size=(ecfg.image_size_h, ecfg.image_size_w),
                    max_episodes=cfg.offline_data.num_episodes,
                )

            _log("Populating offline buffer from CSV episodes...")
            return populate_offline_buffer_from_csv(
                data_dir=cfg.offline_data.csv_data_dir,
                rb=offline_rb,
                split_file=cfg.offline_data.split_file if cfg.offline_data.split_file else None,
                split=cfg.offline_data.split,
                image_keys=image_keys,
                image_size=(ecfg.image_size_h, ecfg.image_size_w),
                max_episodes=cfg.offline_data.num_episodes,
            )

        n_offline = _populate_current_offline_rb()

        n_step_drop_tolerance = max(0, int(acfg.n_step) - 1)
        expected_min_size = max(0, int(n_offline) - n_step_drop_tolerance)
        if len(offline_rb) < expected_min_size:
            _log(
                f"Offline buffer truncated ({len(offline_rb)} < {expected_min_size}, "
                f"raw={n_offline}, n_step={acfg.n_step}); "
                f"rebuilding storage to max_size={n_offline} and repopulating."
            )
            offline_rb = _make_offline_rb(n_offline)
            n_offline = _populate_current_offline_rb()

        _log(f"Offline buffer: {n_offline} transitions, size={len(offline_rb)}")
    else:
        _log("Skipping offline buffer (offline_fraction=0 or no offline_data config)")

    # ── W&B ──
    if wandb is not None:
        if is_dataclass(cfg):
            wandb_cfg = asdict(cfg)
        else:
            try:
                wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                wandb_cfg = {}
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            id=cfg.wandb.continue_run_id,
            resume=None if cfg.wandb.continue_run_id is None else "allow",
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_cfg,
            name=run_name,
            mode=cfg.wandb.mode,
            notes=cfg.wandb.notes,
            group=cfg.wandb.group,
            reinit=True,
        )

    # ── Warm-up ──
    _set_phase("warmup")
    _log(f"Warm-up: filling online buffer with {acfg.learning_starts} random steps...")
    warmup_last_log = time.time()
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
            _log(f"warm-up progress: {len(online_rb)}/{acfg.learning_starts}")
        now = time.time()
        if now - warmup_last_log >= heartbeat_interval_sec:
            _log(f"heartbeat warm-up: online_rb={len(online_rb)}/{acfg.learning_starts}")
            warmup_last_log = now

    _log(f"Warm-up done. Online buffer: {len(online_rb)}")

    # ── Critic warmup ──
    if acfg.critic_warmup_steps > 0:
        _log(f"Critic warmup: {acfg.critic_warmup_steps} steps...")
        critic_last_log = time.time()
        for i in range(acfg.critic_warmup_steps):
            batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
            if offline_batch_size > 0 and len(offline_rb) > 0:
                obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                batch = torch.cat([batch, obatch], dim=0)
            agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if i % 500 == 0:
                _log(f"critic warmup progress: {i}/{acfg.critic_warmup_steps}")
            now = time.time()
            if now - critic_last_log >= heartbeat_interval_sec:
                _log(f"heartbeat critic warmup: {i}/{acfg.critic_warmup_steps}")
                critic_last_log = now
        _log("Critic warmup done.")

    # ── Main training loop ──
    _set_phase("training_loop")
    obs, _ = env.reset()
    global_step = 0
    episode_count = 0
    best_success = 0.0
    eval_metrics = {}
    timer = TrainingTimer()
    train_start = time.time()
    actor_updates = 0
    training_cum_time = 0.0
    outputs_dir = Path(getattr(cfg, "output_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    last_eval_video_path = None

    _log(f"Starting training for {acfg.total_timesteps} steps...")
    train_last_log = time.time()
    wandb_log_every_steps = max(1, int(getattr(cfg, "wandb_log_every_steps", 10)))

    while global_step <= acfg.total_timesteps:
        iter_start = time.time()
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
            if wandb is not None and wandb.run is not None:
                done_rewards = reward[done]
                episode_return = float(done_rewards.mean().detach().cpu().item()) if done_rewards.numel() > 0 else 0.0
                episode_steps = 0.0
                try:
                    final_info = info.get("final_info", None) if isinstance(info, dict) else None
                    if final_info is not None:
                        if (
                            isinstance(final_info, dict)
                            and "episode_steps" in final_info
                            and "_episode_steps" in final_info
                            and done_rewards.numel() > 0
                        ):
                            all_steps = torch.as_tensor(final_info["episode_steps"]).detach().cpu().reshape(-1)
                            episode_indices = torch.as_tensor(final_info["_episode_steps"]).detach().cpu().reshape(-1).bool()
                            if all_steps.numel() == episode_indices.numel() and episode_indices.any():
                                selected_steps = all_steps[episode_indices].float()
                                selected_rewards = reward.detach().cpu().reshape(-1)[episode_indices].float()
                                discount_factor = torch.pow(torch.full_like(selected_steps, float(acfg.gamma)), selected_steps)
                                episode_return = float((discount_factor * selected_rewards).mean().item())
                                episode_steps = float(selected_steps.mean().item())
                        elif isinstance(final_info, (list, tuple)) and done_rewards.numel() > 0:
                            done_indices = torch.where(done)[0].detach().cpu().tolist()
                            selected_steps_list = []
                            for env_idx in done_indices:
                                if env_idx < len(final_info):
                                    item = final_info[env_idx]
                                    if isinstance(item, dict) and "episode_steps" in item:
                                        selected_steps_list.append(float(item["episode_steps"]))
                            if len(selected_steps_list) == int(done_rewards.numel()) and selected_steps_list:
                                selected_steps = np.asarray(selected_steps_list, dtype=np.float32)
                                selected_rewards = done_rewards.detach().cpu().numpy().astype(np.float32)
                                discount_factor = np.power(float(acfg.gamma), selected_steps)
                                episode_return = float(np.mean(discount_factor * selected_rewards))
                                episode_steps = float(np.mean(selected_steps))
                except Exception:
                    pass

                wandb.log(
                    {
                        "training/episode_return": episode_return,
                        "training/episode_steps": episode_steps,
                        "training/episode_count": episode_count,
                    },
                    step=global_step,
                )

        combined_action = info.get("scaled_action", action)
        _add_transitions(
            obs=obs, next_obs=next_obs, actions=combined_action,
            reward=reward, done=done, info=info, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        obs = next_obs

        # (2) Periodic evaluation (same condition pattern as original resfit)
        eval_interval = max(1, int(cfg.eval_interval_every_steps))
        should_eval_now = (global_step % eval_interval == 0) and (bool(cfg.eval_first) or global_step > 0)
        if should_eval_now:
            _set_phase("evaluation")
            _log(
                f"Eval start: episodes={int(cfg.eval_episodes)}, "
                f"save_video={bool(getattr(cfg, 'save_video', False))}"
            )
            eval_video_path = None
            with timer.time("evaluation"):
                if bool(getattr(cfg, "eval_use_iface_runner", False)):
                    eval_metrics, eval_video_path = run_iface_evaluation(
                        csv_base_dir=str(ecfg.csv_base_dir),
                        groot_model_path=str(gcfg.model_path),
                        num_episodes=int(cfg.eval_episodes),
                        max_steps=int(getattr(cfg.isaaclab_env, "max_episode_steps", 1000)),
                        output_dir=outputs_dir,
                        run_name=run_name,
                        global_step=global_step,
                        num_envs=int(ecfg.num_envs),
                        policy_device=gcfg.policy_device,
                        embodiment_tag=str(gcfg.embodiment_tag),
                        task_description=str(gcfg.task_description),
                        language_override=gcfg.language_override,
                        headless=bool(getattr(args_cli, "headless", True)),
                        save_video=bool(getattr(cfg, "save_video", False)),
                    )
                else:
                    eval_metrics = run_dexmg_evaluation(
                        env=eval_env,
                        agent=agent,
                        num_episodes=int(cfg.eval_episodes),
                        device=device,
                        global_step=global_step,
                        save_video=bool(getattr(cfg, "save_video", False)),
                        save_q_plots=bool(getattr(cfg, "save_video", False)),
                        run_name=run_name,
                        output_dir=outputs_dir,
                        force_zero_residual=False,
                        max_steps_per_episode=int(getattr(cfg.isaaclab_env, "max_episode_steps", 1000)),
                    )
            current_success = float(eval_metrics.get("eval/success_rate", 0.0))
            last_eval_video_path = eval_video_path
            if current_success > best_success:
                _log(f"🎉 New best success rate: {current_success:.4f} (prev: {best_success:.4f})")
                best_success = current_success
            _log(
                f"Eval done: success_rate={float(eval_metrics.get('eval/success_rate', 0.0)):.3f} "
                f"mean_return={float(eval_metrics.get('eval/mean_return', 0.0)):.3f}"
            )
            obs, _ = env.reset()
            _set_phase("training_loop")

        global_step += num_envs

        # (3) Update
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
                    actor_lr_warmup_steps = int(getattr(acfg, "actor_lr_warmup_steps", 0))
                    if actor_lr_warmup_steps > 0:
                        warmup_progress = min(1.0, actor_updates / max(1, actor_lr_warmup_steps))
                        current_lr = float(cfg.agent.actor_lr) * warmup_progress
                        for param_group in agent.actor_opt.param_groups:
                            param_group["lr"] = current_lr
                    actor_updates += 1

                with timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

        training_cum_time += time.time() - iter_start

        # (4) Logging
        should_log_console = (global_step % 500 == 0)
        should_log_wandb = (
            wandb is not None
            and wandb.run is not None
            and (global_step % wandb_log_every_steps == 0)
        )

        if should_log_console or should_log_wandb:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0
            ts = timer.get_timing_stats()
            env_pct = ts.get("timing/env_step_percentage", 0)
            grad_pct = ts.get("timing/gradient_update_percentage", 0)

            if should_log_console:
                msg = (
                    f"[{global_step}] episodes={int(episode_count)} SPS={sps} "
                    f"critic_loss={metrics.get('train/critic_loss', 0):.4f} "
                    f"| time%: env={env_pct:.1f} grad={grad_pct:.1f}"
                )
                if "train/actor_loss_base" in metrics:
                    msg += f" actor_loss={metrics['train/actor_loss_base']:.4f}"
                _log(msg)

            if should_log_wandb:
                log_dict = {
                    "training/global_step": global_step,
                    "training/SPS": sps,
                    "training/episode_count": episode_count,
                    "buffer/online_size": len(online_rb),
                    "buffer/offline_size": len(offline_rb),
                    "timing/training_total_time": time.time() - train_start,
                    "timing/aggregate_steps_per_second": global_step / max(1e-6, (time.time() - train_start)),
                    "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                }
                log_dict.update({k: v for k, v in metrics.items() if not k.startswith("_")})
                if "_actions" in metrics:
                    actions = metrics["_actions"]
                    log_dict["train/residual_l1_magnitude"] = torch.mean(torch.abs(actions)).item()
                    log_dict["train/residual_l2_magnitude"] = torch.mean(torch.square(actions)).item()
                    log_dict["histograms/residual_actions"] = wandb.Histogram(actions.detach().cpu().numpy().reshape(-1))
                if "_target_q" in metrics:
                    target_q = metrics["_target_q"]
                    log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.detach().cpu().numpy().reshape(-1))
                if acfg.progressive_clipping_steps > 0:
                    log_dict["training/progressive_clipping_factor"] = min(1.0, global_step / acfg.progressive_clipping_steps)
                log_dict.update(eval_metrics)
                log_dict.update(ts)
                if should_eval_now and bool(getattr(cfg, "save_video", False)) and last_eval_video_path is not None and last_eval_video_path.exists():
                    log_dict["eval/video"] = wandb.Video(str(last_eval_video_path), format="mp4")
                wandb.log(log_dict, step=global_step)

        now = time.time()
        if now - train_last_log >= heartbeat_interval_sec:
            _log(
                "heartbeat train: "
                f"step={global_step}/{acfg.total_timesteps} "
                f"online_rb={len(online_rb)} episodes={int(episode_count)}"
            )
            train_last_log = now

    total_time = time.time() - train_start
    _log(f"Training finished in {total_time:.1f}s ({global_step} steps, {episode_count} episodes)")

    env.close()
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    cfg = ResidualTD3IsaacLabConfig()
    cfg.isaaclab_env.num_envs = args_cli.num_envs
    cfg.isaaclab_env.device = args_cli.device
    cfg.num_envs = args_cli.num_envs
    cfg.algo.total_timesteps = args_cli.total_timesteps
    if args_cli.learning_starts is not None:
        cfg.algo.learning_starts = args_cli.learning_starts
    if args_cli.critic_warmup_steps is not None:
        cfg.algo.critic_warmup_steps = args_cli.critic_warmup_steps
    if args_cli.update_every_n_steps is not None:
        cfg.algo.update_every_n_steps = args_cli.update_every_n_steps
    if args_cli.num_updates_per_iteration is not None:
        cfg.algo.num_updates_per_iteration = args_cli.num_updates_per_iteration
    if args_cli.offline_fraction is not None:
        cfg.algo.offline_fraction = args_cli.offline_fraction
    if args_cli.batch_size is not None:
        cfg.algo.batch_size = args_cli.batch_size
    if args_cli.buffer_size is not None:
        cfg.algo.buffer_size = args_cli.buffer_size
    if args_cli.gamma is not None:
        cfg.algo.gamma = args_cli.gamma
    if args_cli.n_step is not None:
        cfg.algo.n_step = args_cli.n_step
    if args_cli.stddev_max is not None:
        cfg.algo.stddev_max = args_cli.stddev_max
    if args_cli.stddev_min is not None:
        cfg.algo.stddev_min = args_cli.stddev_min
    if args_cli.random_action_noise_scale is not None:
        cfg.algo.random_action_noise_scale = args_cli.random_action_noise_scale
    if args_cli.actor_lr is not None:
        cfg.agent.actor_lr = args_cli.actor_lr
    if args_cli.action_scale is not None:
        cfg.agent.actor.action_scale = args_cli.action_scale
    cfg.groot_policy.host = args_cli.gr00t_host
    cfg.groot_policy.port = args_cli.gr00t_port
    cfg.groot_policy.model_path = args_cli.groot_model_path
    cfg.groot_policy.embodiment_tag = args_cli.groot_embodiment_tag
    cfg.groot_policy.policy_device = args_cli.groot_policy_device
    cfg.groot_policy.strict = args_cli.groot_policy_strict
    if args_cli.task_description is not None:
        cfg.groot_policy.task_description = args_cli.task_description
    if args_cli.language_override is not None:
        cfg.groot_policy.language_override = args_cli.language_override
    if args_cli.csv_base_dir is not None:
        cfg.isaaclab_env.csv_base_dir = args_cli.csv_base_dir
    if args_cli.csv_init_row_index is not None:
        cfg.isaaclab_env.csv_init_row_index = args_cli.csv_init_row_index
    if args_cli.max_episode_steps is not None:
        cfg.isaaclab_env.max_episode_steps = max(1, int(args_cli.max_episode_steps))
    cfg.seed = args_cli.seed
    cfg.device = args_cli.device
    cfg.wandb.mode = args_cli.wandb_mode
    if args_cli.wandb_project is not None:
        cfg.wandb.project = args_cli.wandb_project
    if args_cli.wandb_entity is not None:
        cfg.wandb.entity = args_cli.wandb_entity
    if args_cli.wandb_name is not None:
        cfg.wandb.name = args_cli.wandb_name
    if args_cli.wandb_group is not None:
        cfg.wandb.group = args_cli.wandb_group
    if args_cli.wandb_notes is not None:
        cfg.wandb.notes = args_cli.wandb_notes
    if args_cli.wandb_continue_run_id is not None:
        cfg.wandb.continue_run_id = args_cli.wandb_continue_run_id
    cfg.wandb_log_every_steps = max(1, int(args_cli.wandb_log_every_steps))
    if args_cli.eval_interval_every_steps is not None:
        cfg.eval_interval_every_steps = max(1, int(args_cli.eval_interval_every_steps))
    if args_cli.eval_num_episodes is not None:
        cfg.eval_episodes = max(1, int(args_cli.eval_num_episodes))
    if args_cli.eval_first:
        cfg.eval_first = True
    if args_cli.save_video:
        cfg.save_video = True
    if args_cli.output_dir is not None:
        cfg.output_dir = args_cli.output_dir
    cfg.heartbeat_interval_sec = args_cli.heartbeat_interval_sec
    cfg.stack_dump_interval_sec = args_cli.stack_dump_interval_sec

    main(cfg)
    simulation_app.close()
