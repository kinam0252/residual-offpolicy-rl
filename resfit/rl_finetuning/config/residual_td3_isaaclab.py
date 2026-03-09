"""
Hydra-structured config for Residual TD3 on IsaacLab + GR00T.

Inherits from the existing ``ResidualTD3DexmgConfig`` but overrides
environment, camera, and base-policy settings for the Franka Pickup task
with GR00T as the base policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.residual_td3 import (
    OfflineDataConfig,
    ResidualTD3AlgoConfig,
    ResidualTD3DexmgConfig,
    WandBConfig,
)
from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig


# ── GR00T-specific base policy config (replaces W&B ACT download) ──
@dataclass
class GR00TBasePolicyConfig:
    host: str = "127.0.0.1"
    port: int = 5555
    task_description: str = "Pick up the white object."
    language_override: str | None = None
    action_horizon: int = 16
    model_path: str | None = None
    embodiment_tag: str = "new_embodiment"
    strict: bool = False
    policy_device: str | None = None


# ── IsaacLab environment config ──
@dataclass
class IsaacLabEnvConfig:
    task: str = "Isaac-Franka-Pickup-Direct-v0"
    num_envs: int = 1
    enable_cameras: bool = True
    image_size_h: int = 84
    image_size_w: int = 84
    device: str = "cuda:0"
    # CSV for initial pose
    csv_base_dir: str = "/home/kinam/Desktop/DATA/dataset_from_Namiko/0_Raw_dataset/pickMushroom/pickMushroom_20251209_082448_104"
    csv_init_row_index: int = 0
    # Episode
    max_episode_steps: int = 1000


# ── Offline data (CSV episodes or local LeRobot-format dataset folder) ──
@dataclass
class IsaacLabOfflineDataConfig:
    csv_data_dir: str = "/home/t-kinamkim/Repos/VLA_RL/Data/lerobot/pickMushroom_train_libero_replay_poseinit"
    split_file: str = ""
    split: str = "training"  # "training" or "testing"
    num_episodes: int | None = None  # None = use all


# ── Top-level experiment config ──
@dataclass
class ResidualTD3IsaacLabConfig:
    """Top-level config for residual TD3 on IsaacLab + GR00T."""

    # ------------------------------------------------------------------
    # Task / environment
    # ------------------------------------------------------------------
    isaaclab_env: IsaacLabEnvConfig = field(default_factory=IsaacLabEnvConfig)

    # ------------------------------------------------------------------
    # Base policy (GR00T)
    # ------------------------------------------------------------------
    groot_policy: GR00TBasePolicyConfig = field(default_factory=GR00TBasePolicyConfig)

    # ------------------------------------------------------------------
    # Algorithm & optimisation
    # ------------------------------------------------------------------
    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=300_000,
            batch_size=256,
            buffer_size=200_000,
            learning_starts=5_000,
            gamma=0.99,
            n_step=3,
            offline_fraction=0.0,
            critic_warmup_steps=5_000,
            use_base_policy_for_warmup=True,
            random_action_noise_scale=0.1,
            stddev_max=0.05,
            stddev_min=0.05,
        )
    )

    # ------------------------------------------------------------------
    # Network architectures
    # ------------------------------------------------------------------
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,
            ),
        )
    )

    # ------------------------------------------------------------------
    # Cameras for RL encoder
    # ------------------------------------------------------------------
    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.front",
            "observation.images.back",
            "observation.images.wrist",
        ]
    )
    camera_size: int = 84

    # ------------------------------------------------------------------
    # Offline data (CSV episodes)
    # ------------------------------------------------------------------
    offline_data: IsaacLabOfflineDataConfig | None = field(
        default_factory=IsaacLabOfflineDataConfig
    )

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="isaaclab-franka-pickup-residual-td3",
            mode="online",
        )
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_interval_every_steps: int = 10_000
    eval_episodes: int = 20
    eval_first: bool = True

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 50_000

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    device: str = "cuda:0"
    seed: int = 42
    num_envs: int = 1


# ── Register with Hydra ──
cs = ConfigStore.instance()
cs.store(name="residual_td3_isaaclab_config", node=ResidualTD3IsaacLabConfig)
