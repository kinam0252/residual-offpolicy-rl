"""
Hydra-structured config for Residual TD3 on MuJoCo + GR00T.

Mirrors ``residual_td3_isaaclab.py`` but replaces IsaacLab environment
config with MuJoCo-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resfit.rl_finetuning.config.residual_td3 import (
    OfflineDataConfig,
    ResidualTD3AlgoConfig,
    WandBConfig,
)
from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig


# ── GR00T base policy config (same as IsaacLab version) ──
@dataclass
class GR00TConfig:
    checkpoint_path: str = ""
    embodiment_tag: str = "NEW_EMBODIMENT"
    task_description: str = "lift the cube"
    action_horizon: int = 16
    open_loop_horizon: int = 16
    policy_device: str = "cuda:0"


# ── MuJoCo environment config ──
@dataclass
class MuJoCoEnvConfig:
    num_envs: int = 1
    # Cube
    cube_pos: list[float] = field(default_factory=lambda: [0.45, -0.05, 0.02])
    cube_yaw_deg: float = 0.0
    cube_size: list[float] = field(default_factory=lambda: [0.06, 0.02, 0.02])
    # Scene
    scene_xml: str | None = None
    calib_path: str | None = None
    # Episode
    max_episode_steps: int = 300
    success_threshold: float = 0.005  # metres
    reward_type: str = "sparse"  # "sparse" | "dense"
    # Rendering
    rl_img_size: int = 84
    device: str = "cuda:0"
    # Residual scales
    residual_pos_scale: float = 0.02   # metres
    residual_rot_scale: float = 0.05   # radians
    residual_grip_scale: float = 0.1


# ── Offline data config ──
@dataclass
class MuJoCoOfflineDataConfig:
    csv_data_dir: str = ""
    num_episodes: int | None = None


# ── Top-level experiment config ──
@dataclass
class ResidualTD3MuJoCoConfig:
    """Top-level config for residual TD3 on MuJoCo + GR00T."""

    seed: int | None = None
    device: str = "cuda:0"

    # Environment
    mujoco_env: MuJoCoEnvConfig = field(default_factory=MuJoCoEnvConfig)

    # Base policy (GR00T)
    groot: GR00TConfig = field(default_factory=GR00TConfig)

    # Algorithm
    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=50_000,
            batch_size=64,
            buffer_size=50_000,
            learning_starts=1_000,
            gamma=0.99,
            n_step=3,
            offline_fraction=0.0,  # no offline data by default
            critic_warmup_steps=1_000,
            use_base_policy_for_warmup=True,
            random_action_noise_scale=0.1,
            stddev_max=0.05,
            stddev_min=0.05,
        )
    )

    # Network
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=3e-7,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            clip_q_target_to_reward_range=True,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,
                action_l2_reg_weight=10.0,
            ),
        )
    )

    # RL cameras
    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.front",
            "observation.images.back",
            "observation.images.wrist",
        ]
    )
    camera_size: int = 84

    # Offline data
    offline_data: MuJoCoOfflineDataConfig | None = None

    # Evaluation
    eval_interval_every_steps: int = 5_000
    eval_num_episodes: int = 10
    eval_first: bool = False

    # Output
    output_dir: str = "outputs"

    # W&B
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="mujoco-franka-residual-td3",
            mode="online",
        )
    )
