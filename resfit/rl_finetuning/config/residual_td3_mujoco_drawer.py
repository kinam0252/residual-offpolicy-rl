"""
Config for Residual TD3 on MuJoCo Close Drawer + GR00T.

Follows cube lift config structure but adapted for drawer task:
- State-only (no images for actor/critic)
- ActionScaler enabled
- 6D action (pos3 + rot3, grip disabled)
- Drawer face center in observation.object_state
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resfit.rl_finetuning.config.residual_td3 import (
    ResidualTD3AlgoConfig,
    WandBConfig,
)
from resfit.rl_finetuning.config.rlpd import ActorConfig, CriticConfig, QAgentConfig


@dataclass
class DrawerEnvConfig:
    num_envs: int = 1
    active_drawers: list[int] = field(default_factory=lambda: [2, 3, 4])
    contact_z_gate: bool = True
    max_episode_steps: int = 500
    reward_type: str = "dense"
    device: str = "cuda:0"
    rl_img_size: int = 84
    # Residual scales
    residual_pos_scale: float = 0.02
    residual_rot_scale: float = 0.05
    residual_grip_scale: float = 0.004  # effectively disabled (very small)


@dataclass
class ResidualTD3MuJoCoDrawerConfig:
    """Top-level config for residual TD3 on MuJoCo Close Drawer + GR00T."""

    seed: int | None = None
    device: str = "cuda:0"

    # Environment
    drawer_env: DrawerEnvConfig = field(default_factory=DrawerEnvConfig)

    # Algorithm (same hyperparams as cube lift)
    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=50_000,
            batch_size=256,
            buffer_size=200_000,
            learning_starts=1_000,
            gamma=0.99,
            n_step=3,
            offline_fraction=0.3,
            critic_warmup_steps=1_000,
            use_base_policy_for_warmup=True,
            random_action_noise_scale=0.0,
            stddev_max=0.05,
            stddev_min=0.05,
        )
    )

    # Network
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-5,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            clip_q_target_to_reward_range=True,
            actor=ActorConfig(
                action_scale=0.1,
                hidden_dim=512,
                actor_last_layer_init_scale=0.0,
                action_l2_reg_weight=10.0,
            ),
            critic=CriticConfig(
                hidden_dim=1024,
            ),
        )
    )

    # Evaluation
    eval_interval_every_steps: int = 5_000
    eval_num_episodes: int = 10

    # Output
    output_dir: str = "outputs"

    # W&B
    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(
            project="mujoco-drawer-residual-td3",
            mode="online",
        )
    )
