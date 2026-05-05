"""
Task configurations for unified residual TD3 training.

Each task defines:
- vec_env import path and class name
- wrapper kwargs (grip range, latch, scales)
- camera keys for GR00T observation config shim
- default hyperparameters
- reward type choices
- env creation function
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    """Per-task configuration for the unified residual RL pipeline."""

    # Identity
    name: str
    task_description: str

    # VecEnv
    vec_env_module: str  # e.g. "resfit.rl_finetuning.wrappers.mujoco_vec_env_cup"
    vec_env_class: str   # e.g. "MuJoCoVecEnvCup"

    # Wrapper settings
    grip_min: float = 0.0
    grip_max: float = 1.0
    use_gripper_latch: bool = False
    residual_pos_scale: float = 0.02
    residual_rot_scale: float = 0.05
    residual_grip_scale: float = 0.1

    # Cameras (for FakeImageFeaturesConfig shim)
    camera_keys: dict = field(default_factory=lambda: {
        "observation.images.front": None,
        "observation.images.wrist": None,
    })

    # Training defaults
    default_num_envs: int = 15
    default_reward_type: str = "dense"
    reward_choices: list = field(default_factory=lambda: ["sparse", "dense"])
    default_max_episode_steps: int = 500
    default_gamma: float = 0.95
    default_action_scale: float = 0.1
    default_action_l2_reg: float = 1.0
    default_actor_lr: float = 3e-4
    default_critic_lr: float = 3e-4
    default_buffer_size: int = 500_000
    default_total_timesteps: int = 500_000
    default_actor_hidden_dim: int = 256
    default_critic_hidden_dim: int = 256

    # Offline data
    default_offline_fraction: float = 0.75

    # W&B
    wandb_project: str = "mujoco-franka-residual-td3"

    # GR00T
    groot_checkpoint_hint: str = ""

    # Object state
    object_state_dim: int = 0
    supports_asymmetric_critic: bool = False

    # Image keys for RL (when using images)
    rl_image_keys: list = field(default_factory=list)

    # Extra lowdim keys beyond state + base_action
    extra_lowdim_keys: list = field(default_factory=list)

    # Env-specific defaults
    success_threshold: float = 0.03
    default_scene_xml: str = ""


# ══════════════════════════════════════════════════════════════════
# Task definitions
# ══════════════════════════════════════════════════════════════════

CUP_CONFIG = TaskConfig(
    name="cup",
    task_description="Pick up the cup and stand it upright.",
    vec_env_module="resfit.rl_finetuning.wrappers.mujoco_vec_env_cup",
    vec_env_class="MuJoCoVecEnvCup",
    grip_min=0.0,
    grip_max=0.04,
    use_gripper_latch=True,
    residual_grip_scale=0.004,
    camera_keys={
        "observation.images.cam_base": None,
        "observation.images.cam_wrist": None,
    },
    default_num_envs=27,
    default_reward_type="dense",
    reward_choices=["sparse", "dense", "dense_bonus"],
    default_max_episode_steps=500,
    default_gamma=0.95,
    default_action_l2_reg=1.0,
    default_offline_fraction=0.75,
    wandb_project="mujoco-franka-cup-residual-td3",
    groot_checkpoint_hint="~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000",
    object_state_dim=8,
    extra_lowdim_keys=["observation.object_state"],
)

PNP_CONFIG = TaskConfig(
    name="pnp",
    task_description="Pick up the red cube and place it onto the plate.",
    vec_env_module="resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp",
    vec_env_class="MuJoCoVecEnvPnP",
    grip_min=0.0,
    grip_max=1.0,
    use_gripper_latch=False,
    residual_grip_scale=0.1,
    camera_keys={
        "observation.images.front": None,
        "observation.images.back": None,
        "observation.images.wrist": None,
    },
    default_num_envs=15,
    default_reward_type="dense_v3",
    reward_choices=["sparse", "dense", "dense_clipped", "dense_v2", "dense_v3"],
    default_max_episode_steps=500,
    default_gamma=0.99,
    default_action_scale=0.2,
    default_action_l2_reg=0.01,
    default_actor_lr=3e-4,
    default_critic_lr=3e-4,
    default_buffer_size=500_000,
    default_total_timesteps=500_000,
    default_actor_hidden_dim=256,
    default_critic_hidden_dim=256,
    default_offline_fraction=0.5,
    wandb_project="mujoco-franka-pnp-residual-td3",
    groot_checkpoint_hint="~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000",
    object_state_dim=10,
    supports_asymmetric_critic=True,
    rl_image_keys=["observation.images.front", "observation.images.back", "observation.images.wrist"],
    extra_lowdim_keys=["observation.object_state"],
    success_threshold=0.095,
    default_scene_xml="~/Repos/Intern/Mujoco_Franka/mujoco_menagerie/franka_fr3/fr3_with_hand.xml",
)

LIFT_CONFIG = TaskConfig(
    name="lift",
    task_description="Pick up the red cube and lift it.",
    vec_env_module="resfit.rl_finetuning.wrappers.mujoco_vec_env",
    vec_env_class="MuJoCoVecEnv",
    grip_min=0.0,
    grip_max=1.0,
    use_gripper_latch=False,
    residual_grip_scale=0.1,
    camera_keys={
        "observation.images.front": None,
        "observation.images.wrist": None,
    },
    default_num_envs=10,
    default_reward_type="dense",
    reward_choices=["sparse", "dense"],
    default_max_episode_steps=300,
    default_gamma=0.99,
    default_action_l2_reg=10.0,
    default_actor_lr=1e-5,
    default_critic_lr=1e-4,
    default_buffer_size=200_000,
    default_total_timesteps=50_000,
    default_offline_fraction=0.5,
    wandb_project="mujoco-franka-residual-td3",
    groot_checkpoint_hint="~/DATA/INTERN/training/gr00t_sim_66ep/checkpoint-300000",
    supports_asymmetric_critic=True,
    rl_image_keys=["observation.images.front", "observation.images.wrist"],
    object_state_dim=7,
    extra_lowdim_keys=["observation.object_state"],
)

STACK_CONFIG = TaskConfig(
    name="stack",
    task_description="Pick up the green cube and stack it on the white cube.",
    vec_env_module="resfit.rl_finetuning.wrappers.mujoco_vec_env_stack",
    vec_env_class="MuJoCoVecEnvStack",
    grip_min=0.0,
    grip_max=0.04,
    use_gripper_latch=False,
    residual_grip_scale=0.004,
    camera_keys={
        "observation.images.front": None,
        "observation.images.back": None,
        "observation.images.wrist": None,
    },
    default_num_envs=15,
    default_reward_type="dense",
    reward_choices=["sparse", "dense"],
    default_max_episode_steps=500,
    default_gamma=0.95,
    default_action_scale=0.05,
    default_action_l2_reg=1.0,
    default_offline_fraction=0.75,
    wandb_project="mujoco-franka-stack-residual-td3",
    groot_checkpoint_hint="~/DATA/INTERN/training/groot_stack_sim/checkpoint-100000",
    object_state_dim=10,  # white_cube pos(3)+quat(4) + green_pos(3) = 10
    supports_asymmetric_critic=True,
    rl_image_keys=["observation.images.front", "observation.images.back", "observation.images.wrist"],
    extra_lowdim_keys=["observation.object_state"],
)

DRAWER_CONFIG = TaskConfig(
    name="drawer",
    task_description="Close the open drawer by pushing it.",
    vec_env_module="resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer",
    vec_env_class="MuJoCoVecEnvDrawer",
    grip_min=0.0,
    grip_max=0.0,  # always closed (push task — gripper not used)
    use_gripper_latch=False,
    residual_grip_scale=0.0,  # no grip residual
    camera_keys={
        "observation.images.cam_base": None,
        "observation.images.cam_wrist": None,
    },
    default_num_envs=10,
    default_reward_type="dense",
    reward_choices=["sparse", "dense"],
    default_max_episode_steps=500,
    default_gamma=0.99,
    default_action_scale=0.1,
    default_action_l2_reg=10.0,
    default_offline_fraction=0.3,
    wandb_project="mujoco-franka-drawer-residual-td3",
    groot_checkpoint_hint="~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000",
    object_state_dim=0,  # offline data lacks obs_object_state; online env provides it but we skip for consistency
    supports_asymmetric_critic=False,
    rl_image_keys=["observation.images.cam_base", "observation.images.cam_wrist"],
    extra_lowdim_keys=[],
    success_threshold=0.01,
)

# Registry
TASK_REGISTRY: dict[str, TaskConfig] = {
    "cup": CUP_CONFIG,
    "pnp": PNP_CONFIG,
    "lift": LIFT_CONFIG,
    "stack": STACK_CONFIG,
    "drawer": DRAWER_CONFIG,
}


def get_task_config(task_name: str) -> TaskConfig:
    """Get task config by name. Raises KeyError if not found."""
    if task_name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise KeyError(f"Unknown task '{task_name}'. Available: {available}")
    return TASK_REGISTRY[task_name]
