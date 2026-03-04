# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--max_iteration", type=int, default=None, help="Alias of --max_iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--api_url", type=str, default=None, help="Override latent_api_url (e.g. http://host:5500/api/inference). If not set, uses env default.")
parser.add_argument(
    "--gr00t_debug_compare",
    action="store_true",
    default=False,
    help="Enable detailed GR00T behavior comparison debug logs in Franka pickup env.",
)
parser.add_argument(
    "--gr00t_debug_interval",
    type=int,
    default=120,
    help="Logging interval (sim steps) for GR00T comparison debug.",
)
parser.add_argument(
    "--gr00t_debug_dump",
    action="store_true",
    default=False,
    help="Dump GR00T comparison debug as JSONL under logs/skrl/debug.",
)
parser.add_argument(
    "--csv_base_dir",
    type=str,
    default=None,
    help="CSV base directory for deterministic initialization (same data source as reference script).",
)
parser.add_argument(
    "--gr00t_reference_match",
    action="store_true",
    default=False,
    help="Match reference GR00T script behavior for debugging (CSV init + no RL/noise overlays).",
)
parser.add_argument(
    "--no_gr00t_reference_match",
    action="store_true",
    default=False,
    help="Disable automatic reference-match defaults for Franka pickup task.",
)
parser.add_argument(
    "--disable_wandb",
    action="store_true",
    default=False,
    help="Disable wandb logging during debug runs.",
)
parser.add_argument(
    "--single_episode_debug_steps",
    type=int,
    default=None,
    help="If set, force skrl collection to this many timesteps and exit (for one-episode CSV debug runs).",
)
parser.add_argument(
    "--save_right_view_frames",
    action="store_true",
    default=False,
    help="Save right_view(camera_front) frames as PNGs for video export instead of relying on global render video.",
)
parser.add_argument(
    "--right_view_frame_dir",
    type=str,
    default="logs/skrl/right_view_frames",
    help="Directory to save right_view(camera_front) frames when --save_right_view_frames is enabled.",
)
parser.add_argument(
    "--single_env_repeat",
    action="store_true",
    default=False,
    help="Force single-environment deterministic repeat mode (num_envs=1 with fixed seed defaults).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime
from pathlib import Path

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    if args_cli.single_env_repeat:
        env_cfg.scene.num_envs = 1
    else:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.api_url is not None and hasattr(env_cfg, "latent_api_url"):
        env_cfg.latent_api_url = args_cli.api_url
    if hasattr(env_cfg, "debug_compare_mode"):
        env_cfg.debug_compare_mode = bool(args_cli.gr00t_debug_compare)
    if hasattr(env_cfg, "debug_compare_interval") and args_cli.gr00t_debug_interval is not None:
        env_cfg.debug_compare_interval = int(args_cli.gr00t_debug_interval)
    if hasattr(env_cfg, "debug_compare_dump"):
        env_cfg.debug_compare_dump = bool(args_cli.gr00t_debug_dump)
    if args_cli.csv_base_dir is not None and hasattr(env_cfg, "csv_base_dir"):
        env_cfg.csv_base_dir = args_cli.csv_base_dir
    if args_cli.task == "Isaac-Franka-Pickup-Direct-v0" and hasattr(env_cfg, "csv_base_dir"):
        if env_cfg.csv_base_dir is None:
            csv_candidates = [
                Path("/home/kinam/Desktop/DATA/dataset_from_Namiko/0_Raw_dataset/pickMushroom/pickMushroom_20251209_082334_546"),
                Path("/home/kinam/Desktop/DATA/dataset_from_Namiko/0_Raw_dataset/pickMushroom"),
            ]
            for candidate in csv_candidates:
                if candidate.exists():
                    env_cfg.csv_base_dir = str(candidate)
                    print(f"[INFO] csv_base_dir auto-set for Franka pickup: {env_cfg.csv_base_dir}")
                    break
            if env_cfg.csv_base_dir is None:
                print("[WARNING] csv_base_dir is not set; robot/cube will NOT initialize from CSV.")
    reference_match_enabled = bool(args_cli.gr00t_reference_match)
    if args_cli.task == "Isaac-Franka-Pickup-Direct-v0" and not args_cli.no_gr00t_reference_match:
        reference_match_enabled = True
    if args_cli.single_env_repeat:
        reference_match_enabled = True
    if hasattr(env_cfg, "reference_match_mode"):
        env_cfg.reference_match_mode = reference_match_enabled
    if reference_match_enabled:
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "dt"):
            env_cfg.sim.dt = 0.01
        if hasattr(env_cfg, "decimation"):
            env_cfg.decimation = 1
        print("[INFO] reference_match timing patch: sim.dt=0.01, decimation=1 (match my_test script)")
    if args_cli.save_right_view_frames:
        if hasattr(env_cfg, "save_camera_images"):
            env_cfg.save_camera_images = True
        if hasattr(env_cfg, "camera_save_dir"):
            env_cfg.camera_save_dir = args_cli.right_view_frame_dir
        print(f"[INFO] right_view frame saving enabled: {args_cli.right_view_frame_dir}")
    if args_cli.disable_wandb and hasattr(env_cfg, "use_wandb"):
        env_cfg.use_wandb = False

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    effective_max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else args_cli.max_iteration
    if effective_max_iterations:
        agent_cfg["trainer"]["timesteps"] = effective_max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.single_episode_debug_steps is not None and args_cli.single_episode_debug_steps > 0:
        forced_steps = int(args_cli.single_episode_debug_steps)
        agent_cfg["agent"]["rollouts"] = forced_steps
        agent_cfg["trainer"]["timesteps"] = forced_steps
        print(f"[INFO] single_episode_debug_steps enabled: forcing rollouts={forced_steps}, timesteps={forced_steps}")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    if args_cli.single_env_repeat and args_cli.seed is None:
        args_cli.seed = 0
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    if args_cli.single_env_repeat:
        if hasattr(env_cfg, "joint_reset_noise"):
            env_cfg.joint_reset_noise = 0.0
        print("[INFO] single_env_repeat enabled: num_envs=1, reference_match_mode=True, seed=0(default), joint_reset_noise=0.0")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
