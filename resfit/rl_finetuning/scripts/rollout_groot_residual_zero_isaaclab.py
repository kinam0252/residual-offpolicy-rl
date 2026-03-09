"""
Run minimal resfit-style rollout with GR00T base policy and residual=0.

Goal: bootstrap integration path before full residual RL training.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ensure resfit isaaclab_tasks path wins
_resfit_tasks_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "isaaclab", "source", "isaaclab_tasks")
_resfit_tasks_path = os.path.abspath(_resfit_tasks_path)
if os.path.isdir(_resfit_tasks_path) and _resfit_tasks_path not in sys.path:
    sys.path.insert(0, _resfit_tasks_path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="GR00T residual-zero rollout on IsaacLab")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--heartbeat_interval", type=int, default=25)

parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
parser.add_argument("--gr00t_port", type=int, default=5555)

parser.add_argument("--groot_model_path", type=str, default=None)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default=None)
parser.add_argument("--groot_policy_strict", action="store_true")

parser.add_argument("--task_description", type=str, default="Pick up the white object.")
parser.add_argument("--language_override", type=str, default=None)

parser.add_argument("--csv_base_dir", type=str, default=None)
parser.add_argument("--csv_init_row_index", type=int, default=0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

print(
    f"[bootstrap] args parsed headless={getattr(args_cli, 'headless', None)} device={args_cli.device} "
    f"num_envs={args_cli.num_envs} max_steps={args_cli.max_steps}",
    flush=True,
)
print("[bootstrap] creating AppLauncher...", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[bootstrap] AppLauncher ready", flush=True)

import torch

from resfit.rl_finetuning.wrappers.isaaclab_env_wrapper import create_isaaclab_env
from resfit.rl_finetuning.wrappers.isaaclab_groot_rollout_env import IsaacLabGrootRolloutEnv
from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy


def main() -> None:
    print("[rollout] main() start", flush=True)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    if not args_cli.groot_model_path:
        raise ValueError(
            "Local GR00T mode requires --groot_model_path. "
            "Server mode is disabled for this rollout entrypoint."
        )

    extra_cfg_overrides = {
        "use_api_for_pose": False,
        "gr00t_host": args_cli.gr00t_host,
        "gr00t_port": args_cli.gr00t_port,
    }
    if args_cli.csv_base_dir:
        extra_cfg_overrides["csv_base_dir"] = args_cli.csv_base_dir
        extra_cfg_overrides["csv_init_row_index"] = args_cli.csv_init_row_index

    vec_env = create_isaaclab_env(
        task="Isaac-Franka-Pickup-Direct-v0",
        num_envs=args_cli.num_envs,
        enable_cameras=True,
        device=args_cli.device,
        image_size=(84, 84),
        extra_cfg_overrides=extra_cfg_overrides,
    )
    print("[rollout] env created", flush=True)

    policy_device = args_cli.groot_policy_device if args_cli.groot_policy_device else args_cli.device
    print("[rollout] creating base policy", flush=True)
    base_policy = GR00TBasePolicy(
        host=args_cli.gr00t_host,
        port=args_cli.gr00t_port,
        num_envs=args_cli.num_envs,
        device=policy_device,
        task_description=args_cli.task_description,
        language_override=args_cli.language_override,
        model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        strict=args_cli.groot_policy_strict,
    )
    print("[rollout] base policy created", flush=True)

    env = IsaacLabGrootRolloutEnv(vec_env=vec_env, base_policy=base_policy)
    print("[rollout] wrapper ready, resetting env", flush=True)

    obs, _ = env.reset()
    print("[rollout] reset done", flush=True)
    total_reward = torch.zeros(args_cli.num_envs, device=args_cli.device, dtype=torch.float32)
    done_total = 0

    for step in range(args_cli.max_steps):
        obs, reward, terminated, truncated, _info = env.step(None)
        total_reward += reward
        done = terminated | truncated
        done_total += int(done.sum().item())

        if (step + 1) == 1 or ((step + 1) % max(1, args_cli.heartbeat_interval) == 0):
            print(
                f"[rollout] step={step + 1} avg_reward={total_reward.mean().item():.4f} done_total={done_total}"
            , flush=True)

    print("[rollout] finished", flush=True)
    print(f"[rollout] steps={args_cli.max_steps}", flush=True)
    print(f"[rollout] avg_reward={total_reward.mean().item():.6f}", flush=True)
    print(f"[rollout] done_total={done_total}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"[rollout] fatal: {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        print("[rollout] closing simulation app", flush=True)
        simulation_app.close()
