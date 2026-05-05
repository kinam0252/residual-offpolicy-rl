from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture one in-process reset GR00T input dump for parity debug")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--csv_init_row_index", type=int, default=1)
parser.add_argument("--groot_model_path", type=str, default=None)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--task_description", type=str, default="Pick up the white object.")
parser.add_argument("--language_override", type=str, default="Pick up the white object.")
parser.add_argument("--groot_input_dump_jsonl", type=str, required=True)
parser.add_argument("--action_dump_json", type=str, default=None)
parser.add_argument("--skip_groot_model_load", action="store_true")
parser.add_argument("--force_iface_csv_state", action="store_true")
parser.add_argument("--iface_state_csv_dir", type=str, default=None)
parser.add_argument("--iface_state_row", type=int, default=1)
parser.add_argument("--iface_camera_warmup_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.skip_groot_model_load:
    os.environ["RESFIT_SKIP_GROOT_MODEL_LOAD"] = "1"
os.environ["RESFIT_GROOT_INPUT_DUMP"] = os.path.abspath(os.path.expanduser(args_cli.groot_input_dump_jsonl))
os.environ["RESFIT_MAX_GROOT_DUMPS"] = "1"
os.environ["RESFIT_IFACE_CAMERA_WARMUP_STEPS"] = str(int(args_cli.iface_camera_warmup_steps))

if args_cli.force_iface_csv_state:
    os.environ["RESFIT_FORCE_IFACE_CSV_STATE"] = "1"
    if args_cli.iface_state_csv_dir:
        os.environ["RESFIT_IFACE_STATE_CSV_DIR"] = os.path.abspath(os.path.expanduser(args_cli.iface_state_csv_dir))
    os.environ["RESFIT_IFACE_STATE_ROW"] = str(int(args_cli.iface_state_row))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from resfit.rl_finetuning.wrappers.isaaclab_env_wrapper import create_isaaclab_env
from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy


def main() -> None:
    env = create_isaaclab_env(
        task="Isaac-Franka-Pickup-Direct-v0",
        num_envs=int(args_cli.num_envs),
        enable_cameras=True,
        device=str(args_cli.device),
        extra_cfg_overrides={
            "csv_base_dir": os.path.abspath(os.path.expanduser(args_cli.csv_base_dir)),
            "csv_init_row_index": int(args_cli.csv_init_row_index),
            "use_api_for_pose": False,
            "reference_match_mode": True,
            "post_reset_settle_steps": 200,
            "gr00t_host": "127.0.0.1",
            "gr00t_port": 5555,
        },
    )

    policy = GR00TBasePolicy(
        host="127.0.0.1",
        port=5555,
        num_envs=int(args_cli.num_envs),
        device=str(args_cli.groot_policy_device),
        task_description=str(args_cli.task_description),
        language_override=str(args_cli.language_override),
        model_path=args_cli.groot_model_path,
        embodiment_tag=str(args_cli.groot_embodiment_tag),
        strict=False,
    )

    obs, _ = env.reset()
    action = policy.select_action(obs)

    if args_cli.action_dump_json:
        arr = action.detach().cpu().numpy().astype("float32")
        arr64 = arr.astype("float64", copy=False)
        row = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "md5": hashlib.md5(arr.tobytes()).hexdigest(),
            "mean": float(arr64.mean()),
            "std": float(arr64.std()),
            "min": float(arr64.min()),
            "max": float(arr64.max()),
            "sample": arr.reshape(-1)[:16].tolist(),
            "full": arr.tolist(),
        }
        out_path = Path(args_cli.action_dump_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
