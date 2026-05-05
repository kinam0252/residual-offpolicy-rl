from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


def _bootstrap_typing_extensions_override() -> None:
    """Force-load newer typing_extensions from ISO_DEPS_DIR when available."""
    try:
        workspace_dir = Path(__file__).resolve().parents[3] / "workspace"
        deps_dir = Path(
            os.environ.get("ISO_DEPS_DIR", str(workspace_dir / ".pydeps_groot_iso"))
        ).expanduser().resolve()
        te_file = deps_dir / "typing_extensions.py"
        if not te_file.exists():
            return
        deps_dir_str = str(deps_dir)
        if deps_dir_str in sys.path:
            sys.path.remove(deps_dir_str)
        sys.path.insert(0, deps_dir_str)
        spec = importlib.util.spec_from_file_location("typing_extensions", str(te_file))
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "NoExtraItems"):
            sys.modules["typing_extensions"] = module
    except Exception:
        return


_bootstrap_typing_extensions_override()

_resfit_tasks_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "isaaclab", "source", "isaaclab_tasks")
_resfit_tasks_path = os.path.abspath(_resfit_tasks_path)
if os.path.isdir(_resfit_tasks_path) and _resfit_tasks_path not in sys.path:
    sys.path.insert(0, _resfit_tasks_path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Persistent iface residual step server")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Franka-Pickup-Direct-v0")
parser.add_argument("--image_h", type=int, default=84)
parser.add_argument("--image_w", type=int, default=84)
parser.add_argument("--csv_base_dir", type=str, default=None)
parser.add_argument("--csv_init_row_index", type=int, default=1)
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_host", type=str, default="127.0.0.1")
parser.add_argument("--groot_port", type=int, default=5555)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default=None)
parser.add_argument("--task_description", type=str, default="Pick up the white object.")
parser.add_argument("--language_override", type=str, default=None)
parser.add_argument("--groot_policy_strict", action="store_true")
parser.add_argument("--ipc_host", type=str, default="127.0.0.1")
parser.add_argument("--ipc_port", type=int, required=True)
parser.add_argument("--ipc_auth", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils

from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper


def _tensor_to_numpy_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _tensor_to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_tensor_to_numpy_tree(v) for v in x]
    return x


def _numpy_to_tensor_action(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    action = np.asarray(arr, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    return torch.as_tensor(action, device=device, dtype=torch.float32)


def main() -> None:
    # ── Use IfaceEnvWrapper: same scene/physics/IK as online iface script ──
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    policy_device = args_cli.groot_policy_device or str(args_cli.device)
    env = IfaceEnvWrapper(
        sim=sim,
        csv_dir=args_cli.csv_base_dir or "",
        groot_model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        policy_device=policy_device,
        policy_strict=bool(args_cli.groot_policy_strict),
        task_description=args_cli.task_description,
        language_override=args_cli.language_override,
        max_episode_steps=int(args_cli.max_episode_steps),
    )
    device = torch.device(str(args_cli.device) if torch.cuda.is_available() else "cpu")

    listener = Listener((args_cli.ipc_host, int(args_cli.ipc_port)), authkey=args_cli.ipc_auth.encode("utf-8"))
    conn = listener.accept()

    try:
        while True:
            req = conn.recv()
            cmd = str(req.get("cmd", "")).strip().lower()

            if cmd == "close":
                conn.send({"ok": True})
                break

            if cmd == "reset":
                obs, info = env.reset()
                conn.send({
                    "ok": True,
                    "obs": _tensor_to_numpy_tree(obs),
                    "info": _tensor_to_numpy_tree(info if isinstance(info, dict) else {}),
                })
                continue

            if cmd == "step":
                residual = _numpy_to_tensor_action(req.get("action", np.zeros((1, 7), dtype=np.float32)), device)
                obs, reward, terminated, truncated, info = env.step(residual_action=residual)
                done = terminated | truncated
                info_payload = info if isinstance(info, dict) else {}
                info_payload["scaled_action"] = residual
                conn.send({
                    "ok": True,
                    "obs": _tensor_to_numpy_tree(obs),
                    "reward": _tensor_to_numpy_tree(reward),
                    "terminated": _tensor_to_numpy_tree(terminated),
                    "truncated": _tensor_to_numpy_tree(truncated),
                    "done": _tensor_to_numpy_tree(done),
                    "info": _tensor_to_numpy_tree(info_payload),
                })
                continue

            conn.send({"ok": False, "error": f"unknown cmd: {cmd}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            listener.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
