"""Quick base policy SR measurement (zero residual) across difficulty levels."""
import argparse, json, sys, os, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import types as _types, importlib
if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)
    _hf_val.validate_repo_id = _patched
except: pass

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack
from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--episode_positions_file", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--num_rounds", type=int, default=2, help="Number of eval rounds (episodes per env)")
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--difficulty", type=str, default="easy", choices=["easy", "normal", "hard"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_file", type=str, default=None)
    args = p.parse_args()

    # Load positions
    with open(args.episode_positions_file) as f:
        pos_list = json.load(f)
    seen = set()
    unique_pos = []
    for pp in pos_list:
        if pp["episode"] not in seen:
            seen.add(pp["episode"])
            unique_pos.append(pp)

    # Use first num_envs positions (or all if fewer)
    n = min(args.num_envs, len(unique_pos))
    white_positions = [pp["white_cube_pos"] for pp in unique_pos[:n]]
    green_positions = [pp["green_cube_pos"] for pp in unique_pos[:n]]

    # Difficulty -> random_cube_range
    if args.difficulty == "easy":
        rcr = None
    elif args.difficulty == "normal":
        rcr = [{"dx": (-0.02, 0.02), "dy": (-0.02, 0.02), "yaw": (-15, 15)}, None]
    elif args.difficulty == "hard":
        rcr = [{"dx": (-0.04, 0.04), "dy": (-0.04, 0.04), "yaw": (-30, 30)},
               {"dx": (-0.02, 0.02), "dy": (-0.02, 0.02)}]

    # Auto-detect calib
    calib_path = None
    if "66ep" in args.groot_checkpoint or "100ep" in args.groot_checkpoint:
        c = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
        if Path(c).exists():
            calib_path = c

    print(f"[eval_base] difficulty={args.difficulty}, num_envs={n}, rounds={args.num_rounds}, max_steps={args.max_episode_steps}")

    env = MuJoCoVecEnvStack(
        num_envs=n,
        white_cube_positions=white_positions,
        green_cube_positions=green_positions,
        calib_path=calib_path,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        device=args.device,
        random_cube_range=rcr,
    )

    wrapper = MuJoCoResidualWrapperStack(
        vec_env=env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=args.device,
        task_description="Pick up the white cube and stack it on the green cube.",
        open_loop_horizon=16,
    )

    total_success = 0
    total_episodes = 0
    round_results = []

    for rd in range(args.num_rounds):
        obs, _ = wrapper.reset()
        done_mask = np.zeros(n, dtype=bool)
        success_mask = np.zeros(n, dtype=bool)
        
        for step in range(args.max_episode_steps):
            residual = torch.zeros((n, 7), dtype=torch.float32, device=args.device)
            next_obs, reward, terminated, truncated, info = wrapper.step(residual)
            
            term_np = terminated.cpu().numpy()
            trunc_np = truncated.cpu().numpy()
            
            newly_done = (term_np | trunc_np) & ~done_mask
            success_mask[newly_done & term_np] = True
            done_mask[term_np | trunc_np] = True
            
            obs = next_obs
            if done_mask.all():
                break

        rd_sr = success_mask.sum() / n
        round_results.append(rd_sr)
        total_success += success_mask.sum()
        total_episodes += n
        print(f"  Round {rd+1}/{args.num_rounds}: SR={rd_sr:.1%} ({success_mask.sum()}/{n}) steps={step+1}")

    overall_sr = total_success / total_episodes
    print(f"\n[RESULT] difficulty={args.difficulty} SR={overall_sr:.1%} ({total_success}/{total_episodes})")
    print(f"  Per-round: {[f'{r:.1%}' for r in round_results]}")

    if args.output_file:
        result = {
            "difficulty": args.difficulty,
            "sr": float(overall_sr),
            "total_success": int(total_success),
            "total_episodes": int(total_episodes),
            "per_round": [float(r) for r in round_results],
            "num_envs": n,
            "num_rounds": args.num_rounds,
            "checkpoint": args.groot_checkpoint,
        }
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {args.output_file}")

    wrapper.close()

if __name__ == "__main__":
    main()
