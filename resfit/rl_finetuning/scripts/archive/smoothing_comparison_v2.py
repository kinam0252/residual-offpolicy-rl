"""Compare smoothing methods — independent episodes with proper reset."""
import sys, os, importlib, types as _types
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

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
    _orig = _hf_val.validate_repo_id
    def _p(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _p
except: pass

_repo = str(Path(__file__).resolve().parents[3])
if _repo not in sys.path: sys.path.insert(0, _repo)

import numpy as np
import cv2
import torch
torch.backends.cuda.enable_cudnn_sdp(False)
import imageio.v2 as imageio

from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

SCENE_XML = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")
GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000")
MAX_STEPS = 300
FRAME_H, FRAME_W = 360, 480

METHODS = [
    ("No smoothing", 0.0),
    ("EMA a=0.3", 0.3),
    ("EMA a=0.5", 0.5),
    ("EMA a=0.7", 0.7),
]

all_method_frames = {}

for method_name, ema_alpha in METHODS:
    print(f"\n=== Running: {method_name} ===", flush=True)

    # Fresh env for each method
    mujoco_env = MuJoCoVecEnv(
        num_envs=1,
        scene_xml=SCENE_XML,
        max_episode_steps=500,
        reward_type="dense",
        device="cuda:0",
    )

    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=GROOT_CKPT,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device="cuda:0",
        task_description="lift the cube",
        open_loop_horizon=16,
        action_horizon=16,
        ema_alpha=ema_alpha,
    )

    obs, _ = env.reset()
    zero_res = torch.zeros(1, 7, device="cuda:0")

    frames = []
    actions_log = []

    for step in range(MAX_STEPS):
        # Render
        frame = env.vec_env.get_frame(0, camera="back", size=(FRAME_H, FRAME_W))

        # Label (ASCII only)
        cv2.putText(frame, method_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"step {step}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        frames.append(frame)

        # Log base action before step
        cached = env._cached_chunks[0]
        cidx = env._chunk_idx[0]
        if cached is not None and cidx < cached["eef_pos"].shape[0]:
            pos = cached["eef_pos"][cidx].copy()
            actions_log.append(pos)

        obs, reward, term, trunc, info = env.step(zero_res)
        if term.any() or trunc.any():
            print(f"  Done at step {step}", flush=True)
            break

    all_method_frames[method_name] = frames
    print(f"  {method_name}: {len(frames)} frames", flush=True)

    # Jitter stats
    if len(actions_log) > 1:
        acts = np.array(actions_log)
        deltas = np.diff(acts, axis=0)
        sign_y = np.sum(deltas[1:, 1] * deltas[:-1, 1] < 0)
        print(f"  dy_std={np.std(deltas[:,1]):.5f} reversals_y={sign_y}", flush=True)

    # Free GPU memory
    del env, mujoco_env
    torch.cuda.empty_cache()

# Pad all to same length
max_len = max(len(f) for f in all_method_frames.values())
for name in all_method_frames:
    while len(all_method_frames[name]) < max_len:
        all_method_frames[name].append(all_method_frames[name][-1].copy())

# Compose 2x2 grid
print("\nComposing grid video...", flush=True)
names = [m[0] for m in METHODS]
grid_frames = []
for t in range(max_len):
    top = np.hstack([all_method_frames[names[0]][t], all_method_frames[names[1]][t]])
    bot = np.hstack([all_method_frames[names[2]][t], all_method_frames[names[3]][t]])
    grid = np.vstack([top, bot])
    grid_frames.append(grid)

output_dir = Path("outputs/eval_videos/smoothing_comparison")
output_dir.mkdir(parents=True, exist_ok=True)
vid_path = output_dir / "smoothing_comparison_v2.mp4"
imageio.mimwrite(str(vid_path), grid_frames, fps=30)
print(f"Video saved: {vid_path} ({len(grid_frames)} frames)", flush=True)
print("DONE", flush=True)
