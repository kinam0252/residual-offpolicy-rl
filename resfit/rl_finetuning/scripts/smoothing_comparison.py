"""Compare smoothing methods on GR00T base policy with grid video."""
import sys, os, importlib, types as _types, copy
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
FRAME_SIZE = (360, 480)  # h, w per cell
METHODS = [
    ("No smoothing", 0.0),
    ("EMA α=0.3", 0.3),
    ("EMA α=0.5", 0.5),
    ("EMA α=0.7", 0.7),
]

# We'll run each method sequentially with the same initial state
# by recording the GR00T chunks first, then replaying with different smoothing

print("Creating env (1 env)...", flush=True)
mujoco_env = MuJoCoVecEnv(
    num_envs=1,
    scene_xml=SCENE_XML,
    max_episode_steps=500,
    reward_type="dense",
    device="cuda:0",
)

print("Creating GR00T wrapper...", flush=True)
env = MuJoCoResidualWrapper(
    vec_env=mujoco_env,
    groot_checkpoint=GROOT_CKPT,
    embodiment_tag="NEW_EMBODIMENT",
    policy_device="cuda:0",
    task_description="lift the cube",
    open_loop_horizon=16,
    action_horizon=16,
    ema_alpha=0.0,
)

# First pass: collect raw GR00T actions over one episode
print("Collecting raw GR00T actions...", flush=True)
obs, _ = env.reset()

# Save initial state for replay
import mujoco
init_qpos = env.vec_env._envs[0]["data"].qpos.copy()
init_qvel = env.vec_env._envs[0]["data"].qvel.copy()

raw_actions = []  # (step, 8) absolute actions from GR00T
zero_res = torch.zeros(1, 7, device="cuda:0")

for step in range(MAX_STEPS):
    # Get base action
    base = env._get_base_actions()
    raw_actions.append(base[0].copy())  # (8,)
    obs, reward, term, trunc, info = env.step(zero_res)
    if term.any() or trunc.any():
        print(f"  Episode ended at step {step}", flush=True)
        break

raw_actions = np.array(raw_actions)  # (T, 8)
T = len(raw_actions)
print(f"Collected {T} raw actions", flush=True)

# Apply smoothing methods
def apply_ema(actions, alpha):
    """Apply EMA smoothing to actions. alpha=weight of previous."""
    smoothed = np.zeros_like(actions)
    smoothed[0] = actions[0]
    for t in range(1, len(actions)):
        smoothed[t, :3] = alpha * smoothed[t-1, :3] + (1 - alpha) * actions[t, :3]
        # Quaternion EMA with renormalization
        smoothed[t, 3:7] = alpha * smoothed[t-1, 3:7] + (1 - alpha) * actions[t, 3:7]
        smoothed[t, 3:7] /= np.linalg.norm(smoothed[t, 3:7]) + 1e-8
        smoothed[t, 7] = alpha * smoothed[t-1, 7] + (1 - alpha) * actions[t, 7]
    return smoothed

smoothed_actions = {}
for name, alpha in METHODS:
    if alpha == 0:
        smoothed_actions[name] = raw_actions.copy()
    else:
        smoothed_actions[name] = apply_ema(raw_actions, alpha)
    print(f"  {name}: ready", flush=True)

# Now replay each method with video recording
all_frames = {name: [] for name in smoothed_actions}

for name, actions in smoothed_actions.items():
    print(f"Replaying: {name}...", flush=True)
    # Reset env to initial state
    env.vec_env._envs[0]["data"].qpos[:] = init_qpos
    env.vec_env._envs[0]["data"].qvel[:] = init_qvel
    mujoco.mj_forward(env.vec_env._envs[0]["model"], env.vec_env._envs[0]["data"])

    for step in range(T):
        # Render frame
        frame = env.vec_env.get_frame(0, camera="back", size=FRAME_SIZE)
        # Add label
        cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"step {step}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        all_frames[name].append(frame)

        # Apply action directly to MuJoCo
        action_t = torch.as_tensor(actions[step:step+1], device="cuda:0", dtype=torch.float32)
        env.vec_env.step(action_t, render_mode="rl_only")

    print(f"  {name}: {len(all_frames[name])} frames", flush=True)

# Compose grid video (2x2)
print("Composing grid video...", flush=True)
output_dir = Path("outputs/eval_videos/smoothing_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

method_names = list(smoothed_actions.keys())
grid_frames = []
for t in range(T):
    top = np.hstack([all_frames[method_names[0]][t], all_frames[method_names[1]][t]])
    bot = np.hstack([all_frames[method_names[2]][t], all_frames[method_names[3]][t]])
    grid = np.vstack([top, bot])
    grid_frames.append(grid)

vid_path = output_dir / "smoothing_comparison.mp4"
imageio.mimwrite(str(vid_path), grid_frames, fps=30)
print(f"Video saved: {vid_path} ({len(grid_frames)} frames, {grid_frames[0].shape})", flush=True)

# Also print jitter stats
print("\n=== Jitter Statistics (position delta std) ===", flush=True)
for name, actions in smoothed_actions.items():
    deltas = np.diff(actions[:, :3], axis=0)
    dx_std = np.std(deltas[:, 0])
    dy_std = np.std(deltas[:, 1])
    dz_std = np.std(deltas[:, 2])
    total_std = np.std(np.linalg.norm(deltas, axis=1))
    sign_x = np.sum(deltas[1:, 0] * deltas[:-1, 0] < 0)
    sign_y = np.sum(deltas[1:, 1] * deltas[:-1, 1] < 0)
    sign_z = np.sum(deltas[1:, 2] * deltas[:-1, 2] < 0)
    print(f"  {name:20s}: dx_std={dx_std:.5f} dy_std={dy_std:.5f} dz_std={dz_std:.5f} | reversals x={sign_x} y={sign_y} z={sign_z}", flush=True)

print("\nDONE", flush=True)
