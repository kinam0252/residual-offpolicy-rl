"""Save raw realistic renderer output (original 1280x720) for first frame."""
import os, sys, importlib, types as _types
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []; _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import json
import torch
from PIL import Image
from resfit.rl_finetuning.rendering.realistic_renderer import RealisticImageRenderer
from resfit.rl_finetuning.configs.task_configs import get_task_config

TASK = "lift"
NUM_ENVS = 1
SAVE_DIR = Path("outputs/lift_rl_1803/raw_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

task_cfg = get_task_config(TASK)
mod = importlib.import_module(task_cfg.vec_env_module)
VecEnvClass = getattr(mod, task_cfg.vec_env_class)

# Load positions
with open("configs/lift_66ep_positions.json") as f:
    all_pos = json.load(f)
cube_pos = [all_pos[-1]["cube_pos"]]

vec_env = VecEnvClass(
    num_envs=NUM_ENVS,
    cube_positions=cube_pos,
    max_episode_steps=300,
    reward_type="dense",
    device="cpu",
)
print("VecEnv created")

# Create realistic renderer
renderer = RealisticImageRenderer(task=TASK, vecenv=vec_env, rl_img_size=84)
print("Renderer created")

# Reset env
obs, _ = vec_env.reset()
print("Env reset")

# Use sync_and_render to get processed images (this handles qpos sync internally)
result = renderer.sync_and_render(vec_env, env_idx=0)

# Now get raw images by manually syncing and rendering without resize
import mujoco
src_env = vec_env._envs[0]
src_model = src_env["model"]
src_data = src_env["data"]

nq = min(renderer.model.nq, src_model.nq)
nv = min(renderer.model.nv, src_model.nv)
renderer.data.qpos[:nq] = src_data.qpos[:nq]
renderer.data.qvel[:nv] = src_data.qvel[:nv]
mujoco.mj_forward(renderer.model, renderer.data)

# Get raw BGR images (original resolution)
img_base_bgr = renderer.helper.render_base(renderer.data)
img_wrist_bgr = renderer.helper.render_wrist(renderer.data)

print(f"Raw back: shape={img_base_bgr.shape}, dtype={img_base_bgr.dtype}, range=[{img_base_bgr.min()}, {img_base_bgr.max()}]")
print(f"Raw wrist: shape={img_wrist_bgr.shape}, dtype={img_wrist_bgr.dtype}, range=[{img_wrist_bgr.min()}, {img_wrist_bgr.max()}]")

# Save raw BGR (original resolution)
cv2.imwrite(str(SAVE_DIR / "raw_back_bgr_1280x720.png"), img_base_bgr)
cv2.imwrite(str(SAVE_DIR / "raw_wrist_bgr_1280x720.png"), img_wrist_bgr)

# Save RGB version (original resolution)
img_base_rgb = cv2.cvtColor(img_base_bgr, cv2.COLOR_BGR2RGB)
img_wrist_rgb = cv2.cvtColor(img_wrist_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(img_base_rgb).save(str(SAVE_DIR / "raw_back_rgb_1280x720.png"))
Image.fromarray(img_wrist_rgb).save(str(SAVE_DIR / "raw_wrist_rgb_1280x720.png"))

# Save resized 84x84 RGB (what model sees)
img_base_84 = cv2.resize(img_base_rgb, (84, 84))
img_wrist_84 = cv2.resize(img_wrist_rgb, (84, 84))
Image.fromarray(img_base_84).save(str(SAVE_DIR / "model_input_back_rgb_84x84.png"))
Image.fromarray(img_wrist_84).save(str(SAVE_DIR / "model_input_wrist_rgb_84x84.png"))

# Also save what the wrapper produces (CHW uint8 → HWC for viewing)
result = renderer.sync_and_render(vec_env, env_idx=0)
for cam in ["back", "wrist"]:
    chw = result[cam]  # (3, 84, 84) uint8
    hwc = np.transpose(chw, (1, 2, 0))
    Image.fromarray(hwc).save(str(SAVE_DIR / f"wrapper_output_{cam}_84x84.png"))
    print(f"Wrapper {cam}: shape={chw.shape}, dtype={chw.dtype}, range=[{chw.min()}, {chw.max()}]")
    print(f"  R mean={hwc[:,:,0].mean():.1f}, G={hwc[:,:,1].mean():.1f}, B={hwc[:,:,2].mean():.1f}")

print(f"\nSaved all images to {SAVE_DIR}")

renderer.close()
vec_env.close()
