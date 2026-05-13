"""Save raw realistic renderer output for drawer task — verify cabinet rendering."""
import os, sys, importlib, types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

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
import mujoco
import torch
from PIL import Image
from resfit.rl_finetuning.rendering.realistic_renderer import RealisticImageRenderer
from resfit.rl_finetuning.configs.task_configs import get_task_config

TASK = "drawer"
NUM_ENVS = 1
SAVE_DIR = Path("outputs/drawer_realistic_test/raw_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

task_cfg = get_task_config(TASK)
mod = importlib.import_module(task_cfg.vec_env_module)
VecEnvClass = getattr(mod, task_cfg.vec_env_class)

vec_env = VecEnvClass(
    num_envs=NUM_ENVS,
    max_episode_steps=300,
    reward_type="dense",
    device="cpu",
    parallel_envs=False,
)
print(f"VecEnv created: nq={vec_env._envs[0]['model'].nq}")

# Create realistic renderer
renderer = RealisticImageRenderer(task=TASK, vecenv=vec_env, rl_img_size=84)
print("Renderer created")

# Reset env
obs, _ = vec_env.reset()
print(f"Env reset. Obs keys: {list(obs.keys())}")

# Sync and render
src_env = vec_env._envs[0]
nq = min(renderer.model.nq, src_env["model"].nq)
nv = min(renderer.model.nv, src_env["model"].nv)
renderer.data.qpos[:nq] = src_env["data"].qpos[:nq]
renderer.data.qvel[:nv] = src_env["data"].qvel[:nv]
mujoco.mj_forward(renderer.model, renderer.data)

# Raw images (full resolution)
img_base_bgr = renderer.helper.render_base(renderer.data)
img_wrist_bgr = renderer.helper.render_wrist(renderer.data)

print(f"Raw back: shape={img_base_bgr.shape}, range=[{img_base_bgr.min()}, {img_base_bgr.max()}]")
print(f"Raw wrist: shape={img_wrist_bgr.shape}, range=[{img_wrist_bgr.min()}, {img_wrist_bgr.max()}]")

# Save raw
cv2.imwrite(str(SAVE_DIR / "raw_back_bgr.png"), img_base_bgr)
cv2.imwrite(str(SAVE_DIR / "raw_wrist_bgr.png"), img_wrist_bgr)

# Save RGB
img_base_rgb = cv2.cvtColor(img_base_bgr, cv2.COLOR_BGR2RGB)
img_wrist_rgb = cv2.cvtColor(img_wrist_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(img_base_rgb).save(str(SAVE_DIR / "back_rgb_full.png"))
Image.fromarray(img_wrist_rgb).save(str(SAVE_DIR / "wrist_rgb_full.png"))

# Save 84x84
for name, img in [("back", img_base_rgb), ("wrist", img_wrist_rgb)]:
    small = cv2.resize(img, (84, 84))
    Image.fromarray(small).save(str(SAVE_DIR / f"{name}_rgb_84x84.png"))
    print(f"  {name} 84x84: R={small[:,:,0].mean():.1f} G={small[:,:,1].mean():.1f} B={small[:,:,2].mean():.1f}")

# Wrapper output
result = renderer.sync_and_render(vec_env, env_idx=0)
for cam in ["back", "wrist"]:
    chw = result[cam]
    hwc = np.transpose(chw, (1, 2, 0))
    Image.fromarray(hwc).save(str(SAVE_DIR / f"wrapper_{cam}_84x84.png"))
    print(f"  wrapper {cam}: shape={chw.shape}, range=[{chw.min()}, {chw.max()}]")

print(f"\nSaved all images to {SAVE_DIR}")
renderer.close()
vec_env.close()
