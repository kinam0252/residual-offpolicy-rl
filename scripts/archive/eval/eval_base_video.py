import sys, os, json, time, imageio
sys.path.insert(0, ".")
sys.path.insert(0, "../Mujoco_Franka/src")
os.environ["MUJOCO_GL"] = "egl"

import torch, numpy as np, mujoco, cv2
torch.backends.cuda.enable_cudnn_sdp(False)

import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__="0.0.0"
    _ds.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); _ds.__path__=[]; _ds.__file__=__file__
    _dz = _types.ModuleType("deepspeed.zero"); _dz.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    _dz.Init=lambda *a,**kw:(lambda f:f); _ds.zero=_dz
    sys.modules["deepspeed"]=_ds; sys.modules["deepspeed.zero"]=_dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except: pass

from utils import setup_egl
setup_egl()
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

CUBE_POS = [0.504, -0.042, 0.02]
CKPT = os.environ["EVAL_CKPT"]
CKPT_NAME = os.environ["EVAL_NAME"]
USE_CALIB = os.environ.get("EVAL_CALIB", "") != ""
CALIB_PATH = os.environ.get("EVAL_CALIB", "")
OUT_DIR = os.environ["EVAL_OUT"]

N_ENVS = 10
N_EPISODES = 10
MAX_STEPS = 500

env_kwargs = dict(
    num_envs=N_ENVS,
    cube_positions=[CUBE_POS] * N_ENVS,
    max_episode_steps=MAX_STEPS,
    reward_type="dense_clipped",
    device="cpu",
)
if USE_CALIB:
    env_kwargs["calib_path"] = CALIB_PATH
    env_kwargs["use_calibrated_wrist"] = True

print(f"=== {CKPT_NAME} | cube={CUBE_POS} | {N_EPISODES} eps, {MAX_STEPS} steps ===", flush=True)
env = MuJoCoVecEnv(**env_kwargs)
wrapper = MuJoCoResidualWrapper(
    vec_env=env, groot_checkpoint=CKPT,
    embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0",
    task_description="lift the cube",
)

os.makedirs(OUT_DIR, exist_ok=True)

total_success = 0
total_episodes = 0
ep_frames = [[] for _ in range(N_ENVS)]
ep_step = np.zeros(N_ENVS, dtype=int)
ep_success = np.zeros(N_ENVS, dtype=bool)

obs, _ = wrapper.reset()
t0 = time.time()

while total_episodes < N_EPISODES:
    residual = torch.zeros((N_ENVS, 7), dtype=torch.float32)
    obs, reward, term, trunc, info = wrapper.step(residual)
    ep_step += 1

    # Render video frames for each env
    for i in range(N_ENVS):
        if total_episodes + i >= N_EPISODES:
            break
        e = env._envs[i]
        data = e["data"]
        e["renderer_base"].update_scene(data, camera=e["cam_base_id"], scene_option=e["opt_base"])
        frame_base = e["renderer_base"].render().copy()
        e["renderer_wrist"].update_scene(data, camera=e["cam_wrist_id"], scene_option=e["opt_wrist"])
        frame_wrist = e["renderer_wrist"].render().copy()
        h1, w1 = frame_base.shape[:2]
        h2, w2 = frame_wrist.shape[:2]
        h = max(h1, h2)
        combined = np.zeros((h, w1+w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = frame_base
        combined[:h2, w1:w1+w2] = frame_wrist
        cv2.putText(combined, f"ep{total_episodes+i+1} s{ep_step[i]}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if ep_success[i]:
            cv2.putText(combined, "SUCCESS", (w1-100, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        ep_frames[i].append(combined)

    # FIX: terminated = success (cube lifted), truncated = timeout (fail)
    for i in range(N_ENVS):
        t_i = term[i] if hasattr(term, '__getitem__') else bool(term)
        if t_i:
            ep_success[i] = True

    done_mask = term | trunc | (ep_step >= MAX_STEPS)
    any_done = False
    for i in range(N_ENVS):
        d = done_mask[i] if hasattr(done_mask, '__getitem__') else bool(done_mask)
        if d and total_episodes < N_EPISODES:
            any_done = True
            total_episodes += 1
            ep_num = total_episodes
            succ = ep_success[i]
            if succ:
                total_success += 1
            if ep_frames[i]:
                tag = "succ" if succ else "fail"
                vpath = os.path.join(OUT_DIR, f"ep{ep_num:02d}_{tag}.mp4")
                writer = imageio.get_writer(vpath, fps=20)
                for fr in ep_frames[i]:
                    writer.append_data(fr)
                writer.close()
                print(f"  Saved {vpath} ({len(ep_frames[i])} frames, {tag})", flush=True)
            ep_frames[i] = []
            ep_success[i] = False
            ep_step[i] = 0

    if any_done:
        rate = total_success / total_episodes * 100
        print(f"  ep={total_episodes}/{N_EPISODES} success={total_success} rate={rate:.1f}% t={time.time()-t0:.0f}s", flush=True)
        if total_episodes >= N_EPISODES:
            break
        obs, _ = wrapper.reset()
        ep_step[:] = 0
        ep_success[:] = False
        ep_frames = [[] for _ in range(N_ENVS)]

rate = total_success / total_episodes * 100
elapsed = time.time() - t0
print(f"\n>>> FINAL {CKPT_NAME}: {total_success}/{total_episodes} = {rate:.1f}% in {elapsed:.0f}s", flush=True)

with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
    json.dump({"checkpoint": CKPT_NAME, "success": total_success, "total": total_episodes,
                "rate": rate, "time": elapsed, "cube_pos": CUBE_POS}, f, indent=2)
