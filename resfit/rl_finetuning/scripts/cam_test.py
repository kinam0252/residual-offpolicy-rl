import sys, os, importlib, types as _types
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__="0.0.0"; m.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); m.__path__=[]; m.__file__="fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None); z.Init=lambda *a,**kw:(lambda f:f); m.zero=z
    sys.modules["deepspeed"]=m; sys.modules["deepspeed.zero"]=z
try:
    import huggingface_hub.utils._validators as v; o=v.validate_repo_id
    def p(r):
        if r and (r.startswith("/") or r.startswith(".")): return
        return o(r)
    v.validate_repo_id=p
except: pass
_repo = str(Path(__file__).resolve().parents[3])
if _repo not in sys.path: sys.path.insert(0, _repo)
import numpy as np, torch, cv2
torch.backends.cuda.enable_cudnn_sdp(False)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

NE = 1
np.random.seed(42); torch.manual_seed(42)

rr = dict(dx=[-0.08, 0.14], dy=[-0.15, 0.15], yaw=[-10, 10])
me = MuJoCoVecEnv(num_envs=NE, random_cube_range=rr,
    scene_xml=os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"),
    max_episode_steps=500, reward_type="dense", device="cuda:0")
env = MuJoCoResidualWrapper(vec_env=me,
    groot_checkpoint=os.path.expanduser("~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000"),
    embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0",
    task_description="lift the cube", open_loop_horizon=16, ema_alpha=0.7)

obs, _ = env.reset()
print("GR00T only + EMA 0.7, zero residual", flush=True)

frames_base = []
frames_wrist = []
for st in range(500):
    zero_action = torch.zeros(NE, env.action_dim, device="cuda:0")
    obs, r, tm, tr, inf = env.step(zero_action)
    fb = me.get_frame(env_id=0, camera="back", size=(480, 480))
    fw = me.get_frame(env_id=0, camera="wrist", size=(480, 480))
    frames_base.append(fb)
    frames_wrist.append(fw)
    if tm[0] or tr[0]:
        status = "SUCCESS" if tm[0] else "TIMEOUT"
        print(f"Episode done at step {st}: {status}", flush=True)
        break

outdir = "outputs/eval_videos/cam_test"
os.makedirs(outdir, exist_ok=True)
H, W = 480, 480
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(f"{outdir}/cam_test_groot_ema07.mp4", fourcc, 30, (W*2, H))
for fb, fw in zip(frames_base, frames_wrist):
    fb_bgr = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
    fw_bgr = cv2.cvtColor(fw, cv2.COLOR_RGB2BGR)
    cv2.putText(fb_bgr, "cam_base", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(fw_bgr, "cam_wrist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    combined = np.hstack([fb_bgr, fw_bgr])
    out.write(combined)
out.release()
print(f"Saved {outdir}/cam_test_groot_ema07.mp4 ({len(frames_base)} frames)", flush=True)
