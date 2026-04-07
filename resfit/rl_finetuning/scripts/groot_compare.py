import sys, os, importlib, json, types as _types
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
import numpy as np, torch, cv2, mujoco
torch.backends.cuda.enable_cudnn_sdp(False)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

SCENE_XML = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")
CUBE_INFO_DIR = os.path.expanduser("~/DATA/INTERN/datasets/cube_info/")
OUTDIR = "outputs/eval_videos/groot_compare"
os.makedirs(OUTDIR, exist_ok=True)

# Load first episode cube info
cube_dirs = sorted([d for d in os.listdir(CUBE_INFO_DIR) if d.startswith("kinam_v2_20260406")])
ci = json.load(open(os.path.join(CUBE_INFO_DIR, cube_dirs[0], "cube_info.json")))
cube_pos = ci["cube_table_pos"]
cube_quat = ci["cube_table_quat_wxyz"]
print(f"Cube pos={cube_pos}, yaw={ci['yaw_deg']:.1f}", flush=True)

configs = [
    ("20k",         os.path.expanduser("~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-20000"), 0.0),
    ("20k+EMA0.7",  os.path.expanduser("~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-20000"), 0.7),
    ("70k",         os.path.expanduser("~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-70000"), 0.0),
    ("70k+EMA0.7",  os.path.expanduser("~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-70000"), 0.7),
]

all_frames = {}  # label -> list of frames
max_steps = 500

for label, ckpt, ema in configs:
    print(f"\n=== {label} (ema={ema}) ===", flush=True)
    print(f"  Loading {ckpt}...", flush=True)

    me = MuJoCoVecEnv(num_envs=1, cube_positions=[cube_pos], cube_yaw_deg=0.0,
        random_cube_range=None, scene_xml=SCENE_XML, max_episode_steps=max_steps,
        reward_type="dense", device="cuda:0")

    # Set cube quat
    env_data = me._envs[0]
    cqa = env_data["cube_qposadr"]
    env_data["data"].qpos[cqa + 3:cqa + 7] = cube_quat
    env_data["cube_pos_init"] = np.array(cube_pos)
    mujoco.mj_forward(env_data["model"], env_data["data"])

    env = MuJoCoResidualWrapper(vec_env=me, groot_checkpoint=ckpt,
        embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0",
        task_description="lift the cube", open_loop_horizon=16, ema_alpha=ema)

    obs, _ = env.reset()
    # Fix cube after reset
    env_data["data"].qpos[cqa:cqa + 3] = cube_pos
    env_data["data"].qpos[cqa + 3:cqa + 7] = cube_quat
    env_data["cube_pos_init"] = np.array(cube_pos)
    mujoco.mj_forward(env_data["model"], env_data["data"])

    frames = []
    for st in range(max_steps):
        zero_action = torch.zeros(1, env.action_dim, device="cuda:0")
        obs, r, tm, tr, inf = env.step(zero_action)
        fb = me.get_frame(env_id=0, camera="back", size=(360, 480))
        fb_bgr = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
        # Add label
        cv2.putText(fb_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(fb_bgr, f"step={st}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        frames.append(fb_bgr)
        if tm[0] or tr[0]:
            status = "SUCCESS" if tm[0] else "TIMEOUT"
            print(f"  {status} at step {st}", flush=True)
            break

    all_frames[label] = frames
    del env, me
    torch.cuda.empty_cache()

# Pad all to same length
max_len = max(len(f) for f in all_frames.values())
for label in all_frames:
    while len(all_frames[label]) < max_len:
        last = all_frames[label][-1].copy()
        cv2.putText(last, "DONE", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        all_frames[label].append(last)

# Create 2x2 grid video
labels = [c[0] for c in configs]
H, W = all_frames[labels[0]][0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(f"{OUTDIR}/groot_compare_4way.mp4", fourcc, 30, (W*2, H*2))

for i in range(max_len):
    top = np.hstack([all_frames[labels[0]][i], all_frames[labels[1]][i]])
    bot = np.hstack([all_frames[labels[2]][i], all_frames[labels[3]][i]])
    grid = np.vstack([top, bot])
    out.write(grid)
out.release()
print(f"\nSaved {OUTDIR}/groot_compare_4way.mp4 ({max_len} frames)", flush=True)
