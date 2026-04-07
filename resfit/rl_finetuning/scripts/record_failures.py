"""Record failure cases from v87 best checkpoint as grid video."""
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

import numpy as np, cv2, torch
torch.backends.cuda.enable_cudnn_sdp(False)
import imageio.v2 as imageio

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

FRAME_H, FRAME_W = 360, 480
MAX_FAILURES = 4  # collect 4 failure cases for 2x2 grid

np.random.seed(123)  # different seed to get some failures
torch.manual_seed(123)
dev = torch.device("cuda:0")

rr = dict(dx=[-0.08, 0.14], dy=[-0.15, 0.15], yaw=[-10, 10])
me = MuJoCoVecEnv(
    num_envs=1, random_cube_range=rr,
    scene_xml=os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"),
    max_episode_steps=500, reward_type="dense", device="cuda:0",
)
env = MuJoCoResidualWrapper(
    vec_env=me,
    groot_checkpoint=os.path.expanduser("~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000"),
    embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0",
    task_description="lift the cube", open_loop_horizon=16, ema_alpha=0.0,
)
obs, _ = env.reset()

ik = ["observation.depth.front", "observation.depth.wrist"]
ld = env.observation_space["observation.state"].shape[1]
cfg = ResidualTD3MuJoCoConfig()
cfg.agent.actor.action_scale = 0.2; cfg.agent.actor.hidden_dim = 512; cfg.agent.critic.hidden_dim = 1024
ag = QAgent(obs_shape=(1,84,84), prop_shape=(ld,), action_dim=env.action_dim,
            rl_cameras=ik, cfg=cfg.agent, residual_actor=True,
            object_state_dim=7, asymmetric_critic=True)
ag.to(dev)
ck = torch.load("outputs/mujoco_td3/v87_off02_tau005/checkpoints/best.pt", map_location=dev, weights_only=False)
ag.load_state_dict(ck, strict=False); ag.eval()
print("Loaded checkpoint", flush=True)

failure_videos = []
episode = 0

while len(failure_videos) < MAX_FAILURES and episode < 30:
    obs, _ = env.reset()
    # Get cube position for label
    qa = me._envs[0]["cube_qposadr"]; cube_pos = me._envs[0]["data"].qpos[qa:qa+3].copy()
    base_pos = np.array([0.45, -0.05, 0.02])
    dx_cm = (cube_pos[0] - base_pos[0]) * 100
    dy_cm = (cube_pos[1] - base_pos[1]) * 100
    label = f"ep{episode} dx={dx_cm:.1f}cm dy={dy_cm:.1f}cm"
    
    frames = []
    success = False
    
    for step in range(500):
        frame = env.vec_env.get_frame(0, camera="back", size=(FRAME_H, FRAME_W))
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"step {step}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        frames.append(frame)
        
        with torch.no_grad():
            action = ag.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        obs, reward, term, trunc, info = env.step(action)
        
        if term.any():
            success = True
            # Add a few more frames to show the lift
            for extra in range(10):
                frame = env.vec_env.get_frame(0, camera="back", size=(FRAME_H, FRAME_W))
                cv2.putText(frame, f"{label} - SUCCESS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                frames.append(frame)
            break
        if trunc.any():
            break
    
    if not success:
        # Mark last frames as failure
        for f in frames[-20:]:
            cv2.putText(f, "FAIL", (FRAME_W - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        failure_videos.append(frames)
        print(f"  FAIL ep{episode}: {label} ({len(frames)} frames)", flush=True)
    else:
        print(f"  OK   ep{episode}: {label}", flush=True)
    
    episode += 1

if len(failure_videos) < MAX_FAILURES:
    print(f"Only found {len(failure_videos)} failures in {episode} episodes", flush=True)

# Pad to same length
if failure_videos:
    max_len = max(len(v) for v in failure_videos)
    for v in failure_videos:
        while len(v) < max_len:
            v.append(v[-1].copy())

    # If less than 4 failures, pad with empty panels
    while len(failure_videos) < 4:
        blank = [np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)] * max_len
        failure_videos.append(blank)

    # 2x2 grid
    grid_frames = []
    for t in range(max_len):
        top = np.hstack([failure_videos[0][t], failure_videos[1][t]])
        bot = np.hstack([failure_videos[2][t], failure_videos[3][t]])
        grid_frames.append(np.vstack([top, bot]))

    out_dir = Path("outputs/eval_videos/failure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_path = out_dir / "failure_cases.mp4"
    imageio.mimwrite(str(vid_path), grid_frames, fps=30)
    print(f"\nVideo saved: {vid_path} ({len(grid_frames)} frames)", flush=True)

print(f"\nTotal: {episode} episodes, {len([v for v in failure_videos if any(f.any() for f in v)])} failures", flush=True)
print("DONE", flush=True)
