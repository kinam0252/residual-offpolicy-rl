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
import numpy as np, torch
torch.backends.cuda.enable_cudnn_sdp(False)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

EMA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
NE = 10; NEP = 5
np.random.seed(42); torch.manual_seed(42)
print(f"GR00T ONLY eval | EMA alpha = {EMA} | {NE}x{NEP} = {NE*NEP} episodes", flush=True)

rr = dict(dx=[-0.08, 0.14], dy=[-0.15, 0.15], yaw=[-10, 10])
me = MuJoCoVecEnv(num_envs=NE, random_cube_range=rr,
    scene_xml=os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"),
    max_episode_steps=500, reward_type="dense", device="cuda:0")
env = MuJoCoResidualWrapper(vec_env=me,
    groot_checkpoint=os.path.expanduser("~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000"),
    embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0",
    task_description="lift the cube", open_loop_horizon=16, ema_alpha=EMA)

obs, _ = env.reset()
ts = 0; te = 0
for ep in range(NEP):
    obs, _ = env.reset()
    ed = [False]*NE; es = [False]*NE
    for st in range(500):
        # Zero residual action — GR00T base only
        zero_action = torch.zeros(NE, env.action_dim, device="cuda:0")
        obs, r, tm, tr, inf = env.step(zero_action)
        d = tm | tr
        for i in range(NE):
            if not ed[i] and d[i]:
                ed[i] = True
                if tm[i]: es[i] = True
            if all(ed): break
        if all(ed): break
    sr = sum(es)/NE; ts += sum(es); te += NE
    print(f"  Batch {ep}: SR={sr:.0%} {es}", flush=True)
print(f"\nOVERALL SR (GR00T only, EMA={EMA}): {ts/te:.1%} ({ts}/{te})", flush=True)
