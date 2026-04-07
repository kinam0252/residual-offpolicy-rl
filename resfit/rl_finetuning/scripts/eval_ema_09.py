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
import numpy as np, torch; torch.backends.cuda.enable_cudnn_sdp(False)
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
EMA=0.9; NE=10; NEP=5
np.random.seed(42); torch.manual_seed(42); dev=torch.device("cuda:0")
print(f"EMA alpha = {EMA}", flush=True)
rr = dict(dx=[-0.08,0.14], dy=[-0.15,0.15], yaw=[-10,10])
me = MuJoCoVecEnv(num_envs=NE, random_cube_range=rr, scene_xml=os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"), max_episode_steps=500, reward_type="dense", device="cuda:0")
env = MuJoCoResidualWrapper(vec_env=me, groot_checkpoint=os.path.expanduser("~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000"), embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0", task_description="lift the cube", open_loop_horizon=16, ema_alpha=EMA)
obs,_=env.reset()
ik=["observation.depth.front","observation.depth.wrist"]
ld=env.observation_space["observation.state"].shape[1]
cfg=ResidualTD3MuJoCoConfig(); cfg.agent.actor.action_scale=0.2; cfg.agent.actor.hidden_dim=512; cfg.agent.critic.hidden_dim=1024
ag=QAgent(obs_shape=(1,84,84),prop_shape=(ld,),action_dim=env.action_dim,rl_cameras=ik,cfg=cfg.agent,residual_actor=True,object_state_dim=7,asymmetric_critic=True)
ag.to(dev)
ck=torch.load("outputs/mujoco_td3/v87_off02_tau005/checkpoints/best.pt",map_location=dev,weights_only=False)
ag.load_state_dict(ck,strict=False); ag.eval(); print("Loaded", flush=True)
ts=0; te=0
for ep in range(NEP):
    obs,_=env.reset(); ed=[False]*NE; es=[False]*NE
    for st in range(500):
        with torch.no_grad(): a=ag.act(obs,eval_mode=True,stddev=0.0,cpu=False)
        obs,r,tm,tr,inf=env.step(a); d=tm|tr
        for i in range(NE):
            if not ed[i] and d[i]:
                ed[i]=True
                if tm[i]: es[i]=True
        if all(ed): break
    sr=sum(es)/NE; ts+=sum(es); te+=NE
    print(f"  Ep {ep}: SR={sr:.0%} {es}", flush=True)
print(f"\nOVERALL SR (EMA={EMA}): {ts/te:.1%} ({ts}/{te})", flush=True)
