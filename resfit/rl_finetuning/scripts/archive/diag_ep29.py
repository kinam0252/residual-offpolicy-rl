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
import numpy as np, torch, mujoco
torch.backends.cuda.enable_cudnn_sdp(False)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

EMA = 0.7
EP_IDX = 29
GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-20000")
SCENE_XML = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")
CUBE_INFO_DIR = os.path.expanduser("~/DATA/INTERN/datasets/cube_info/")

cube_dirs = sorted([d for d in os.listdir(CUBE_INFO_DIR) if d.startswith("kinam_v2_20260406")])
ci = json.load(open(os.path.join(CUBE_INFO_DIR, cube_dirs[EP_IDX], "cube_info.json")))
cube_pos = ci["cube_table_pos"]
cube_quat = ci["cube_table_quat_wxyz"]
print(f"EP{EP_IDX} cube_pos={cube_pos} yaw={ci['yaw_deg']:.1f}", flush=True)

me = MuJoCoVecEnv(num_envs=1, cube_positions=[cube_pos], cube_yaw_deg=0.0,
    random_cube_range=None, scene_xml=SCENE_XML, max_episode_steps=500,
    reward_type="dense", device="cuda:0")

env_data = me._envs[0]
cqa = env_data["cube_qposadr"]
env_data["data"].qpos[cqa + 3:cqa + 7] = cube_quat
env_data["cube_pos_init"] = np.array(cube_pos)
mujoco.mj_forward(env_data["model"], env_data["data"])

env = MuJoCoResidualWrapper(vec_env=me, groot_checkpoint=GROOT_CKPT,
    embodiment_tag="NEW_EMBODIMENT", policy_device="cuda:0",
    task_description="lift the cube", open_loop_horizon=16, ema_alpha=EMA)

obs, _ = env.reset()
env_data["data"].qpos[cqa:cqa + 3] = cube_pos
env_data["data"].qpos[cqa + 3:cqa + 7] = cube_quat
env_data["cube_pos_init"] = np.array(cube_pos)
mujoco.mj_forward(env_data["model"], env_data["data"])

header = f"{'step':>4} {'grip_w':>8} {'grip_cmd':>8} {'L':>3} {'R':>3} {'cube_z':>7} {'ee_z':>7} {'cube_x':>7} {'cube_y':>7} {'ee_x':>7} {'ee_y':>7} {'nfc':>3} {'force':>10}"
print(header, flush=True)
print("-" * len(header), flush=True)

model = env_data["model"]
data = env_data["data"]
cube_geom_id = env_data["cube_geom_id"]
finger_geom_ids = env_data["finger_geom_ids"]

for st in range(500):
    zero_action = torch.zeros(1, env.action_dim, device="cuda:0")
    obs, r, tm, tr, inf = env.step(zero_action)

    fid = env_data["ids"]["finger_ids"][0]
    gripper_w = data.qpos[model.jnt_qposadr[fid]] * 2 if fid >= 0 else -1
    grip_cmd = env._held_base_action[0, 7] if env._held_base_action is not None else -1

    contacts = me._get_contacts(0)
    l_con, r_con = contacts[0], contacts[1]

    cube_xyz = data.qpos[cqa:cqa + 3].copy()

    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if ee_site >= 0:
        ee_pos = data.site_xpos[ee_site].copy()
    else:
        ee_pos = obs["observation.state"][0, :3].cpu().numpy()

    total_force = 0.0
    cube_contacts = 0
    for ci_idx in range(data.ncon):
        c = data.contact[ci_idx]
        if cube_geom_id in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == cube_geom_id else c.geom1
            if other in finger_geom_ids:
                cube_contacts += 1
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci_idx, f)
                total_force += np.linalg.norm(f[:3])

    if st % 10 == 0 or l_con > 0 or r_con > 0:
        print(f"{st:4d} {gripper_w:8.4f} {grip_cmd:8.4f} {l_con:3.0f} {r_con:3.0f} {cube_xyz[2]:7.4f} {ee_pos[2]:7.4f} {cube_xyz[0]:7.4f} {cube_xyz[1]:7.4f} {ee_pos[0]:7.4f} {ee_pos[1]:7.4f} {cube_contacts:3d} {total_force:10.3f}", flush=True)

    if tm[0] or tr[0]:
        status = "SUCCESS" if tm[0] else "TIMEOUT"
        print(f"\n=== {status} at step {st} ===", flush=True)
        break

print(f"\nFinal cube_z={cube_xyz[2]:.4f}, gripper_w={gripper_w:.4f}", flush=True)
