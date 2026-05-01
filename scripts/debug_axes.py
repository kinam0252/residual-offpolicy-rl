"""Debug: identify correct gripper grasp axis and cube axis for alignment reward."""
import sys, os, importlib, types as _types
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_BUILD_OPS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__ = "0.0.0"
    m.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None); m.__path__ = []; m.__file__ = "fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    z.Init = lambda *a, **kw: (lambda f: f); m.zero = z
    sys.modules["deepspeed"] = m; sys.modules["deepspeed.zero"] = z

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

_MF = str(Path(__file__).resolve().parents[1] / "Mujoco_Franka" / "src")
if _MF not in sys.path:
    sys.path.insert(0, _MF)
from utils import get_tcp_pose

GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000")

vec_env = MuJoCoVecEnvPnP(
    num_envs=1, reward_type="dense_v3",
    random_cube_range=None, random_bowl_range=None,
    max_episode_steps=300,
)
wrapper = MuJoCoResidualWrapperPnP(
    vec_env=vec_env, groot_checkpoint=GROOT_CKPT,
)
obs, _ = wrapper.reset()

d = vec_env._envs[0]
model = d["model"]; data = d["data"]; ids = d["ids"]
hand_id = ids["hand_id"]
cube_qposadr = d["cube_qposadr"]

print(f"hand body: id={hand_id}, name={model.body(hand_id).name}")
print(f"cube_qposadr={cube_qposadr}")

# Print finger geom info
for i in range(model.ngeom):
    name = model.geom(i).name
    if 'finger' in name.lower():
        print(f"Geom '{name}' (id={i}): pos={data.geom_xpos[i].round(4)}")

print("\n=== Running base policy (fixed env, 300 steps) ===")
grasped_step = None
for step in range(300):
    action = np.zeros((1, wrapper.action_dim))
    obs, rew, term, trunc, info = wrapper.step(action)
    
    tcp_pos, hand_R = get_tcp_pose(model, data, hand_id)
    cube_pos = data.qpos[cube_qposadr:cube_qposadr+3].copy()
    cube_quat_wxyz = data.qpos[cube_qposadr+3:cube_qposadr+7]
    cube_R = Rotation.from_quat([cube_quat_wxyz[1],cube_quat_wxyz[2],cube_quat_wxyz[3],cube_quat_wxyz[0]]).as_matrix()
    
    dist = np.linalg.norm(tcp_pos - cube_pos)
    grasped = d["grasp_state"]["grasped"]
    if grasped and grasped_step is None:
        grasped_step = step

    combos = {}
    for gi, gname in enumerate(['x','y','z']):
        gdir = hand_R[:2, gi]; gdir_n = gdir / (np.linalg.norm(gdir)+1e-8)
        for ci, cname in enumerate(['x','y','z']):
            cdir = cube_R[:2, ci]; cdir_n = cdir / (np.linalg.norm(cdir)+1e-8)
            combos[f"h{gname}_c{cname}"] = abs(float(np.dot(gdir_n, cdir_n)))

    if step % 25 == 0 or (dist < 0.06 and step % 5 == 0) or (grasped and step == grasped_step):
        g = '★' if grasped else ' '
        print(f"\nstep={step:3d} {g} dist={dist:.4f} rew={rew[0]:.4f}")
        print(f"  hand_R: x={hand_R[:,0].round(3)} y={hand_R[:,1].round(3)} z={hand_R[:,2].round(3)}")
        print(f"  cube_R: x={cube_R[:,0].round(3)} y={cube_R[:,1].round(3)} z={cube_R[:,2].round(3)}")
        for gn in ['x','y','z']:
            vals = " ".join(f"c{cn}={combos[f'h{gn}_c{cn}']:.3f}" for cn in ['x','y','z'])
            print(f"  hand_{gn}: {vals}")
        
        # Finger direction analysis
        finger_positions = []
        for i in range(model.ngeom):
            if 'finger' in model.geom(i).name.lower():
                finger_positions.append((model.geom(i).name, data.geom_xpos[i].copy()))
        if len(finger_positions) >= 2:
            fvec = finger_positions[1][1] - finger_positions[0][1]
            fvec_n = fvec / (np.linalg.norm(fvec)+1e-8)
            print(f"  finger_dir ({finger_positions[0][0]}→{finger_positions[1][0]}): {fvec_n.round(3)}")
            for gi, gname in enumerate(['x','y','z']):
                print(f"    |cos(finger, hand_{gname})| = {abs(float(np.dot(fvec_n, hand_R[:,gi]))):.3f}")

vec_env.close()
print("\nDone.")
