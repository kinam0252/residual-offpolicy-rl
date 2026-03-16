"""Debug script: 3 parallel envs with batched GR00T inference + concat video.

Usage:
  isaaclab.sh -p resfit/rl_finetuning/scripts/debug_parallel_envs.py \
    --headless --csv_dir ... --groot_model_path ... --max_steps 500
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _bootstrap_te():
    try:
        ws = Path(__file__).resolve().parents[3] / "workspace"
        d = Path(os.environ.get("ISO_DEPS_DIR", str(ws / ".pydeps_groot_iso"))).resolve()
        f = d / "typing_extensions.py"
        if not f.exists(): return
        s = str(d)
        if s in sys.path: sys.path.remove(s)
        sys.path.insert(0, s)
        sp = importlib.util.spec_from_file_location("typing_extensions", str(f))
        if sp and sp.loader:
            m = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(m)
            if hasattr(m, "NoExtraItems"): sys.modules["typing_extensions"] = m
    except Exception: pass

_bootstrap_te()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--csv_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default=None)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--output_dir", type=str, default="/tmp/debug_parallel")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import pandas as pd
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

NUM_ENVS = int(args.num_envs)
CSV_DIR = Path(args.csv_dir).expanduser().resolve()
DEVICE = args.device
LANGUAGE = args.language_override or "pick up mushroom"

# ── Scene (same as online iface) ──
@configclass
class _SceneCfg(InteractiveSceneCfg):
    floor = AssetBaseCfg(prim_path="/World/Environment/floor", spawn=sim_utils.CuboidCfg(size=(20,20,0.02), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45,0.45,0.45))), init_state=AssetBaseCfg.InitialStateCfg(pos=(0,0,-0.01)))
    dome_light = AssetBaseCfg(prim_path="/World/Lights/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0,1.0,1.0)))
    sun_light = AssetBaseCfg(prim_path="/World/Lights/SunLight", spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(1.0,0.98,0.95), angle=0.53), init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0,0.0,3.0), rot=(0.892,0.239,0.369,0.0)))
    cube = RigidObjectCfg(prim_path="/World/envs/env_.*/cube", spawn=sim_utils.CuboidCfg(size=(0.05,0.12,0.04), rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=2.0, disable_gravity=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1, enable_gyroscopic_forces=True, retain_accelerations=False, max_linear_velocity=1000.0, max_angular_velocity=1000.0), mass_props=sim_utils.MassPropertiesCfg(mass=0.1), collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.02, rest_offset=0.0), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75,0.15,0.15))), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2,0.0,0.06), rot=(0.707,0.0,0.0,0.707)))
    robot = ArticulationCfg(prim_path="/World/envs/env_.*/Robot", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd", activate_contact_sensors=False), init_state=ArticulationCfg.InitialStateCfg(joint_pos={"panda_joint1":1.157,"panda_joint2":-1.066,"panda_joint3":-0.155,"panda_joint4":-2.239,"panda_joint5":-1.841,"panda_joint6":1.003,"panda_joint7":0.469,"panda_finger_joint.*":0.035}), actuators={"panda_shoulder":ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"],effort_limit_sim=87,stiffness=8000,damping=400),"panda_forearm":ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"],effort_limit_sim=12,stiffness=8000,damping=400),"panda_hand":ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"],effort_limit_sim=500,stiffness=2e4,damping=1500)})

def _make_scene_cfg():
    cfg = _SceneCfg(num_envs=NUM_ENVS, env_spacing=2.0)
    q0 = np.array([0.70710678,0,0,0.70710678],dtype=np.float32)
    th = np.deg2rad(-20.0)
    qp = np.array([np.cos(th/2),np.sin(th/2),0,0],dtype=np.float32)
    bx = np.array([q0[1],q0[2],q0[3],q0[0]])
    px = np.array([qp[1],qp[2],qp[3],qp[0]])
    qn = (Rotation.from_quat(bx)*Rotation.from_quat(px)).as_quat()
    qw = np.array([qn[3],qn[0],qn[1],qn[2]],dtype=np.float32)
    cfg.camera_front = CameraCfg(prim_path="/World/envs/env_.*/DebugCamera_front",width=640,height=480,offset=CameraCfg.OffsetCfg(pos=(0.4,-0.7,0.8),rot=(1,0,0,0),convention="world"),spawn=sim_utils.PinholeCameraCfg(focal_length=22,focus_distance=1,horizontal_aperture=20.955,clipping_range=(0.1,10)),data_types=["rgb"])
    cfg.camera_back = CameraCfg(prim_path="/World/envs/env_.*/DebugCamera_back",width=640,height=480,offset=CameraCfg.OffsetCfg(pos=(0.4,0.7,0.8),rot=(1,0,0,0),convention="world"),spawn=sim_utils.PinholeCameraCfg(focal_length=22,focus_distance=1,horizontal_aperture=20.955,clipping_range=(0.1,10)),data_types=["rgb"])
    cfg.camera_wrist = CameraCfg(prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",width=640,height=480,offset=CameraCfg.OffsetCfg(pos=(0.06,0,0),rot=(float(qw[0]),float(qw[1]),float(qw[2]),float(qw[3])),convention="ros"),spawn=sim_utils.PinholeCameraCfg(focal_length=22,focus_distance=0.5,horizontal_aperture=20.955,clipping_range=(0.05,5)),data_types=["rgb"])
    return cfg

def _rgb_to_uint8(rgb_tensor, env_id):
    rgb = rgb_tensor[env_id]
    if rgb.is_cuda: rgb = rgb.cpu()
    arr = rgb.numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr,0,1)*255).astype(np.uint8) if arr.max()<=1 else np.clip(arr,0,255).astype(np.uint8)
    if arr.ndim==3 and arr.shape[-1]==4: arr = arr[..., :3]
    if arr.ndim==3 and arr.shape[0]==3 and arr.shape[-1]!=3: arr = np.transpose(arr,(1,2,0))
    return arr

def _quat_mul(q1, q2):
    w1,x1,y1,z1 = q1[:,0],q1[:,1],q1[:,2],q1[:,3]
    w2,x2,y2,z2 = q2[:,0],q2[:,1],q2[:,2],q2[:,3]
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2],dim=-1)

def main():
    t0 = time.time()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sim + Scene ──
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=DEVICE)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(_make_scene_cfg())
    sim.reset()

    robot = scene["robot"]; cube = scene["cube"]
    cam_f = scene["camera_front"]; cam_b = scene["camera_back"]; cam_w = scene["camera_wrist"]
    sim_dt = sim.get_physics_dt()
    N = NUM_ENVS
    env_ids = torch.arange(N, device=DEVICE, dtype=torch.long)
    origins = scene.env_origins[env_ids]

    # ── Load GR00T ──
    tag = None
    for m in EmbodimentTag:
        if m.value == args.embodiment_tag: tag = m; break
    groot = Gr00tPolicy(embodiment_tag=tag, model_path=args.groot_model_path, device=args.policy_device, strict=True)
    mod_cfg = groot.get_modality_config()
    vid_keys = list(getattr(mod_cfg["video"],"modality_keys",[]))
    state_keys = list(getattr(mod_cfg["state"],"modality_keys",[]))
    lang_keys = list(getattr(mod_cfg["language"],"modality_keys",[]))
    lang_key = lang_keys[0] if lang_keys else "annotation.human.action.task_description"
    action_horizon = len(mod_cfg["action"].delta_indices)
    steps_per_action = max(1, int(round(0.05/sim_dt)))
    inference_interval = steps_per_action * action_horizon
    print(f"[PAR] N={N}, steps_per_action={steps_per_action}, action_horizon={action_horizon}, inference_interval={inference_interval}")

    # ── IK ──
    rcfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    rcfg.resolve(scene)
    # Per-env IK controllers (avoids shape mismatch with batched scene)
    diff_iks = []
    for _ in range(N):
        ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose",use_relative_mode=False,ik_method="dls"), num_envs=1, device=DEVICE)
        ik.reset()
        diff_iks.append(ik)
    arm_names = sorted([n for n in robot.data.joint_names if "panda_joint" in n and "finger" not in n])
    arm_ids = [robot.data.joint_names.index(n) for n in arm_names]
    hand_names = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
    hand_ids = [robot.data.joint_names.index(n) for n in hand_names]
    all_ids = arm_ids + hand_ids

    # ── Friction ──
    mats = robot.root_physx_view.get_material_properties().to("cpu")
    mats[...,0]=5e3; mats[...,1]=5e3
    robot.root_physx_view.set_material_properties(mats, torch.arange(N,device="cpu",dtype=torch.long))

    # ── CSV init (same for all envs) ──
    js = pd.read_csv(CSV_DIR/"franka_joint_states.csv",header=0).values.astype(np.float32)[1:,10:17]
    gs = pd.read_csv(CSV_DIR/"gripper_joint_states.csv",header=0).values.astype(np.float32)[1:,4:5]
    arm_init = js[1,:7]
    fp = float(np.clip(float(gs[1,0]),0,1))*0.04
    q_init = np.concatenate([arm_init,[fp]*len(hand_ids)]).astype(np.float32)
    q_t = torch.tensor(q_init,device=DEVICE).unsqueeze(0).repeat(N,1)
    qd_t = torch.zeros_like(q_t)
    robot.write_joint_state_to_sim(q_t, qd_t, joint_ids=all_ids)
    robot.set_joint_position_target(q_t, joint_ids=all_ids)

    # Cube
    op = CSV_DIR/"object_pos.csv"
    if op.exists():
        od=pd.read_csv(op,header=0).values.astype(np.float32); obj=od[0,3:7].copy()
    else:
        obj=np.array([0.2,0,0.06,0],dtype=np.float32)
    obj_pos = origins + torch.tensor(obj[:3],device=DEVICE).unsqueeze(0)
    dr = cube.data.default_root_state[0,3:7].cpu().numpy()
    from resfit.rl_finetuning.wrappers.iface_env_wrapper import _yaw_to_quat_wxyz
    yq = _yaw_to_quat_wxyz(float(obj[3]))
    br = Rotation.from_quat([dr[1],dr[2],dr[3],dr[0]])
    yr = Rotation.from_quat([yq[1],yq[2],yq[3],yq[0]])
    cq = (yr*br).as_quat()
    oq = torch.tensor([[cq[3],cq[0],cq[1],cq[2]]],device=DEVICE,dtype=torch.float32).repeat(N,1)
    cube.write_root_pose_to_sim(torch.cat([obj_pos,oq],dim=-1), env_ids=env_ids)

    # Camera warmup + settle
    for _ in range(10):
        scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
        for c in (cam_f,cam_b,cam_w): c.update(dt=sim_dt)
    eye_f = torch.tensor([[0.4,-0.7,0.8]],device=DEVICE).repeat(N,1)+origins
    eye_b = torch.tensor([[0.4,0.7,0.8]],device=DEVICE).repeat(N,1)+origins
    tp = torch.tensor([[0.25,0,-0.05]],device=DEVICE).repeat(N,1)+origins
    cam_f.set_world_poses_from_view(eye_f,tp); cam_b.set_world_poses_from_view(eye_b,tp)
    for _ in range(200):
        scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
        robot.set_joint_position_target(q_t, joint_ids=all_ids)
        for c in (cam_f,cam_b,cam_w): c.update(dt=sim_dt)

    print(f"[PAR] Init done in {time.time()-t0:.1f}s")

    # ── Per-env state ──
    action_windows = [None]*N  # (H,7) per env
    window_idx = [0]*N
    smoothed_dp = [None]*N
    smoothed_drot = [None]*N
    last_grip = [1.0]*N
    steps_since_infer = [inference_interval]*N  # trigger on first step
    joint_pos_des = q_t.clone()

    # Video frames: list of (H,W,3) per env
    all_frames = [[] for _ in range(N)]

    def _do_batched_groot_inference(env_list):
        """Batched GR00T inference for given env indices."""
        B = len(env_list)
        # Build batched obs: video (B,1,H,W,3), state (B,1,D), language (B,1)
        vid_batch = {}
        for vk in vid_keys:
            frames = []
            for eid in env_list:
                lk = vk.lower()
                if "wrist" in lk or "ego" in lk: f = _rgb_to_uint8(cam_w.data.output["rgb"], eid)
                elif "left" in lk: f = _rgb_to_uint8(cam_b.data.output["rgb"], eid)
                else: f = _rgb_to_uint8(cam_f.data.output["rgb"], eid)
                frames.append(f)
            vid_batch[vk] = np.stack(frames)[:,None,...].astype(np.uint8)  # (B,1,H,W,3)

        state_batch = {}
        for sk in state_keys:
            vals = []
            for eid in env_list:
                if "joint" in sk.lower():
                    v = robot.data.joint_pos[eid, arm_ids].detach().cpu().numpy().astype(np.float32)
                elif "gripper" in sk.lower():
                    gj = float(robot.data.joint_pos[eid, hand_ids[0]].item()) if hand_ids else 0.0
                    v = np.array([float(np.clip(gj/0.04,0,1))], dtype=np.float32)
                else:
                    v = np.zeros(1, dtype=np.float32)
                vals.append(v)
            state_batch[sk] = np.stack(vals)[:,None,:].astype(np.float32)  # (B,1,D)

        lang_batch = {lang_key: [[LANGUAGE]]*B}

        obs = {"video": vid_batch, "state": state_batch, "language": lang_batch}
        pred_action, _ = groot.get_action(obs)

        # Parse actions per env
        if isinstance(pred_action, dict):
            if "action" in pred_action:
                arr = np.array(pred_action["action"], dtype=np.float32)
                chunks = arr if arr.ndim==3 else arr[None,...]  # (B,H,7)
            elif "action.ee_delta" in pred_action and "action.gripper_pos" in pred_action:
                ee = np.array(pred_action["action.ee_delta"],dtype=np.float32)
                gr = np.array(pred_action["action.gripper_pos"],dtype=np.float32)
                if ee.ndim==2: ee=ee[None,...]; gr=gr[None,...]
                chunks = np.concatenate([ee,gr],axis=-1)
            else:
                chunks = np.zeros((B,action_horizon,7),dtype=np.float32)
        else:
            arr = np.array(pred_action,dtype=np.float32)
            chunks = arr if arr.ndim==3 else arr[None,...]

        return chunks  # (B, H, 7)

    # ── Main loop ──
    t_start = time.time()
    for step in range(args.max_steps):
        # Camera update
        for c in (cam_f,cam_b,cam_w): c.update(dt=sim_dt)
        try:
            cam_f.set_world_poses_from_view(eye_f,tp)
            cam_b.set_world_poses_from_view(eye_b,tp)
        except: pass

        # Check which envs need inference
        need_infer = [eid for eid in range(N) if steps_since_infer[eid] >= inference_interval]
        if need_infer:
            t_infer = time.time()
            chunks = _do_batched_groot_inference(need_infer)
            for i, eid in enumerate(need_infer):
                action_windows[eid] = chunks[i][:action_horizon]
                window_idx[eid] = 0
                steps_since_infer[eid] = 0
            dt_infer = time.time()-t_infer
            if step < 5 or step % 100 == 0:
                print(f"[PAR] step={step} batched_infer B={len(need_infer)} dt={dt_infer:.2f}s")
        for eid in range(N):
            steps_since_infer[eid] += 1

        # Control tick
        control_tick = (step % steps_per_action) == 0
        for eid in range(N):
            if action_windows[eid] is None: continue
            if control_tick:
                wi = window_idx[eid]
                interp = action_windows[eid][wi]
                if wi < len(action_windows[eid])-1: window_idx[eid] += 1

                dp = interp[:3].astype(np.float32)
                drot = interp[3:6].astype(np.float32)
                rn = float(np.linalg.norm(drot))
                if rn > 0.35 and rn > 1e-6: drot *= (0.35/rn)
                grip = float(interp[6])

                a = 0.35
                if smoothed_dp[eid] is None:
                    smoothed_dp[eid] = torch.tensor(dp,device=DEVICE).unsqueeze(0)
                    smoothed_drot[eid] = torch.tensor(drot,device=DEVICE).unsqueeze(0)
                dp_t = torch.tensor(dp,device=DEVICE).unsqueeze(0)
                drot_t = torch.tensor(drot,device=DEVICE).unsqueeze(0)
                smoothed_dp[eid] = (1-a)*smoothed_dp[eid]+a*dp_t
                smoothed_drot[eid] = (1-a)*smoothed_drot[eid]+a*drot_t

                ee = robot.data.body_state_w[eid:eid+1, rcfg.body_ids[0], :7]
                tgt_p = ee[:,:3] + smoothed_dp[eid]
                dq_xyzw = Rotation.from_rotvec(smoothed_drot[eid][0].detach().cpu().numpy()).as_quat()
                dq_wxyz = np.array([dq_xyzw[3],dq_xyzw[0],dq_xyzw[1],dq_xyzw[2]],dtype=np.float32)
                dq = torch.tensor(dq_wxyz,device=DEVICE).unsqueeze(0)
                tgt_q = _quat_mul(dq, ee[:,3:7])
                tgt_q = tgt_q / tgt_q.norm(dim=-1,keepdim=True).clamp(min=1e-6)
                cmd = torch.cat([tgt_p[0],tgt_q[0]]).unsqueeze(0)  # (1,7)
                diff_iks[eid].set_command(cmd)
                last_grip[eid] = float(np.clip(grip,0,1))

            # IK every step
            ee = robot.data.body_state_w[eid:eid+1, rcfg.body_ids[0], :7]
            jac = robot.root_physx_view.get_jacobians()[eid:eid+1, rcfg.body_ids[0]-1, :, :][:,  :, arm_ids]
            jp = robot.data.joint_pos[eid:eid+1][:, arm_ids]
            jpa = diff_iks[eid].compute(ee[:,:3], ee[:,3:7], jac, jp)
            prev = robot.data.joint_pos_target[eid:eid+1][:, arm_ids]
            jpa = prev + torch.clamp(jpa-prev, -0.05, 0.05)
            fg = float(last_grip[eid])*0.04
            jg = torch.full((1,len(hand_ids)),fg,device=DEVICE)
            joint_pos_des[eid] = torch.cat([jpa[0],jg[0]])

        # Sim step
        robot.set_joint_position_target(joint_pos_des, joint_ids=all_ids)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)

        # Capture frames (front camera, downscaled)
        if step % steps_per_action == 0:
            for eid in range(N):
                f = _rgb_to_uint8(cam_f.data.output["rgb"], eid)
                # Downscale to 160x120 for compact concat video
                ft = torch.from_numpy(f).permute(2,0,1).float().unsqueeze(0)
                ft = F.interpolate(ft, size=(120,160), mode="bilinear", align_corners=False)
                all_frames[eid].append(ft.squeeze(0).permute(1,2,0).to(torch.uint8).numpy())

    elapsed = time.time()-t_start
    print(f"[PAR] Done: {args.max_steps} steps in {elapsed:.1f}s ({args.max_steps/elapsed:.1f} SPS)")

    # ── Save individual + concat videos ──
    for eid in range(N):
        p = out_dir / f"env{eid}.mp4"
        w = imageio.get_writer(str(p), fps=20)
        for f in all_frames[eid]: w.append_data(f)
        w.close()
        print(f"[PAR] Saved {p} ({len(all_frames[eid])} frames)")

    # Concat: side by side
    min_len = min(len(f) for f in all_frames)
    concat_path = out_dir / f"concat_{N}envs.mp4"
    w = imageio.get_writer(str(concat_path), fps=20)
    for i in range(min_len):
        row = np.concatenate([all_frames[eid][i] for eid in range(N)], axis=1)  # (H, W*N, 3)
        w.append_data(row)
    w.close()
    print(f"[PAR] Concat video: {concat_path} ({min_len} frames, {N} envs side-by-side)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n\n{'='*60}\nFATAL ERROR:\n{'='*60}", flush=True)
        traceback.print_exc()
        print(f"{'='*60}\n", flush=True)
    finally:
        simulation_app.close()
