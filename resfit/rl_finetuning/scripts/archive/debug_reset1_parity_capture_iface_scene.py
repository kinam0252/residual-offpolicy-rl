from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture one reset-time GR00T input using iface-like scene")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--csv_dir", type=str, required=True)
parser.add_argument("--csv_row", type=int, default=1)
parser.add_argument("--groot_model_path", type=str, default=None)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--task_description", type=str, default="Pick up the white object.")
parser.add_argument("--language_override", type=str, default="Pick up the white object.")
parser.add_argument("--groot_input_dump_jsonl", type=str, required=True)
parser.add_argument("--action_dump_json", type=str, default=None)
parser.add_argument("--skip_groot_model_load", action="store_true")
parser.add_argument("--force_iface_csv_state", action="store_true")
parser.add_argument("--iface_state_csv_dir", type=str, default=None)
parser.add_argument("--iface_state_row", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=10)
parser.add_argument("--settle_steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.skip_groot_model_load:
    os.environ["RESFIT_SKIP_GROOT_MODEL_LOAD"] = "1"
os.environ["RESFIT_GROOT_INPUT_DUMP"] = os.path.abspath(os.path.expanduser(args_cli.groot_input_dump_jsonl))
os.environ["RESFIT_MAX_GROOT_DUMPS"] = "1"
if args_cli.force_iface_csv_state:
    os.environ["RESFIT_FORCE_IFACE_CSV_STATE"] = "1"
    if args_cli.iface_state_csv_dir:
        os.environ["RESFIT_IFACE_STATE_CSV_DIR"] = os.path.abspath(os.path.expanduser(args_cli.iface_state_csv_dir))
    os.environ["RESFIT_IFACE_STATE_ROW"] = str(int(args_cli.iface_state_row))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    floor = AssetBaseCfg(
        prim_path="/World/Environment/floor",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0)),
    )

    sun_light = AssetBaseCfg(
        prim_path="/World/Lights/SunLight",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 0.98, 0.95), angle=0.53),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0), rot=(0.892, 0.239, 0.369, 0.0)),
    )

    try:
        key_light = AssetBaseCfg(
            prim_path="/World/Lights/KeyLight",
            spawn=sim_utils.RectLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95), width=1.2, height=0.8),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, -1.0, 2.0), rot=(0.92, 0.23, 0.30, 0.0)),
        )
    except AttributeError:
        pass

    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.12, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=2.0,
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                enable_gyroscopic_forces=True,
                retain_accelerations=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.06),
            rot=(0.707, 0.0, 0.0, 0.707),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=8000.0,
                damping=400.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=8000.0,
                damping=400.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=500.0,
                stiffness=2e4,
                damping=1500.0,
            ),
        },
    )


def _make_scene_cfg(num_envs: int, env_spacing: float) -> TableTopSceneCfg:
    cfg = TableTopSceneCfg(num_envs=num_envs, env_spacing=env_spacing)

    cfg.camera_front = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_front",
        width=640,
        height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.4, -0.7, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        data_types=["rgb"],
    )

    cfg.camera_back = CameraCfg(
        prim_path="/World/envs/env_.*/DebugCamera_back",
        width=640,
        height=480,
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.7, 0.8), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        data_types=["rgb"],
    )

    q0 = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float32)
    theta = np.deg2rad(-20.0)
    q_pitch = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0, 0.0], dtype=np.float32)
    base_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=np.float32)
    pitch_xyzw = np.array([q_pitch[1], q_pitch[2], q_pitch[3], q_pitch[0]], dtype=np.float32)
    q_new_xyzw = (Rotation.from_quat(base_xyzw) * Rotation.from_quat(pitch_xyzw)).as_quat()
    q_new_wxyz = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]], dtype=np.float32)

    cfg.camera_wrist = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCamera",
        width=640,
        height=480,
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.0),
            rot=(float(q_new_wxyz[0]), float(q_new_wxyz[1]), float(q_new_wxyz[2]), float(q_new_wxyz[3])),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=0.5,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        data_types=["rgb"],
    )

    return cfg


def _rgb_to_uint8(rgb_tensor: torch.Tensor) -> np.ndarray:
    rgb = rgb_tensor[0]
    if rgb.is_cuda:
        rgb = rgb.cpu()
    arr = rgb.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    q_xyzw = Rotation.from_euler("z", yaw, degrees=False).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = _make_scene_cfg(num_envs=int(args_cli.num_envs), env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    cube = scene["cube"]
    camera_front = scene["camera_front"]
    camera_back = scene["camera_back"]
    camera_wrist = scene["camera_wrist"]

    sim_dt = sim.get_physics_dt()

    for _ in range(max(0, int(args_cli.warmup_steps))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        for cam in (camera_front, camera_back, camera_wrist):
            cam.update(dt=sim_dt)

    env_ids_gpu = torch.arange(scene.num_envs, device=sim.device, dtype=torch.long)
    env_origins = scene.env_origins[env_ids_gpu]
    eye_front = torch.tensor([[0.4, -0.7, 0.8]], device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1) + env_origins
    eye_back = torch.tensor([[0.4, 0.7, 0.8]], device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1) + env_origins
    target_pos = torch.tensor([[0.25, 0.0, -0.05]], device=sim.device, dtype=torch.float32).repeat(scene.num_envs, 1) + env_origins
    camera_front.set_world_poses_from_view(eye_front, target_pos)
    camera_back.set_world_poses_from_view(eye_back, target_pos)

    csv_dir = Path(args_cli.csv_dir).expanduser().resolve()
    joint_states_csv = csv_dir / "franka_joint_states.csv"
    gripper_states_csv = csv_dir / "gripper_joint_states.csv"
    object_pos_csv = csv_dir / "object_pos.csv"

    js = pd.read_csv(joint_states_csv, header=0).values.astype(np.float32)
    gs = pd.read_csv(gripper_states_csv, header=0).values.astype(np.float32)
    js = js[1:, 10:17]
    gs = gs[1:, 4:5]

    row = max(0, min(int(args_cli.csv_row), js.shape[0] - 1, gs.shape[0] - 1))
    arm_init = js[row, :7]
    grip_width = float(np.clip(float(gs[row, 0]), 0.0, 1.0))
    finger_pos = grip_width * 0.04

    arm_joint_names = sorted([n for n in robot.data.joint_names if ("panda_joint" in n and "finger" not in n)])
    arm_joint_ids = [robot.data.joint_names.index(n) for n in arm_joint_names]
    hand_joint_names = sorted([n for n in robot.data.joint_names if "panda_finger_joint" in n])
    hand_joint_ids = [robot.data.joint_names.index(n) for n in hand_joint_names]
    all_joint_ids = arm_joint_ids + hand_joint_ids

    q_init = np.concatenate([arm_init, [finger_pos] * len(hand_joint_ids)]).astype(np.float32)
    q_init_t = torch.tensor(q_init, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    qd_init_t = torch.zeros_like(q_init_t)
    robot.write_joint_state_to_sim(q_init_t, qd_init_t, joint_ids=all_joint_ids)
    robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)

    if object_pos_csv.exists():
        od = pd.read_csv(object_pos_csv, header=0).values.astype(np.float32)
        obj = od[0, 3:7].copy()
    else:
        obj = np.array([0.2, 0.0, 0.06, 0.0], dtype=np.float32)

    obj_pos_local = torch.tensor(obj[:3], device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    obj_pos_w = env_origins + obj_pos_local

    default_root_state = cube.data.default_root_state[env_ids_gpu].clone()
    base_rot = default_root_state[0, 3:7].cpu().numpy()
    yaw = float(obj[3])
    yaw_quat = _yaw_to_quat_wxyz(yaw)

    base_rot_scipy = np.array([base_rot[1], base_rot[2], base_rot[3], base_rot[0]])
    yaw_quat_scipy = np.array([yaw_quat[1], yaw_quat[2], yaw_quat[3], yaw_quat[0]])
    combined_r = Rotation.from_quat(yaw_quat_scipy) * Rotation.from_quat(base_rot_scipy)
    combined_xyzw = combined_r.as_quat()
    combined_wxyz = np.array([combined_xyzw[3], combined_xyzw[0], combined_xyzw[1], combined_xyzw[2]], dtype=np.float32)

    obj_quat_w = torch.tensor(combined_wxyz, device=sim.device, dtype=torch.float32).unsqueeze(0).repeat(scene.num_envs, 1)
    cube_pose = torch.cat([obj_pos_w, obj_quat_w], dim=-1)
    cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids_gpu)

    for _ in range(max(0, int(args_cli.settle_steps))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        robot.set_joint_position_target(q_init_t, joint_ids=all_joint_ids)
        camera_front.update(dt=sim_dt)
        camera_back.update(dt=sim_dt)
        camera_wrist.update(dt=sim_dt)

    wrist = _rgb_to_uint8(camera_wrist.data.output["rgb"])
    left = _rgb_to_uint8(camera_back.data.output["rgb"])
    right = _rgb_to_uint8(camera_front.data.output["rgb"])

    def to_chw_t(frame_hwc: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.transpose(frame_hwc, (2, 0, 1)).copy()).to(torch.uint8)

    joint_pos_np = robot.data.joint_pos[0, arm_joint_ids].detach().cpu().numpy().astype(np.float32)
    gripper_joint = float(robot.data.joint_pos[0, hand_joint_ids[0]].detach().cpu().item()) if hand_joint_ids else 0.0
    gripper_frac = float(np.clip(gripper_joint / 0.04, 0.0, 1.0))

    obs = {
        "observation.images.wrist_hires": to_chw_t(wrist).unsqueeze(0),
        "observation.images.back_hires": to_chw_t(left).unsqueeze(0),
        "observation.images.front_hires": to_chw_t(right).unsqueeze(0),
        "observation.raw_joint_pos": torch.from_numpy(joint_pos_np).unsqueeze(0),
        "observation.raw_gripper_frac": torch.tensor([[gripper_frac]], dtype=torch.float32),
        "observation.state": torch.zeros((1, 34), dtype=torch.float32),
    }

    policy = GR00TBasePolicy(
        host="127.0.0.1",
        port=5555,
        num_envs=1,
        device=str(args_cli.groot_policy_device),
        task_description=str(args_cli.task_description),
        language_override=str(args_cli.language_override),
        model_path=args_cli.groot_model_path,
        embodiment_tag=str(args_cli.groot_embodiment_tag),
        strict=False,
    )
    action = policy.select_action(obs)

    if args_cli.action_dump_json:
        arr = action.detach().cpu().numpy().astype("float32")
        arr64 = arr.astype("float64", copy=False)
        row = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "md5": hashlib.md5(arr.tobytes()).hexdigest(),
            "mean": float(arr64.mean()),
            "std": float(arr64.std()),
            "min": float(arr64.min()),
            "max": float(arr64.max()),
            "sample": arr.reshape(-1)[:16].tolist(),
            "full": arr.tolist(),
        }
        out_path = Path(args_cli.action_dump_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
