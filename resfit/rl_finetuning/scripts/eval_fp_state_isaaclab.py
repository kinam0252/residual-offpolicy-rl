"""
Eval v31a (asymmetric state) checkpoint with FoundationPose predicted cube pose.

Replaces GT observation.object_state with FoundationPose 6D pose prediction
via TCP connection to fp_pose_server.py.

Requires: fp_pose_server.py running on --fp_host:--fp_port

Usage:
  isaac_python eval_fp_state_isaaclab.py --headless --enable_cameras \
    --checkpoint_path /path/to/agent_step34000.pt \
    --fp_host 127.0.0.1 --fp_port 5560 \
    --num_episodes 3 --num_envs 20 ...
"""
from __future__ import annotations
import argparse, os, sys, time, json, socket, pickle, struct
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval with FoundationPose state prediction")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default=None, help="Output JSON path")
parser.add_argument("--fp_host", type=str, default="127.0.0.1")
parser.add_argument("--fp_port", type=int, default=5560)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--cube_perturb_table_path", type=str, default=None)
parser.add_argument("--action_scale", type=float, default=0.1)
parser.add_argument("--reward_type", type=str, default="dense_clipped")
parser.add_argument("--depth_norm_path", type=str, default=None)
parser.add_argument("--object_state_mode", type=str, default="raw")
parser.add_argument("--critic_hidden_dim", type=int, default=1024)
parser.add_argument("--actor_hidden_dim", type=int, default=512)
parser.add_argument("--fp_register_every_n", type=int, default=0,
                    help="Re-register FP every N steps (0=only on reset)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper, _rgb_to_uint8
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.rlpd import QAgentConfig, ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
import isaaclab.sim as sim_utils
from scipy.spatial.transform import Rotation

_log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [fp-eval] {msg}", flush=True)

# ── Camera params ──
FL, HA = 22.0, 20.955
CAM_W, CAM_H = 640, 480
FX = FY = FL * CAM_W / HA
CX, CY = CAM_W / 2.0, CAM_H / 2.0
K_CAM = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# Camera offset relative to env origin (from CameraCfg)
CAM_FRONT_OFFSET = np.array([0.4, -0.7, 0.8])
CAM_TARGET_OFFSET = np.array([0.25, 0.0, -0.05])


def compute_T_cv_w(cam_eye, cam_target):
    """Compute OpenCV camera extrinsics (world→camera transform)."""
    z_cam = cam_eye - cam_target
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(np.array([0., 0., 1.]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R_gl = np.array([x_cam, y_cam, z_cam]).T  # cam-to-world OpenGL
    M_gl2cv = np.diag([1.0, -1.0, -1.0])
    R_cv = M_gl2cv @ R_gl.T
    t_cv = -R_cv @ cam_eye
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cv
    T[:3, 3] = t_cv
    return T


def make_registration_mask(gt_pos_world, K, T_cv_w, H, W):
    """Project GT cube position onto image and create convex hull mask."""
    R_obj = np.eye(3)  # rough approx — just a sphere around center
    hx, hy, hz = 0.025, 0.06, 0.02
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    corners_world = corners + gt_pos_world
    # Project to image
    pts_2d = []
    for c in corners_world:
        p_cam = T_cv_w[:3, :3] @ c + T_cv_w[:3, 3]
        if p_cam[2] <= 0.01:
            continue
        u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
        v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
        pts_2d.append([u, v])
    mask = np.zeros((H, W), dtype=np.uint8)
    if len(pts_2d) >= 3:
        hull = cv2.convexHull(np.array(pts_2d))
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    else:
        # Fallback: circle at projected center
        p_cam = T_cv_w[:3, :3] @ gt_pos_world + T_cv_w[:3, 3]
        if p_cam[2] > 0.01:
            u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
            v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
            cv2.circle(mask, (u, v), 40, 255, -1)
    return mask


class FPClient:
    """TCP client for FoundationPose pose server."""

    def __init__(self, host, port, timeout=60):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))
        _log(f"Connected to FP server at {host}:{port}")

    def _send_recv(self, request):
        data = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack('>I', len(data)) + data)
        header = self._recv_exact(4)
        if not header:
            raise ConnectionError("FP server disconnected")
        length = struct.unpack('>I', header)[0]
        resp_data = self._recv_exact(length)
        return pickle.loads(resp_data)

    def _recv_exact(self, n):
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(min(n - len(data), 1 << 20))
            if not chunk:
                return None
            data += chunk
        return data

    def reset(self, env_id):
        return self._send_recv({'env_id': env_id, 'mode': 'reset'})

    def reset_all(self):
        return self._send_recv({'env_id': -1, 'mode': 'reset_all'})

    def register(self, env_id, rgb, depth, K, T_cv_w, ob_mask):
        return self._send_recv({
            'env_id': env_id, 'mode': 'register',
            'rgb': rgb, 'depth': depth, 'K': K,
            'T_cv_w': T_cv_w, 'ob_mask': ob_mask,
        })

    def track(self, env_id, rgb, depth, K, T_cv_w):
        return self._send_recv({
            'env_id': env_id, 'mode': 'track',
            'rgb': rgb, 'depth': depth, 'K': K,
            'T_cv_w': T_cv_w,
        })

    def close(self):
        self.sock.close()


def main():
    device = torch.device(args.groot_policy_device)
    N = args.num_envs

    _log("Creating environment...")
    _sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    env = IfaceEnvWrapper(
        sim=_sim, csv_dir=args.csv_base_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args.language_override,
        max_episode_steps=args.max_episode_steps,
        num_envs=N,
        success_threshold=args.success_threshold,
        cube_perturb_table_path=args.cube_perturb_table_path,
        reward_type=args.reward_type,
        use_depth=True,  # need cameras for FP
        depth_norm_path=args.depth_norm_path,
        use_state=True,
        object_state_mode=args.object_state_mode,
    )
    _log(f"Env ready: {N} envs")

    # Agent setup (asymmetric critic)
    image_keys = ["observation.depth.front", "observation.depth.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    object_state_dim = 0
    if "observation.object_state" in env.observation_space.spaces:
        object_state_dim = env.observation_space["observation.object_state"].shape[1]

    agent_cfg = QAgentConfig(
        actor_lr=1e-5, critic_lr=1e-4, critic_target_tau=0.005,
        actor=ActorConfig(action_scale=args.action_scale,
                          actor_last_layer_init_scale=0.0,
                          action_l2_reg_weight=1.0),
    )
    agent_cfg.critic.hidden_dim = args.critic_hidden_dim
    agent_cfg.actor.hidden_dim = args.actor_hidden_dim

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=agent_cfg,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=True,
    )
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    agent.load_checkpoint_compat(ckpt)
    agent.eval()
    agent.to(device)
    _log(f"Agent loaded from {args.checkpoint_path}")

    # Connect to FP server
    fp = FPClient(args.fp_host, args.fp_port)

    # Per-env camera extrinsics
    env_origins = env.scene.env_origins[:N].cpu().numpy()  # (N, 3)
    T_cv_w_per_env = []
    for eid in range(N):
        eye = env_origins[eid] + CAM_FRONT_OFFSET
        target = env_origins[eid] + CAM_TARGET_OFFSET
        T_cv_w_per_env.append(compute_T_cv_w(eye, target))
    _log(f"Camera extrinsics computed for {N} envs")

    # Eval loop
    all_rates = []
    all_returns = []
    fp_errors = []

    for ep_i in range(args.num_episodes):
        _log(f"Running episode {ep_i+1}/{args.num_episodes}...")
        env.reset()

        # Reset FP tracking for all envs
        fp.reset_all()
        fp_initialized = [False] * N

        obs = env._build_obs()
        ep_rewards = torch.zeros(N, device=device)
        ep_ever_success = torch.zeros(N, device=device)
        ep_fp_errors = []

        for s in range(args.max_episode_steps):
            # ── FoundationPose prediction for each env ──
            # Camera data is valid (render=True ensures cameras have fresh data)
            for eid in range(N):
                try:
                    # Get RGB from front camera
                    rgb = _rgb_to_uint8(env.camera_front.data.output["rgb"], eid)  # (H, W, 3) uint8

                    # Get depth from front camera
                    depth_raw = env.camera_front.data.output["depth"][eid]  # (H, W, 1) or (H, W)
                    if depth_raw.dim() == 3:
                        depth_raw = depth_raw[:, :, 0]
                    depth = depth_raw.cpu().numpy().astype(np.float32)  # (H, W) in meters

                    T_cv_w = T_cv_w_per_env[eid]

                    if not fp_initialized[eid]:
                        # Registration: use GT pose to create mask
                        gt_pos = env.cube.data.root_state_w[eid, :3].cpu().numpy()
                        ob_mask = make_registration_mask(gt_pos, K_CAM, T_cv_w, CAM_H, CAM_W)
                        result = fp.register(eid, rgb, depth, K_CAM, T_cv_w, ob_mask)
                        fp_initialized[eid] = True
                    else:
                        result = fp.track(eid, rgb, depth, K_CAM, T_cv_w)

                    if result.get('success', False):
                        fp_pos = result['pos']    # (3,) float32
                        fp_quat = result['quat_wxyz']  # (4,) float32

                        # Compute tracking error vs GT
                        gt_pos = env.cube.data.root_state_w[eid, :3].cpu().numpy()
                        trans_err = np.linalg.norm(fp_pos - gt_pos) * 100  # cm
                        ep_fp_errors.append(trans_err)

                        # Build object state based on mode
                        if args.object_state_mode == "raw":
                            fp_state = np.concatenate([fp_pos, fp_quat]).astype(np.float32)
                        elif args.object_state_mode == "relative":
                            hand_idx = env.robot.find_bodies("panda_hand")[0][0]
                            ee_pos = env.robot.data.body_pos_w[eid, hand_idx].cpu().numpy()
                            rel_pos = fp_pos - ee_pos
                            contact = (env.left_finger_force[eid].norm().item() > 0.1)
                            fp_state = np.concatenate([rel_pos, [float(contact)]]).astype(np.float32)
                        elif args.object_state_mode == "full":
                            hand_idx = env.robot.find_bodies("panda_hand")[0][0]
                            ee_pos = env.robot.data.body_pos_w[eid, hand_idx].cpu().numpy()
                            rel_pos = fp_pos - ee_pos
                            contact = (env.left_finger_force[eid].norm().item() > 0.1)
                            fp_state = np.concatenate([rel_pos, fp_quat, [float(contact)]]).astype(np.float32)
                        else:
                            fp_state = np.concatenate([fp_pos, fp_quat]).astype(np.float32)

                        obs["observation.object_state"][eid] = torch.tensor(
                            fp_state, device=device, dtype=torch.float32)
                    else:
                        # FP failed — keep GT state (no override)
                        if s < 5:
                            _log(f"  FP failed env {eid} step {s}: {result.get('error', '?')}")

                except Exception as e:
                    if s < 5:
                        _log(f"  FP error env {eid} step {s}: {e}")

            # Agent acts on (potentially FP-modified) obs
            with torch.no_grad(), utils.eval_mode(agent):
                residual = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Step env (render=True — need fresh camera frames for FP next step)
            next_obs, reward, term, trunc, info = env.step(residual, render=True)
            ep_rewards += reward[:N].to(device)

            # Success check
            cube_z = env.cube.data.root_state_w[:N, 2].to(device)
            cube_init_z = env._initial_cube_z[:N].to(device)
            lifted = ((cube_z - cube_init_z) >= args.success_threshold).float()
            _ch = getattr(env, '_contact_history', None)
            if _ch is not None:
                lifted = lifted * _ch.any(dim=1).float().to(device)
            ep_ever_success = torch.max(ep_ever_success, lifted)

            obs = next_obs
            if (term | trunc)[:N].all():
                break

            # Log progress
            if s == 0 or (s + 1) % 200 == 0:
                mean_err = np.mean(ep_fp_errors[-N:]) if ep_fp_errors else 0
                _log(f"  Ep{ep_i+1} step {s+1}: fp_err={mean_err:.2f}cm "
                     f"success_so_far={ep_ever_success.mean().item()*100:.1f}%")

        rate = ep_ever_success.mean().item()
        ret = ep_rewards.mean().item()
        mean_fp_err = np.mean(ep_fp_errors) if ep_fp_errors else 0
        all_rates.append(rate)
        all_returns.append(ret)
        fp_errors.append(mean_fp_err)
        _log(f"  Episode {ep_i+1}: success={rate*100:.1f}% return={ret:.1f} "
             f"fp_err={mean_fp_err:.2f}cm")

    avg_rate = np.mean(all_rates)
    avg_ret = np.mean(all_returns)
    avg_fp_err = np.mean(fp_errors)

    _log(f"RESULT: success_rate={avg_rate*100:.1f}% "
         f"({'/'.join(f'{r*100:.0f}%' for r in all_rates)}) "
         f"fp_err={avg_fp_err:.2f}cm")

    # Save results
    result = {
        "checkpoint": str(args.checkpoint_path),
        "success_rate_3ep": avg_rate,
        "mean_return_3ep": avg_ret,
        "all_rates": all_rates,
        "all_returns": all_returns,
        "fp_mean_error_cm": fp_errors,
        "num_episodes": args.num_episodes,
        "num_envs": N,
        "mode": "fp_predicted_state",
        "object_state_mode": args.object_state_mode,
    }
    out_path = args.output_path or str(
        Path(args.checkpoint_path).parent.parent / "fp_eval_result.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    _log(f"Results saved to {out_path}")

    fp.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
