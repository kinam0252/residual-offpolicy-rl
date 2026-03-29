#!/usr/bin/env python3
"""
Compare FP pose predictions using GT depth vs DA-V2 predicted depth.
3-panel video: GT pose | FP+GT depth | FP+DA-V2 depth (fixed align)

Usage:
  CUDA_VISIBLE_DEVICES=2 /home/nas_main/kinamkim/.conda/envs/foundationpose/bin/python \
    compare_fp_gt_vs_da2_depth.py --episode 38 --max_frames 100
"""
import argparse, sys, os, time, json
import numpy as np
import cv2
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=38)
parser.add_argument("--max_frames", type=int, default=0, help="0=all frames")
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()

BASE = Path("/home/nas_main/kinamkim/Repos/Intern/assets/lerobot_data/pickMushroom_train_dense_clipped_3cm_depth_pose")
FP_ROOT = Path("/home/nas_main/kinamkim/Repos/Intern/FoundationPose")
OUT_DIR = Path(args.output_dir or "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/report/visualize")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ep = args.episode
ep_name = f"episode_{ep:06d}"

# ── Load data ──
print(f"Loading episode {ep}...")
import pandas as pd
pose_df = pd.read_parquet(str(BASE / f"data/chunk-000/{ep_name}.parquet"))
gt_depth_npz = np.load(str(BASE / f"depth_front/{ep_name}.npz"))['frames']
rgb_path = str(BASE / f"videos/chunk-000/right_image/{ep_name}.mp4")
T = len(pose_df)
if args.max_frames > 0:
    T = min(T, args.max_frames)
print(f"  T={T}")

# ── Camera params ──
cam_eye = np.array([0.4, -0.7, 0.8])
cam_target = np.array([0.25, 0.0, -0.05])
fl, ha = 22.0, 20.955
W, H = 640, 480
fx = fy = fl * W / ha
cx, cy = W / 2.0, H / 2.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

z_cam = cam_eye - cam_target; z_cam /= np.linalg.norm(z_cam)
x_cam = np.cross(np.array([0., 0., 1.]), z_cam); x_cam /= np.linalg.norm(x_cam)
y_cam = np.cross(z_cam, x_cam)
R_view = np.array([x_cam, y_cam, z_cam])

# DA-V2 fixed alignment params (from offline analysis)
DA2_FIXED_S = -0.153
DA2_FIXED_T = 1.35

depth_norm = json.loads(Path("/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/depth_min_max/depth_normalization.json").read_text())


def project(pt_world):
    p = R_view @ (pt_world - cam_eye)
    if -p[2] <= 0.01: return None
    u = int(fx * p[0] / (-p[2]) + cx)
    v = int(fy * (-p[1]) / (-p[2]) + cy)
    if -50 <= u < W + 50 and -50 <= v < H + 50:
        return (u, v)
    return None


def draw_wireframe(frame, pos, R_or_quat, color, label=""):
    if isinstance(R_or_quat, np.ndarray) and R_or_quat.shape == (3, 3):
        R_obj = R_or_quat
    else:
        quat_wxyz = R_or_quat
        R_obj = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    hx, hy, hz = 0.025, 0.06, 0.02
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    corners_world = (R_obj @ corners.T).T + pos
    pts_2d = [project(c) for c in corners_world]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        if pts_2d[i] and pts_2d[j]:
            cv2.line(frame, pts_2d[i], pts_2d[j], color, 2)
    center = project(pos)
    if center and label:
        cv2.putText(frame, label, (center[0]-20, center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def fp_pose_to_world(ob_in_cam):
    """Convert FP ob_in_cam (4x4) to world frame pos + rot."""
    R_gl = np.array([x_cam, y_cam, z_cam]).T
    M_gl2cv = np.diag([1.0, -1.0, -1.0])
    R_cv = M_gl2cv @ R_gl.T
    t_cv = -R_cv @ cam_eye
    T_cv_w = np.eye(4)
    T_cv_w[:3, :3] = R_cv
    T_cv_w[:3, 3] = t_cv
    T_w_cv = np.linalg.inv(T_cv_w)
    T_obj_w = T_w_cv @ ob_in_cam
    R_correction = np.diag([1.0, -1.0, -1.0])
    T_obj_w[:3, :3] = T_obj_w[:3, :3] @ R_correction
    return T_obj_w[:3, 3], T_obj_w[:3, :3]


# ── Load FoundationPose ──
print("Loading FoundationPose...")
sys.path.insert(0, str(FP_ROOT))
sys.path.insert(0, str(FP_ROOT / "mycpp" / "build"))
import trimesh
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
mesh = trimesh.creation.box(extents=[0.05, 0.12, 0.04])

# Two separate FP instances (independent tracking state)
fp_gt = FoundationPose(
    model_pts=mesh.vertices.astype(np.float32),
    model_normals=mesh.vertex_normals.astype(np.float32),
    mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug=0,
    debug_dir="/tmp/fp_gt_debug",
)
fp_da2 = FoundationPose(
    model_pts=mesh.vertices.astype(np.float32),
    model_normals=mesh.vertex_normals.astype(np.float32),
    mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug=0,
    debug_dir="/tmp/fp_da2_debug",
)
print("FoundationPose ready (2 instances)")

# ── Load DA-V2 ──
print("Loading DA-V2...")
sys.path.insert(0, "/home/nas_main/kinamkim/Repos/Intern/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2
da2 = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
da2.load_state_dict(torch.load("/home/nas_main/kinamkim/Repos/Intern/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth", map_location='cpu'))
da2 = da2.to('cuda:0').eval()
print("DA-V2 ready")

# ── Process ──
print(f"Processing {T} frames (3 panels: GT | FP+GT_depth | FP+DA2_depth)...")
cap = cv2.VideoCapture(rgb_path)
FFMPEG = "/home/nas_main/kinamkim/.local/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
panel_w = W // 2  # smaller panels for 3-up
panel_h = H // 2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
canvas_w = panel_w * 3
canvas_h = panel_h
tmp_path = str(OUT_DIR / f"tmp_fp_gt_vs_da2_ep{ep}.mp4")
writer = cv2.VideoWriter(tmp_path, fourcc, 10, (canvas_w, canvas_h))

cube_pos_list = pose_df['cube_pos'].tolist()
cube_quat_list = pose_df['cube_quat_wxyz'].tolist()
fp_gt_initialized = False
fp_da2_initialized = False

errors_gt = []
errors_da2 = []

for t in range(T):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ret, frame_bgr = cap.read()
    if not ret:
        break

    gt_pos = np.array(cube_pos_list[t])
    gt_quat = np.array(cube_quat_list[t])
    rgb_fp = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # GT depth (upscale from 84x84)
    gt_depth_84 = gt_depth_npz[t]
    gt_depth_full = cv2.resize(gt_depth_84, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # DA-V2 predicted depth (fixed alignment)
    with torch.no_grad():
        pred_rel = da2.infer_image(frame_bgr)
    pred_full = cv2.resize(pred_rel, (W, H), interpolation=cv2.INTER_LINEAR)
    da2_depth = np.clip(DA2_FIXED_S * pred_full + DA2_FIXED_T, 0.01, 20.0).astype(np.float32)

    # Registration mask from GT pose
    if not fp_gt_initialized:
        R_obj = Rotation.from_quat([gt_quat[1], gt_quat[2], gt_quat[3], gt_quat[0]]).as_matrix()
        hx, hy, hz = 0.025, 0.06, 0.02
        corners = np.array([[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                            [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]])
        corners_w = (R_obj @ corners.T).T + gt_pos
        pts_2d = [p for c in corners_w if (p := project(c)) is not None]
        ob_mask = np.zeros((H, W), dtype=np.uint8)
        if len(pts_2d) >= 4:
            hull = cv2.convexHull(np.array(pts_2d))
            cv2.fillConvexPoly(ob_mask, hull, 255)
            ob_mask = cv2.dilate(ob_mask, np.ones((15, 15), np.uint8))

    try:
        # FP with GT depth
        if not fp_gt_initialized:
            pose_gt = fp_gt.register(K=K, rgb=rgb_fp, depth=gt_depth_full, ob_mask=ob_mask, iteration=5)
            fp_gt_initialized = True
        else:
            pose_gt = fp_gt.track_one(rgb=rgb_fp, depth=gt_depth_full, K=K, iteration=2)
        pos_gt, rot_gt = fp_pose_to_world(pose_gt)
        err_gt = np.linalg.norm(pos_gt - gt_pos) * 100
        errors_gt.append(err_gt)

        # FP with DA-V2 depth
        if not fp_da2_initialized:
            pose_da2 = fp_da2.register(K=K, rgb=rgb_fp, depth=da2_depth, ob_mask=ob_mask, iteration=5)
            fp_da2_initialized = True
        else:
            pose_da2 = fp_da2.track_one(rgb=rgb_fp, depth=da2_depth, K=K, iteration=2)
        pos_da2, rot_da2 = fp_pose_to_world(pose_da2)
        err_da2 = np.linalg.norm(pos_da2 - gt_pos) * 100
        errors_da2.append(err_da2)

        # Draw 3 panels
        p1 = frame_bgr.copy()
        draw_wireframe(p1, gt_pos, gt_quat, (0, 255, 0), "GT")
        cv2.putText(p1, "GT Pose", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        p2 = frame_bgr.copy()
        draw_wireframe(p2, gt_pos, gt_quat, (0, 255, 0))
        draw_wireframe(p2, pos_gt, rot_gt, (255, 0, 0), "FP")
        cv2.putText(p2, f"FP+GT depth ({err_gt:.1f}cm)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        p3 = frame_bgr.copy()
        draw_wireframe(p3, gt_pos, gt_quat, (0, 255, 0))
        draw_wireframe(p3, pos_da2, rot_da2, (0, 0, 255), "FP")
        cv2.putText(p3, f"FP+DA2 depth ({err_da2:.1f}cm)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.putText(p1, f"t={t}/{T}", (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    except Exception as e:
        p1 = frame_bgr.copy()
        draw_wireframe(p1, gt_pos, gt_quat, (0, 255, 0), "GT")
        p2 = p1.copy()
        p3 = frame_bgr.copy()
        cv2.putText(p3, f"Error: {str(e)[:50]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        if t < 3:
            import traceback; traceback.print_exc()

    # Resize and combine
    p1s = cv2.resize(p1, (panel_w, panel_h))
    p2s = cv2.resize(p2, (panel_w, panel_h))
    p3s = cv2.resize(p3, (panel_w, panel_h))
    combined = np.hstack([p1s, p2s, p3s])
    writer.write(combined)

    if t % 20 == 0:
        eg = errors_gt[-1] if errors_gt else 0
        ed = errors_da2[-1] if errors_da2 else 0
        print(f"  t={t}: GT_depth_err={eg:.2f}cm  DA2_depth_err={ed:.2f}cm")

writer.release()
cap.release()

# Re-encode
final_path = str(OUT_DIR / f"fp_gt_vs_da2_depth_ep{ep}.mp4")
os.system(f'{FFMPEG} -y -i {tmp_path} -c:v libx264 -pix_fmt yuv420p {final_path} 2>/dev/null')
if os.path.exists(tmp_path):
    os.remove(tmp_path)

print(f"\n=== Results ===")
print(f"  FP + GT depth:  mean_err={np.mean(errors_gt):.2f}cm  std={np.std(errors_gt):.2f}cm")
print(f"  FP + DA2 depth: mean_err={np.mean(errors_da2):.2f}cm  std={np.std(errors_da2):.2f}cm")
print(f"  Gap: {np.mean(errors_da2)-np.mean(errors_gt):+.2f}cm")
print(f"\nSaved: {final_path}")

del da2, fp_gt, fp_da2
torch.cuda.empty_cache()
