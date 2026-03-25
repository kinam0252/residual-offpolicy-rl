#!/usr/bin/env python3
"""
Compare GT 6D pose vs FoundationPose predicted pose on RGB video.
Renders GT (green) and FP predicted (red) cube wireframes side by side.

Usage:
  CUDA_VISIBLE_DEVICES=2 /home/nas_main/kinamkim/.conda/envs/foundationpose/bin/python \
    compare_gt_vs_fp_pose.py --episode 38 --max_frames 50
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
FFMPEG = "/home/nas_main/kinamkim/.local/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

ep = args.episode
ep_name = f"episode_{ep:06d}"

# ── Load data ──
print(f"Loading episode {ep}...")
import pandas as pd
pose_df = pd.read_parquet(str(BASE / f"data/chunk-000/{ep_name}.parquet"))
gt_depth_npz = np.load(str(BASE / f"depth_front/{ep_name}.npz"))['frames']  # (T, 84, 84)
rgb_path = str(BASE / f"videos/chunk-000/right_image/{ep_name}.mp4")
T = len(pose_df)
if args.max_frames > 0:
    T = min(T, args.max_frames)
print(f"  T={T}, GT depth shape={gt_depth_npz.shape}")

# ── Camera params (from iface_env_wrapper) ──
cam_eye = np.array([0.4, -0.7, 0.8])
cam_target = np.array([0.25, 0.0, -0.05])
fl, ha = 22.0, 20.955
W, H = 640, 480
fx = fy = fl * W / ha  # ~671.92
cx, cy = W / 2.0, H / 2.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

# Build view matrix (OpenGL look-at → OpenCV)
z_cam = cam_eye - cam_target; z_cam /= np.linalg.norm(z_cam)
x_cam = np.cross(np.array([0., 0., 1.]), z_cam); x_cam /= np.linalg.norm(x_cam)
y_cam = np.cross(z_cam, x_cam)
R_view = np.array([x_cam, y_cam, z_cam])  # cam axes as rows

def project(pt_world):
    p = R_view @ (pt_world - cam_eye)
    if -p[2] <= 0.01: return None
    u = int(fx * p[0] / (-p[2]) + cx)
    v = int(fy * (-p[1]) / (-p[2]) + cy)
    if -50 <= u < W + 50 and -50 <= v < H + 50:
        return (u, v)
    return None

def world_to_cam_pose(pos, quat_wxyz):
    """Convert world pose to camera-frame 4x4 (OpenCV convention for FP)."""
    R_obj = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    T_obj_w = np.eye(4)
    T_obj_w[:3, :3] = R_obj
    T_obj_w[:3, 3] = pos
    
    # World → OpenCV camera
    # OpenGL look-at: z_cam backward, x_cam right, y_cam up
    # OpenCV: x right, y down, z forward
    R_gl = np.array([x_cam, y_cam, z_cam]).T  # world-to-cam (OpenGL col convention)
    # OpenGL → OpenCV: flip Y and Z
    M_gl2cv = np.diag([1, -1, -1]).astype(np.float64)
    R_cv = M_gl2cv @ R_gl.T
    t_cv = -R_cv @ cam_eye
    T_cv_w = np.eye(4)
    T_cv_w[:3, :3] = R_cv
    T_cv_w[:3, 3] = t_cv
    
    return T_cv_w @ T_obj_w  # ob_in_cam

def draw_wireframe(frame, pos, R_or_quat, color, label=""):
    """Draw 3D cube wireframe on frame. R_or_quat can be 3x3 matrix or wxyz quaternion."""
    if isinstance(R_or_quat, np.ndarray) and R_or_quat.shape == (3, 3):
        R_obj = R_or_quat
    else:
        quat_wxyz = R_or_quat
        R_obj = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    # Cube corners (5x12x4 cm)
    hx, hy, hz = 0.025, 0.06, 0.02
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    corners_world = (R_obj @ corners.T).T + pos
    
    pts_2d = []
    for c in corners_world:
        p = project(c)
        pts_2d.append(p)
    
    # Draw edges
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        if pts_2d[i] and pts_2d[j]:
            cv2.line(frame, pts_2d[i], pts_2d[j], color, 2)
    
    # Draw axes at center
    center = project(pos)
    if center:
        axis_labels = ['X', 'Y', 'Z']
        axis_colors = [(0,0,255),(0,255,0),(255,0,0)]  # BGR: Red, Green, Blue
        for ax_i in range(3):
            tip = pos + 0.04 * R_obj[:, ax_i]
            tip_px = project(tip)
            if tip_px:
                cv2.arrowedLine(frame, center, tip_px, axis_colors[ax_i], 2, tipLength=0.25)
                cv2.putText(frame, axis_labels[ax_i], (tip_px[0]+3, tip_px[1]-3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, axis_colors[ax_i], 1)
        if label:
            cv2.putText(frame, label, (center[0]-20, center[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# ── Load FoundationPose ──
print("Loading FoundationPose...")
sys.path.insert(0, str(FP_ROOT))
sys.path.insert(0, str(FP_ROOT / "mycpp" / "build"))

# Import FP modules
import trimesh
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

scorer = ScorePredictor()
refiner = PoseRefinePredictor()

try:
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()
except Exception as e:
    print(f"nvdiffrast not available: {e}")
    glctx = None

# Create cube mesh programmatically
mesh = trimesh.creation.box(extents=[0.05, 0.12, 0.04])
print(f"  Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

fp_est = FoundationPose(
    model_pts=mesh.vertices.astype(np.float32),
    model_normals=mesh.vertex_normals.astype(np.float32),
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx,
    debug=0,
    debug_dir=str(OUT_DIR / "fp_debug"),
)
print("FoundationPose ready")

# ── Load depth normalization ──
depth_norm = json.loads(Path("/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/depth_min_max/depth_normalization.json").read_text())

# ── DA-V2 for metric depth ──
print("Loading DA-V2...")
sys.path.insert(0, str(Path("/home/nas_main/kinamkim/Repos/Intern/Depth-Anything-V2")))
from depth_anything_v2.dpt import DepthAnythingV2
da2 = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
da2.load_state_dict(torch.load("/home/nas_main/kinamkim/Repos/Intern/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth", map_location='cpu'))
da2 = da2.to('cuda:0').eval()
print("DA-V2 ready")

# ── Process frames ──
print(f"Processing {T} frames...")
cap = cv2.VideoCapture(rgb_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
tmp_path = str(OUT_DIR / f"tmp_gt_vs_fp_ep{ep}.mp4")
writer = cv2.VideoWriter(tmp_path, fourcc, 10, (W * 2, H))  # side-by-side

cube_pos_list = pose_df['cube_pos'].tolist()
cube_quat_list = pose_df['cube_quat_wxyz'].tolist()
fp_initialized = False

for t in range(T):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ret, frame = cap.read()
    if not ret: break
    
    gt_pos = np.array(cube_pos_list[t])
    gt_quat = np.array(cube_quat_list[t])
    
    # Get depth via DA-V2 + alignment
    with torch.no_grad():
        pred_rel = da2.infer_image(frame)
    gt_depth_84 = gt_depth_npz[t]
    d_min, d_max = depth_norm['front']['min'], depth_norm['front']['max']
    
    # Upscale GT depth to 640x480 for FP
    gt_depth_full = cv2.resize(gt_depth_84, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Align DA-V2 to GT scale
    pred_resized = cv2.resize(pred_rel, (W, H), interpolation=cv2.INTER_LINEAR)
    valid = gt_depth_full > 0.01
    if valid.sum() > 100:
        p = pred_resized[valid].flatten().astype(np.float64)
        g = gt_depth_full[valid].flatten().astype(np.float64)
        A = np.stack([p, np.ones_like(p)], axis=1)
        s, shift = np.linalg.lstsq(A, g, rcond=None)[0]
        da2_depth = np.clip(s * pred_resized + shift, 0.01, 20).astype(np.float32)
    else:
        da2_depth = gt_depth_full  # fallback
    
    # FoundationPose inference
    rgb_fp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        if not fp_initialized:
            # First frame: create mask from GT pose projection
            hx, hy, hz = 0.025, 0.06, 0.02
            corners = np.array([
                [-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz],
            ])
            R_obj = Rotation.from_quat([gt_quat[1],gt_quat[2],gt_quat[3],gt_quat[0]]).as_matrix()
            corners_w = (R_obj @ corners.T).T + gt_pos
            pts_2d = []
            for c in corners_w:
                p = project(c)
                if p: pts_2d.append(p)
            ob_mask = np.zeros((H, W), dtype=np.uint8)
            if len(pts_2d) >= 4:
                hull = cv2.convexHull(np.array(pts_2d))
                cv2.fillConvexPoly(ob_mask, hull, 255)
                # Dilate mask slightly
                ob_mask = cv2.dilate(ob_mask, np.ones((15,15), np.uint8))
            
            fp_pose = fp_est.register(K=K, rgb=rgb_fp, depth=da2_depth, ob_mask=ob_mask, iteration=5)
            fp_initialized = True
        else:
            # Track
            fp_pose = fp_est.track_one(rgb=rgb_fp, depth=da2_depth, K=K, iteration=2)
        
        # Extract FP predicted pose in world frame
        # Use the same transform as world_to_cam_pose but inverted
        ob_in_cam = fp_pose  # 4x4 from FoundationPose
        
        # Build T_cv_w (same as world_to_cam_pose)
        R_gl = np.array([x_cam, y_cam, z_cam]).T  # cam-to-world OpenGL (columns = axes)
        M_gl2cv = np.diag([1.0, -1.0, -1.0])
        R_cv = M_gl2cv @ R_gl.T  # world-to-cam rotation (OpenCV)
        t_cv = -R_cv @ cam_eye
        T_cv_w = np.eye(4)
        T_cv_w[:3, :3] = R_cv
        T_cv_w[:3, 3] = t_cv
        
        # ob_in_world = T_w_cv @ ob_in_cam
        T_w_cv = np.linalg.inv(T_cv_w)
        T_obj_w = T_w_cv @ ob_in_cam
        
        # Apply symmetry correction: FP canonical frame has X-axis flipped (180° around X)
        # This is due to trimesh box vs Isaac Sim cube axis convention difference
        R_correction = np.diag([1.0, -1.0, -1.0])  # 180° rotation around X
        T_obj_w[:3, :3] = T_obj_w[:3, :3] @ R_correction
        
        fp_pos = T_obj_w[:3, 3]
        fp_rot = T_obj_w[:3, :3]
        fp_quat = Rotation.from_matrix(fp_rot).as_quat()  # xyzw
        fp_quat_wxyz = np.array([fp_quat[3], fp_quat[0], fp_quat[1], fp_quat[2]])
        
        # Distance error
        trans_err = np.linalg.norm(fp_pos - gt_pos) * 100  # cm
        
        # Side-by-side: left=GT, right=FP
        frame_gt = frame.copy()
        frame_fp = frame.copy()
        draw_wireframe(frame_gt, gt_pos, gt_quat, (0, 255, 0), "GT")
        draw_wireframe(frame_fp, fp_pos, fp_rot, (0, 0, 255), "FP")
        
        cv2.putText(frame_gt, f"GT Pose", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_fp, f"FP Pred (err={trans_err:.1f}cm)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame_gt, f"Ep{ep} t={t}/{T}", (10, H-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        combined = np.hstack([frame_gt, frame_fp])
        
        if t % 20 == 0:
            print(f"  t={t}: trans_err={trans_err:.2f}cm")
    except Exception as e:
        frame_gt = frame.copy()
        draw_wireframe(frame_gt, gt_pos, gt_quat, (0, 255, 0), "GT")
        frame_err = frame.copy()
        cv2.putText(frame_err, f"FP error: {str(e)[:40]}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        combined = np.hstack([frame_gt, frame_err])
        if t < 3:
            import traceback; traceback.print_exc()
    
    writer.write(combined)

writer.release()
cap.release()

# Re-encode
final_path = str(OUT_DIR / f"gt_vs_fp_pose_ep{ep}_v2.mp4")
os.system(f'{FFMPEG} -y -i {tmp_path} -c:v libx264 -pix_fmt yuv420p {final_path} 2>/dev/null')
os.remove(tmp_path)
print(f"\nSaved: {final_path}")

del da2, fp_est; torch.cuda.empty_cache()
