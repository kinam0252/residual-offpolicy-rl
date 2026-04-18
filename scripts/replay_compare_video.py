"""Replay collected trajectories in MuJoCo and compare with eval sweep grid video.

Creates:
  1. Per-position replay videos from collected offline data
  2. Per-position cells extracted from eval_grid.mp4  
  3. Side-by-side comparison videos (collection vs eval_sweep)
  4. Combined comparison grid
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


def find_best_episode(data_dir, pos_idx, episodes_per_pos=34):
    """Find shortest success episode for a position, or first fail if none."""
    best_ep, best_steps = None, 999
    for offset in range(episodes_per_pos):
        ep_idx = pos_idx * episodes_per_pos + offset
        path = os.path.join(data_dir, f'ep_{ep_idx:04d}.npz')
        if not os.path.exists(path):
            continue
        d = np.load(path)
        n_steps = d['step_idx'].shape[0]
        if d['success'][0] and n_steps < best_steps:
            best_ep, best_steps = ep_idx, n_steps
    if best_ep is None:
        best_ep = pos_idx * episodes_per_pos
    return best_ep


def replay_trajectory(vec_env, env_id, actions, camera='back', size=(240, 320), skip=3):
    """Step env with recorded actions and capture frames."""
    frames = []
    frames.append(vec_env.get_frame(env_id, camera=camera, size=size))
    for t in range(len(actions)):
        # Apply action to single env
        act = actions[t]
        vec_env._step_single_env(env_id, act)
        if t % skip == 0:
            frames.append(vec_env.get_frame(env_id, camera=camera, size=size))
    return frames


def extract_grid_cells(grid_video_path, n_envs=15, cols=5):
    """Extract per-cell videos from a grid video."""
    cap = cv2.VideoCapture(grid_video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = (n_envs + cols - 1) // cols
    cell_w, cell_h = w // cols, h // rows

    cell_frames = [[] for _ in range(n_envs)]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i in range(n_envs):
            r, c = divmod(i, cols)
            cell = frame[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            cell_frames[i].append(cell)
    cap.release()
    return cell_frames, (cell_h, cell_w)


def make_side_by_side(frames_a, frames_b, label_a="Collection", label_b="Eval Sweep"):
    """Create side-by-side video frames with labels."""
    max_len = max(len(frames_a), len(frames_b))
    h_a, w_a = frames_a[0].shape[:2]
    h_b, w_b = frames_b[0].shape[:2]
    
    # Resize to same height
    target_h = max(h_a, h_b)
    combined = []
    
    for t in range(max_len):
        fa = frames_a[min(t, len(frames_a)-1)].copy()
        fb = frames_b[min(t, len(frames_b)-1)].copy()
        
        # Resize to same height
        if fa.shape[0] != target_h:
            scale = target_h / fa.shape[0]
            fa = cv2.resize(fa, (int(fa.shape[1]*scale), target_h))
        if fb.shape[0] != target_h:
            scale = target_h / fb.shape[0]
            fb = cv2.resize(fb, (int(fb.shape[1]*scale), target_h))
        
        # Add labels
        cv2.putText(fa, label_a, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(fb, label_b, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Separator
        sep = np.ones((target_h, 4, 3), dtype=np.uint8) * 128
        combined.append(np.concatenate([fa, sep, fb], axis=1))
    
    return combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True, help='Collection npz dir')
    p.add_argument('--positions_file', type=str, required=True, help='Eval positions JSON')
    p.add_argument('--eval_grid_video', type=str, required=True, help='Eval sweep grid video')
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--episodes_per_pos', type=int, default=34)
    p.add_argument('--camera', type=str, default='back')
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--frame_skip', type=int, default=3)
    p.add_argument('--frame_size', type=str, default='240,320', help='H,W')
    args = p.parse_args()

    frame_h, frame_w = [int(x) for x in args.frame_size.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load positions
    with open(args.positions_file) as f:
        positions = json.load(f)
    num_envs = len(positions)
    print(f"Positions: {num_envs}", flush=True)

    # Find best episode per position
    best_eps = []
    for i in range(num_envs):
        ep = find_best_episode(args.data_dir, i, args.episodes_per_pos)
        d = np.load(os.path.join(args.data_dir, f'ep_{ep:04d}.npz'))
        succ = "✓" if d['success'][0] else "✗"
        print(f"  pos{i:2d}: ep_{ep:04d} ({d['step_idx'].shape[0]} steps) {succ}", flush=True)
        best_eps.append(ep)

    # Extract eval grid cells
    print(f"\nExtracting eval grid cells from {args.eval_grid_video}...", flush=True)
    eval_cells, (cell_h, cell_w) = extract_grid_cells(args.eval_grid_video, num_envs)
    print(f"  Cell size: {cell_h}x{cell_w}, frames per cell: {[len(c) for c in eval_cells]}", flush=True)

    # Save eval cell videos
    for i in range(num_envs):
        if eval_cells[i]:
            path = os.path.join(args.output_dir, f'eval_sweep_pos{i:02d}.mp4')
            imageio.mimwrite(path, eval_cells[i], fps=args.fps, macro_block_size=1)

    # Create MuJoCo env (1 at a time) and replay
    print("\nCreating MuJoCo environment...", flush=True)
    import torch
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

    cube_positions = [p['cube_pos'] for p in positions]
    bowl_positions = [p['bowl_pos'] for p in positions]
    cube_yaw_degs = [p.get('cube_yaw_deg', 0.0) for p in positions]

    # Replay each position one by one using a single-env vec_env
    print("\nReplaying trajectories...", flush=True)
    collection_frames = []

    for i in range(num_envs):
        ep_idx = best_eps[i]
        d = np.load(os.path.join(args.data_dir, f'ep_{ep_idx:04d}.npz'))
        actions = d['action']  # (T, 7)
        success = d['success'][0]
        n_steps = d['step_idx'].shape[0]

        cp = cube_positions[i]
        if len(cp) == 2:
            cp = cp + [0.02]

        # Create env with this position's config
        vec_env = MuJoCoVecEnvPnP(
            num_envs=1,
            cube_pos=cp,
            bowl_positions=[bowl_positions[i]],
            cube_yaw_degs=[cube_yaw_degs[i]],
            random_cube_range=None,
            episode_positions_file=None,
            max_episode_steps=500,
            image_height=84,
            image_width=84,
            use_calibrated_wrist=True,
        )
        vec_env.reset()

        # obs_base_action is 7D: pos(3) + euler_xyz(3) + grip(1)
        # env.step() needs 8D: pos(3) + quat_xyzw(4) + grip(1)
        base_actions = d['obs_base_action']  # (T, 7)

        # Replay and capture frames
        frames = []
        frames.append(vec_env.get_frame(0, camera=args.camera, size=(frame_h, frame_w)))

        for t in range(n_steps):
            ba7 = base_actions[t]
            # Convert 7D → 8D
            act8 = np.zeros(8, dtype=np.float32)
            act8[:3] = ba7[:3]  # pos
            act8[3:7] = Rotation.from_euler("xyz", ba7[3:6]).as_quat()  # euler → quat_xyzw
            act8[7] = ba7[6]  # grip
            act_tensor = torch.tensor(act8, dtype=torch.float32).unsqueeze(0)
            vec_env.step(act_tensor, render_mode="none")
            if t % args.frame_skip == 0:
                frames.append(vec_env.get_frame(0, camera=args.camera, size=(frame_h, frame_w)))

        vec_env.close()

        tag = "succ" if success else "fail"
        print(f"  pos{i:2d}: {len(frames)} frames ({tag})", flush=True)

        # Save individual replay video
        path = os.path.join(args.output_dir, f'collect_pos{i:02d}_{tag}.mp4')
        imageio.mimwrite(path, frames, fps=args.fps, macro_block_size=1)
        collection_frames.append(frames)

    # Create side-by-side comparison videos
    print("\nCreating comparison videos...", flush=True)
    for i in range(num_envs):
        if not eval_cells[i]:
            continue
        sbs = make_side_by_side(collection_frames[i], eval_cells[i],
                                label_a="Collection(1env)", label_b="EvalSweep(15env)")
        path = os.path.join(args.output_dir, f'compare_pos{i:02d}.mp4')
        imageio.mimwrite(path, sbs, fps=args.fps, macro_block_size=1)

    # Create combined comparison grid (5 cols: collect | eval for each pos, 3 rows)
    print("\nCreating combined grid...", flush=True)
    cols = 5
    rows = (num_envs + cols - 1) // cols
    max_t = max(
        max((len(f) for f in collection_frames), default=1),
        max((len(f) for f in eval_cells), default=1),
    )
    
    # Each cell: collection on left, eval on right, stacked
    cell_size = (frame_h, frame_w)
    grid_frames = []
    for t in range(max_t):
        grid_rows = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                idx = r * cols + c
                if idx < num_envs:
                    # Collection frame
                    cf = collection_frames[idx]
                    ef = eval_cells[idx]
                    fa = cf[min(t, len(cf)-1)] if cf else np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    fb = ef[min(t, len(ef)-1)] if ef else np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    # Resize to same size
                    fa = cv2.resize(fa, (frame_w, frame_h))
                    fb = cv2.resize(fb, (frame_w, frame_h))
                    # Stack vertically: collection on top, eval on bottom
                    cv2.putText(fa, f"Col p{idx}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(fb, f"Eval p{idx}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cell = np.concatenate([fa, fb], axis=0)
                else:
                    cell = np.zeros((frame_h * 2, frame_w, 3), dtype=np.uint8)
                row_cells.append(cell)
            grid_rows.append(np.concatenate(row_cells, axis=1))
        grid_frames.append(np.concatenate(grid_rows, axis=0))

    path = os.path.join(args.output_dir, 'comparison_grid.mp4')
    imageio.mimwrite(path, grid_frames, fps=args.fps, macro_block_size=1)
    print(f"\nSaved comparison grid: {path} ({len(grid_frames)} frames)", flush=True)

    print("\nDone!", flush=True)
    vec_env.close()


if __name__ == '__main__':
    main()
