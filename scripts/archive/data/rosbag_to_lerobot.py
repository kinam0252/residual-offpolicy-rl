#!/usr/bin/env python3
"""Convert FR3 teleop ROS2 bags to LeRobot v3.0 dataset format.

Usage:
    python rosbag_to_lerobot.py \
        --input-dir  /path/to/Teleop/original/Lift \
        --output-dir /path/to/Teleop/lerobot/Lift \
        --fps 15 \
        --repo-id kinamkim/fr3_lift_teleop
"""

import argparse
import io
import logging
import shutil
import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── ROS topic mapping ────────────────────────────────────────────────
TOPICS = {
    "current_pose": "/current_pose",
    "target_pose": "/target_pose",
    "gripper": "/gripper/joint_states",
    "cam_base": "/cam_base/camera/color/image_raw/compressed",
    "cam_wrist": "/cam_hand/camera/color/image_raw/compressed",
}


# ── helpers ──────────────────────────────────────────────────────────
def find_nearest(timestamps: list[int], target: int) -> int:
    """Return index of nearest timestamp (nanosecond precision)."""
    idx = bisect_left(timestamps, target)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    if (timestamps[idx] - target) < (target - timestamps[idx - 1]):
        return idx
    return idx - 1


def pose_to_array(msg) -> np.ndarray:
    """PoseStamped → [x, y, z, qx, qy, qz, qw]."""
    p = msg.pose.position
    o = msg.pose.orientation
    return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)


def decode_compressed_image(msg, resize=None) -> Image.Image:
    """sensor_msgs/CompressedImage → PIL.Image (RGB), optionally resized."""
    img = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    if resize:
        img = img.resize((resize[1], resize[0]), Image.LANCZOS)  # PIL uses (W, H)
    return img


# ── bag reader ───────────────────────────────────────────────────────
def read_bag(bag_path: Path, typestore):
    """Read relevant topics from a ROS2 bag.

    Returns dict[str, list[(timestamp_ns, deserialized_msg)]].
    """
    data = {k: [] for k in TOPICS}
    topic_to_key = {v: k for k, v in TOPICS.items()}

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in topic_to_key]
        for conn, ts_ns, rawdata in reader.messages(connections=connections):
            key = topic_to_key[conn.topic]
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            data[key].append((ts_ns, msg))

    return data


# ── frame alignment ─────────────────────────────────────────────────
def align_frames(bag_data: dict, task: str, image_size=None) -> list[dict]:
    """Align all modalities to cam_base timestamps (reference clock).

    Returns list of frame dicts ready for dataset.add_frame().
    """
    ref_entries = bag_data["cam_base"]
    if not ref_entries:
        logger.warning("No cam_base images found – skipping episode.")
        return []

    # Pre-compute timestamp arrays for fast lookup
    ts_lookup = {
        key: [ts for ts, _ in bag_data[key]]
        for key in ["current_pose", "target_pose", "gripper", "cam_wrist"]
    }

    frames = []
    for i, (ref_ts, base_msg) in enumerate(ref_entries):
        frame = {}

        # ── images ───────────────────────────────────────────────
        frame["observation.images.cam_base"] = decode_compressed_image(base_msg, resize=image_size)

        wrist_idx = find_nearest(ts_lookup["cam_wrist"], ref_ts)
        _, wrist_msg = bag_data["cam_wrist"][wrist_idx]
        frame["observation.images.cam_wrist"] = decode_compressed_image(wrist_msg, resize=image_size)

        # ── gripper ──────────────────────────────────────────────
        gripper_idx = find_nearest(ts_lookup["gripper"], ref_ts)
        _, gripper_msg = bag_data["gripper"][gripper_idx]
        gripper_val = np.float32(gripper_msg.position[0])

        # ── observation.state = [ee_pose(7), gripper(1)] ────────
        pose_idx = find_nearest(ts_lookup["current_pose"], ref_ts)
        _, pose_msg = bag_data["current_pose"][pose_idx]
        ee_state = pose_to_array(pose_msg)
        frame["observation.state"] = np.concatenate([ee_state, [gripper_val]])

        # ── action = [target_pose(7), gripper(1)] ────────────────
        target_idx = find_nearest(ts_lookup["target_pose"], ref_ts)
        _, target_msg = bag_data["target_pose"][target_idx]
        ee_action = pose_to_array(target_msg)
        frame["action"] = np.concatenate([ee_action, [gripper_val]])

        frame["task"] = task

        frames.append(frame)

    return frames


# ── main entry ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ROS2 bag → LeRobot v3.0")
    parser.add_argument("--input-dir", type=str, required=True, help="Dir with episode bag folders")
    parser.add_argument("--output-dir", type=str, required=True, help="Output LeRobot dataset root")
    parser.add_argument("--repo-id", type=str, default="kinamkim/fr3_lift_teleop")
    parser.add_argument("--fps", type=int, default=15, help="Dataset fps (cam_base rate)")
    parser.add_argument("--task", type=str, default="lift")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=None,
                        help="Resize images to (H, W), e.g. --image-size 224 224")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Remove existing output dir (LeRobotDataset.create requires non-existent dir)
    if output_dir.exists():
        logger.info(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # ── discover episodes ────────────────────────────────────────
    episode_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("kinam")
    ])
    logger.info(f"Found {len(episode_dirs)} episode directories in {input_dir}")

    # ── define features ──────────────────────────────────────────
    img_h, img_w = args.image_size if args.image_size else (720, 1280)
    features = {
        "observation.images.cam_base": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        },
    }

    # ── create dataset ───────────────────────────────────────────
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=str(output_dir),
        robot_type="fr3",
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )
    logger.info(f"Created LeRobot dataset at {output_dir}")

    # ── process episodes ─────────────────────────────────────────
    t_start = time.time()
    n_total = len(episode_dirs)
    for ep_idx, ep_dir in enumerate(episode_dirs):
        t_ep_start = time.time()
        logger.info(f"[{ep_idx+1}/{n_total}] Processing {ep_dir.name} ...")

        # Read task
        task_file = ep_dir / "task.txt"
        task = task_file.read_text().strip() if task_file.exists() else args.task

        # Read bag
        bag_data = read_bag(ep_dir, typestore)

        # Align frames
        frames = align_frames(bag_data, task, image_size=args.image_size)
        if not frames:
            logger.warning(f"  Skipping {ep_dir.name} (no frames)")
            continue

        logger.info(f"  {len(frames)} frames aligned")

        # Add frames
        for frame in frames:
            dataset.add_frame(frame)

        # Save episode
        dataset.save_episode()

        # ETA calculation
        elapsed = time.time() - t_start
        ep_elapsed = time.time() - t_ep_start
        done = ep_idx + 1
        avg_per_ep = elapsed / done
        remaining = avg_per_ep * (n_total - done)
        eta_min, eta_sec = divmod(int(remaining), 60)
        logger.info(
            f"  Episode saved ({done}/{n_total}) | "
            f"this ep: {ep_elapsed:.1f}s | "
            f"elapsed: {elapsed:.0f}s | "
            f"ETA: {eta_min}m {eta_sec}s"
        )

        # Free memory
        del bag_data, frames

    # ── finalize ─────────────────────────────────────────────────
    dataset.finalize()
    logger.info(
        f"Done. {dataset.meta.info['total_episodes']} episodes, "
        f"{dataset.meta.info['total_frames']} frames."
    )


if __name__ == "__main__":
    main()
