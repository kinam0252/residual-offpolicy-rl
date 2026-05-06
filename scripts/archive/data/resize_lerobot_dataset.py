#!/usr/bin/env python3
"""
Resize a LeRobot v3.0 video dataset from 640x360 to 224x224.
Copies parquet data (unchanged) and re-encodes videos at 224x224.
Updates meta/info.json and meta/stats.json accordingly.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def resize_video(src_path: str, dst_path: str, target_size: tuple[int, int], fps: int):
    """Resize a video file to target_size (width, height) using OpenCV."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    w, h = target_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(resized)

    cap.release()
    writer.release()

    # Re-encode with h264 using ffmpeg if available, otherwise keep mp4v
    try:
        import subprocess
        tmp_path = dst_path + ".tmp.mp4"
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", dst_path,
                "-c:v", "libx264", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-an", tmp_path,
            ],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0:
            os.replace(tmp_path, dst_path)
        else:
            # ffmpeg failed, keep the mp4v version
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # ffmpeg not available, keep mp4v encoded file
        pass


def main():
    parser = argparse.ArgumentParser(description="Resize LeRobot dataset videos to 224x224")
    parser.add_argument("--src", type=str, required=True, help="Source dataset directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination dataset directory")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224],
                        help="Target size (width height), default: 224 224")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    target_w, target_h = args.size

    print(f"Resizing dataset: {src_dir} -> {dst_dir}")
    print(f"Target size: {target_w}x{target_h}")

    # 1. Copy data/ directory (parquet files, unchanged)
    src_data = src_dir / "data"
    dst_data = dst_dir / "data"
    if src_data.exists():
        print("Copying parquet data files...")
        shutil.copytree(src_data, dst_data, dirs_exist_ok=True)

    # 2. Copy meta/ directory
    src_meta = src_dir / "meta"
    dst_meta = dst_dir / "meta"
    if src_meta.exists():
        print("Copying meta files...")
        shutil.copytree(src_meta, dst_meta, dirs_exist_ok=True)

    # 3. Update meta/info.json with new image dimensions
    info_path = dst_meta / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for key, feat in info["features"].items():
        if feat.get("dtype") == "video":
            feat["shape"] = [target_h, target_w, 3]
            if "video_info" in feat:
                feat["video_info"]["video.height"] = target_h
                feat["video_info"]["video.width"] = target_w

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Updated info.json: image shape -> [{target_h}, {target_w}, 3]")

    # 4. Update meta/stats.json - remove image stats (they'll be recomputed)
    stats_path = dst_meta / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        # Keep image stats but note they may need recomputation
        # The normalization stats for images from ImageNet are standard
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

    # 5. Resize all video files
    src_videos = src_dir / "videos"
    dst_videos = dst_dir / "videos"
    fps = info.get("fps", 15)

    video_files = sorted(src_videos.rglob("*.mp4"))
    print(f"Resizing {len(video_files)} video files...")

    for vf in tqdm(video_files, desc="Resizing videos"):
        rel = vf.relative_to(src_videos)
        dst_vf = dst_videos / rel
        dst_vf.parent.mkdir(parents=True, exist_ok=True)
        resize_video(str(vf), str(dst_vf), (target_w, target_h), fps)

    print(f"\nDone! New dataset at: {dst_dir}")
    print(f"Image size: {target_w}x{target_h}")


if __name__ == "__main__":
    main()
