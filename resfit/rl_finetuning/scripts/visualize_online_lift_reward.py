from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def _to_hwc(frame_chw: np.ndarray, size: int = 384) -> np.ndarray:
    frame = np.transpose(frame_chw, (1, 2, 0)).astype(np.uint8)
    if frame.shape[0] != size or frame.shape[1] != size:
        y_idx = np.linspace(0, frame.shape[0] - 1, size).astype(np.int32)
        x_idx = np.linspace(0, frame.shape[1] - 1, size).astype(np.int32)
        frame = frame[np.ix_(y_idx, x_idx)]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize online buffer with lift/reward overlay")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_mp4", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    d = np.load(args.input_npz, allow_pickle=True)
    frames = d["front_image"].astype(np.uint8)
    reward = d["reward"].astype(np.float32)
    lift_delta = d["lift_delta"].astype(np.float32) if "lift_delta" in d.files else np.zeros((frames.shape[0],), dtype=np.float32)
    threshold_m = float(d["lift_reward_threshold_m"].item()) if "lift_reward_threshold_m" in d.files else 0.01
    done = d["done"].astype(np.bool_) if "done" in d.files else np.zeros((frames.shape[0],), dtype=np.bool_)

    out = Path(args.output_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out.as_posix(), fps=args.fps, codec="libx264", macro_block_size=1) as writer:
        for i in range(frames.shape[0]):
            rgb = _to_hwc(frames[i], size=384)
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)

            lift_cm = float(lift_delta[i] * 100.0)
            threshold_cm = threshold_m * 100.0
            rew = float(reward[i])
            done_i = bool(done[i])

            draw.rectangle([(0, 0), (383, 82)], fill=(0, 0, 0))
            draw.text((8, 6), f"step={i}", fill=(255, 255, 255))
            draw.text((8, 26), f"lift_cm={lift_cm:.3f}  threshold_cm={threshold_cm:.3f}", fill=(255, 255, 255))
            draw.text((8, 46), f"reward={rew:.1f}  done={int(done_i)}", fill=(255, 255, 255))

            if rew >= 0.5:
                draw.rectangle([(290, 8), (378, 38)], fill=(0, 128, 0))
                draw.text((300, 16), "R=1", fill=(255, 255, 255))
            else:
                draw.rectangle([(290, 8), (378, 38)], fill=(128, 0, 0))
                draw.text((300, 16), "R=0", fill=(255, 255, 255))

            writer.append_data(np.asarray(img, dtype=np.uint8))

    max_lift_cm = float(lift_delta.max() * 100.0) if lift_delta.size > 0 else 0.0
    first_reward_step = int(np.argmax(reward >= 1.0)) if np.any(reward >= 1.0) else -1
    print(f"saved_video={out}")
    print(f"frames={frames.shape[0]}")
    print(f"threshold_cm={threshold_cm:.3f}")
    print(f"max_lift_cm={max_lift_cm:.3f}")
    print(f"first_reward_step={first_reward_step}")


if __name__ == "__main__":
    main()
