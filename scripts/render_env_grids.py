"""Render initial frames for easy/normal/hard eval configs as 5x3 grids."""
import json, sys, os
import numpy as np
import mujoco
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)
RENDER_W, RENDER_H = 320, 240
OUTPUT_DIR = "outputs/env_grids"


def render_grid(positions, title, output_path):
    """Render 15 envs as 5x3 grid (5 base positions × 3 perturbations)."""
    n = len(positions)
    assert n == 15, f"Expected 15 positions, got {n}"

    # Create env with these positions
    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]
    cube_yaw_degs = [p.get("cube_yaw_deg", 0.0) for p in positions]

    from scipy.spatial.transform import Rotation
    cube_quats = []
    for yaw in cube_yaw_degs:
        q_xyzw = Rotation.from_euler("z", np.radians(yaw)).as_quat()
        cube_quats.append([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # wxyz

    vec_env = MuJoCoVecEnvPnP(
        scene_xml=SCENE_XML,
        num_envs=n,
        cube_positions=cube_positions,
        bowl_positions=bowl_positions,
    )
    # Set cube yaws
    for ei in range(n):
        env = vec_env._envs[ei]
        cube_qposadr = env.get("cube_qposadr")
        if cube_qposadr is not None:
            env["data"].qpos[cube_qposadr + 3:cube_qposadr + 7] = cube_quats[ei]
    # Forward to update rendering
    for ei in range(n):
        env = vec_env._envs[ei]
        mujoco.mj_forward(env["model"], env["data"])

    # Render each env
    frames = []
    for ei in range(n):
        env = vec_env._envs[ei]
        renderer = env["renderer_base"]
        renderer.update_scene(env["data"], camera=env["cam_base_id"])
        frame = renderer.render().copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (RENDER_W, RENDER_H))

        # Add label
        p = positions[ei]
        yaw = p.get("cube_yaw_deg", 0)
        dist = p.get("distance", 0)
        label = f"env{ei} yaw={yaw:+.0f} d={dist:.2f}"
        cv2.putText(frame_bgr, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        frames.append(frame_bgr)

    # Arrange as 5 rows × 3 cols (5 base positions, 3 perturbations each)
    rows = []
    for r in range(5):
        row_frames = [frames[r * 3 + c] for c in range(3)]
        rows.append(np.hstack(row_frames))
    grid = np.vstack(rows)

    # Add title
    title_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    grid = np.vstack([title_bar, grid])

    cv2.imwrite(output_path, grid)
    print(f"Saved {output_path} ({grid.shape[1]}x{grid.shape[0]})")

    # Cleanup
    for env in vec_env._envs:
        for key in list(env.keys()):
            if "renderer" in key:
                del env[key]
    del vec_env


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for level in ["easy", "normal", "hard"]:
        config_path = f"configs/pnp_eval_{level}.json"
        with open(config_path) as f:
            positions = json.load(f)

        output_path = os.path.join(OUTPUT_DIR, f"env_grid_{level}.png")
        render_grid(positions, f"Eval Envs: {level.upper()} (5 base x 3 perturb)", output_path)


if __name__ == "__main__":
    main()
