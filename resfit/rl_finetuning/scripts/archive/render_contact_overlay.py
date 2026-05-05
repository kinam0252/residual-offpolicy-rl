#!/usr/bin/env python3
"""Render drawer front-view from MuJoCo and overlay contact points.

For each drawer (D2, D3, D4):
  1. Place a virtual camera looking straight at the drawer face
  2. Render the MuJoCo scene
  3. Project training-data and policy-eval contact points onto the image
  4. Save annotated image
"""
import sys, os, json, pathlib, argparse
import numpy as np
import mujoco
import cv2

# ── Paths ──
REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    CABINET_LONG, CABINET_SHORT, CABINET_HEIGHT, WALL_THICK, NUM_DRAWERS,
    DRAWER_SLIDE, DRAWER_GAP, DRAWER_SLOT_HEIGHT,
    ACTIVE_DRAWERS, FLOOR_TO_DRAWER,
    load_cabinet_placement, make_model_with_cabinet,
)

# Face geometry (cabinet-local frame)
D_HX = (CABINET_LONG - 2 * WALL_THICK) / 2 - DRAWER_GAP
FACE_X_CLOSED = D_HX + WALL_THICK / 2
FACE_HY = CABINET_SHORT / 2 - DRAWER_GAP
FACE_HZ = DRAWER_SLOT_HEIGHT / 2 - DRAWER_GAP


def drawer_z_center(drawer_idx: int) -> float:
    """Cabinet-local Z center of a drawer face."""
    slot = WALL_THICK + DRAWER_SLOT_HEIGHT / 2 + drawer_idx * (DRAWER_SLOT_HEIGHT + WALL_THICK)
    return slot


def load_training_contacts(data_dir: str) -> dict:
    """Load training demo contact points per drawer.

    observation.state is in world frame: [x, y, z, qx, qy, qz, qw, grip].
    Cabinet orientation: local_Y = world_X - cab_x, local_Z = world_Z - cab_z.
    """
    import pandas as pd
    data_path = pathlib.Path(data_dir)

    # Cabinet world position (from cabinet_placement.json, verified)
    CAB_X_WORLD = 0.5177
    CAB_Z_WORLD = 0.005
    # Drawer face Z centers in cabinet-local frame
    DZ_LOCAL = {d: drawer_z_center(d) for d in [2, 3, 4]}
    # Corresponding world Z = CAB_Z_WORLD + dz_local
    DRAWER_Z_WORLD = {d: CAB_Z_WORLD + DZ_LOCAL[d] for d in [2, 3, 4]}

    contacts = {d: {"y": [], "z": []} for d in [2, 3, 4]}
    for i in range(33):
        fp = data_path / f"episode_{i:06d}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        states = np.stack(df["observation.state"].values)
        n = len(states)
        late_z = states[int(n * 0.5):, 2].mean()
        drawer = min(DRAWER_Z_WORLD, key=lambda d: abs(late_z - DRAWER_Z_WORLD[d]))

        # Push = TCP world_x beyond cabinet front face
        # Cabinet local_x = -(world_y - cab_y), but for push detection use world position
        push_mask = states[:, 0] > 0.45
        if push_mask.sum() > 0:
            # Cabinet local Y = world X - cab_x (side position)
            local_y = states[push_mask, 0] - CAB_X_WORLD
            y_norm = local_y / FACE_HY
            # Cabinet local Z = world Z - cab_z, then normalize to drawer face
            local_z = states[push_mask, 2] - CAB_Z_WORLD
            z_norm = (local_z - DZ_LOCAL[drawer]) / FACE_HZ
            contacts[drawer]["y"].extend(y_norm.tolist())
            contacts[drawer]["z"].extend(z_norm.tolist())
    return contacts


def load_eval_contacts(eval_dir: str) -> dict:
    """Load policy eval contact points per drawer."""
    contacts = {d: {"y": [], "z": []} for d in [2, 3, 4]}
    for f in sorted(pathlib.Path(eval_dir).glob("ep*.json")):
        d = json.load(open(f))
        drawer = d["drawer"]
        for c in d["contact_points"]:
            contacts[drawer]["y"].append(c["y_norm"])
            contacts[drawer]["z"].append(c["z_norm"])
    return contacts


def render_drawer_front(drawer_idx: int, img_w: int = 800, img_h: int = 600):
    """Render the drawer from the front using a virtual camera.
    Returns (image, cam_params) where cam_params has info for projection."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
        load_calib, _DEFAULT_CALIB_DRAWER
    )

    T_base_cam = load_calib(_DEFAULT_CALIB_DRAWER)
    cab_pos, cab_euler = load_cabinet_placement()

    xml_str = make_model_with_cabinet(T_base_cam, cab_pos, cab_euler)
    model = xml_str  # make_model_with_cabinet returns MjModel directly
    data = mujoco.MjData(model)

    # Open the drawer slightly so face is visible
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import set_drawer_pos
    set_drawer_pos(model, data, drawer_idx, DRAWER_SLIDE)
    mujoco.mj_forward(model, data)

    # Get cabinet world position/orientation
    cab_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cabinet")
    cab_pos_w = data.xpos[cab_body_id].copy()
    cab_mat_w = data.xmat[cab_body_id].reshape(3, 3).copy()

    # Drawer face center in world frame
    dz = drawer_z_center(drawer_idx)
    face_center_local = np.array([FACE_X_CLOSED + DRAWER_SLIDE, 0.0, dz])
    face_center_world = cab_pos_w + cab_mat_w @ face_center_local

    # Camera: look at face from front, offset in cabinet X direction
    cam_dist = 0.25
    cam_lookat = face_center_world.copy()

    # Cabinet X-axis in world (front direction)
    cab_x_world = cab_mat_w[:, 0]

    # Compute azimuth for MuJoCo free camera
    # From test: cabinet front is world -Y, and az=90 gives the front view.
    # MuJoCo free cam: az=90 places camera on -Y side looking at +Y.
    # General formula: az = 90 + angle of cab_front from -Y axis
    # cab_x_world = (0, -1, 0) → angle_from_negY = 0 → az = 90
    angle_from_neg_y = np.arctan2(-cab_x_world[0], -cab_x_world[1])
    cam_azimuth = 90.0 + np.degrees(angle_from_neg_y)

    # Create offscreen renderer with custom camera
    renderer = mujoco.Renderer(model, height=img_h, width=img_w)

    # Set up scene camera
    scene_option = mujoco.MjvOption()
    # Hide robot body to show only cabinet
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    for i in range(mujoco.mjtRndFlag.mjNRNDFLAG):
        pass  # Keep defaults

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_lookat
    cam.distance = cam_dist
    cam.azimuth = cam_azimuth
    cam.elevation = 0.0

    # Make robot invisible by setting alpha to 0 for robot geoms
    # Save and restore later
    orig_rgba = {}
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name is None:
            name = ""
        # Cabinet geoms start with "cab_" or "drawer_"
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        # Keep cabinet, table, and world. Hide everything else (robot).
        if (not name.startswith("cab_") and not name.startswith("drawer_")
                and not name.startswith("table") and body_name not in ("world", "cabinet")
                and "drawer" not in body_name):
            orig_rgba[gid] = model.geom_rgba[gid].copy()
            model.geom_rgba[gid] = [0, 0, 0, 0]  # invisible

    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    img = renderer.render().copy()

    # Restore
    for gid, rgba in orig_rgba.items():
        model.geom_rgba[gid] = rgba

    renderer.close()

    return img, {
        "cab_pos_w": cab_pos_w,
        "cab_mat_w": cab_mat_w,
        "face_center_world": face_center_world,
        "cam_lookat": cam_lookat,
        "cam_dist": cam_dist,
        "dz": dz,
        "img_w": img_w,
        "img_h": img_h,
    }


def project_contacts_to_image(y_norms, z_norms, cam_params, drawer_idx):
    """Project normalised contact coords (Y,Z on face) to pixel coords.

    The front-view camera looks straight at the drawer face.
    Cabinet-local Y maps to camera horizontal, Z maps to vertical.
    Camera horizontal is along -cabinet_Y when viewed from front (+cab_X side).
    """
    img_w = cam_params["img_w"]
    img_h = cam_params["img_h"]
    cam_dist = cam_params["cam_dist"]

    fovy = 45.0
    fovy_rad = np.radians(fovy)

    visible_h = 2 * cam_dist * np.tan(fovy_rad / 2)
    visible_w = visible_h * (img_w / img_h)

    ppm_v = img_h / visible_h
    ppm_h = img_w / visible_w

    cx, cy = img_w / 2, img_h / 2

    y_arr = np.asarray(y_norms)
    z_arr = np.asarray(z_norms)

    # At az=90, camera horizontal = world +X = cabinet +Y
    # So cabinet Y_norm positive → pixel right (positive px offset)
    px = cx + y_arr * FACE_HY * ppm_h
    # +Z is up in world, -pixel is up on screen
    py = cy - z_arr * FACE_HZ * ppm_v

    return px, py


def draw_face_rect(img, cam_params):
    """Draw the drawer face boundary rectangle on the image."""
    img_w = cam_params["img_w"]
    img_h = cam_params["img_h"]
    cam_dist = cam_params["cam_dist"]
    fovy_rad = np.radians(45.0)
    visible_h = 2 * cam_dist * np.tan(fovy_rad / 2)
    visible_w = visible_h * (img_w / img_h)
    ppm_v = img_h / visible_h
    ppm_h = img_w / visible_w
    cx, cy = img_w / 2, img_h / 2

    x1 = int(cx - FACE_HY * ppm_h)
    x2 = int(cx + FACE_HY * ppm_h)
    y1 = int(cy - FACE_HZ * ppm_v)
    y2 = int(cy + FACE_HZ * ppm_v)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return img


def make_overlay(drawer_idx, train_contacts, eval_contacts, img_w=800, img_h=600):
    """Create one image: MuJoCo render + contact overlay for a single drawer."""
    img, cam_params = render_drawer_front(drawer_idx, img_w, img_h)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw face boundary
    img = draw_face_rect(img, cam_params)

    # Crosshair at center
    cx, cy = img_w // 2, img_h // 2
    cv2.line(img, (cx - 30, cy), (cx + 30, cy), (200, 200, 200), 1)
    cv2.line(img, (cx, cy - 30), (cx, cy + 30), (200, 200, 200), 1)

    # --- Training data contacts (blue) ---
    ty = train_contacts[drawer_idx]["y"]
    tz = train_contacts[drawer_idx]["z"]
    if len(ty) > 0:
        tpx, tpy = project_contacts_to_image(ty, tz, cam_params, drawer_idx)
        for x, y in zip(tpx.astype(int), tpy.astype(int)):
            cv2.circle(img, (x, y), 2, (255, 150, 50), -1)  # blue-ish
        # Mean
        mx, my = int(np.mean(tpx)), int(np.mean(tpy))
        cv2.drawMarker(img, (mx, my), (255, 100, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
        cv2.putText(img, "Demo mean", (mx + 10, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    # --- Eval contacts (red) ---
    ey = eval_contacts[drawer_idx]["y"]
    ez = eval_contacts[drawer_idx]["z"]
    if len(ey) > 0:
        epx, epy = project_contacts_to_image(ey, ez, cam_params, drawer_idx)
        for x, y in zip(epx.astype(int), epy.astype(int)):
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # red
        # Mean
        mx, my = int(np.mean(epx)), int(np.mean(epy))
        cv2.drawMarker(img, (mx, my), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 15, 2)
        cv2.putText(img, "Policy mean", (mx + 10, my + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    # Title
    drawer_names = {2: "D2 (Top)", 3: "D3 (Middle)", 4: "D4 (Bottom)"}
    title = f"Drawer {drawer_idx} - {drawer_names.get(drawer_idx, '')} - Contact Overlay"
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Legend
    ly = img_h - 60
    cv2.circle(img, (15, ly), 4, (255, 150, 50), -1)
    cv2.putText(img, "Training demo", (25, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 50), 1)
    cv2.circle(img, (15, ly + 20), 4, (0, 0, 255), -1)
    cv2.putText(img, "Policy eval (ckpt 100k)", (25, ly + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.rectangle(img, (140, ly + 32), (170, ly + 42), (255, 255, 0), 2)
    cv2.putText(img, "Drawer face boundary", (175, ly + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return img


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/datasets/CloseDrawer_sim_33ep/data/chunk-000"))
    parser.add_argument("--eval_dir", type=str,
                        default="outputs/drawer_contact_analysis")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/Repos/Intern/.VIEW/drawer_contact_v2"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training demo contacts...")
    train_contacts = load_training_contacts(args.train_data)
    print("Loading policy eval contacts...")
    eval_contacts = load_eval_contacts(args.eval_dir)

    images = []
    for drawer_idx in [2, 3, 4]:
        print(f"Rendering D{drawer_idx}...")
        img = make_overlay(drawer_idx, train_contacts, eval_contacts, args.width, args.height)
        save_path = out_dir / f"contact_overlay_d{drawer_idx}.png"
        cv2.imwrite(str(save_path), img)
        print(f"  Saved: {save_path}")
        images.append(img)

    # Also create a combined 3-panel image
    combined = np.hstack(images)
    combined_path = out_dir / "contact_overlay_all.png"
    cv2.imwrite(str(combined_path), combined)
    print(f"Combined saved: {combined_path}")
    print("Done.")


if __name__ == "__main__":
    main()
