import argparse
import os
import random

import numpy as np
import pyvista as pv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random superquadric OBJ+MTL pairs."
    )
    parser.add_argument("output_dir", type=str, help="Directory to write OBJ/MTL files")
    parser.add_argument(
        "--num",
        type=int,
        default=200,
        help="Number of random shapes to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.3,
        help="Minimum uniform scale factor (default: 0.05)",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.0,
        help="Maximum uniform scale factor (default: 0.25)",
    )
    parser.add_argument(
        "--scale-min-x",
        type=float,
        default=None,
        help="Minimum X scale factor (fallback: --scale-min)",
    )
    parser.add_argument(
        "--scale-max-x",
        type=float,
        default=None,
        help="Maximum X scale factor (fallback: --scale-max)",
    )
    parser.add_argument(
        "--scale-min-y",
        type=float,
        default=None,
        help="Minimum Y scale factor (fallback: --scale-min)",
    )
    parser.add_argument(
        "--scale-max-y",
        type=float,
        default=None,
        help="Maximum Y scale factor (fallback: --scale-max)",
    )
    parser.add_argument(
        "--scale-min-z",
        type=float,
        default=None,
        help="Minimum Z scale factor (fallback: --scale-min)",
    )
    parser.add_argument(
        "--scale-max-z",
        type=float,
        default=None,
        help="Maximum Z scale factor (fallback: --scale-max)",
    )
    parser.add_argument(
        "--parts-min",
        type=int,
        default=1,
        help="Minimum number of parts to combine per shape (default: 1)",
    )
    parser.add_argument(
        "--parts-max",
        type=int,
        default=3,
        help="Maximum number of parts to combine per shape (default: 3)",
    )
    parser.add_argument(
        "--offset-max",
        type=float,
        default=0.2,
        help="Max translation (meters) per part around the origin when combining parts",
    )
    parser.add_argument(
        "--shape-types",
        type=str,
        default="superquadric,box,sphere,cylinder,cone",
        help=(
            "Comma-separated shape types for extra parts (first part is always superquadric). "
            "Supported: superquadric, box, sphere, cylinder, cone"
        ),
    )
    return parser.parse_args()


def float_to_str(x: float) -> str:
    """Convert float to 2-digit string with no dot (e.g., 0.3 → '030', 1.25 → '125')."""
    return f"{x:.2f}".replace(".", "")


COLORS = {
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}


def write_mtl(mtl_path: str, material_name: str, rgb: tuple[float, float, float]) -> None:
    """Write a simple MTL with a single diffuse color."""
    r, g, b = rgb
    with open(mtl_path, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")  # Diffuse color
        f.write("Ka 0.0 0.0 0.0\n")  # Ambient black
        f.write("Ks 0.0 0.0 0.0\n")  # Specular black
        f.write("d 1.0\n")  # Full opacity
        f.write("Ns 10.0\n")  # Specular exponent


def generate_superquadric(
    output_dir: str,
    n1: float,
    n2: float,
    index: int,
    color_name: str,
    scale: np.ndarray,
    parts: int,
    offsets: list[np.ndarray],
    part_types: list[str],
) -> None:
    """Generate a (possibly compound) superquadric OBJ and MTL pair with a chosen color."""
    meshes = []
    shape_tokens = []
    for part_idx in range(parts):
        shape = part_types[part_idx]
        if shape == "superquadric":
            mesh = pv.ParametricSuperEllipsoid(n1=n1, n2=n2)
            shape_tokens.append("sq")
        elif shape == "box":
            mesh = pv.Box()  # unit cube
            shape_tokens.append("bx")
        elif shape == "sphere":
            mesh = pv.Sphere(radius=0.5)
            shape_tokens.append("sp")
        elif shape == "cylinder":
            mesh = pv.Cylinder(radius=0.5, height=1.0)
            shape_tokens.append("cy")
        elif shape == "cone":
            mesh = pv.Cone(radius=0.5, height=1.0)
            shape_tokens.append("co")
        else:
            raise ValueError(f"Unsupported shape type: {shape}")

        mesh.scale(scale, inplace=True)
        offset = offsets.pop()
        mesh.translate(offset, inplace=True)
        meshes.append(mesh)

    # Combine parts into a single mesh
    mesh = pv.merge(meshes)  # combine parts
    mesh.clean(inplace=True)
    n1_str = float_to_str(n1)
    n2_str = float_to_str(n2)
    sx, sy, sz = scale.tolist()

    base_name = (
        f"sq_{index:03}_n1_{n1_str}_n2_{n2_str}_"
        f"sx_{sx:.3f}_sy_{sy:.3f}_sz_{sz:.3f}_{color_name}_p{parts}_t_{'-'.join(shape_tokens)}"
    )
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    mtl_path = os.path.join(output_dir, f"{base_name}.mtl")

    mesh.save(obj_path)

    with open(obj_path, "r") as f:
        lines = f.readlines()
    with open(obj_path, "w") as f:
        f.write(f"mtllib {base_name}.mtl\n")
        f.write(f"usemtl {color_name}_material\n")
        f.writelines(lines)

    write_mtl(mtl_path, f"{color_name}_material", COLORS[color_name])
    print(f"Saved: {obj_path} and {mtl_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Sample exponents in a reasonable superquadric range.
    n1_samples = np.random.uniform(0.05, 2.0, size=args.num)
    n2_samples = np.random.uniform(0.05, 2.0, size=args.num)

    scale_min_x = args.scale_min_x if args.scale_min_x is not None else args.scale_min
    scale_max_x = args.scale_max_x if args.scale_max_x is not None else args.scale_max
    scale_min_y = args.scale_min_y if args.scale_min_y is not None else args.scale_min
    scale_max_y = args.scale_max_y if args.scale_max_y is not None else args.scale_max
    scale_min_z = args.scale_min_z if args.scale_min_z is not None else args.scale_min
    scale_max_z = args.scale_max_z if args.scale_max_z is not None else args.scale_max

    sx_samples = np.random.uniform(scale_min_x, scale_max_x, size=args.num)
    sy_samples = np.random.uniform(scale_min_y, scale_max_y, size=args.num)
    sz_samples = np.random.uniform(scale_min_z, scale_max_z, size=args.num)
    scale_samples = np.stack([sx_samples, sy_samples, sz_samples], axis=1)
    parts_samples = np.random.randint(args.parts_min, args.parts_max + 1, size=args.num)
    offsets = np.random.uniform(-args.offset_max, args.offset_max, size=(args.num, args.parts_max, 3))
    extra_shape_types = [s.strip() for s in args.shape_types.split(",") if s.strip()]

    for idx, (n1, n2, scale, part_count) in enumerate(
        zip(n1_samples, n2_samples, scale_samples, parts_samples)
    ):
        color = random.choice(list(COLORS.keys()))
        # select as many offsets as parts for this sample
        part_offsets = [offsets[idx, j] for j in range(part_count)]
        # first part always superquadric to preserve n1/n2 meaning; remaining parts sampled
        part_types = ["superquadric"]
        if part_count > 1:
            sampled = [random.choice(extra_shape_types) for _ in range(part_count - 1)]
            part_types.extend(sampled)
        generate_superquadric(
            args.output_dir, n1, n2, idx, color, scale, part_count, part_offsets, part_types
        )


if __name__ == "__main__":
    main()
