import pyvista as pv
import os
import sys
import numpy as np

# Read output directory from command-line argument
if len(sys.argv) < 2:
    print("Usage: python generate_superquadrics_with_mtl.py <output_obj_directory>")
    sys.exit(1)

output_dir = sys.argv[1]
os.makedirs(output_dir, exist_ok=True)

# Fixed values for e1 and e2
e1_values = [1e-10]
e2_values = [1e-10] + np.arange(0.1, 2.0, 0.1).tolist() + [2.0]

def float_to_str(x):
    """Convert float to 2-digit string with no dot (e.g., 0.3 → '030', 1.25 → '125')"""
    return f"{x:.2f}".replace(".", "")

def generate_superquadric(e1, e2, index):
    """Generate a superquadric OBJ and MTL pair with red color"""
    mesh = pv.ParametricSuperEllipsoid(n1=e1, n2=e2)
    e1_str = float_to_str(e1)
    e2_str = float_to_str(e2)

    base_name = f"sq_{index:03}_e1_{e1_str}_e2_{e2_str}"
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    mtl_path = os.path.join(output_dir, f"{base_name}.mtl")

    # Export OBJ file
    mesh.save(obj_path)

    # Inject mtllib and usemtl into OBJ file
    with open(obj_path, "r") as f:
        lines = f.readlines()
    with open(obj_path, "w") as f:
        f.write(f"mtllib {base_name}.mtl\n")
        f.write("usemtl red_material\n")
        f.writelines(lines)

    # Write MTL file (red material)
    with open(mtl_path, "w") as f:
        f.write("newmtl red_material\n")
        f.write("Kd 1.0 0.0 0.0\n")  # Diffuse red
        f.write("Ka 0.0 0.0 0.0\n")  # Ambient black
        f.write("Ks 0.0 0.0 0.0\n")  # Specular black
        f.write("d 1.0\n")           # Full opacity
        f.write("Ns 10.0\n")         # Specular exponent

    print(f"Saved: {obj_path} and {mtl_path}")

# Generate all combinations
index = 0
for e1 in e1_values:
    for e2 in e2_values:
        generate_superquadric(e1, e2, index)
        index += 1
