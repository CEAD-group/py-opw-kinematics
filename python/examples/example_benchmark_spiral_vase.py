# %%
"""
Benchmark: Inverse kinematics for 250k positions on a spiral vase path.

Simulates a robot following a continuous 3D-printing vase-mode toolpath:
spiral ascent with a sinusoidal radius profile, tool pointing straight down.

Expected duration is about 1s.
"""

import time

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation

from py_opw_kinematics import KinematicModel, Robot

np.set_printoptions(precision=2, suppress=True)

# %%
# Robot setup (Comau NJ165 parameters)
kinematic_model = KinematicModel(
    a1=400,
    a2=-250,
    b=0,
    c1=830,
    c2=1175,
    c3=1444,
    c4=230,
    offsets=(0, 0, 0, 0, 0, 0),
    flip_axes=(True, False, True, True, False, True),
)
robot = Robot(kinematic_model, degrees=True)

# End effector: tool pointing straight down (-Z world direction)
ee_rotation = Rotation.from_euler("XYZ", [180, 0, 0], degrees=True)
ee_transform = RigidTransform.from_components(rotation=ee_rotation, translation=[0, 0, 0])

# %%
# Generate spiral vase path
N = 250_000
print(f"Generating {N:,} spiral vase positions...")
t0_gen = time.perf_counter()

turns = 200
t = np.linspace(0, 1, N)

# Vase radius profile: narrow at base and lip, widest in the middle
r_base, r_waist = 120, 320
radius = r_base + (r_waist - r_base) * np.sin(np.pi * t)

# Spiral angles
theta = 2 * np.pi * turns * t

# TCP positions: vase centred at x=1800mm, base at z=900mm, height 600mm
cx, base_z, vase_height = 1800, 900, 600
positions = np.column_stack([
    cx + radius * np.cos(theta),
    radius * np.sin(theta),
    base_z + vase_height * t,
])

# Constant orientation: tool pointing straight down
rot_down = Rotation.from_euler("XYZ", [180, 0, 0], degrees=True)
rotations = Rotation.from_quat(np.tile(rot_down.as_quat(), (N, 1)))

trajectory = RigidTransform.from_components(rotation=rotations, translation=positions)

t_gen = time.perf_counter() - t0_gen
print(f"  Generation time : {t_gen:.3f} s")
print(f"  X range         : {positions[:, 0].min():.0f} – {positions[:, 0].max():.0f} mm")
print(f"  Y range         : {positions[:, 1].min():.0f} – {positions[:, 1].max():.0f} mm")
print(f"  Z range         : {positions[:, 2].min():.0f} – {positions[:, 2].max():.0f} mm")
print(f"  Radius range    : {radius.min():.0f} – {radius.max():.0f} mm")

# %%
# Benchmark batch inverse kinematics
print(f"\nRunning batch IK for {N:,} poses...")

t0_ik = time.perf_counter()
joints = robot.batch_inverse(
    poses=trajectory,
    current_joints=(0, 0, -90, 0, 0, 0),
    ee_transform=ee_transform,
)
t_ik = time.perf_counter() - t0_ik

solved_mask = ~np.isnan(joints[:, 0])
n_solved = int(solved_mask.sum())

print(f"  IK time         : {t_ik:.3f} s  ({N / t_ik / 1e6:.2f} M poses/s)")
print(f"  Solved          : {n_solved:,} / {N:,}  ({100 * n_solved / N:.1f}%)")

# %%
# Accuracy check on a random sample of solved poses
print("\nAccuracy check (50 random solved poses):")

solved_idx = np.where(solved_mask)[0]
rng = np.random.default_rng(42)
sample_idx = rng.choice(solved_idx, size=min(50, len(solved_idx)), replace=False)

if len(sample_idx) == 0:
    print("  No solved poses available; skipping accuracy check.")
else:
    errors_t, errors_r = [], []
    for i in sample_idx:
        pose_orig = trajectory[i]
        pose_fk = robot.forward(tuple(joints[i]), ee_transform=ee_transform)
        errors_t.append(np.linalg.norm(pose_orig.translation - pose_fk.translation))
        errors_r.append(np.degrees((pose_orig.rotation.inv() * pose_fk.rotation).magnitude()))

    print(f"  Max translation error : {max(errors_t):.2e} mm")
    print(f"  Max rotation error    : {max(errors_r):.2e} deg")
