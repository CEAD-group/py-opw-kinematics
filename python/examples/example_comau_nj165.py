# %%
"""
Example: Basic forward and inverse kinematics with Comau NJ165 robot parameters
"""

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation

from py_opw_kinematics import KinematicModel, Robot

# %%
# Define robot geometry (Comau NJ165 parameters)
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
    has_parallelogram=True,
)

# Create robot with joint angles in degrees
robot = Robot(kinematic_model, degrees=True)

np.set_printoptions(precision=2, suppress=True)

# %%
# Define end effector transform (tool offset)
ee_rotation = Rotation.from_euler("XYZ", [0, -90, 0], degrees=True)
ee_transform = RigidTransform.from_components(rotation=ee_rotation, translation=[0, 0, 0])

# %%
# Test forward kinematics with expected values
print("Forward Kinematics Tests:")
# fmt: off
test_cases = [
    (( 0,  0, -90,   0,  0,  0), (2074.00,    0.00, 2255.00), (  0.0,   0.00,   0.00)),
    ((10,  0, -90,   0,  0,  0), (2042.49, -360.15, 2255.00), (  0.0,   0.00, -10.00)),
    (( 0, 10, -90,   0,  0,  0), (2278.04,    0.00, 2237.15), (  0.0,   0.00,   0.00)),
    (( 0,  0, -80,   0,  0,  0), (2005.16,    0.00, 2541.89), (  0.0, -10.00,   0.00)),
    (( 0,  0, -90,  10,  0,  0), (2074.00,    0.00, 2255.00), (-10.0,   0.00,   0.00)),
    (( 0,  0, -90,   0, 10,  0), (2070.51,    0.00, 2215.06), (  0.0,  10.00,   0.00)),
    (( 0,  0, -90,   0,  0, 10), (2074.00,    0.00, 2255.00), (-10.0,   0.00,   0.00)),
    ((10, 20, -90,  30, 20, 10), (2417.77, -466.26, 2116.01), (-37.35, 25.99,  -4.81)),
]
# fmt: on

for angles, t_exp, r_exp in test_cases:
    pose = robot.forward(joints=angles, ee_transform=ee_transform)
    t = pose.translation
    r = pose.rotation.as_euler("XYZ", degrees=True)

    pos_ok = np.allclose(t, t_exp, atol=1)
    rot_ok = np.allclose(r, r_exp, atol=1)

    status = "ok" if (pos_ok and rot_ok) else "FAIL"
    print(f"  {angles} -> r={r} {status}")

# %%
# Inverse kinematics example
print("\nInverse Kinematics:")
joints = (10, 0, -90, 0, 0, 0)
pose = robot.forward(joints, ee_transform=ee_transform)
print(f"  Input joints: {joints}")
print(f"  FK pose: {np.round(pose.translation, 2)}")

solutions = robot.inverse(pose, ee_transform=ee_transform)
print(f"  IK solutions ({len(solutions)} found):")
for i, sol in enumerate(solutions):
    print(f"    {i+1}: {np.round(sol, 2)}")

# %%
# Create pose using scipy RigidTransform
print("\nCreating pose with scipy RigidTransform:")
target_pose = RigidTransform.from_components(
    rotation=Rotation.from_euler("XYZ", [0, 0, 0], degrees=True),
    translation=[2000, 0, 2000],
)
print(f"  Created pose at: {target_pose.translation}")

solutions = robot.inverse(target_pose, ee_transform=ee_transform)
if solutions:
    pose_verify = robot.forward(solutions[0], ee_transform=ee_transform)
    euler = pose_verify.rotation.as_euler("XYZ", degrees=True)
    print(f"  FK verification: X={pose_verify.translation[0]:.1f} Y={pose_verify.translation[1]:.1f} Z={pose_verify.translation[2]:.1f}")

# %%
# Get all link frames
print("\nLink frames:")
frames = robot.forward_frames(joints, ee_transform=ee_transform)
frame_names = ["Base", "J1", "J2", "J3", "J4", "J5", "J6", "TCP"]
for i, name in enumerate(frame_names):
    print(f"  {name}: {np.round(frames[i].translation, 1)}")
