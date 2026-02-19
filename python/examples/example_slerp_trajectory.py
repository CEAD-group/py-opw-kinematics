# %%
"""
Example: SLERP trajectory interpolation with py-opw-kinematics

This example demonstrates:
1. Creating poses from Euler angles using scipy
2. Interpolating between poses using SLERP (for rotation) and linear (for translation)
3. Computing inverse kinematics for the interpolated trajectory
"""

import numpy as np
from scipy.spatial.transform import Rotation, RigidTransform, Slerp

from py_opw_kinematics import KinematicModel, Robot, interpolate_poses

# %%
# Create robot with Comau NJ165 parameters
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

# %%
# Define start and end poses using scipy RigidTransform
pose_start = RigidTransform.from_components(
    rotation=Rotation.from_euler("XYZ", [0, 0, -10], degrees=True),
    translation=[2000, -200, 2000],
)

pose_end = RigidTransform.from_components(
    rotation=Rotation.from_euler("XYZ", [-30, 20, 10], degrees=True),
    translation=[2200, 200, 1800],
)

print("Start pose:", pose_start.translation, pose_start.rotation.as_euler("XYZ", degrees=True))
print("End pose:", pose_end.translation, pose_end.rotation.as_euler("XYZ", degrees=True))

# %%
# Interpolate between poses using SLERP for rotation + linear for translation
# This provides smooth rotation interpolation without gimbal lock issues

keyframes = RigidTransform.concatenate([pose_start, pose_end])
xn = np.linspace(0, 1, 11)  # 11 points = 10 segments
trajectory = interpolate_poses([0, 1], keyframes, xn)

print(f"\nInterpolated trajectory ({len(trajectory)} poses):")
for i in range(len(trajectory)):
    pos = trajectory[i].translation
    euler = trajectory[i].rotation.as_euler("XYZ", degrees=True)
    print(f"  {i:2d}: X={pos[0]:7.1f} Y={pos[1]:7.1f} Z={pos[2]:7.1f} "
          f"A={euler[0]:6.1f} B={euler[1]:6.1f} C={euler[2]:6.1f}")

# %%
# Compute inverse kinematics for the trajectory
joints_array = robot.batch_inverse(trajectory)
print("\nJoint angles for trajectory:")
print(joints_array)

# %%
# Verify by computing forward kinematics and comparing
# This demonstrates the round-trip accuracy

print("\nVerification (FK from computed joints):")
for i, row in enumerate(joints_array):
    pose_original = trajectory[i]
    joints = tuple(row)
    pose_computed = robot.forward(joints)

    # Compare translations
    trans_error = np.linalg.norm(
        pose_original.translation - pose_computed.translation
    )

    # Compare rotations (using angular distance)
    rot_diff = pose_original.rotation.inv() * pose_computed.rotation
    angle_error = np.degrees(rot_diff.magnitude())

    print(f"  {i:2d}: translation error = {trans_error:.4f} mm, "
          f"rotation error = {angle_error:.4f} deg")

# %%
# Advanced: Custom interpolation with ease-in-ease-out

# Extract rotations and create SLERP interpolator
rotations = Rotation.concatenate([pose_start.rotation, pose_end.rotation])
slerp = Slerp([0, 1], rotations)

# Interpolate at custom time values with smoothstep for ease-in-ease-out
t_values = np.linspace(0, 1, 21)
t_smooth = 3 * t_values**2 - 2 * t_values**3

interpolated_rotations = slerp(t_smooth)

# Linear interpolation for translation with same easing
t_col = t_smooth[:, np.newaxis]
positions = (1 - t_col) * pose_start.translation + t_col * pose_end.translation

custom_trajectory = RigidTransform.from_components(
    rotation=interpolated_rotations,
    translation=positions,
)

print(f"\nCustom trajectory with ease-in-ease-out ({len(custom_trajectory)} poses):")
custom_joints = robot.batch_inverse(custom_trajectory)
print(custom_joints[:5])
print("...")
print(custom_joints[-5:])

# %%
# Working directly with scipy Rotation for more complex operations

# Create rotation from quaternion (scipy uses x, y, z, w order)
quat_xyzw = [0, 0.707, 0, 0.707]  # 90 degree rotation around Y
rotation = Rotation.from_quat(quat_xyzw)

pose_from_quat = RigidTransform.from_components(
    rotation=rotation,
    translation=np.array([2000, 0, 2000]),
)

# Convert to different representations
print("\nRotation representations:")
print(f"  Quaternion (x,y,z,w): {rotation.as_quat()}")
print(f"  Euler XYZ (deg): {rotation.as_euler('XYZ', degrees=True)}")
print(f"  Rotation vector: {rotation.as_rotvec()}")

# Compute IK for this pose
solutions = robot.inverse(pose_from_quat)
print(f"\nIK solutions: {len(solutions)}")
for i, sol in enumerate(solutions):
    print(f"  {i}: {np.round(sol, 2)}")
