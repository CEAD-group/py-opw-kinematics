# %%
"""
Example: Batch operations for forward and inverse kinematics
"""

from py_opw_kinematics import KinematicModel, Robot, interpolate_poses
from scipy.spatial.transform import RigidTransform, Rotation
import numpy as np

np.set_printoptions(precision=2, suppress=True)

# %%
kinematic_model = KinematicModel(
    a1=400.333,
    a2=-251.449,
    b=0,
    c1=830,
    c2=1177.556,
    c3=1443.593,
    c4=230,
    offsets=(0, 0, 0, 0, 0, 0),
    flip_axes=(True, False, True, True, False, True),
)

robot = Robot(kinematic_model, degrees=True)

# Create end effector transformation
ee_rotation = Rotation.from_euler("xyz", [0, -90, 0], degrees=True)
ee_transform = RigidTransform.from_components(
    rotation=ee_rotation, translation=[0, 0, 0]
)

# %%
# Create a trajectory of poses using interpolation
n = 1000
print(f"Creating {n} poses...")

pose_start = RigidTransform.from_components(
    rotation=Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
    translation=[1800, -500, 1000],
)
pose_end = RigidTransform.from_components(
    rotation=Rotation.from_euler("xyz", [10, 10, 10], degrees=True),
    translation=[2100, 500, 2000],
)

keyframes = RigidTransform.concatenate([pose_start, pose_end])
trajectory = interpolate_poses([0, 1], keyframes, np.linspace(0, 1, n))

# %%
# Batch inverse kinematics
print("Running batch inverse kinematics...")
result = robot.batch_inverse(
    poses=trajectory, current_joints=(0, 0, -90, 0, 0, 0), ee_transform=ee_transform
)

print(f"Result shape: {result.shape}")
print("First 5 rows:")
print(result[:5])
print("...")
print("Last 5 rows:")
print(result[-5:])

# %%
# Batch forward kinematics
print("\nRunning batch forward kinematics...")
poses_result = robot.batch_forward(result, ee_transform=ee_transform)
print(f"Got {len(poses_result)} poses back")

# Verify first and last
print(f"\nFirst pose translation: {np.round(poses_result[0].translation, 2)}")
print(f"Last pose translation: {np.round(poses_result[-1].translation, 2)}")
