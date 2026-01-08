# %%
from py_opw_kinematics import KinematicModel, Robot, EulerConvention
from scipy.spatial.transform import RigidTransform, Rotation
import numpy as np
import polars as pl

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
    has_parallelogram=True,
)
euler_convention = EulerConvention("XYZ", extrinsic=True, degrees=True)
robot = Robot(kinematic_model, euler_convention)

# Create the EE transformation that was previously set in constructor
ee_rotation = Rotation.from_euler("XYZ", [0, -90, 0], degrees=True)
ee_transform = RigidTransform.from_components(
    rotation=ee_rotation, translation=[0, 0, 0]
)

n = 1000
poses_df = pl.DataFrame(
    {
        "X": np.linspace(1800, 2100, n),
        "Y": np.linspace(-500, 500, n),
        "Z": np.linspace(1000, 2000, n),
        "A": np.linspace(0, 10, n),
        "B": np.linspace(0, 10, n),
        "C": np.linspace(0, 10, n),
    }
)

# Convert DataFrame to list of RigidTransform objects
# Note: The new batch_inverse API expects List[RigidTransform] instead of DataFrame
poses = []
for row in poses_df.iter_rows(named=True):
    translation = [row["X"], row["Y"], row["Z"]]
    rotation = Rotation.from_euler("xyz", [row["A"], row["B"], row["C"]], degrees=True)
    pose = RigidTransform.from_components(rotation=rotation, translation=translation)
    poses.append(pose)

res = robot.batch_inverse(
    poses=poses, current_joints=(0, 0, -90, 0, 0, 0), ee_transform=ee_transform
)

print(res)
# %%
