#!/usr/bin/env python3
"""
Example demonstrating the use of relative axis limits in py_opw_kinematics.

This example shows how to set relative constraints where one axis's limits
depend on another axis's position, which is common in parallelogram mechanisms.
Also demonstrates constraint unit handling (degrees input, radians storage).
"""
# %%
from py_opw_kinematics import KinematicModel, Robot, EulerConvention
import polars as pl
import numpy as np

# Create kinematic model with relative constraints during initialization
kinematic_model = KinematicModel(
    a1=400,  # $MC_ROBX_MAIN_LENGTH_AB[0]
    a2=-250,  # - $MC_ROBX_TX3P3_POS[2]
    b=0,
    c1=830,  # $MC_ROBX_TIRORO_POS[2]
    c2=1175,  # $MC_ROBX_MAIN_LENGTH_AB[1]
    c3=1444,  # $MC_ROBX_TX3P3_POS[0]
    c4=230,  # $MC_ROBX_TFLWP_POS[2]
    offsets=(0, 0, 0, 0, 0, 0),
    flip_axes=(False, False, True, False, False, False),
    has_parallelogram=True,
    relative_constraints=[
        (2, 1, -150, -50),  # J3 relative to J2: parallelogram limits in degrees
    ],
)  # Create kinematic model for Comau NJ165-3.0

# Verify constraint storage (always in radians internally)
print("NJ165 Constraint Unit Verification:")
stored_constraints = kinematic_model.relative_constraints
if stored_constraints:
    j3_j2_constraint = stored_constraints[0]
    print(f"Input: J3-J2 range [-150°, -50°]")
    print(f"Stored: [{j3_j2_constraint[2]:.4f}, {j3_j2_constraint[3]:.4f}] rad")
    print(
        f"Verify: [{np.rad2deg(j3_j2_constraint[2]):.1f}°, {np.rad2deg(j3_j2_constraint[3]):.1f}°]\n"
    )

axis_limits_nj165 = (  # lower and upper limits per axis in degrees
    (-175, 175),
    (-95, 75),
    (-256, -10),
    (-2700, 2700),
    (-125, 125),
    (-2700, 2700),
)
# Add absolute constraints using degrees parameter for clarity
for i, (lower, upper) in enumerate(axis_limits_nj165):
    kinematic_model.set_absolute_constraint(
        i, lower, upper, degrees=True
    )  # Input in degrees

# Alternative approach: set relative constraint after initialization
# kinematic_model.set_relative_constraint(2, 1, -150, -50)  # Now defaults to degrees

parallelogram_limits = (-160.0, -30.0)  # relative limits for J3 relative to J2

euler = EulerConvention("XYZ", extrinsic=False, degrees=True)  # Create Euler convention
# ee_translation = (245, 0, -428)  # set the end-effector translation (TCP offset)
ee_translation = (145.5, -353, -330.5)  # set the end-effector translation (TCP offset)
ee_rotation = (0, -90, 0)  # set the end-effector rotation ($MC_ROBX_TFLWP_RPY)
start_position_joints = (
    0,
    0,
    -100,
    0,
    10,
    0,
)  # set the current joint angles so that the robot is not in a singularity

data = [
    {"X": 2000.0, "Y": 500.0, "Z": 1200.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2100.0, "Y": 200.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2200.0, "Y": 200.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2300.0, "Y": 200.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2400.0, "Y": 200.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2400.0, "Y": 300.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2400.0, "Y": 400.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2400.0, "Y": 500.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
    {"X": 2400.0, "Y": 600.0, "Z": 800.0, "A": 0.0, "B": 0.0, "C": 0.0},
]
poses = pl.DataFrame(data)

# %%
robot_no_limits = Robot(
    kinematic_model,
    euler,
    ee_rotation=ee_rotation,
    ee_translation=ee_translation,
)
# %%
print("=== Without Any Constraints ===")
solutions_no_limits = robot_no_limits.batch_inverse(
    poses, current_joints=start_position_joints
)
print(f"Found {len(solutions_no_limits)} solutions:")
for i, sol in enumerate(solutions_no_limits.transpose()):
    print(
        f"  Solution {i+1}: {[f'{angle:7.2f}°' for angle in sol]} delta J3-J2={sol[2]-sol[1]:7.2f}°"
    )

# %%
kinematic_model.clear_all_constraints()
# Now set a relative constraint: J3 = J2 + offset, where offset is between -160° and -30°
kinematic_model.set_sum_constraint(  # J3 is axis 2, J2 is axis 1
    axis=2,
    reference_axis=1,
    min_offset=parallelogram_limits[0],
    max_offset=parallelogram_limits[1],
    degrees=True,
)
kinematic_model.set_axis_limits(limits=axis_limits_nj165)
robot_with_constraints = Robot(
    kinematic_model,
    euler,
    ee_rotation=ee_rotation,
    ee_translation=ee_translation,
)
# %%
print("\n=== With Relative Constraint (J3 relative to J2 in [-160°, -30°]) ===")
solutions_with_constraints = robot_with_constraints.batch_inverse(
    poses, current_joints=start_position_joints
)
print(f"Found {len(solutions_with_constraints)} solutions:")
for i, sol in enumerate(solutions_with_constraints.transpose()):
    print(
        f"  Pose {i+1}: {[f'{angle:7.2f}°' for angle in sol]} delta J3-J2={sol[2]-sol[1]:7.2f}°"
    )
