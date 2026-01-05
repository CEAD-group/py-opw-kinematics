# %%
"""
Utility to find correct sign values for kinematic model parameters.

This does a brute-force search to find the correct sign values for the kinematic
model parameters that match known joint positions and expected end-effector poses.
"""

from py_opw_kinematics import Robot, KinematicModel
from scipy.spatial.transform import RigidTransform, Rotation
import numpy as np
import itertools


def find_correct_configuration(observations, model: KinematicModel, degrees: bool):
    """
    Find robot configurations that match observed FK results.

    Parameters:
        observations: List of (joints, xyz, euler_abc) tuples
        model: Base kinematic model with absolute parameter values
        degrees: Whether joint angles and Euler angles are in degrees

    Returns:
        List of Robot instances that match all observations
    """
    sign_values = [-1, 1]
    flip_axes_values = [False, True]
    offset_values = [0, 180]
    ee_rotation_values = [-90.0, 0.0]

    parameter_combinations = itertools.product(
        sign_values,
        sign_values,
        sign_values,
        sign_values,
        sign_values,
        sign_values,
        sign_values,  # a1_sign, a2_sign, b_sign, c1_sign, c2_sign, c3_sign, c4_sign
        offset_values,
        offset_values,
        offset_values,
        offset_values,
        offset_values,
        offset_values,  # offsets 1 to 6
        flip_axes_values,
        flip_axes_values,
        flip_axes_values,
        flip_axes_values,
        flip_axes_values,
        flip_axes_values,  # f1 to f6
        ee_rotation_values,
        ee_rotation_values,
        ee_rotation_values,  # ee_A, ee_B, ee_C
    )

    total_combinations = (
        len(sign_values) ** 7
        * len(offset_values) ** 6
        * len(flip_axes_values) ** 6
        * len(ee_rotation_values) ** 3
    )
    print(f"Testing {total_combinations} configurations")

    working_configurations: list[tuple[Robot, tuple[float, float, float]]] = []

    for i, params in enumerate(parameter_combinations):
        if (i + 1) % 10000 == 0:
            print(
                f"Testing configuration {i+1}/{total_combinations}, "
                f"Found {len(working_configurations)} candidates",
                end="\r",
            )

        (
            a1_sign,
            a2_sign,
            b_sign,
            c1_sign,
            c2_sign,
            c3_sign,
            c4_sign,
            offset1,
            offset2,
            offset3,
            offset4,
            offset5,
            offset6,
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            ee_A,
            ee_B,
            ee_C,
        ) = params

        robot = Robot(
            KinematicModel(
                a1=a1_sign * model.a1,
                a2=a2_sign * model.a2,
                b=b_sign * model.b,
                c1=c1_sign * model.c1,
                c2=c2_sign * model.c2,
                c3=c3_sign * model.c3,
                c4=c4_sign * model.c4,
                offsets=(offset1, offset2, offset3, offset4, offset5, offset6),
                flip_axes=(
                    bool(f1),
                    bool(f2),
                    bool(f3),
                    bool(f4),
                    bool(f5),
                    bool(f6),
                ),
            ),
            degrees=degrees,
        )

        # Create EE transform
        ee_rotation = Rotation.from_euler("XYZ", [ee_A, ee_B, ee_C], degrees=degrees)
        ee_transform = RigidTransform.from_components(
            rotation=ee_rotation, translation=[0, 0, 0]
        )

        # Test each observation
        for joints, xyz, abc in observations:
            pose = robot.forward(joints=joints, ee_transform=ee_transform)
            t = pose.translation
            r = pose.rotation.as_euler("XYZ", degrees=degrees)

            if not (np.allclose(t, xyz, atol=1e-2) and np.allclose(r, abc, atol=1e-2)):
                break
        else:
            # All observations matched
            working_configurations.append((robot, (ee_A, ee_B, ee_C)))

    print()  # Newline after progress
    return working_configurations


if __name__ == "__main__":
    observations = [
        # (joints, xyz, euler_abc)
        (
            [0, 0, -90, 0, 0, 0],
            [2073.926, 0.0, 2259.005],
            [0, 0, 0],
        ),
        (
            [10, 20, -90, 30, 20, 10],
            [2418.558, -466.396, 2119.864],
            [-37.346, 25.987, -4.814],
        ),
    ]

    model = KinematicModel(
        a1=400.333,
        a2=-251.449,
        b=0,
        c1=830,
        c2=1177.556,
        c3=1443.593,
        c4=230,
    )

    results = find_correct_configuration(
        observations=observations, model=model, degrees=True
    )

    print(f"\nFound {len(results)} matching configurations:")
    for robot, ee_angles in results:
        print(f"\nRobot: {repr(robot)}")
        print(f"EE rotation (XYZ): {ee_angles}")
