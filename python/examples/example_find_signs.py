# It can be a bit of a hassle to find the correct sign values for the kinematic model parameters. This function does a bruteforce search to find the correct sign values for the kinematic model parameters that match the known joint positions and expected end-effector positions and rotations. The function takes the known joint positions and expected end-effector positions and rotations as input, and returns the correct sign values for the kinematic model parameters. If no valid configuration is found, it returns None.
# %%
from py_opw_kinematics import Robot, EulerConvention, KinematicModel
import numpy as np
import itertools


def find_correct_configuration(observations, model: KinematicModel, degrees: bool):
    # Define possible parameter values
    sign_values = [-1, 1]
    flip_axes_values = [False, True]
    offset_values = [0, 180]
    ee_rotation_values = [-90, 0]
    extrinsic_values = [False]

    # Generate all combinations of parameters using itertools.product
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
        extrinsic_values,  # extrinsic
    )
    total_combinations = (
        len(sign_values) ** 7
        * len(offset_values) ** 6
        * len(flip_axes_values) ** 6
        * len(ee_rotation_values) ** 3
        * len(extrinsic_values)
    )
    print(f"Testing {total_combinations} configurations")
    working_configurations: list[Robot] = []
    # Iterate over each parameter combination
    for i, params in enumerate(parameter_combinations):
        if (i + 1) % 10000 == 0:
            print(
                f"Testing configuration {i+1}/{total_combinations}, Found {len(working_configurations)} candidates",
                end="\r",
            )

        # Unpack the parameters
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
            extrinsic,
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
                has_parallellogram=model.has_parallellogram,
                offsets=[offset1, offset2, offset3, offset4, offset5, offset6],
                flip_axes=[f1, f2, f3, f4, f5, f6],
            ),
            EulerConvention("XYZ", extrinsic=extrinsic, degrees=degrees),
            ee_rotation=[ee_A, ee_B, ee_C],
        )

        # Test each known joint position against the expected transformation
        for joints, xyz, abc in observations:
            t, r = robot.forward(joints=joints)

            # Assert that computed and expected values are approximately equal
            if not (np.allclose(t, xyz, atol=1e-2) and np.allclose(r, abc, atol=1e-2)):
                break  # Break if the configuration does not match the observation

        else:
            working_configurations.append(robot)

    return working_configurations


if __name__ == "__main__":
    observations = [
        # sets of joints, xyz, abc,
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
        has_parallellogram=True,
    )

    # Run the function to find the correct configuration
    results = find_correct_configuration(
        observations=observations, model=model, degrees=True
    )
    for result in results:
        print(repr(result))
