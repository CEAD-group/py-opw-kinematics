# %%
from py_opw_kinematics import Robot, EulerConvention, KinematicModel
from math import pi
import numpy as np
import pytest


@pytest.fixture
def known_robot():
    # Initialize Kinematic Model with known parameters and inlined signs
    kinematic_model = KinematicModel(
        a1=400.333,
        a2=-251.449,
        b=0,
        c1=830,
        c2=1177.556,
        c3=1443.593,
        c4=230,
        offsets=[0] * 6,
        sign_corrections=[-1, 1, -1, -1, 1, -1],
        has_parallellogram=True,
    )

    # Define Euler convention and create robot
    euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
    ee_rotation = [0, -90, 0]
    return Robot(kinematic_model, euler_convention, ee_rotation=ee_rotation)


@pytest.mark.parametrize(
    "joints, expected_position, expected_orientation",
    [
        ([0, 0, -90, 0, 0, 0], [2073.926, 0.0, 2259.005], None),
        ([10, 0, -90, 0, 0, 0], [2042.418, -360.133, 2259.005], None),
        ([10, 10, -90, 0, 0, 0], [2243.792, -395.641, 2241.115], None),
        ([0, 0, -90, 0, 10, 0], [2070.432, 0.0, 2219.066], None),
        ([0, 0, -90, 10, 10, 0], [2070.432, -6.935, 2219.673], None),
        (
            [10, 20, -90, 30, 20, 10],
            [2418.558, -466.396, 2119.864],
            [-37.346, 25.987, -4.814],
        ),
    ],
)
def test_robot_forward_kinematics(
    known_robot, joints, expected_position, expected_orientation
):
    robot = known_robot

    # Calculate forward kinematics
    t, r_rad = robot.forward(joints=joints)

    # Assert the translation vector is close to the expected position
    assert np.allclose(t, expected_position, atol=1e-3)
    print("R_rad", r_rad)
    if expected_orientation:
        # Assert the rotation vector is close to the expected orientation
        assert np.allclose(r_rad, expected_orientation, atol=1e-3)


@pytest.mark.parametrize(
    "joints", [[-10, 0, -30, 10, 10, -10], [10, 20, -90, 30, 20, 10]]
)
@pytest.mark.parametrize("has_parallellogram", [True, False])
@pytest.mark.parametrize("extrinsic", [True, False])
@pytest.mark.parametrize("ee_rotation", [[0, 0, 0], [0, -90, 0], [30, 40, 60]])
@pytest.mark.parametrize("degrees", [True, False])
@pytest.mark.parametrize("sign_corrections", [[1, 1, 1, 1, 1, 1], [1, -1, 1, 1, -1, 1]])
@pytest.mark.parametrize("offsets", [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
def test_robot_kinematics_roundtrip(
    joints,
    has_parallellogram,
    extrinsic,
    ee_rotation,
    degrees,
    sign_corrections,
    offsets,
):
    # Initialize Kinematic Model with known parameters and inlined signs
    kinematic_model = KinematicModel(
        a1=1,
        a2=2,
        b=3,
        c1=4,
        c2=5,
        c3=6,
        c4=7,
        offsets=offsets,
        sign_corrections=sign_corrections,
        has_parallellogram=has_parallellogram,
    )

    # Define Euler convention and create robot
    euler_convention = EulerConvention("XYZ", extrinsic=extrinsic, degrees=degrees)
    robot = Robot(kinematic_model, euler_convention, ee_rotation=ee_rotation)

    joints = joints if degrees else np.deg2rad(joints)

    # Perform forward kinematics to get the pose
    position, orientation = robot.forward(joints=joints)

    # Calculate inverse kinematics to retrieve joint angles from the given pose
    joint_solutions = robot.inverse((position, orientation))

    # Ensure at least one valid solution matches the original joint angles
    assert any(
        np.allclose(solution, joints, atol=1e-3) for solution in joint_solutions
    ), f"No valid joint solution found for joints: {joints}, {joint_solutions}"
