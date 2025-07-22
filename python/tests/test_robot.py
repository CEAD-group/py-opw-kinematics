from py_opw_kinematics import Robot, KinematicModel, BaseConfig, ToolConfig
import numpy as np
import pytest


@pytest.fixture
def example_robot():
    # Initialize Kinematic Model with known parameters and inlined signs
    kinematic_model = KinematicModel(
        a1=0.150,
        a2=-0.110,
        b=0.0,
        c1=0.4865,
        c2=0.700,
        c3=0.678,
        c4=0.135,
        offsets=(0, 0, -np.pi / 2, 0, 0, 0),
        sign_corrections=(1, 1, 1, 1, 1, 1),
    )

    base_config = BaseConfig(translation=[0, 0, 2.3], rotation=[0, 1, 0, 0])
    tool_config = ToolConfig(
        translation=[0, 0, 0.095],
        rotation=[
            -0.00012991440873552217,
            -0.968154906938256,
            -0.0004965996111545046,
            0.2503407964804168,
        ],
    )

    return Robot(kinematic_model, base_config, tool_config)


@pytest.mark.parametrize(
    "joints, expected_position, expected_orientation",
    [
        (
            [-103.1, -85.03, 19.06, -70.19, -35.87, 185.01],
            [0.200, -0.3, 0.9],
            [0.8518, 0.13766, -0.46472, -0.19852],
        ),
        (
            [-116.97, -85.69, 16.82, -63.5, -39.63, 192.76],
            [0.300, -0.3, 0.9],
            [0.8518, 0.13766, -0.46472, -0.19852],
        ),
        (
            [-128.14, -86.43, 13.04, -59.66, -40.66, 201.57],
            [0.400, -0.3, 0.9],
            [0.8518, 0.13766, -0.46472, -0.19852],
        ),
        (
            [-124.68, -61.16, -20.4, 56.41, -38.79, -24.56],
            [0.200, -0.5, 0.6],
            [0.69636, 0.12279, 0.12279, 0.69636],
        ),
        (
            [-127.36, -62.29, -16.83, 59.35, -35.14, -23.42],
            [0.201, -0.451, 0.604],
            [0.70106, 0.15305, 0.0923, 0.69034],
        ),
        # TODO: Add some samples that are invalid
    ],
)
def test_robot_forward_kinematics(
    example_robot, joints, expected_position, expected_orientation
):
    robot = example_robot

    # Calculate forward kinematics
    t, r = robot.forward(joints=joints)

    # NOTE: Tolerance is pretty high because I didn't measure it very accurately
    # Assert the translation vector is close to the expected position
    assert np.allclose(t, expected_position, atol=1e-1)
    if expected_orientation:
        # Assert the rotation vector is close to the expected orientation
        assert r == pytest.approx(expected_orientation, abs=1e-1)


def test_batch_forward_random(example_robot):
    robot = example_robot

    # Create test joint configurations as list of tuples
    test_joints = [
        [-103.1, -85.03, 19.06, -70.19, -35.87, 185.01],
        [-116.97, -85.69, 16.82, -63.5, -39.63, 192.76],
        [-128.14, -86.43, 13.04, -59.66, -40.66, 201.57],
    ]

    # Use batch_forward to compute poses
    batch_poses = robot.batch_forward(test_joints)

    # Verify that we get poses for each joint configuration
    assert len(batch_poses) == len(test_joints), (
        "Mismatch in number of joint configurations"
    )

    # Check that each result is a valid pose
    for i, pose in enumerate(batch_poses):
        assert len(pose) == 2, f"Pose {i} should have position and orientation"
        assert len(pose[0]) == 3, f"Position {i} should have 3 components"
        assert len(pose[1]) == 4, f"Orientation {i} should have 4 quaternion components"

        # Verify by comparing with individual forward kinematics
        individual_pose = robot.forward(test_joints[i])
        assert np.allclose(pose[0], individual_pose[0], atol=1e-10), (
            f"Position mismatch for joint config {i}"
        )
        assert np.allclose(pose[1], individual_pose[1], atol=1e-10), (
            f"Orientation mismatch for joint config {i}"
        )


def test_batch_inverse_random(example_robot):
    robot = example_robot

    # Create test poses as list of tuples
    test_poses = [
        ([0.2, -0.3, 0.9], [0.8518, 0.13766, -0.46472, -0.19852]),
        ([0.3, -0.3, 0.9], [0.8518, 0.13766, -0.46472, -0.19852]),
        ([0.4, -0.3, 0.9], [0.8518, 0.13766, -0.46472, -0.19852]),
    ]

    # Use batch_inverse to compute joint angles
    batch_solutions = robot.batch_inverse(test_poses)

    # Verify that we get solutions for each pose
    assert len(batch_solutions) == len(test_poses), "Mismatch in number of poses"

    # Check that each pose has solutions
    for i, solutions in enumerate(batch_solutions):
        assert len(solutions) > 0, f"No solutions found for pose {i}"

        # Verify solutions by forward kinematics
        for solution in solutions:
            computed_pose = robot.forward(solution)
            original_pose = test_poses[i]

            # Check position with tolerance
            assert np.allclose(computed_pose[0], original_pose[0], atol=1e-6), (
                f"Position mismatch for pose {i}"
            )
