from py_opw_kinematics import Robot, KinematicModel, BaseConfig, ToolConfig
import numpy as np
import pytest
import polars as pl


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

    # Generate random joint angles within typical ranges
    num_samples = 50
    np.random.seed(42)
    joint_data = {
        "J1": np.random.uniform(-180, 180, num_samples),
        "J2": np.random.uniform(-90, 90, num_samples),
        "J3": np.random.uniform(-180, 180, num_samples),
        "J4": np.random.uniform(-180, 180, num_samples),
        "J5": np.random.uniform(-90, 90, num_samples),
        "J6": np.random.uniform(-180, 180, num_samples),
    }
    joints_df = pl.DataFrame(joint_data)

    # Use batch_forward to compute positions and orientations
    result_df = robot.batch_forward(joints_df)

    # Verify that the output DataFrame has the expected length
    assert len(result_df) == num_samples, "Mismatch in number of samples"


def test_batch_inverse_random(example_robot):
    robot = example_robot

    # Generate random positions and orientations
    num_samples = 50
    np.random.seed(42)
    pose_data = {
        "X": np.random.uniform(1500, 2500, num_samples),
        "Y": np.random.uniform(-1000, 1000, num_samples),
        "Z": np.random.uniform(1000, 2500, num_samples),
        "A": np.random.uniform(-1, 1, num_samples),
        "B": np.random.uniform(-1, 1, num_samples),
        "C": np.random.uniform(-1, 1, num_samples),
        "D": np.random.uniform(-1, 1, num_samples),
    }
    poses_df = pl.DataFrame(pose_data)

    # Use batch_inverse to compute joint angles
    result_df = robot.batch_inverse(poses_df)

    # Verify that the output DataFrame has the expected length
    assert len(result_df) <= num_samples, "Mismatch in number of samples"
