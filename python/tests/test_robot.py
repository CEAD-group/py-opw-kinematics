from py_opw_kinematics import Robot, EulerConvention, KinematicModel
import numpy as np
import pytest
import polars as pl


@pytest.fixture
def example_robot():
    # Initialize Kinematic Model with known parameters and inlined signs
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
    example_robot, joints, expected_position, expected_orientation
):
    robot = example_robot

    # Calculate forward kinematics
    t, r = robot.forward(joints=joints)

    # Assert the translation vector is close to the expected position
    assert np.allclose(t, expected_position, atol=1e-3)
    print("R", r)
    if expected_orientation:
        # Assert the rotation vector is close to the expected orientation
        assert np.allclose(r, expected_orientation, atol=1e-3)


@pytest.mark.parametrize(
    "joints, expected_position, expected_orientation",
    [
        ([0, 0, -90, 0, 0, 0], [2308.022, 132.200, 1707.280], [0, 0, 0]),
        (
            [10, 20, -90, 30, 20, 10],
            [2396.467, -743.091, 1572.479],
            [-37.346, 25.987, -4.814],
        ),
    ],
)
def test_robot_forward_kinematics_with_ee_translation(
    example_robot, joints, expected_position, expected_orientation
):
    robot = example_robot
    robot.ee_translation = [234.096, 132.2, -551.725]
    # Calculate forward kinematics
    t, r = robot.forward(joints=joints)

    # Assert the translation vector is close to the expected position
    assert np.allclose(t, expected_position, atol=1e-3)
    print("R", r)
    if expected_orientation:
        # Assert the rotation vector is close to the expected orientation
        assert np.allclose(r, expected_orientation, atol=1e-3)


def test_robot_inverse_with_ee_translation(example_robot):
    robot = example_robot
    robot.ee_translation = [234.096, 132.2, -551.725]
    expected_joints = [10, 20, -90, 30, 20, 10]

    joint_solutions = robot.inverse(
        (
            [2396.467, -743.091, 1572.479],
            [-37.346, 25.987, -4.814],
        )
    )

    # Ensure at least one valid solution matches the original joint angles
    assert any(
        np.allclose(solution, expected_joints, atol=1e-2)
        for solution in joint_solutions
    ), f"No valid joint solution found for joints: {expected_joints}, {joint_solutions}"


@pytest.mark.parametrize(
    "joints", [[-10, 0, -30, 10, 10, -10], [10, 20, -90, 30, 20, 10]]
)
@pytest.mark.parametrize("has_parallelogram", [True, False])
@pytest.mark.parametrize("extrinsic", [True, False])
@pytest.mark.parametrize("ee_translation", [[0, 0, 0], [100, 200, 300]])
@pytest.mark.parametrize("ee_rotation", [[0, 0, 0], [0, -90, 0], [30, 40, 60]])
@pytest.mark.parametrize("degrees", [True, False])
@pytest.mark.parametrize(
    "flip_axes",
    [
        (False, False, False, False, False, False),
        (False, True, False, False, True, False),
    ],
)
@pytest.mark.parametrize("offsets", [(0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6)])
def test_robot_kinematics_roundtrip(
    joints,
    has_parallelogram,
    extrinsic,
    ee_translation,
    ee_rotation,
    degrees,
    flip_axes,
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
        flip_axes=flip_axes,
        has_parallelogram=has_parallelogram,
    )

    # Define Euler convention and create robot
    euler_convention = EulerConvention("XYZ", extrinsic=extrinsic, degrees=degrees)
    robot = Robot(
        kinematic_model,
        euler_convention,
        ee_translation=ee_translation,
        ee_rotation=ee_rotation,
    )
    joints = joints if degrees else np.deg2rad(joints)

    # Perform forward kinematics to get the pose
    position, orientation = robot.forward(joints=joints)

    # Calculate inverse kinematics to retrieve joint angles from the given pose
    joint_solutions = robot.inverse((position, orientation))

    # Ensure at least one valid solution matches the original joint angles
    assert any(
        np.allclose(solution, joints, atol=1e-3) for solution in joint_solutions
    ), f"No valid joint solution found for joints: {joints}, {joint_solutions}"


def test_settings_and_getting_ee_rotation(example_robot):
    robot = example_robot
    robot.ee_rotation = [0, 0, 0]
    assert np.allclose(robot.ee_rotation, [0, 0, 0])

    _translation, rotation = robot.forward(joints=[0, 0, 0, 0, 0, 0])

    assert np.allclose(rotation, robot.ee_rotation)
    robot.ee_rotation = [10, -40, 30]
    assert np.allclose(robot.ee_rotation, [10, -40, 30])

    _translation, rotation = robot.forward(joints=[0, 0, 0, 0, 0, 0])
    assert np.allclose(rotation, robot.ee_rotation)


@pytest.mark.parametrize(
    "initial_translation, joint_angles, expected_diff",
    [
        ([0, 0, 0], [0, 0, -90, 0, 0, 0], [0, 0, 0]),
        ([10, 20, 30], [0, 0, -90, 0, 0, 0], [10, 20, 30]),
        ([10, 20, 30], [90, 0, -90, 0, 0, 0], [20, -10, 30]),
    ],
)
def test_ee_translation(
    example_robot, initial_translation, joint_angles, expected_diff
):
    robot = example_robot
    robot.ee_rotation = [0, -90, 0]
    robot.ee_translation = [0, 0, 0]
    initial_translation_result, _ = robot.forward(joints=joint_angles)
    robot.ee_translation = initial_translation
    updated_translation_result, _ = robot.forward(joints=joint_angles)

    # Calculate translation differences
    translation_diff = np.array(updated_translation_result) - np.array(
        initial_translation_result
    )

    # Assert translation differences
    assert np.allclose(
        translation_diff, expected_diff
    ), f"Expected translation difference {expected_diff}, but got {translation_diff}"


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
        "A": np.random.uniform(-180, 180, num_samples),
        "B": np.random.uniform(-90, 90, num_samples),
        "C": np.random.uniform(-180, 180, num_samples),
    }
    poses_df = pl.DataFrame(pose_data)

    # Use batch_inverse to compute joint angles
    result_df = robot.batch_inverse(poses_df)

    # Verify that the output DataFrame has the expected length
    assert len(result_df) <= num_samples, "Mismatch in number of samples"


def test_batch_roundtrip(example_robot):
    robot = example_robot
    robot.ee_rotation = [0, -90, 0]
    robot.ee_translation = [100, 0, -500]

    num_samples = 21
    np.random.seed(42)
    joint_data = {
        "J1": np.linspace(-180, 100, num_samples),
        "J2": np.linspace(-80, 90, num_samples),
        "J3": np.linspace(-170, 180, num_samples),
        "J4": np.linspace(-600, 0, num_samples),
        "J5": np.linspace(-80, 90, num_samples),
        "J6": np.linspace(600, 0, num_samples),
    }
    joints_df = pl.DataFrame(joint_data)

    poses_df = robot.batch_forward(joints_df)
    result_joints_df = robot.batch_inverse(
        poses_df, current_joints=joints_df[0].to_numpy()[0]
    )

    assert np.isclose(
        joints_df.to_numpy(), result_joints_df.to_numpy(), atol=1e-3
    ).all()
