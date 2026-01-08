from py_opw_kinematics import Robot, EulerConvention, KinematicModel
import numpy as np
import pytest
import polars as pl
from scipy.spatial.transform import RigidTransform, Rotation


@pytest.fixture
def example_robot() -> Robot:
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
    return Robot(kinematic_model, euler_convention)


@pytest.fixture
def example_ee_rotation() -> Rotation:
    ee_rotation = Rotation.from_euler("xyz", [0, -90, 0], degrees=True)
    return ee_rotation


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
    example_robot: Robot,
    example_ee_rotation: Rotation,
    joints,
    expected_position,
    expected_orientation,
):
    robot = example_robot
    ee_transform = RigidTransform.from_components(
        rotation=example_ee_rotation, translation=[0, 0, 0]
    )

    # Calculate forward kinematics
    t, r = robot.forward_legacy(joints=joints, ee_transform=ee_transform)

    # Assert the translation vector is close to the expected position
    assert np.allclose(t, expected_position, atol=1e-3)

    if expected_orientation:
        # Assert the orientation is close to the expected orientation
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
    example_robot: Robot,
    example_ee_rotation: Rotation,
    joints,
    expected_position,
    expected_orientation,
):
    robot = example_robot
    ee_translation = [234.096, 132.2, -551.725]
    ee_transform = RigidTransform.from_components(
        rotation=example_ee_rotation, translation=ee_translation
    )
    # Calculate forward kinematics
    t, r = robot.forward_legacy(joints=joints, ee_transform=ee_transform)
    # Assert the translation vector is close to the expected position
    assert np.allclose(t, expected_position, atol=1e-3)

    if expected_orientation:
        # Convert rotation to euler angles and assert
        euler_angles = r
        assert np.allclose(euler_angles, expected_orientation, atol=1e-3)


def test_robot_inverse_with_ee_translation(
    example_robot: Robot, example_ee_rotation: Rotation
):
    robot = example_robot
    ee_translation = [234.096, 132.2, -551.725]
    expected_joints = [10, 20, -90, 30, 20, 10]

    ee_transform = RigidTransform.from_components(
        rotation=example_ee_rotation, translation=ee_translation
    )

    # Create target pose using the robot's Euler convention properly
    # The expected values are in the robot's Euler convention format
    expected_euler = (-37.346, 25.987, -4.814)
    expected_position = [2396.467, -743.091, 1572.479]

    # Convert robot's Euler angles to rotation matrix using robot's convention
    rotation_matrix = robot._euler_convention.euler_to_matrix(expected_euler)
    target_rotation = Rotation.from_matrix(rotation_matrix)
    target_pose = RigidTransform.from_components(
        rotation=target_rotation, translation=expected_position
    )

    joint_solutions = robot.inverse(
        target_pose,
        ee_transform=ee_transform,
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
    )
    joints = joints if degrees else np.deg2rad(joints)
    ee_rotation_scipy = Rotation.from_euler("xyz", ee_rotation, degrees=True)
    ee_transform = RigidTransform.from_components(
        rotation=ee_rotation_scipy, translation=ee_translation
    )

    # Perform forward kinematics to get the pose
    forward_result = robot.forward(joints=joints, ee_transform=ee_transform)

    # Calculate inverse kinematics to retrieve joint angles from the given pose
    joint_solutions = robot.inverse(forward_result, ee_transform=ee_transform)

    # Ensure at least one valid solution matches the original joint angles
    assert any(
        np.allclose(solution, joints, atol=1e-3) for solution in joint_solutions
    ), f"No valid joint solution found for joints: {joints}, {joint_solutions}"

    # Ensure all forward kinematics from the computed joint angles match the original pose
    for joint_solution in joint_solutions:
        solution_result = robot.forward(
            joints=joint_solution, ee_transform=ee_transform
        )
        assert np.allclose(
            forward_result.translation, solution_result.translation, atol=1e-3
        ), f"Position mismatch: {forward_result.translation} != {solution_result.translation}"

        # Compare rotations using rotation matrices to avoid gimbal lock issues
        orig_rot_matrix = forward_result.rotation.as_matrix()
        sol_rot_matrix = solution_result.rotation.as_matrix()
        assert np.allclose(
            orig_rot_matrix, sol_rot_matrix, atol=1e-3
        ), f"Orientation mismatch in rotation matrices"


@pytest.mark.parametrize(
    "initial_translation, joint_angles, expected_diff",
    [
        ([0, 0, 0], [0, 0, -90, 0, 0, 0], [0, 0, 0]),
        ([10, 20, 30], [0, 0, -90, 0, 0, 0], [10, 20, 30]),
        ([10, 20, 30], [90, 0, -90, 0, 0, 0], [20, -10, 30]),
    ],
)
def test_ee_translation(
    example_robot: Robot, initial_translation, joint_angles, expected_diff
):
    robot = example_robot
    ee_rotation = [0, -90, 0]
    ee_translation = [0, 0, 0]
    ee_rotation_scipy = Rotation.from_euler("xyz", ee_rotation, degrees=True)
    ee_transform = RigidTransform.from_components(
        rotation=ee_rotation_scipy, translation=ee_translation
    )
    initial_result = robot.forward(joints=joint_angles, ee_transform=ee_transform)

    ee_translation = initial_translation
    ee_rotation_scipy = Rotation.from_euler("xyz", ee_rotation, degrees=True)
    ee_transform = RigidTransform.from_components(
        rotation=ee_rotation_scipy, translation=ee_translation
    )
    updated_result = robot.forward(joints=joint_angles, ee_transform=ee_transform)

    # Calculate translation differences
    translation_diff = updated_result.translation - initial_result.translation

    # Assert translation differences
    assert np.allclose(
        translation_diff, expected_diff
    ), f"Expected translation difference {expected_diff}, but got {translation_diff}"


def test_batch_forward_random(example_robot: Robot, example_ee_rotation: Rotation):
    robot = example_robot

    ee_transform = RigidTransform.from_rotation(example_ee_rotation)
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
    result_poses = robot.batch_forward(joints_df, ee_transform=ee_transform)

    # Verify that the output list has the expected length
    assert len(result_poses) == num_samples, "Mismatch in number of samples"


def test_batch_inverse_random(example_robot: Robot, example_ee_rotation: Rotation):
    robot = example_robot
    ee_transform = RigidTransform.from_rotation(example_ee_rotation)

    # Generate random positions and orientations
    num_samples = 50
    np.random.seed(42)
    poses = []
    for _ in range(num_samples):
        position = np.random.uniform([1500, -1000, 1000], [2500, 1000, 2500])
        euler_angles = np.random.uniform([-180, -90, -180], [180, 90, 180])
        rotation = Rotation.from_euler("xyz", euler_angles, degrees=True)
        pose = RigidTransform.from_components(rotation=rotation, translation=position)
        poses.append(pose)

    # Use batch_inverse to compute joint angles
    result_df = robot.batch_inverse(poses, ee_transform=ee_transform)

    # Verify that the output DataFrame has the expected length
    assert len(result_df) <= num_samples, "Mismatch in number of samples"


def test_batch_roundtrip(example_robot: Robot):
    robot = example_robot
    ee_rotation = [0, -90, 0]
    ee_translation = [100, 0, -500]

    ee_rotation_scipy = Rotation.from_euler("xyz", ee_rotation, degrees=True)
    ee_transform = RigidTransform.from_components(
        rotation=ee_rotation_scipy, translation=ee_translation
    )

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

    poses_list = robot.batch_forward(joints_df, ee_transform=ee_transform)
    result_joints_df = robot.batch_inverse(
        poses_list, current_joints=joints_df[0].to_numpy()[0], ee_transform=ee_transform
    )

    assert np.isclose(
        joints_df.to_numpy(), result_joints_df.to_numpy(), atol=1e-3
    ).all()
