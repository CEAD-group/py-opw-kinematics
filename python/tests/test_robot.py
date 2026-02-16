from py_opw_kinematics import Robot, KinematicModel
import numpy as np
import pytest
from scipy.spatial.transform import RigidTransform, Rotation


@pytest.fixture
def example_robot() -> Robot:
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
    return Robot(kinematic_model, degrees=True)


@pytest.fixture
def example_ee_rotation() -> Rotation:
    return Rotation.from_euler("xyz", [0, -90, 0], degrees=True)


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

    result = robot.forward(joints=joints, ee_transform=ee_transform)
    t = result.translation
    r = result.rotation.as_euler("XYZ", degrees=True)

    assert np.allclose(t, expected_position, atol=1e-3)

    if expected_orientation:
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

    result = robot.forward(joints=joints, ee_transform=ee_transform)
    t = result.translation
    r = result.rotation.as_euler("XYZ", degrees=True)

    assert np.allclose(t, expected_position, atol=1e-3)

    if expected_orientation:
        assert np.allclose(r, expected_orientation, atol=1e-3)


def test_robot_inverse_with_ee_translation(
    example_robot: Robot, example_ee_rotation: Rotation
):
    robot = example_robot
    ee_translation = [234.096, 132.2, -551.725]
    expected_joints = [10, 20, -90, 30, 20, 10]

    ee_transform = RigidTransform.from_components(
        rotation=example_ee_rotation, translation=ee_translation
    )

    # Create target pose using scipy's Rotation
    expected_euler = (-37.346, 25.987, -4.814)
    expected_position = [2396.467, -743.091, 1572.479]

    target_rotation = Rotation.from_euler("XYZ", expected_euler, degrees=True)
    target_pose = RigidTransform.from_components(
        rotation=target_rotation, translation=expected_position
    )

    joint_solutions = robot.inverse(target_pose, ee_transform=ee_transform)

    assert any(
        np.allclose(solution, expected_joints, atol=1e-2)
        for solution in joint_solutions
    ), f"No valid joint solution found for joints: {expected_joints}, {joint_solutions}"


@pytest.mark.parametrize(
    "joints", [[-10, 0, -30, 10, 10, -10], [10, 20, -90, 30, 20, 10]]
)
@pytest.mark.parametrize("has_parallelogram", [True, False])
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
    ee_translation,
    ee_rotation,
    degrees,
    flip_axes,
    offsets,
):
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

    robot = Robot(kinematic_model, degrees=degrees)
    joints = joints if degrees else np.deg2rad(joints)
    ee_rotation_scipy = Rotation.from_euler("xyz", ee_rotation, degrees=True)
    ee_transform = RigidTransform.from_components(
        rotation=ee_rotation_scipy, translation=ee_translation
    )

    forward_result = robot.forward(joints=joints, ee_transform=ee_transform)
    joint_solutions = robot.inverse(forward_result, ee_transform=ee_transform)

    assert any(
        np.allclose(solution, joints, atol=1e-3) for solution in joint_solutions
    ), f"No valid joint solution found for joints: {joints}, {joint_solutions}"

    for joint_solution in joint_solutions:
        solution_result = robot.forward(
            joints=joint_solution, ee_transform=ee_transform
        )
        assert np.allclose(
            forward_result.translation, solution_result.translation, atol=1e-3
        ), f"Position mismatch: {forward_result.translation} != {solution_result.translation}"

        orig_rot_matrix = forward_result.rotation.as_matrix()
        sol_rot_matrix = solution_result.rotation.as_matrix()
        assert np.allclose(
            orig_rot_matrix, sol_rot_matrix, atol=1e-3
        ), "Orientation mismatch in rotation matrices"


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

    translation_diff = updated_result.translation - initial_result.translation

    assert np.allclose(
        translation_diff, expected_diff
    ), f"Expected translation difference {expected_diff}, but got {translation_diff}"


def test_batch_forward_random(example_robot: Robot, example_ee_rotation: Rotation):
    robot = example_robot

    ee_transform = RigidTransform.from_rotation(example_ee_rotation)
    num_samples = 50
    np.random.seed(42)
    joints_array = np.column_stack([
        np.random.uniform(-180, 180, num_samples),
        np.random.uniform(-90, 90, num_samples),
        np.random.uniform(-180, 180, num_samples),
        np.random.uniform(-180, 180, num_samples),
        np.random.uniform(-90, 90, num_samples),
        np.random.uniform(-180, 180, num_samples),
    ])

    result_poses = robot.batch_forward(joints_array, ee_transform=ee_transform)

    assert len(result_poses) == num_samples, "Mismatch in number of samples"


def test_batch_inverse_random(example_robot: Robot, example_ee_rotation: Rotation):
    robot = example_robot
    ee_transform = RigidTransform.from_rotation(example_ee_rotation)

    num_samples = 50
    np.random.seed(42)
    positions = np.random.uniform([1500, -1000, 1000], [2500, 1000, 2500], size=(num_samples, 3))
    euler_angles = np.random.uniform([-180, -90, -180], [180, 90, 180], size=(num_samples, 3))
    poses = RigidTransform.from_components(
        rotation=Rotation.from_euler("xyz", euler_angles, degrees=True),
        translation=positions,
    )

    result = robot.batch_inverse(poses, ee_transform=ee_transform)

    assert len(result) <= num_samples, "Mismatch in number of samples"


def test_batch_roundtrip(example_robot: Robot):
    robot = example_robot
    ee_rotation = [0, -90, 0]
    ee_translation = [100, 0, -500]

    ee_rotation_scipy = Rotation.from_euler("xyz", ee_rotation, degrees=True)
    ee_transform = RigidTransform.from_components(
        rotation=ee_rotation_scipy, translation=ee_translation
    )

    num_samples = 21
    joints_array = np.column_stack([
        np.linspace(-180, 100, num_samples),
        np.linspace(-80, 90, num_samples),
        np.linspace(-170, 180, num_samples),
        np.linspace(-600, 0, num_samples),
        np.linspace(-80, 90, num_samples),
        np.linspace(600, 0, num_samples),
    ])

    poses_list = robot.batch_forward(joints_array, ee_transform=ee_transform)
    result_joints = robot.batch_inverse(
        poses_list, current_joints=joints_array[0], ee_transform=ee_transform
    )

    assert np.isclose(joints_array, result_joints, atol=1e-3).all()


def test_batch_inverse_current_joints_signatures(example_robot: Robot):
    """Test batch_inverse with various current_joints input types."""
    import polars as pl
    import pandas as pd

    robot = example_robot
    ee_transform = RigidTransform.from_components(
        rotation=Rotation.from_euler("xyz", [0, -90, 0], degrees=True),
        translation=[0, 0, 0],
    )

    # Create test poses
    joints_start = (0, 0, -90, 0, 0, 0)
    poses = robot.batch_forward(
        np.array([[0, 0, -90, 0, 0, 0], [10, 0, -90, 0, 0, 0]]),
        ee_transform=ee_transform,
    )

    # Test 1: No current_joints - returns numpy array
    result = robot.batch_inverse(poses, ee_transform=ee_transform)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 6)

    # Test 2: Tuple current_joints - returns numpy array
    result = robot.batch_inverse(poses, current_joints=joints_start, ee_transform=ee_transform)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 6)

    # Test 3: Numpy array current_joints - returns numpy array
    result = robot.batch_inverse(
        poses, current_joints=np.array(joints_start), ee_transform=ee_transform
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 6)

    # Test 4: Polars DataFrame current_joints - returns polars DataFrame
    pl_joints = pl.DataFrame({"J1": [0.0], "J2": [0.0], "J3": [-90.0], "J4": [0.0], "J5": [0.0], "J6": [0.0]})
    result = robot.batch_inverse(poses, current_joints=pl_joints, ee_transform=ee_transform)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 6)
    assert list(result.columns) == ["J1", "J2", "J3", "J4", "J5", "J6"]

    # Test 5: Pandas DataFrame current_joints - returns pandas DataFrame
    pd_joints = pd.DataFrame({"J1": [0.0], "J2": [0.0], "J3": [-90.0], "J4": [0.0], "J5": [0.0], "J6": [0.0]})
    result = robot.batch_inverse(poses, current_joints=pd_joints, ee_transform=ee_transform)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 6)
    assert list(result.columns) == ["J1", "J2", "J3", "J4", "J5", "J6"]


def test_interpolate_poses():
    """Test the interpolate_poses function with SLERP + linear interpolation."""
    from py_opw_kinematics import interpolate_poses

    pose_start = RigidTransform.from_translation([0, 0, 0])
    pose_end = RigidTransform.from_components(
        rotation=Rotation.from_euler("X", 90, degrees=True),
        translation=[100, 200, 300],
    )

    keyframes = RigidTransform.concatenate([pose_start, pose_end])
    trajectory = interpolate_poses([0, 1], keyframes, np.linspace(0, 1, 11))

    assert len(trajectory) == 11

    # First and last should match start/end
    assert np.allclose(trajectory[0].translation, pose_start.translation)
    assert np.allclose(trajectory[-1].translation, pose_end.translation)

    # Middle should be interpolated
    assert np.allclose(trajectory[5].translation, [50, 100, 150], atol=1e-6)


def test_interpolate_poses_multi_keyframe():
    """Test interpolate_poses with multiple keyframes."""
    from py_opw_kinematics import interpolate_poses

    # Three keyframes
    poses = RigidTransform.from_components(
        rotation=Rotation.from_euler("ZYX", [0, 90, 180], degrees=True),
        translation=[[0, 0, 0], [100, 0, 0], [100, 100, 0]],
    )

    trajectory = interpolate_poses([0, 1, 2], poses, [0, 0.5, 1, 1.5, 2])

    assert len(trajectory) == 5
    assert np.allclose(trajectory[0].translation, [0, 0, 0])
    assert np.allclose(trajectory[2].translation, [100, 0, 0])
    assert np.allclose(trajectory[4].translation, [100, 100, 0])
