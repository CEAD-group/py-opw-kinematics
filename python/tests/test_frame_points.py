import itertools

import numpy as np
import pytest
from py_opw_kinematics import KinematicModel, Robot
from scipy.spatial.transform import RigidTransform, Rotation

LIMITS = np.array(
    [(-180, 180), (-75, 95), (-160, -30), (-270, 270), (-125, 125), (-270, 270)],
    dtype=float,
)

# Metres with sign flips; millimetres with offsets and a nonzero b; radians.
MODELS = {
    "metres_flipped": (
        KinematicModel(
            a1=0.400, a2=-0.250, b=0.0, c1=0.830, c2=1.175, c3=1.444, c4=0.230,
            flip_axes=(True, False, True, True, False, True),
        ),
        True,
    ),
    "millimetres_offsets": (
        KinematicModel(
            a1=320.0, a2=-195.0, b=45.0, c1=780.0, c2=1075.0, c3=1170.0, c4=215.0,
            offsets=(0.0, 0.0, -90.0, 0.0, 0.0, 180.0),
            flip_axes=(False, True, False, False, True, False),
        ),
        True,
    ),
    "radians": (
        KinematicModel(
            a1=0.150, a2=0.110, b=-0.020, c1=0.450, c2=0.600, c3=0.650, c4=0.100,
            offsets=(0.1, -0.2, 0.3, 0.0, 0.5, -0.4),
        ),
        False,
    ),
}

EE = RigidTransform.from_components(
    rotation=Rotation.from_euler("xyz", [10.0, -90.0, 25.0], degrees=True),
    translation=[0.05, -0.02, 0.3],
)


def _joints(degrees: bool, n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(7)
    q = rng.uniform(LIMITS[:, 0], LIMITS[:, 1], size=(n, 6))
    corners = np.array(list(itertools.product(*LIMITS)), dtype=float)
    q = np.vstack([q, corners, np.zeros((1, 6))])
    return q if degrees else np.radians(q)


def _frames(scale: float) -> list[tuple[int, np.ndarray]]:
    rng = np.random.default_rng(11)
    frames = [(k, rng.normal(size=3) * scale) for k in range(7)]
    frames += [(2, np.zeros(3)), (6, np.zeros(3)), (0, rng.normal(size=3) * scale)]
    return frames


def _reference(robot: Robot, joints: np.ndarray, frames, ee) -> np.ndarray:
    n = len(joints)
    per_config = 6 if ee is None else 7
    poses = robot.batch_joint_poses(joints, ee).as_matrix().reshape(n, per_config, 4, 4)
    out = np.empty((n, len(frames), 3))
    for k, (index, point) in enumerate(frames):
        out[:, k] = poses[:, index, :3, :3] @ point + poses[:, index, :3, 3]
    return out


@pytest.mark.parametrize("name", list(MODELS))
def test_matches_batch_joint_poses(name: str):
    model, degrees = MODELS[name]
    robot = Robot(model, degrees=degrees)
    scale = 100.0 if name == "millimetres_offsets" else 0.1
    ee = RigidTransform.from_components(rotation=EE.rotation, translation=EE.translation * scale * 10)
    joints = _joints(degrees)
    frames = _frames(scale)

    points = robot.batch_frame_points(joints, frames, ee)
    expected = _reference(robot, joints, frames, ee)

    assert points.shape == (len(joints), len(frames), 3)
    # 1e-12 in metre-scale units; batch_joint_poses goes through quaternions, so a few ulp of the coordinate remain.
    atol = max(1e-12, 4e-15 * np.abs(expected).max())
    np.testing.assert_allclose(points, expected, rtol=0, atol=atol)


@pytest.mark.parametrize("deepest", range(6))
def test_shallow_selection(deepest: int):
    # Only the chain up to the deepest selected joint is evaluated; the result must not change.
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 50)
    frames = [(k, np.array([0.01, 0.02, 0.03])) for k in range(deepest + 1)]
    frames.append((deepest, np.zeros(3)))
    np.testing.assert_allclose(
        robot.batch_frame_points(joints, frames),
        _reference(robot, joints, frames, None),
        rtol=0,
        atol=1e-12,
    )


def test_offset_forms_agree():
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 20)
    point = np.array([0.1, -0.2, 0.3])
    offset = RigidTransform.from_components(
        rotation=Rotation.from_euler("z", 30, degrees=True), translation=point
    )
    as_vector = robot.batch_frame_points(joints, [(2, point)])
    np.testing.assert_array_equal(as_vector, robot.batch_frame_points(joints, [(2, offset)]))
    np.testing.assert_array_equal(as_vector, robot.batch_frame_points(joints, [(2, offset.as_matrix())]))
    np.testing.assert_array_equal(as_vector, robot.batch_frame_points(joints, [(2, list(point))]))


def test_nan_rows():
    model, degrees = MODELS["millimetres_offsets"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 20)
    joints[3, 5] = np.nan  # J6 only: still NaN for a point on J1
    joints[7, 0] = np.nan
    frames = [(0, np.array([1.0, 2.0, 3.0])), (6, np.zeros(3))]

    points = robot.batch_frame_points(joints, frames, EE)

    assert np.isnan(points[[3, 7]]).all()
    finite = np.ones(len(joints), dtype=bool)
    finite[[3, 7]] = False
    assert np.isfinite(points[finite]).all()


def test_threads_give_identical_results():
    model, degrees = MODELS["millimetres_offsets"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 5000)
    joints[::97, 2] = np.nan
    frames = _frames(100.0)

    reference = robot.batch_frame_points(joints, frames, EE, threads=1)
    for threads in (None, 4):
        np.testing.assert_array_equal(robot.batch_frame_points(joints, frames, EE, threads=threads), reference)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(robot.batch_frame_points(joints, frames, EE, threads=0), reference)
    with pytest.raises(ValueError):
        robot.batch_frame_points(joints, frames, EE, threads=-1)


def test_empty_inputs():
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    frames = [(2, np.zeros(3)), (6, np.zeros(3))]
    for threads in (None, 1, 4):
        assert robot.batch_frame_points(np.empty((0, 6)), frames, EE, threads=threads).shape == (0, 2, 3)
        assert robot.batch_frame_points(_joints(degrees, 3), [], threads=threads).shape == (3 + 64 + 1, 0, 3)


def test_dataframe_input():
    pd = pytest.importorskip("pandas")
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 10)
    frame = pd.DataFrame(joints, columns=["J1", "J2", "J3", "J4", "J5", "J6"])
    frames = [(4, np.array([0.1, 0.0, 0.0]))]
    np.testing.assert_array_equal(robot.batch_frame_points(frame, frames), robot.batch_frame_points(joints, frames))


def test_invalid_arguments():
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 2)
    with pytest.raises(ValueError, match="ee_transform"):
        robot.batch_frame_points(joints, [(6, np.zeros(3))])
    with pytest.raises(ValueError, match="0..=6"):
        robot.batch_frame_points(joints, [(7, np.zeros(3))], EE)
    with pytest.raises((ValueError, OverflowError)):
        robot.batch_frame_points(joints, [(-1, np.zeros(3))])
    with pytest.raises(ValueError, match="offset"):
        robot.batch_frame_points(joints, [(2, np.zeros(2))])
    with pytest.raises(ValueError, match="shape"):
        robot.batch_frame_points(np.zeros((3, 5)), [(2, np.zeros(3))])
