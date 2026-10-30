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


def _reference(robot: Robot, joints: np.ndarray, frames: list[int], ee) -> np.ndarray:
    # The raw matrices: the public method's RigidTransform re-normalises rotations by about 1e-16.
    per_config = 6 if ee is None else 7
    ee_matrix = None if ee is None else ee.as_matrix()
    raw = robot._robot.batch_joint_poses(joints, ee_matrix)
    poses = raw.reshape(len(joints), per_config, 4, 4)
    return np.ascontiguousarray(poses[:, frames].swapaxes(0, 1))


@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("name", list(MODELS))
def test_identical_to_batch_joint_poses(name: str, homogeneous: bool):
    model, degrees = MODELS[name]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees)
    frames = [0, 1, 2, 3, 4, 5, 6, 2, 6, 0]

    out = robot.batch_frame_poses(joints, frames, EE, homogeneous=homogeneous)

    expected = _reference(robot, joints, frames, EE)
    if not homogeneous:
        expected = expected[..., :3, :]
    assert out.shape == expected.shape
    assert out.dtype == np.float64 and out.flags.c_contiguous
    assert all(out[k].flags.c_contiguous for k in range(len(frames)))
    np.testing.assert_array_equal(out, expected)

    public = robot.batch_joint_poses(joints, EE).as_matrix().reshape(len(joints), 7, 4, 4)
    public = public[:, frames].swapaxes(0, 1)[..., : out.shape[2], :]
    np.testing.assert_allclose(out, public, rtol=0, atol=1e-12)


@pytest.mark.parametrize("deepest", range(6))
def test_shallow_selection(deepest: int):
    # Only the chain up to the deepest selected joint is evaluated; the result must not change.
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 50)
    frames = list(range(deepest + 1))[::-1]
    np.testing.assert_array_equal(
        robot.batch_frame_poses(joints, frames, homogeneous=True),
        _reference(robot, joints, frames, None),
    )


def test_nan_rows():
    model, degrees = MODELS["millimetres_offsets"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 20)
    joints[3, 5] = np.nan  # J6 only: still NaN in the J1 frame
    joints[7, 0] = np.nan

    for homogeneous in (False, True):
        out = robot.batch_frame_poses(joints, [0, 6], EE, homogeneous=homogeneous)
        assert np.isnan(out[:, [3, 7]]).all()
        finite = np.ones(len(joints), dtype=bool)
        finite[[3, 7]] = False
        assert np.isfinite(out[:, finite]).all()


@pytest.mark.parametrize("homogeneous", [False, True])
def test_threads_give_identical_results(homogeneous: bool):
    model, degrees = MODELS["millimetres_offsets"]
    robot = Robot(model, degrees=degrees)
    # Not a multiple of the internal run length, so the last run is partial.
    joints = _joints(degrees, 5000)
    joints[::97, 2] = np.nan
    frames = [2, 6, 0, 5]

    reference = robot.batch_frame_poses(joints, frames, EE, homogeneous=homogeneous, threads=1)
    for threads in (None, 4):
        out = robot.batch_frame_poses(joints, frames, EE, homogeneous=homogeneous, threads=threads)
        np.testing.assert_array_equal(out, reference)
    with pytest.warns(DeprecationWarning):
        out = robot.batch_frame_poses(joints, frames, EE, homogeneous=homogeneous, threads=0)
    np.testing.assert_array_equal(out, reference)
    with pytest.raises(ValueError):
        robot.batch_frame_poses(joints, frames, EE, threads=-1)


def test_empty_inputs():
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    for threads in (None, 1, 4):
        assert robot.batch_frame_poses(np.empty((0, 6)), [2, 6], EE, threads=threads).shape == (2, 0, 3, 4)
        assert robot.batch_frame_poses(
            np.empty((0, 6)), [2], homogeneous=True, threads=threads
        ).shape == (1, 0, 4, 4)
        assert robot.batch_frame_poses(_joints(degrees, 3), [], threads=threads).shape == (0, 3 + 64 + 1, 3, 4)


def test_dataframe_input():
    pd = pytest.importorskip("pandas")
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 10)
    frame = pd.DataFrame(joints, columns=["J1", "J2", "J3", "J4", "J5", "J6"])
    np.testing.assert_array_equal(robot.batch_frame_poses(frame, [4]), robot.batch_frame_poses(joints, [4]))


def test_invalid_arguments():
    model, degrees = MODELS["metres_flipped"]
    robot = Robot(model, degrees=degrees)
    joints = _joints(degrees, 2)
    with pytest.raises(ValueError, match="ee_transform"):
        robot.batch_frame_poses(joints, [6])
    with pytest.raises(ValueError, match="0..=6"):
        robot.batch_frame_poses(joints, [7], EE)
    with pytest.raises((ValueError, OverflowError)):
        robot.batch_frame_poses(joints, [-1])
    with pytest.raises(ValueError, match="shape"):
        robot.batch_frame_poses(np.zeros((3, 5)), [2])
