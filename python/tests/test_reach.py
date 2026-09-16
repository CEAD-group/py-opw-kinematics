import itertools

import numpy as np
import pytest
from py_opw_kinematics import KinematicModel, ReachResult, Robot
from scipy.spatial.transform import RigidTransform, Rotation

# Comau NJ165 in metres, kinematic-frame J3 (no parallelogram conversion).
NJ165 = KinematicModel(
    a1=0.400,
    a2=-0.250,
    b=0.0,
    c1=0.830,
    c2=1.175,
    c3=1.444,
    c4=0.230,
    offsets=(0, 0, 0, 0, 0, 0),
    flip_axes=(True, False, True, True, False, True),
)
NJ165_LIMITS = np.array(
    [(-180, 180), (-75, 95), (-160, -30), (-2700, 2700), (-125, 125), (-2700, 2700)],
    dtype=float,
)
HOME = np.array([0.0, 0.0, -90.0, 0.0, 0.0, 0.0])

Joints = tuple[float, float, float, float, float, float]


def _j(q: np.ndarray) -> Joints:
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]))


@pytest.fixture(scope="module")
def robot() -> Robot:
    return Robot(NJ165, degrees=True)


def _margin(joints: np.ndarray, limits: np.ndarray) -> np.ndarray:
    lo, hi = limits[:, 0], limits[:, 1]
    return np.minimum(joints - lo, hi - joints).min(axis=-1)


def _grid_joints() -> np.ndarray:
    # Inside the limits, away from wrap-around on J1, J4, J6 so IK normalisation
    # cannot move a sample to an equivalent angle.
    j1 = np.linspace(-150, 150, 5)
    j2 = np.linspace(-60, 80, 5)
    j3 = np.linspace(-150, -40, 5)
    j4 = np.array([-120.0, 30.0, 150.0])
    # J5 = 0 is excluded: at the wrist singularity the J4/J6 split is not unique.
    j5 = np.linspace(-110, 110, 6)
    j6 = np.array([-90.0, 0.0, 60.0])
    grid = np.array(list(itertools.product(j1, j2, j3, j4, j5, j6)), dtype=float)
    return grid


def _slot_of(joints_sample: np.ndarray, slots: np.ndarray, tol: float = 1e-6) -> int:
    # Compare modulo a full turn: batch_inverse keeps angles near the previous row.
    diff = (slots - joints_sample + 180.0) % 360.0 - 180.0
    matches = np.where(np.all(np.abs(diff) < tol, axis=1))[0]
    assert len(matches) >= 1, f"{joints_sample} not found in {slots}"
    return int(matches[0])


def test_home_pose(robot: Robot):
    pose = robot.forward(_j(HOME))
    assert np.allclose(pose.translation, [0.400 + 1.444 + 0.230, 0.0, 0.830 + 1.175 + 0.250])
    # Flange z points along world x at home in this crate; only the position is checked.

    result = robot.reach(pose)
    assert isinstance(result, ReachResult)
    assert result.joints.shape == (1, 8, 6)
    slot = _slot_of(HOME, result.joints[0])
    assert slot >= 0

    for s in range(8):
        q = result.joints[0, s]
        if np.isnan(q[0]):
            continue
        back = robot.forward(_j(q))
        assert np.allclose(back.translation, pose.translation, atol=1e-9)
        assert np.allclose(back.rotation.as_matrix(), pose.rotation.as_matrix(), atol=1e-9)

    # J5 == 0 on the home branch: wrist singularity on both flips of that elbow.
    for s in (0, 4):
        assert result.wrist[0, s] < 1e-12
        assert result.sigma_min[0, s] < 1e-6
    exists = ~np.isnan(result.joints[0, :, 0])
    assert np.all(np.isinf(result.limit_margin[0, exists]))


def test_round_trip_through_fk(robot: Robot):
    samples = _grid_joints()
    poses = robot.batch_forward(samples)
    result = robot.reach(poses, NJ165_LIMITS)

    assert result.joints.shape == (len(samples), 8, 6)
    expected_margin = _margin(samples, NJ165_LIMITS)
    slots = np.array([_slot_of(q, result.joints[i]) for i, q in enumerate(samples)])
    found_margin = result.limit_margin[np.arange(len(samples)), slots]
    assert np.allclose(found_margin, expected_margin, atol=1e-6)

    # The slot index is a function of the sign choices only. flip_axes negates
    # J1 and J3, so the raw OPW angles are -J1 and -J3.
    matrices = poses.as_matrix()
    wrist_centre = matrices[:, :3, 3] - 0.230 * matrices[:, :3, 2]
    theta1_front = np.degrees(np.arctan2(wrist_centre[:, 1], wrist_centre[:, 0]))
    d1 = (-samples[:, 0] - theta1_front + 180.0) % 360.0 - 180.0
    shoulder = (np.abs(d1) > 90.0).astype(int)
    raw_theta3 = -samples[:, 2] + np.degrees(np.arctan2(-0.250, 1.444))
    elbow = (raw_theta3 < 0).astype(int)
    wrist_flip = (samples[:, 4] < 0).astype(int)
    assert np.array_equal(slots, 4 * wrist_flip + 2 * shoulder + elbow)
    assert shoulder.any() and not shoulder.all()

    # No sample was near a boundary, so every non-NaN slot reproduces the pose.
    exists = ~np.isnan(result.joints[:, :, 0])
    flat = result.joints[exists]
    back = robot.batch_forward(flat).as_matrix()
    target = np.repeat(poses.as_matrix()[:, None], 8, axis=1)[exists]
    assert np.allclose(back, target, atol=1e-9)


def test_superset_of_batch_inverse(robot: Robot):
    samples = _grid_joints()[::7]
    poses = robot.batch_forward(samples)
    single = np.asarray(robot.batch_inverse(poses, current_joints=_j(HOME)))
    result = robot.reach(poses)

    for i in range(len(samples)):
        if np.isnan(single[i, 0]):
            continue
        _slot_of(single[i], result.joints[i])
        assert np.any(~np.isnan(result.joints[i, :, 0]))


def test_branch_existence_is_order_independent(robot: Robot):
    rng = np.random.default_rng(0)
    samples = _grid_joints()[::5]
    poses = robot.batch_forward(samples)
    perm = rng.permutation(len(samples))
    # Index rather than rebuild from matrices, so the inputs stay bitwise identical.
    shuffled = poses[perm]

    a = robot.reach(poses, NJ165_LIMITS)
    b = robot.reach(shuffled, NJ165_LIMITS)
    inv = np.argsort(perm)
    for name in ("joints", "limit_margin", "extension", "sigma_min", "wrist"):
        assert np.array_equal(getattr(a, name), getattr(b, name)[inv], equal_nan=True)


def test_extension_margin(robot: Robot):
    # Walk the TCP radially outward from the shoulder at the shoulder height.
    rotation = robot.forward(_j(HOME)).rotation
    # Start beyond max(c2, L) from the shoulder so the outer boundary is the active one.
    radius = np.linspace(2.2, 3.5, 1301)
    z = 0.830 + 0.0  # J2 axis height
    translations = np.stack([radius, np.zeros_like(radius), np.full_like(radius, z)], axis=1)
    poses = RigidTransform.from_components(
        rotation=Rotation.concatenate([rotation] * len(radius)), translation=translations
    )
    result = robot.reach(poses)

    ext = result.extension
    assert np.all(np.diff(ext) < 0)
    # d = |wrist centre - J2 axis| in the J1 plane, L = sqrt(a2^2 + c3^2).
    wrist_x = radius - 0.230
    d = wrist_x - 0.400
    L = np.hypot(-0.250, 1.444)
    expected = np.minimum(1.175 + L - d, d - abs(1.175 - L))
    assert np.allclose(ext, expected, atol=1e-12)

    exists = np.any(~np.isnan(result.joints[:, :, 0]), axis=1)
    last = np.max(np.where(exists)[0])
    assert ext[last] >= 0
    assert ext[last + 1] < 0
    assert not np.any(exists[last + 1 :])
    step = np.diff(ext)
    assert np.allclose(step, step[0], atol=1e-12)


def test_extension_back_shoulder(robot: Robot):
    # Reachable only through the back-shoulder branches: extension must not be negative.
    pose = RigidTransform.from_components(
        rotation=robot.forward(_j(HOME)).rotation, translation=[-0.45, 0.0, 0.80]
    )
    result = robot.reach(pose)
    exists = ~np.isnan(result.joints[0, :, 0])
    assert exists[[2, 3, 6, 7]].any() and not exists[[0, 1, 4, 5]].any()
    assert result.extension[0] >= 0


def test_wrist_centre_inside_b_cylinder():
    model = KinematicModel(
        a1=0.400, a2=-0.250, b=0.15, c1=0.830, c2=1.175, c3=1.444, c4=0.230,
        offsets=(0, 0, 0, 0, 0, 0), flip_axes=(True, False, True, True, False, True),
    )
    robot = Robot(model, degrees=True)
    # Wrist centre on the J1 axis, well inside the |b| cylinder.
    pose = RigidTransform.from_components(
        rotation=Rotation.identity(), translation=[0.0, 0.0, 2.0 + 0.230]
    )
    result = robot.reach(pose)
    assert np.isfinite(result.extension[0])
    assert result.extension[0] < 0
    assert np.all(np.isnan(result.joints[0]))


def test_reach_result_equality_does_not_raise(robot: Robot):
    a = robot.reach(robot.forward(_j(HOME)))
    b = robot.reach(robot.forward(_j(HOME)))
    assert bool(a == b) is False
    assert a != b
    hash(a)


def _numeric_jacobian(robot: Robot, q: np.ndarray, h_deg: float = 1e-4) -> np.ndarray:
    # Columns: d(p, rotvec-ish)/d(q_rad) from forward differences via batch_forward.
    jac = np.zeros((6, 6))
    base = robot.forward(_j(q))
    for j in range(6):
        plus = q.copy()
        minus = q.copy()
        plus[j] += h_deg
        minus[j] -= h_deg
        fp = robot.forward(_j(plus))
        fm = robot.forward(_j(minus))
        dp = (fp.translation - fm.translation) / np.radians(2 * h_deg)
        # Angular velocity: (R(q+h) R(q-h)^T) as a rotation vector, per radian.
        dr = (fp.rotation * fm.rotation.inv()).as_rotvec() / np.radians(2 * h_deg)
        jac[:3, j] = dp
        jac[3:, j] = dr
    del base
    return jac


def test_sigma_min_matches_finite_difference(robot: Robot):
    rng = np.random.default_rng(1)
    lo, hi = NJ165_LIMITS[:, 0], NJ165_LIMITS[:, 1]
    lo = np.maximum(lo, -170)
    hi = np.minimum(hi, 170)
    samples = rng.uniform(lo, hi, size=(200, 6))
    # Keep the wrist away from its singularity so sigma_min is not dominated by it.
    samples[:, 4] = np.where(np.abs(samples[:, 4]) < 10, 20.0, samples[:, 4])
    poses = robot.batch_forward(samples)
    result = robot.reach(poses)

    for i, q in enumerate(samples):
        s = _slot_of(q, result.joints[i])
        numeric = np.linalg.svd(_numeric_jacobian(robot, q), compute_uv=False).min()
        analytic = result.sigma_min[i, s]
        assert abs(analytic - numeric) / numeric < 1e-3, (q, analytic, numeric)


def test_sigma_min_with_ee_transform(robot: Robot):
    ee = RigidTransform.from_components(
        rotation=Rotation.from_euler("xyz", [0, -90, 0], degrees=True), translation=[0.3, 0.1, 0.05]
    )
    q = np.array([20.0, 15.0, -70.0, 40.0, 50.0, -30.0])
    pose = robot.forward(_j(q), ee_transform=ee)
    result = robot.reach(pose, ee_transform=ee)
    s = _slot_of(q, result.joints[0])

    # Numeric Jacobian of the TCP including the tool offset.
    jac = np.zeros((6, 6))
    h = 1e-4
    for j in range(6):
        plus, minus = q.copy(), q.copy()
        plus[j] += h
        minus[j] -= h
        fp = robot.forward(_j(plus), ee_transform=ee)
        fm = robot.forward(_j(minus), ee_transform=ee)
        jac[:3, j] = (fp.translation - fm.translation) / np.radians(2 * h)
        jac[3:, j] = (fp.rotation * fm.rotation.inv()).as_rotvec() / np.radians(2 * h)
    numeric = np.linalg.svd(jac, compute_uv=False).min()
    assert abs(result.sigma_min[0, s] - numeric) / numeric < 1e-3


def test_singular_configurations(robot: Robot):
    # Wrist singularity: J5 == 0 on both flips.
    q = np.array([30.0, 20.0, -100.0, 45.0, 0.0, 10.0])
    pose = robot.forward(_j(q))
    result = robot.reach(pose)
    exists = ~np.isnan(result.joints[0, :, 0])
    assert exists[0] and exists[4]
    for s in (0, 4):
        assert result.wrist[0, s] < 1e-12
        assert result.sigma_min[0, s] < 1e-6
        # The J4/J6 split is arbitrary at the singularity; the pose must still hold.
        back = robot.forward(_j(result.joints[0, s]))
        assert np.allclose(back.as_matrix(), pose.as_matrix(), atol=1e-9)

    # Straight arm: place the wrist centre exactly on the outer reach circle.
    L = np.hypot(-0.250, 1.444)
    d = 1.175 + L
    wrist_x = 0.400 + d
    tcp_x = wrist_x + 0.230
    # Tool z along world x (the home orientation) puts the wrist centre c4 behind the TCP.
    pose = RigidTransform.from_components(
        rotation=robot.forward(_j(HOME)).rotation, translation=[tcp_x, 0.0, 0.830]
    )
    result = robot.reach(pose)
    assert abs(result.extension[0]) < 1e-12
    exists = ~np.isnan(result.joints[0, :, 0])
    assert np.any(exists)
    assert np.all(result.sigma_min[0, exists] < 1e-6)


def test_limits_are_reported_not_enforced(robot: Robot):
    q = np.array([10.0, 80.0, -100.0, 20.0, 40.0, 30.0])
    tight = NJ165_LIMITS.copy()
    tight[1] = (-75, 75)
    result = robot.reach(robot.forward(_j(q)), tight)
    s = _slot_of(q, result.joints[0])
    assert np.isclose(result.limit_margin[0, s], -5.0, atol=1e-9)


def test_nan_pose_row(robot: Robot):
    poses = robot.batch_forward(np.array([HOME, HOME]))
    matrices = poses.as_matrix()
    matrices[1, 0, 3] = np.nan
    # Bypass RigidTransform validation by going through the internal call.
    joints, margin, ext, _sigma, _wrist = robot._robot.reach(
        np.ascontiguousarray(matrices.reshape(-1, 16)), None, None
    )
    assert np.all(np.isnan(joints[1]))
    assert np.isnan(ext[1])
    assert np.all(np.isnan(margin[1]))
    assert not np.all(np.isnan(joints[0]))


def test_radians_robot():
    robot_rad = Robot(NJ165, degrees=False)
    q = np.radians(np.array([10.0, 20.0, -70.0, 30.0, 20.0, 10.0]))
    result = robot_rad.reach(robot_rad.forward(_j(q)), np.radians(NJ165_LIMITS))
    s = _slot_of(q, result.joints[0])
    assert np.isclose(result.limit_margin[0, s], _margin(q, np.radians(NJ165_LIMITS)))


def test_joint_limits_shape_validation(robot: Robot):
    with pytest.raises(ValueError):
        robot.reach(robot.forward(_j(HOME)), np.zeros((5, 2)))


def test_sigma_min_skipped_outside_limits(robot: Robot):
    q = np.array([10.0, 20.0, -70.0, 30.0, 40.0, 10.0])
    tight = NJ165_LIMITS.copy()
    # Keeps the front-shoulder branches only; the back-shoulder ones need J1 near 180.
    tight[0] = (-30, 30)
    pose = robot.forward(_j(q))
    limited = robot.reach(pose, tight)
    unlimited = robot.reach(pose)

    outside = limited.limit_margin[0] < 0
    assert np.any(outside)
    assert np.all(np.isnan(limited.sigma_min[0, outside]))
    # Every usable branch keeps the value it has without limits.
    inside = limited.limit_margin[0] >= 0
    assert np.any(inside)
    assert np.array_equal(limited.sigma_min[0, inside], unlimited.sigma_min[0, inside])


@pytest.mark.parametrize("threads", [0, 2, 4])
def test_threads_give_identical_results(robot: Robot, threads: int):
    samples = _grid_joints()[::3]
    poses = robot.batch_forward(samples)
    single = robot.reach(poses, NJ165_LIMITS)
    threaded = robot.reach(poses, NJ165_LIMITS, threads=threads)
    for name in ("joints", "limit_margin", "extension", "sigma_min", "wrist"):
        assert np.array_equal(getattr(single, name), getattr(threaded, name), equal_nan=True)


def test_threads_release_the_gil(robot: Robot):
    # A second Python thread must keep running while reach is in the kernel.
    import threading
    import time

    samples = _grid_joints()
    poses = robot.batch_forward(samples)
    ticks = 0
    done = threading.Event()

    def spin():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        while ticks == 0:
            time.sleep(0.001)
        before = ticks
        t0 = time.perf_counter()
        robot.reach(poses, NJ165_LIMITS)
        elapsed = time.perf_counter() - t0
    finally:
        done.set()
        spinner.join()
    # The spinner sleeps 1 ms per tick, so a GIL-holding kernel would stall it.
    assert elapsed > 0.02, "kernel too fast to tell; use a larger grid"
    assert ticks - before > 5


def test_negative_threads_rejected(robot: Robot):
    with pytest.raises(ValueError):
        robot.reach(robot.forward(_j(HOME)), threads=-1)
