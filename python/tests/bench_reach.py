"""
Benchmark for Robot.reach against batch_inverse on the Comau NJ165.

Not collected by pytest; run with ``uv run python python/tests/bench_reach.py``.
"""

import time

import numpy as np
from py_opw_kinematics import KinematicModel, Robot
from scipy.spatial.transform import RigidTransform, Rotation

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
HOME = (0.0, 0.0, -90.0, 0.0, 0.0, 0.0)


def build_poses(robot: Robot) -> RigidTransform:
    rotation = robot.forward((0, 30, -60, 0, 60, 0)).rotation
    xs = np.linspace(1.0, 3.0, 100)
    ys = np.linspace(-1.0, 1.0, 200)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    translations = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.8)], axis=1)
    return RigidTransform.from_components(
        rotation=Rotation.concatenate([rotation] * len(translations)),
        translation=translations,
    )


def timed(fn, repeats: int = 3):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def main() -> None:
    robot = Robot(NJ165, degrees=True)
    poses = build_poses(robot)
    n = len(poses)

    single, t_single = timed(lambda: robot.batch_inverse(poses, current_joints=HOME))
    result, t_reach = timed(lambda: robot.reach(poses, NJ165_LIMITS))

    reachable_single = np.mean(~np.isnan(single[:, 0]))
    inside = result.limit_margin >= 0
    reachable_any = np.mean(np.any(inside, axis=1))
    sigma = result.sigma_min[inside]

    print(f"poses:                     {n}")
    print(f"batch_inverse:             {t_single / n * 1e6:8.2f} us/pose")
    print(f"reach (with margins):      {t_reach / n * 1e6:8.2f} us/pose")
    print(f"ratio reach/batch_inverse: {t_reach / t_single:8.2f}x")
    print(f"reachable (batch_inverse): {reachable_single:8.3f}")
    print(f"reachable (any branch):    {reachable_any:8.3f}")
    print(
        "sigma_min inside limits:   "
        f"min {sigma.min():.4g}, p1 {np.percentile(sigma, 1):.4g}, median {np.median(sigma):.4g}"
    )


if __name__ == "__main__":
    main()
