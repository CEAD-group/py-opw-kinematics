"""
py-opw-kinematics: Forward and inverse kinematics for six-axis industrial robots.

Rotation handling is delegated to scipy.spatial.transform.Rotation for flexibility.
This library focuses on pure kinematics with 4x4 transformation matrices.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import RigidTransform

from ._internal import KinematicModel
from ._internal import Robot as _RobotInternal

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]
    import polars as pl
    NumpyOrDataFrame = np.ndarray | pd.DataFrame | pl.DataFrame

_JOINT_COLS = ["J1", "J2", "J3", "J4", "J5", "J6"]


# eq=False: the generated __eq__ and __hash__ do not work on ndarray fields.
@dataclass(frozen=True, eq=False)
class ReachResult:
    """
    All eight inverse-kinematics branches for each pose, with per-branch margins.

    Branch slots are stable: slot ``s = 4 * wrist_flip + 2 * shoulder + elbow``,
    where the sign choices are taken on the raw OPW angles (before offsets and
    flip_axes): elbow 0 has ``theta3 = +acos(..) - psi3``, elbow 1 the negative
    root; shoulder 0 has ``theta1 = atan2(cy, cx) - atan2(b, ..)``, shoulder 1
    the arm reaching back (``theta1 + pi``); wrist_flip 0 has ``theta5 >= 0``,
    wrist_flip 1 ``theta5 <= 0``. This is the order rs-opw-kinematics builds
    its candidates in.

    :ivar joints: (n, 8, 6) joint angles in the robot's angle unit. A NaN row
        means the branch does not exist for that pose (wrist centre outside the
        reachable annulus or a complex root). Limits are never applied.
    :ivar limit_margin: (n, 8) ``min(q - lo, hi - q)`` over all joints, in the
        robot's angle unit. Negative means outside the limits. ``+inf`` when
        no limits were given, NaN where the branch does not exist. Joints are
        normalised to (-180, 180] degrees before the comparison, so a limit
        window that is not contained in one turn (for example 0..360) is
        evaluated against the wrapped angle.
    :ivar extension: (n,) signed distance of the wrist centre to the annulus
        reachable by the shoulder, in the model's length unit, taken over both
        the front and back shoulder configurations. Positive inside, zero on
        the reach boundary, negative when no branch can exist. For a wrist
        centre closer than ``|b|`` to the J1 axis it is the negative distance
        to that cylinder instead.
    :ivar sigma_min: (n, 8) smallest singular value of the 6x6 geometric
        Jacobian of the requested TCP with respect to the joints in radians.
        Translation rows are in the model's length unit per radian, rotation
        rows in radians per radian. Zero at a singularity.
    :ivar wrist: (n, 8) ``|sin(theta5)|``; zero at the wrist singularity.
    """

    joints: np.ndarray
    limit_margin: np.ndarray
    extension: np.ndarray
    sigma_min: np.ndarray
    wrist: np.ndarray


class Robot:
    """
    Robot kinematics with scipy RigidTransform integration.

    Joint angles can be in degrees (default) or radians.
    Poses are represented as scipy RigidTransform objects.
    """

    def __init__(
        self,
        kinematic_model: KinematicModel,
        degrees: bool = True,
    ) -> None:
        """
        Initialize a Robot instance.

        :param kinematic_model: The kinematic model defining robot geometry.
        :param degrees: If True, joint angles are in degrees. If False, radians.
        """
        self._robot = _RobotInternal(kinematic_model, degrees)
        self._degrees = degrees
        self._kinematic_model = kinematic_model

    @property
    def degrees(self) -> bool:
        """Whether joint angles are in degrees (True) or radians (False)."""
        return self._degrees

    def __repr__(self) -> str:
        return self._robot.__repr__()

    def forward(
        self,
        joints: tuple[float, float, float, float, float, float],
        ee_transform: Optional["RigidTransform"] = None,
    ) -> "RigidTransform":
        """
        Compute forward kinematics for given joint angles.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation (optional).
        :return: RigidTransform representing the TCP pose.
        """
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        matrix_4x4 = self._robot.forward(joints, ee_matrix)
        return RigidTransform.from_matrix(matrix_4x4)

    def inverse(
        self,
        pose: "RigidTransform",
        current_joints: tuple[float, float, float, float, float, float] | None = None,
        ee_transform: Optional["RigidTransform"] = None,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """
        Compute inverse kinematics for a given pose.

        :param pose: Desired TCP pose as RigidTransform.
        :param current_joints: Current joint configuration for solution ranking.
        :param ee_transform: End effector transformation (optional).
        :return: List of possible joint configurations.
        """
        matrix_4x4 = pose.as_matrix()
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        return self._robot.inverse(matrix_4x4, current_joints, ee_matrix)

    def joint_poses(
        self,
        joints: tuple[float, float, float, float, float, float],
        ee_transform: Optional["RigidTransform"] = None,
    ) -> RigidTransform:
        """
        Compute per-joint poses using the OPW FK chain (consistent with forward()).

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation (optional).
            When provided, an additional TCP+EE pose is appended.
        :return: RigidTransform with 6 poses [J1, J2, J3, J4, J5, J6/TCP],
            or 7 poses [..., TCP+EE] when ee_transform is given.
        """
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        raw = np.array(self._robot.joint_poses(joints, ee_matrix))
        return RigidTransform.from_matrix(raw)

    def batch_joint_poses(
        self,
        joints: "NumpyOrDataFrame",
        ee_transform: Optional["RigidTransform"] = None,
    ) -> RigidTransform:
        """
        Compute per-joint poses for multiple joint configurations.

        :param joints: Joint angles as numpy array (N,6), or DataFrame with columns J1-J6.
        :param ee_transform: End effector transformation (optional).
        :return: RigidTransform with N*6 poses (or N*7 when ee_transform is given).
            Reshape to (N, 6, 4, 4) or (N, 7, 4, 4) for per-config access.
        """
        if hasattr(joints, "to_numpy"):
            if hasattr(joints, "select"):
                arr = joints.select(_JOINT_COLS).to_numpy()  # type: ignore[operator]
            else:
                arr = joints[_JOINT_COLS].to_numpy()  # type: ignore[union-attr,attr-defined]
        else:
            arr = np.asarray(joints)
        joints_array = np.ascontiguousarray(arr, dtype=np.float64)

        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        result_array = self._robot.batch_joint_poses(joints_array, ee_matrix)

        return RigidTransform.from_matrix(result_array.reshape(-1, 4, 4))

    def forward_frames(
        self,
        joints: tuple[float, float, float, float, float, float],
        ee_transform: Optional["RigidTransform"] = None,
    ) -> RigidTransform:
        """
        Compute 4x4 transform matrices for all robot links.

        .. deprecated:: 1.2.0
            Use :meth:`joint_poses` instead. ``forward_frames`` uses incorrect
            rotation axes (X instead of Z for J4/J6), producing incorrect
            orientations. Translations are correct.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation (optional).
        :return: List of RigidTransforms for [Base, J1, J2, J3, J4, J5, J6, TCP].
        """
        import warnings
        warnings.warn(
            "forward_frames() is deprecated and produces incorrect rotations. "
            "Use joint_poses() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        raw_frames = np.array(self._robot.forward_frames(joints, ee_matrix))
        return RigidTransform.from_matrix(raw_frames)

    def batch_forward(
        self,
        joints: "NumpyOrDataFrame",
        ee_transform: Optional["RigidTransform"] = None,
    ) -> RigidTransform:
        """
        Compute forward kinematics for multiple joint configurations.

        :param joints: Joint angles as numpy array (N,6), or DataFrame with columns J1-J6.
        :param ee_transform: End effector transformation (optional).
        :return: List of RigidTransform objects.
        """
        if hasattr(joints, "to_numpy"):
            if hasattr(joints, "select"):
                arr = joints.select(_JOINT_COLS).to_numpy()  # type: ignore[operator]
            else:
                arr = joints[_JOINT_COLS].to_numpy()  # type: ignore[union-attr,attr-defined]
        else:
            arr = np.asarray(joints)
        joints_array = np.ascontiguousarray(arr, dtype=np.float64)

        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        result_array = self._robot.batch_forward(joints_array, ee_matrix)

        return RigidTransform.from_matrix(result_array.reshape(-1, 4, 4))

    def batch_inverse(
        self,
        poses: RigidTransform,
        current_joints: "NumpyOrDataFrame | tuple[float, float, float, float, float, float] | None" = None,
        ee_transform: Optional["RigidTransform"] = None,
    ) -> "NumpyOrDataFrame":
        """
        Compute inverse kinematics for multiple poses.

        :param poses: RigidTransform containing N poses.
        :param current_joints: Starting joint configuration for solution continuity.
            Can be tuple, numpy array, or single-row DataFrame.
        :param ee_transform: End effector transformation (optional).
        :return: Joint angles in same format as current_joints (numpy array if not specified).
        """
        # Detect output type from current_joints
        if current_joints is not None:
            if hasattr(current_joints, "to_numpy"):
                # DataFrame-like inputs (Pandas, Polars)
                is_polars = hasattr(current_joints, "select")
                if is_polars:
                    arr = current_joints.select(_JOINT_COLS).to_numpy()  # type: ignore[union-attr,attr-defined,operator]
                    output_kwargs = {"schema": _JOINT_COLS}
                else:
                    arr = current_joints[_JOINT_COLS].to_numpy()  # type: ignore[union-attr,attr-defined,call-overload]
                    output_kwargs = {"columns": _JOINT_COLS}
                output_type = type(current_joints)
            else:
                arr = np.atleast_2d(current_joints)
                output_type, output_kwargs = None, {}
            current_joints_tuple = tuple(np.ascontiguousarray(arr, dtype=np.float64)[0])
        else:
            current_joints_tuple = None
            output_type, output_kwargs = None, {}

        matrix_array = poses.as_matrix().reshape(-1, 16)

        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        result_array = self._robot.batch_inverse(matrix_array, current_joints_tuple, ee_matrix)

        return output_type(result_array, **output_kwargs) if output_type else result_array  # type: ignore[misc,arg-type,call-overload]

    def reach(
        self,
        poses: RigidTransform,
        joint_limits: ArrayLike | None = None,
        ee_transform: Optional["RigidTransform"] = None,
    ) -> ReachResult:
        """
        Compute all eight inverse-kinematics branches for multiple poses.

        Unlike :meth:`batch_inverse` there is no continuity selection, no limit
        filtering and no sorting: every branch that exists is returned in a
        stable slot, and ``limit_margin`` reports how far inside (or outside)
        the joint limits it is. See :class:`ReachResult` for the slot order.

        :param poses: RigidTransform containing N poses.
        :param joint_limits: (6, 2) lower/upper joint bounds in the robot's
            angle unit (optional). Without limits every margin is ``+inf``.
        :param ee_transform: End effector transformation (optional).
        :return: ReachResult with per-pose, per-branch arrays.
        """
        matrix_array = np.ascontiguousarray(poses.as_matrix().reshape(-1, 16), dtype=np.float64)
        limits = None
        if joint_limits is not None:
            limits_array = np.asarray(joint_limits, dtype=np.float64)
            if limits_array.shape != (6, 2):
                raise ValueError("joint_limits must have shape (6, 2)")
            limits = [tuple(row) for row in limits_array]
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        joints, limit_margin, extension, sigma_min, wrist = self._robot.reach(
            matrix_array, limits, ee_matrix
        )
        return ReachResult(
            joints=joints,
            limit_margin=limit_margin,
            extension=extension,
            sigma_min=sigma_min,
            wrist=wrist,
        )


def interpolate_poses(
    x: ArrayLike,
    poses: "RigidTransform",
    xn: ArrayLike,
) -> "RigidTransform":
    """
    Interpolate poses at new points using SLERP for rotation and linear for translation.

    API follows scipy.interpolate.interp1d(x, y) pattern.

    :param x: Array of N values corresponding to each keyframe (e.g., times or distances).
    :param poses: RigidTransform with N keyframe poses.
    :param xn: Array of M values where interpolation is desired.
    :return: RigidTransform containing M interpolated poses.

    Example:
        >>> keyframes = RigidTransform.concatenate([pose_start, pose_end])
        >>> trajectory = interpolate_poses([0, 1], keyframes, np.linspace(0, 1, 11))
    """
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp

    x = np.asarray(x)
    xn = np.asarray(xn)

    # SLERP for rotations
    interp_rotations = Slerp(x, poses.rotation)(xn)

    # Linear interpolation for translations
    interp_translations = interp1d(x, poses.translation, axis=0)(xn)

    return RigidTransform.from_components(rotation=interp_rotations, translation=interp_translations)  # type: ignore[arg-type]


__all__ = [
    "KinematicModel",
    "ReachResult",
    "RigidTransform",
    "Robot",
    "interpolate_poses",
]
