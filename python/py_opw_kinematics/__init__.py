"""
py-opw-kinematics: Forward and inverse kinematics for six-axis industrial robots.

Rotation handling is delegated to scipy.spatial.transform.Rotation for flexibility.
This library focuses on pure kinematics with 4x4 transformation matrices.
"""

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from ._internal import KinematicModel
from ._internal import Robot as _RobotInternal

from scipy.spatial.transform import RigidTransform

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]
    import polars as pl
    NumpyOrDataFrame = Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]

_JOINT_COLS = ["J1", "J2", "J3", "J4", "J5", "J6"]


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
        joints: Tuple[float, float, float, float, float, float],
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
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
        ee_transform: Optional["RigidTransform"] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
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

    def forward_frames(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional["RigidTransform"] = None,
    ) -> RigidTransform:
        """
        Compute 4x4 transform matrices for all robot links.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation (optional).
        :return: List of RigidTransforms for [Base, J1, J2, J3, J4, J5, J6, TCP].
        """
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
            arr = joints.select(_JOINT_COLS).to_numpy() if hasattr(joints, "select") else joints[_JOINT_COLS].to_numpy()  # type: ignore[union-attr]
        else:
            arr = np.asarray(joints)
        joints_array = np.ascontiguousarray(arr, dtype=np.float64)

        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        result_array = self._robot.batch_forward(joints_array, ee_matrix)

        return RigidTransform.from_matrix(result_array.reshape(-1, 4, 4))

    def batch_inverse(
        self,
        poses: RigidTransform,
        current_joints=None,
        ee_transform: Optional["RigidTransform"] = None,
    ):
        """
        Compute inverse kinematics for multiple poses.

        :param poses: List of RigidTransform objects.
        :param current_joints: Starting joint configuration for solution continuity.
            Can be tuple, numpy array, or DataFrame row.
        :param ee_transform: End effector transformation (optional).
        :return: Joint angles in same format as current_joints (numpy array if not specified).
        """
        # Detect output type from current_joints
        if current_joints is not None:
            cj = np.atleast_2d(current_joints)
            if hasattr(cj, "to_numpy"):
                # convenience methods for DataFrame-like inputs (Pandas, Polars)
                is_polars = hasattr(cj, "select")
                arr = cj.select(_JOINT_COLS).to_numpy() if is_polars else cj[_JOINT_COLS].to_numpy()  # type: ignore[union-attr,attr-defined]
                output_type, output_kwargs = type(current_joints), {"schema" if is_polars else "columns": _JOINT_COLS}
            else:
                arr = np.asarray(cj)
                output_type, output_kwargs = None, {}
            current_joints_tuple = tuple(np.ascontiguousarray(arr, dtype=np.float64)[0])
        else:
            current_joints_tuple = None
            output_type, output_kwargs = None, {}

        matrix_array = poses.as_matrix().reshape(-1, 16)

        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        result_array = self._robot.batch_inverse(matrix_array, current_joints_tuple, ee_matrix)

        return output_type(result_array, **output_kwargs) if output_type else result_array


def interpolate_poses(
    x: np.ndarray,
    poses: "RigidTransform",
    xn: np.ndarray,
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
    "Robot",
    "RigidTransform",
    "interpolate_poses",
]
