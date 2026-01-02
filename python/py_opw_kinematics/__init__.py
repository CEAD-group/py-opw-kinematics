from typing import List, Optional, Tuple

import numpy as np
import polars as pl

from ._internal import CheckMode, CollisionBody, EulerConvention, JointLimits, KinematicModel, RobotBody, SafetyDistances
from ._internal import Robot as _RobotInternal


class Robot:
    """
    Robot kinematics wrapper that provides a Polars DataFrame interface
    for batch operations while using NumPy arrays internally for performance.

    Supports optional collision detection (via robot_body) and joint limits.
    """

    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        robot_body: Optional[RobotBody] = None,
        joint_limits: Optional[JointLimits] = None,
    ) -> None:
        self._robot = _RobotInternal(
            kinematic_model, euler_convention, ee_rotation, ee_translation,
            robot_body, joint_limits
        )

    def __repr__(self) -> str:
        return self._robot.__repr__()

    @property
    def ee_rotation(self) -> Tuple[float, float, float]:
        return self._robot.ee_rotation

    @ee_rotation.setter
    def ee_rotation(self, value: Tuple[float, float, float]) -> None:
        self._robot.ee_rotation = value

    @property
    def ee_translation(self) -> Tuple[float, float, float]:
        return self._robot.ee_translation

    @ee_translation.setter
    def ee_translation(self, value: Tuple[float, float, float]) -> None:
        self._robot.ee_translation = value

    @property
    def ee_rotation_quat(self) -> Tuple[float, float, float, float]:
        """End-effector rotation as quaternion (w, x, y, z) - scalar first."""
        return self._robot.ee_rotation_quat

    @ee_rotation_quat.setter
    def ee_rotation_quat(self, value: Tuple[float, float, float, float]) -> None:
        self._robot.ee_rotation_quat = value

    def forward(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position (x, y, z) and orientation as Euler angles.
        """
        return self._robot.forward(joints)

    def forward_quat(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Computes the forward kinematics with quaternion output.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position (x, y, z) and quaternion (w, x, y, z).
        """
        return self._robot.forward_quat(joints)

    def forward_matrix(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], List[List[float]]]:
        """
        Computes the forward kinematics with rotation matrix output.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position (x, y, z) and 3x3 rotation matrix.
        """
        return self._robot.forward_matrix(joints)

    def inverse(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics for a given pose.

        :param pose: Desired pose (position and Euler angles) of the end-effector.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        return self._robot.inverse(pose, current_joints)

    def inverse_quat(
        self,
        position: Tuple[float, float, float],
        quaternion: Tuple[float, float, float, float],
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics with quaternion input.

        :param position: Desired position (x, y, z) of the end-effector.
        :param quaternion: Desired orientation as quaternion (w, x, y, z) - scalar first.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        return self._robot.inverse_quat(position, quaternion, current_joints)

    def inverse_matrix(
        self,
        position: Tuple[float, float, float],
        rotation_matrix: List[List[float]],
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics with rotation matrix input.

        :param position: Desired position (x, y, z) of the end-effector.
        :param rotation_matrix: Desired orientation as 3x3 rotation matrix.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        return self._robot.inverse_matrix(position, rotation_matrix, current_joints)

    def batch_inverse(
        self,
        poses: pl.DataFrame,
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> pl.DataFrame:
        """
        Computes the inverse kinematics for multiple poses in batch mode.

        :param poses: DataFrame containing desired poses with columns X, Y, Z, A, B, C.
        :param current_joints: Current joint configuration (optional).
        :return: DataFrame containing the computed joint configurations with columns J1-J6.
        """
        # Convert Polars DataFrame to NumPy array
        poses_array = poses.select(["X", "Y", "Z", "A", "B", "C"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        poses_array = np.ascontiguousarray(poses_array, dtype=np.float64)

        # Call Rust with NumPy array
        result_array = self._robot.batch_inverse(poses_array, current_joints)

        # Convert back to Polars DataFrame
        return pl.DataFrame(
            {
                "J1": result_array[:, 0],
                "J2": result_array[:, 1],
                "J3": result_array[:, 2],
                "J4": result_array[:, 3],
                "J5": result_array[:, 4],
                "J6": result_array[:, 5],
            }
        )

    def batch_forward(self, joints: pl.DataFrame) -> pl.DataFrame:
        """
        Computes the forward kinematics for multiple sets of joint angles in batch mode.

        :param joints: DataFrame containing joint configurations with columns J1-J6.
        :return: DataFrame containing the computed poses with columns X, Y, Z, A, B, C.
        """
        # Convert Polars DataFrame to NumPy array
        joints_array = joints.select(["J1", "J2", "J3", "J4", "J5", "J6"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        joints_array = np.ascontiguousarray(joints_array, dtype=np.float64)

        # Call Rust with NumPy array
        result_array = self._robot.batch_forward(joints_array)

        # Convert back to Polars DataFrame
        return pl.DataFrame(
            {
                "X": result_array[:, 0],
                "Y": result_array[:, 1],
                "Z": result_array[:, 2],
                "A": result_array[:, 3],
                "B": result_array[:, 4],
                "C": result_array[:, 5],
            }
        )

    def batch_forward_quat(self, joints: pl.DataFrame) -> pl.DataFrame:
        """
        Computes the forward kinematics with quaternion output for multiple sets of joint angles.

        :param joints: DataFrame containing joint configurations with columns J1-J6.
        :return: DataFrame with columns X, Y, Z, W, QX, QY, QZ (quaternion scalar first).
        """
        # Convert Polars DataFrame to NumPy array
        joints_array = joints.select(["J1", "J2", "J3", "J4", "J5", "J6"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        joints_array = np.ascontiguousarray(joints_array, dtype=np.float64)

        # Call Rust with NumPy array
        result_array = self._robot.batch_forward_quat(joints_array)

        # Convert back to Polars DataFrame
        return pl.DataFrame(
            {
                "X": result_array[:, 0],
                "Y": result_array[:, 1],
                "Z": result_array[:, 2],
                "W": result_array[:, 3],
                "QX": result_array[:, 4],
                "QY": result_array[:, 5],
                "QZ": result_array[:, 6],
            }
        )

    def batch_inverse_quat(
        self,
        poses: pl.DataFrame,
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> pl.DataFrame:
        """
        Computes the inverse kinematics with quaternion input for multiple poses.

        :param poses: DataFrame with columns X, Y, Z, W, QX, QY, QZ (quaternion scalar first).
        :param current_joints: Current joint configuration (optional).
        :return: DataFrame containing the computed joint configurations with columns J1-J6.
        """
        # Convert Polars DataFrame to NumPy array
        poses_array = poses.select(["X", "Y", "Z", "W", "QX", "QY", "QZ"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        poses_array = np.ascontiguousarray(poses_array, dtype=np.float64)

        # Call Rust with NumPy array
        result_array = self._robot.batch_inverse_quat(poses_array, current_joints)

        # Convert back to Polars DataFrame
        return pl.DataFrame(
            {
                "J1": result_array[:, 0],
                "J2": result_array[:, 1],
                "J3": result_array[:, 2],
                "J4": result_array[:, 3],
                "J5": result_array[:, 4],
                "J6": result_array[:, 5],
            }
        )

    # ==================== Collision Detection Properties ====================

    @property
    def has_collision_geometry(self) -> bool:
        """Check if the robot has collision geometry configured."""
        return self._robot.has_collision_geometry

    @property
    def has_joint_limits(self) -> bool:
        """Check if the robot has joint limits configured."""
        return self._robot.has_joint_limits

    # ==================== Collision Detection Methods ====================

    def collides(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> bool:
        """
        Check if a joint configuration collides with itself or the environment.

        Requires robot_body to be configured. Returns False if no collision geometry.

        :param joints: Tuple of 6 joint angles (in the EulerConvention's units).
        :return: True if collision detected, False otherwise.
        """
        return self._robot.collides(joints)

    def collision_details(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> List[Tuple[int, int]]:
        """
        Get detailed collision information for a joint configuration.

        Requires robot_body to be configured. Returns empty list if no collision geometry.

        :param joints: Tuple of 6 joint angles (in the EulerConvention's units).
        :return: List of (i, j) tuples where i and j are indices of colliding bodies.
                 Joint indices: 0-5 for J1-J6, 100 for tool, 101 for base, 102+ for environment.
        """
        return self._robot.collision_details(joints)

    def joints_compliant(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> bool:
        """
        Check if a joint configuration is within the configured joint limits.

        Requires joint_limits to be configured. Returns True if no limits configured.

        :param joints: Tuple of 6 joint angles (in the EulerConvention's units).
        :return: True if all joints are within limits, False otherwise.
        """
        return self._robot.joints_compliant(joints)

    def near(
        self,
        joints: Tuple[float, float, float, float, float, float],
        safety: SafetyDistances,
    ) -> List[Tuple[int, int]]:
        """
        Check for objects within a specified safety distance.

        Requires robot_body to be configured. Returns empty list if no collision geometry.

        :param joints: Tuple of 6 joint angles (in the EulerConvention's units).
        :param safety: SafetyDistances configuration to use for distance checking.
        :return: List of (i, j) tuples where i and j are indices of bodies within the safety distance.
        """
        return self._robot.near(joints, safety)

    def batch_collides(self, joints: np.ndarray) -> np.ndarray:
        """
        Batch collision checking using NumPy arrays.

        Requires robot_body to be configured. Returns all False if no collision geometry.
        NaN input rows are treated as non-colliding (returns False).

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :return: Boolean array of shape (n,) where True indicates collision.
        """
        joints_array = np.ascontiguousarray(joints, dtype=np.float64)
        return self._robot.batch_collides(joints_array)

    def batch_joints_compliant(self, joints: np.ndarray) -> np.ndarray:
        """
        Batch joint limits checking using NumPy arrays.

        Requires joint_limits to be configured. Returns all True if no limits configured.
        NaN input rows are treated as non-compliant (returns False).

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :return: Boolean array of shape (n,) where True indicates within limits.
        """
        joints_array = np.ascontiguousarray(joints, dtype=np.float64)
        return self._robot.batch_joints_compliant(joints_array)


__all__ = ["CheckMode", "CollisionBody", "EulerConvention", "JointLimits", "KinematicModel", "Robot", "RobotBody", "SafetyDistances"]
