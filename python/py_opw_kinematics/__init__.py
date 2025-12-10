from typing import Optional, Tuple

import numpy as np
import polars as pl

from ._internal import EulerConvention, KinematicModel
from ._internal import Robot as _RobotInternal


class Robot:
    """
    Robot kinematics wrapper that provides a Polars DataFrame interface
    for batch operations while using NumPy arrays internally for performance.
    """

    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._robot = _RobotInternal(
            kinematic_model, euler_convention, ee_rotation, ee_translation
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

    def forward(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position and orientation of the end-effector.
        """
        return self._robot.forward(joints)

    def inverse(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> list[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics for a given pose.

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        return self._robot.inverse(pose, current_joints)

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


__all__ = ["EulerConvention", "KinematicModel", "Robot"]
