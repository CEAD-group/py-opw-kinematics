from typing import Optional, Tuple

import numpy as np
import polars as pl
from scipy.spatial.transform import RigidTransform

from ._internal import EulerConvention, KinematicModel
from ._internal import Robot as _RobotInternal


class Robot:
    """
    Robot kinematics wrapper that provides a Polars DataFrame interface
    for batch operations with RigidTransform support for end effector configuration.
    """

    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
    ) -> None:
        self._robot = _RobotInternal(kinematic_model, euler_convention)

    def __repr__(self) -> str:
        return self._robot.__repr__()

    def forward(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[RigidTransform] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: A tuple containing the position and orientation of the end-effector.
        """
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        return self._robot.forward(joints, ee_matrix)

    def inverse(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[RigidTransform] = None,
    ) -> list[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics for a given pose.

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        return self._robot.inverse(pose, current_joints, ee_matrix)

    def batch_inverse(
        self,
        poses: pl.DataFrame,
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[RigidTransform] = None,
    ) -> pl.DataFrame:
        """
        Computes the inverse kinematics for multiple poses in batch mode.

        :param poses: DataFrame containing desired poses with columns X, Y, Z, A, B, C.
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: DataFrame containing the computed joint configurations with columns J1-J6.
        """
        # Convert Polars DataFrame to NumPy array
        poses_array = poses.select(["X", "Y", "Z", "A", "B", "C"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        poses_array = np.ascontiguousarray(poses_array, dtype=np.float64)

        # Convert RigidTransform to matrix for Rust interface
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()

        # Call Rust with NumPy array
        result_array = self._robot.batch_inverse(poses_array, current_joints, ee_matrix)

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

    def batch_forward(
        self,
        joints: pl.DataFrame,
        ee_transform: Optional[RigidTransform] = None,
    ) -> pl.DataFrame:
        """
        Computes the forward kinematics for multiple sets of joint angles in batch mode.

        :param joints: DataFrame containing joint configurations with columns J1-J6.
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: DataFrame containing the computed poses with columns X, Y, Z, A, B, C.
        """
        # Convert Polars DataFrame to NumPy array
        joints_array = joints.select(["J1", "J2", "J3", "J4", "J5", "J6"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        joints_array = np.ascontiguousarray(joints_array, dtype=np.float64)

        # Convert RigidTransform to matrix for Rust interface
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()

        # Call Rust with NumPy array
        result_array = self._robot.batch_forward(joints_array, ee_matrix)

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

    def forward_frames(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[RigidTransform] = None,
    ) -> pl.DataFrame:
        """
        Compute 4x4 transform matrices for all robot links.

        :param joints: Joint angles of the robot
        :param ee_transform: End effector transformation as RigidTransform (optional)
        :return: DataFrame with columns 'link' and 'transform'
                 where 'transform' contains RigidTransform objects
        """
        # Convert RigidTransform to matrix for Rust interface
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()

        # Get raw matrices from Rust (Vec<[[f64; 4]; 4]>)
        raw_frames = self._robot.forward_frames(joints, ee_matrix)

        # Link names in order
        link_names = [
            "Base",
            "J1",
            "J2",
            "J3",
            "J4",
            "J5",
            "J6",
            "TCP",
        ]

        # Convert all matrices to numpy array at once
        matrices_array = np.array(raw_frames)  # Shape: (8, 4, 4)

        # Create RigidTransforms efficiently
        transforms = [
            RigidTransform.from_matrix(matrices_array[i])
            for i in range(len(link_names))
        ]

        return pl.DataFrame({"link": link_names, "transform": transforms})


__all__ = ["EulerConvention", "KinematicModel", "Robot"]
