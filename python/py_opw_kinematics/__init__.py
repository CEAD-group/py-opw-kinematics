from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.spatial.transform import RigidTransform, Rotation

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
        self._euler_convention = euler_convention

    def __repr__(self) -> str:
        return self._robot.__repr__()

    def rigid_transform_to_pose(
        self, rigid_transform: RigidTransform
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Converts a RigidTransform object to the legacy pose format.

        :param rigid_transform: RigidTransform object to convert
        :return: Tuple of (translation, orientation) where both are 3-element tuples
        """
        # Extract translation
        translation = tuple(rigid_transform.translation)

        # Extract rotation and convert to Euler angles using robot's convention
        rotation_matrix = rigid_transform.rotation.as_matrix()
        orientation = self._euler_convention.matrix_to_euler(rotation_matrix.tolist())

        return (translation, tuple(orientation))

    def forward_legacy(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[RigidTransform] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Legacy forward kinematics method that returns pose as tuples.

        :param joints: Joint angles of the robot.
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: Tuple of (translation, orientation) where both are 3-element tuples
        """
        rigid_transform = self.forward(joints, ee_transform)
        return self.rigid_transform_to_pose(rigid_transform)

    def inverse_legacy(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[RigidTransform] = None,
    ) -> list[Tuple[float, float, float, float, float, float]]:
        """
        Legacy inverse kinematics method that accepts pose as tuples.

        :param pose: Desired pose as (translation, orientation) tuples.
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        # Convert legacy pose format to RigidTransform
        translation, orientation = pose

        # Convert orientation (Euler angles) to rotation matrix
        rotation_matrix = self._euler_convention.euler_to_matrix(orientation)
        rotation = Rotation.from_matrix(rotation_matrix)

        # Create RigidTransform
        rigid_transform = RigidTransform.from_components(
            rotation=rotation, translation=np.array(translation)
        )

        return self.inverse(rigid_transform, current_joints, ee_transform)

    def forward(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[RigidTransform] = None,
    ) -> RigidTransform:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: RigidTransform representing the end-effector pose.
        """
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        matrix_4x4 = self._robot.forward(joints, ee_matrix)

        # Convert 4x4 matrix to RigidTransform
        return RigidTransform.from_matrix(matrix_4x4)

    def inverse(
        self,
        pose: RigidTransform,
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[RigidTransform] = None,
    ) -> list[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics for a given pose.

        :param pose: Desired pose as RigidTransform.
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        # Convert RigidTransform to 4x4 matrix
        matrix_4x4 = pose.as_matrix()

        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()
        return self._robot.inverse(matrix_4x4, current_joints, ee_matrix)

    def batch_inverse(
        self,
        poses: List[RigidTransform],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[RigidTransform] = None,
    ) -> pl.DataFrame:
        """
        Computes the inverse kinematics for multiple poses in batch mode.

        :param poses: List of RigidTransform objects representing desired poses.
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: DataFrame containing the computed joint configurations with columns J1-J6.
        """
        # Convert RigidTransforms to 4x4 matrices
        n_poses = len(poses)
        matrix_array = np.zeros((n_poses, 16), dtype=np.float64)

        for i, pose in enumerate(poses):
            matrix_4x4 = pose.as_matrix()
            matrix_array[i] = matrix_4x4.flatten()

        # Convert RigidTransform to matrix for Rust interface
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()

        # Call Rust with matrix array
        result_array = self._robot.batch_inverse(
            matrix_array, current_joints, ee_matrix
        )

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
    ) -> List[RigidTransform]:
        """
        Computes the forward kinematics for multiple joint configurations in batch mode.

        :param joints: DataFrame containing joint configurations with columns J1-J6.
        :param ee_transform: End effector transformation as RigidTransform (optional).
        :return: List of RigidTransform objects representing the computed poses.
        """
        # Convert Polars DataFrame to NumPy array
        joints_array = joints.select(["J1", "J2", "J3", "J4", "J5", "J6"]).to_numpy()

        # Ensure float64 dtype and C-contiguous layout
        joints_array = np.ascontiguousarray(joints_array, dtype=np.float64)

        # Convert RigidTransform to matrix for Rust interface
        ee_matrix = None if ee_transform is None else ee_transform.as_matrix()

        # Call Rust with NumPy array
        result_array = self._robot.batch_forward(joints_array, ee_matrix)

        # Convert 4x4 matrices back to RigidTransform objects
        poses = []
        for i in range(result_array.shape[0]):
            # Reshape flattened matrix back to 4x4
            matrix_4x4 = result_array[i].reshape(4, 4)
            rigid_transform = RigidTransform.from_matrix(matrix_4x4)
            poses.append(rigid_transform)

        return poses

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
