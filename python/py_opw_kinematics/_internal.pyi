from typing import List, Tuple, Literal, Optional
import numpy as np
import numpy.typing as npt

EulerSequence = Literal[
    "XYX", "XYZ", "XZX", "XZY", "YXY", "YXZ", "YZX", "YZY", "ZXY", "ZXZ", "ZYX", "ZYZ"
]

class EulerConvention:
    sequence: EulerSequence
    extrinsic: bool
    degrees: bool

    def __init__(self, sequence: EulerSequence, extrinsic: bool, degrees: bool) -> None:
        """
        Initializes an EulerConvention instance.

        :param sequence: The Euler sequence (e.g., 'XYZ', 'ZYX').
        :param extrinsic: Whether the rotation is extrinsic.
        :param degrees: Whether angles are in degrees or radians.
        """
        ...

    def convert(
        self, other: "EulerConvention", angles: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Converts angles to another Euler convention.

        :param other: Target Euler convention to convert to.
        :param angles: Angles in the current convention.
        :return: Angles in the target convention.
        """
        ...

    def matrix_to_euler(self, rot: List[List[float]]) -> Tuple[float, float, float]:
        """
        Converts a rotation matrix to Euler angles based on the current convention.

        :param rot: 3x3 rotation matrix.
        :return: Euler angles in the current convention.
        """
        ...

    def euler_to_matrix(self, angles: Tuple[float, float, float]) -> List[List[float]]:
        """
        Converts Euler angles to a rotation matrix.

        :param angles: Euler angles in the current convention.
        :return: Corresponding 3x3 rotation matrix.
        """
        ...

    def matrix_to_quaternion(
        self, rot: List[List[float]]
    ) -> Tuple[float, float, float, float]:
        """
        Converts a rotation matrix to a quaternion.

        :param rot: 3x3 rotation matrix.
        :return: Corresponding quaternion in the order (w, i, j, k).
        """
        ...

    def quaternion_to_euler(
        self, quat: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Converts a quaternion to Euler angles based on the current convention.

        :param quat: Quaternion in the order (w, i, j, k).
        :return: Euler angles in the current convention.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the EulerConvention instance.
        """
        ...

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the EulerConvention instance.
        """
        ...

class KinematicModel:
    a1: float
    a2: float
    b: float
    c1: float
    c2: float
    c3: float
    c4: float
    offsets: Tuple[float, float, float, float, float, float]
    flip_axes: Optional[Tuple[bool, bool, bool, bool, bool, bool]]
    has_parallelogram: bool

    def __init__(
        self,
        a1: float = 0,
        a2: float = 0,
        b: float = 0,
        c1: float = 0,
        c2: float = 0,
        c3: float = 0,
        c4: float = 0,
        offsets: Tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        flip_axes: Optional[Tuple[bool, bool, bool, bool, bool, bool]] = (
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        has_parallelogram: bool = False,
    ) -> None:
        """
        Initializes a KinematicModel instance.

        :param a1, a2, b, c1, c2, c3, c4: Kinematic parameters.
        :param offsets: Joint offsets.
        :param flip_axes: Boolean flags for flipping axes.
        :param has_parallelogram: Indicates if the model has a parallelogram linkage.
        """
        ...

class Robot:
    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
    ) -> None:
        """
        Initializes a Robot instance.

        :param kinematic_model: The kinematic model of the robot.
        :param euler_convention: Euler convention used for end-effector rotation.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the Robot instance.
        """
        ...

    def forward(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: A tuple containing the position and orientation of the end-effector.
        """
        ...

    def inverse(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics for a given pose.

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        ...

    def batch_inverse(
        self,
        poses: npt.NDArray[np.float64],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the inverse kinematics for multiple poses in batch mode.

        :param poses: NumPy array of shape (n, 6) with columns [X, Y, Z, A, B, C].
        :param current_joints: Current joint configuration (optional).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        """
        ...

    def batch_forward(
        self,
        joints: npt.NDArray[np.float64],
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the forward kinematics for multiple sets of joint angles in batch mode.

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n, 6) with columns [X, Y, Z, A, B, C].
        """
        ...

    def forward_frames(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[npt.NDArray[np.float64]]:
        """
        Computes the forward kinematics for all joint frames.

        :param joints: Joint angles of the robot.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: A list of 4x4 transformation matrices for each joint frame.
        """
        ...

__all__: List[str] = ["EulerConvention", "KinematicModel", "Robot"]
