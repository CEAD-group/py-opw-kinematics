from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt

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

    def __init__(
        self,
        a1: float = 0,
        a2: float = 0,
        b: float = 0,
        c1: float = 0,
        c2: float = 0,
        c3: float = 0,
        c4: float = 0,
        offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes: Optional[Tuple[bool, bool, bool, bool, bool, bool]] = (False, False, False, False, False, False),
    ) -> None:
        """
        Initialize a KinematicModel instance.

        :param a1, a2, b, c1, c2, c3, c4: Kinematic parameters.
        :param offsets: Joint offsets.
        :param flip_axes: Boolean flags for flipping axes.
        """
        ...

class Robot:
    def __init__(
        self,
        kinematic_model: KinematicModel,
        degrees: bool = True,
    ) -> None:
        """
        Initialize a Robot instance.

        :param kinematic_model: The kinematic model of the robot.
        :param degrees: Whether joint angles are in degrees (True) or radians (False).
        """
        ...

    def __repr__(self) -> str: ...

    def forward(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute forward kinematics for given joint angles.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: 4x4 transformation matrix.
        """
        ...

    def inverse(
        self,
        pose: npt.NDArray[np.float64],
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Compute inverse kinematics for a given pose.

        :param pose: 4x4 transformation matrix.
        :param current_joints: Current joint configuration for solution ranking.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: List of possible joint configurations.
        """
        ...

    def batch_inverse(
        self,
        poses: npt.NDArray[np.float64],
        current_joints: Optional[Tuple[float, float, float, float, float, float]] = None,
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute inverse kinematics for multiple poses.

        :param poses: NumPy array of shape (n, 16) with flattened 4x4 matrices.
        :param current_joints: Starting joint configuration for solution continuity.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n, 6) with joint angles.
        """
        ...

    def batch_forward(
        self,
        joints: npt.NDArray[np.float64],
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute forward kinematics for multiple joint configurations.

        :param joints: NumPy array of shape (n, 6) with joint angles.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n, 16) with flattened 4x4 matrices.
        """
        ...

    def forward_frames(
        self,
        joints: Tuple[float, float, float, float, float, float],
        ee_transform: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[npt.NDArray[np.float64]]:
        """
        Compute 4x4 transform matrices for all robot links.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: List of 4x4 transformation matrices for [Base, J1, J2, J3, J4, J5, J6, TCP].
        """
        ...

__all__: List[str] = ["KinematicModel", "Robot"]
