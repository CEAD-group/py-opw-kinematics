
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
    offsets: tuple[float, float, float, float, float, float]
    flip_axes: tuple[bool, bool, bool, bool, bool, bool] | None

    def __init__(
        self,
        a1: float = 0,
        a2: float = 0,
        b: float = 0,
        c1: float = 0,
        c2: float = 0,
        c3: float = 0,
        c4: float = 0,
        offsets: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes: tuple[bool, bool, bool, bool, bool, bool] | None = (False, False, False, False, False, False),
    ) -> None:
        """
        Initialize a KinematicModel instance.

        :param a1, a2, b, c1, c2, c3, c4: Kinematic parameters.
        :param offsets: Joint offsets.
        :param flip_axes: Boolean flags for flipping axes.
        """

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


    def forward(
        self,
        joints: tuple[float, float, float, float, float, float],
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute forward kinematics for given joint angles.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: 4x4 transformation matrix.
        """

    def inverse(
        self,
        pose: npt.NDArray[np.float64],
        current_joints: tuple[float, float, float, float, float, float] | None = None,
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """
        Compute inverse kinematics for a given pose.

        :param pose: 4x4 transformation matrix.
        :param current_joints: Current joint configuration for solution ranking.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: List of possible joint configurations.
        """

    def batch_inverse(
        self,
        poses: npt.NDArray[np.float64],
        current_joints: tuple[float, float, float, float, float, float] | None = None,
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute inverse kinematics for multiple poses.

        :param poses: NumPy array of shape (n, 16) with flattened 4x4 matrices.
        :param current_joints: Starting joint configuration for solution continuity.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n, 6) with joint angles.
        """

    def reach(
        self,
        poses: npt.NDArray[np.float64],
        joint_limits: list[tuple[float, float]] | None = None,
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """
        Compute all eight inverse-kinematics branches for multiple poses.

        :param poses: NumPy array of shape (n, 16) with flattened 4x4 matrices.
        :param joint_limits: Six (lower, upper) pairs in the robot's angle unit (optional).
            With limits, sigma_min is NaN for branches outside them.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: Tuple of (joints (n, 8, 6), limit_margin (n, 8), extension (n,),
            sigma_min (n, 8), wrist (n, 8)).
        """

    def batch_forward(
        self,
        joints: npt.NDArray[np.float64],
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute forward kinematics for multiple joint configurations.

        :param joints: NumPy array of shape (n, 6) with joint angles.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n, 16) with flattened 4x4 matrices.
        """

    def joint_poses(
        self,
        joints: tuple[float, float, float, float, float, float],
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> list[npt.NDArray[np.float64]]:
        """
        Compute per-joint poses using the OPW FK chain (consistent with forward()).

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: List of 4x4 transformation matrices for [J1, J2, J3, J4, J5, J6/TCP].
            When ee_transform is given, a 7th TCP+EE pose is appended.
        """

    def batch_joint_poses(
        self,
        joints: npt.NDArray[np.float64],
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute per-joint poses for multiple joint configurations.

        :param joints: NumPy array of shape (n, 6) with joint angles.
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: NumPy array of shape (n*6, 16) or (n*7, 16) with flattened 4x4 matrices.
        """

    def forward_frames(
        self,
        joints: tuple[float, float, float, float, float, float],
        ee_transform: npt.NDArray[np.float64] | None = None,
    ) -> list[npt.NDArray[np.float64]]:
        """
        Compute 4x4 transform matrices for all robot links.

        :param joints: Joint angles (J1-J6).
        :param ee_transform: End effector transformation matrix (4x4) (optional).
        :return: List of 4x4 transformation matrices for [Base, J1, J2, J3, J4, J5, J6, TCP].
        """

__all__: list[str] = ["KinematicModel", "Robot"]
