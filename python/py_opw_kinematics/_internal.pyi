
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
        *,
        threads: int | None = None,
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
        :param threads: Keyword-only worker cap: None for the global pool, 1 for the
            calling thread, n >= 2 for a cached pool of n. Below 1 raises ValueError,
            except 0, a deprecated alias for None.
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

    def batch_frame_points(
        self,
        joints: npt.NDArray[np.float64],
        frames: list[int],
        points: npt.NDArray[np.float64],
        ee_transform: npt.NDArray[np.float64] | None = None,
        *,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        World positions of fixed points on selected frames, for many joint rows.

        :param joints: NumPy array of shape (n, 6) with joint angles.
        :param frames: K frame indices: 0..5 for J1..J6, 6 for J6 with ee_transform.
        :param points: NumPy array of shape (K, 3), each point in its frame's local coordinates.
        :param ee_transform: End effector transformation matrix (4x4), required for frame 6.
        :param threads: Keyword-only worker cap, same convention as reach.
        :return: NumPy array of shape (n, K, 3) with positions in the base frame.
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
