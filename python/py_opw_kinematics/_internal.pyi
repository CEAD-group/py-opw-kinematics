from typing import Dict, List, Tuple, Literal, Optional
from enum import IntEnum
import numpy as np
import numpy.typing as npt

EulerSequence = Literal[
    "XYX", "XYZ", "XZX", "XZY", "YXY", "YXZ", "YZX", "YZY", "ZXY", "ZXZ", "ZYX", "ZYZ"
]

class CheckMode(IntEnum):
    """Collision checking mode."""

    FirstCollisionOnly: int
    """Stop after finding the first collision (faster)."""
    AllCollisions: int
    """Find all collisions (slower but complete)."""
    NoCheck: int
    """Disable collision checking entirely."""


class SafetyDistances:
    """
    Safety distances configuration for collision detection.

    Defines tolerance bounds between robot parts and environmental objects.
    """

    # Class constants for joint indices
    J1: int
    J2: int
    J3: int
    J4: int
    J5: int
    J6: int
    J_TOOL: int
    J_BASE: int
    NEVER_COLLIDES: float
    TOUCH_ONLY: float

    to_environment: float
    to_robot_default: float
    special_distances: Dict[Tuple[int, int], float]
    mode: CheckMode

    def __init__(
        self,
        to_environment: float = 0.0,
        to_robot_default: float = 0.0,
        special_distances: Optional[Dict[Tuple[int, int], float]] = None,
        mode: CheckMode = CheckMode.FirstCollisionOnly,
    ) -> None:
        """
        Create a new SafetyDistances configuration.

        :param to_environment: Minimum distance to environment objects. Default: 0.0 (touch only)
        :param to_robot_default: Default minimum distance between robot parts. Default: 0.0 (touch only)
        :param special_distances: Optional dict mapping (joint_id, joint_id) pairs to specific distances.
                                  Use the J1-J6, J_TOOL, J_BASE constants for joint indices.
                                  Use NEVER_COLLIDES (-1.0) to skip checking between specific parts.
        :param mode: Collision checking mode. Default: FirstCollisionOnly
        """
        ...

    def min_distance(self, id1: int, id2: int) -> float:
        """
        Get the minimum allowed distance between two objects.

        Returns the special distance if defined, otherwise the default robot distance.
        Order of indices doesn't matter (symmetric lookup).
        """
        ...

    def set_special_distance(self, id1: int, id2: int, distance: float) -> None:
        """Set a special distance for a specific joint pair."""
        ...

    def __repr__(self) -> str: ...


class CollisionBody:
    """
    A collision body representing an environment obstacle or robot part.

    Loaded from STL/PLY/OBJ mesh files with optional position and orientation.
    """

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    scale: float
    num_triangles: int

    def __init__(
        self,
        mesh_path: str,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float, float]] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Create a CollisionBody from a mesh file.

        :param mesh_path: Path to mesh file (STL, PLY, or OBJ format)
        :param position: Position (x, y, z) in meters. Default: (0, 0, 0)
        :param orientation: Orientation as quaternion (w, x, y, z) - scalar first. Default: (1, 0, 0, 0)
        :param scale: Scale factor to apply to the mesh. Default: 1.0
        """
        ...

    def __repr__(self) -> str: ...


class RobotBody:
    """
    Robot body configuration for collision detection.

    Represents the geometry of a 6-axis robot including joint meshes,
    optional tool and base, environment obstacles, and safety distances.
    """

    has_tool: bool
    has_base: bool
    num_environment_bodies: int
    safety: SafetyDistances

    def __init__(
        self,
        joint_meshes: Tuple[str, str, str, str, str, str],
        scale: float = 1.0,
        safety: Optional[SafetyDistances] = None,
    ) -> None:
        """
        Create a RobotBody from individual mesh files for each joint.

        :param joint_meshes: Tuple of 6 paths to mesh files (STL/PLY/OBJ), one per joint
        :param scale: Scale factor to apply to all meshes. Default: 1.0
        :param safety: Safety distances configuration. Default: touch-only
        """
        ...

    def with_tool(self, mesh_path: str, scale: float = 1.0) -> "RobotBody":
        """
        Add a tool mesh to the robot.

        :param mesh_path: Path to the tool mesh file
        :param scale: Scale factor for the mesh. Default: 1.0
        :return: Self for method chaining
        """
        ...

    def with_base(
        self,
        mesh_path: str,
        position: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float, float]] = None,
        scale: float = 1.0,
    ) -> "RobotBody":
        """
        Add a base mesh to the robot.

        :param mesh_path: Path to the base mesh file
        :param position: Position (x, y, z) of the base. Default: (0, 0, 0)
        :param orientation: Orientation as quaternion (w, x, y, z). Default: (1, 0, 0, 0)
        :param scale: Scale factor for the mesh. Default: 1.0
        :return: Self for method chaining
        """
        ...

    def add_environment(self, body: CollisionBody) -> "RobotBody":
        """
        Add an environment collision body.

        :param body: CollisionBody to add to the environment
        :return: Self for method chaining
        """
        ...

    def with_safety(self, safety: SafetyDistances) -> "RobotBody":
        """
        Set the safety distances configuration.

        :param safety: SafetyDistances configuration
        :return: Self for method chaining
        """
        ...

    def __repr__(self) -> str: ...


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
    ee_rotation: Tuple[float, float, float]
    ee_rotation_quat: Tuple[float, float, float, float]
    ee_translation: Tuple[float, float, float]
    has_collision_geometry: bool
    has_joint_limits: bool

    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        robot_body: Optional[RobotBody] = None,
        joint_limits: Optional[JointLimits] = None,
    ) -> None:
        """
        Initializes a Robot instance.

        :param kinematic_model: The kinematic model of the robot.
        :param euler_convention: Euler convention used for end-effector rotation.
        :param ee_rotation: Initial rotation of the end-effector as Euler angles.
        :param ee_translation: Initial translation of the end-effector.
        :param robot_body: Optional collision geometry configuration.
        :param joint_limits: Optional joint limits configuration.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the Robot instance.
        """
        ...

    def forward(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position (x, y, z) and orientation as Euler angles.
        """
        ...

    def forward_quat(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Computes the forward kinematics with quaternion output.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position (x, y, z) and quaternion (w, x, y, z).
        """
        ...

    def forward_matrix(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], List[List[float]]]:
        """
        Computes the forward kinematics with rotation matrix output.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position (x, y, z) and 3x3 rotation matrix.
        """
        ...

    def inverse(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics for a given pose.

        :param pose: Desired pose (position and Euler angles) of the end-effector.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        ...

    def inverse_quat(
        self,
        position: Tuple[float, float, float],
        quaternion: Tuple[float, float, float, float],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics with quaternion input.

        :param position: Desired position (x, y, z) of the end-effector.
        :param quaternion: Desired orientation as quaternion (w, x, y, z) - scalar first.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        ...

    def inverse_matrix(
        self,
        position: Tuple[float, float, float],
        rotation_matrix: List[List[float]],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Computes the inverse kinematics with rotation matrix input.

        :param position: Desired position (x, y, z) of the end-effector.
        :param rotation_matrix: Desired orientation as 3x3 rotation matrix.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        ...

    def batch_inverse(
        self,
        poses: npt.NDArray[np.float64],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the inverse kinematics for multiple poses in batch mode.

        :param poses: NumPy array of shape (n, 6) with columns [X, Y, Z, A, B, C].
        :param current_joints: Current joint configuration (optional).
        :return: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        """
        ...

    def batch_forward(
        self, joints: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes the forward kinematics for multiple sets of joint angles in batch mode.

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :return: NumPy array of shape (n, 6) with columns [X, Y, Z, A, B, C].
        """
        ...

    def batch_forward_quat(
        self, joints: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes the forward kinematics with quaternion output in batch mode.

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :return: NumPy array of shape (n, 7) with columns [X, Y, Z, W, QX, QY, QZ].
        """
        ...

    def batch_inverse_quat(
        self,
        poses: npt.NDArray[np.float64],
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Computes the inverse kinematics with quaternion input in batch mode.

        :param poses: NumPy array of shape (n, 7) with columns [X, Y, Z, W, QX, QY, QZ].
        :param current_joints: Current joint configuration (optional).
        :return: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        """
        ...

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
        ...

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
        ...

    def joints_compliant(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> bool:
        """
        Check if a joint configuration is within the configured joint limits.

        Requires joint_limits to be configured. Returns True if no limits configured.

        :param joints: Tuple of 6 joint angles (in the EulerConvention's units).
        :return: True if all joints are within limits, False otherwise.
        """
        ...

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
        ...

    def batch_collides(
        self, joints: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Batch collision checking using NumPy arrays.

        Requires robot_body to be configured. Returns all False if no collision geometry.
        NaN input rows are treated as non-colliding (returns False).

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :return: Boolean array of shape (n,) where True indicates collision.
        """
        ...

    def batch_joints_compliant(
        self, joints: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Batch joint limits checking using NumPy arrays.

        Requires joint_limits to be configured. Returns all True if no limits configured.
        NaN input rows are treated as non-compliant (returns False).

        :param joints: NumPy array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6].
        :return: Boolean array of shape (n,) where True indicates within limits.
        """
        ...


class JointLimits:
    """
    Joint limits wrapper around rs-opw-kinematics Constraints.
    Supports wrap-around ranges for continuous joints.
    """

    from_limits: Tuple[float, float, float, float, float, float]
    to_limits: Tuple[float, float, float, float, float, float]
    centers: Tuple[float, float, float, float, float, float]
    sorting_weight: float

    def __init__(
        self,
        limits: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
        ],
        sorting_weight: float = 0.0,
        degrees: bool = True,
    ) -> None:
        """
        Create joint limits from (min, max) pairs for each of the 6 joints.

        :param limits: Tuple of 6 (min, max) pairs defining the range for each joint.
                       Wrap-around is supported: if min > max, the range wraps through 0.
        :param sorting_weight: Weight for sorting IK solutions (0.0 = prefer previous joints,
                               1.0 = prefer center of constraints). Default: 0.0
        :param degrees: Whether the limits are specified in degrees. Default: True
        """
        ...

    def compliant(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> bool:
        """
        Check if the given joint configuration satisfies all joint limits.

        :param joints: Tuple of 6 joint angles.
        :return: True if all joints are within their limits, False otherwise.
        """
        ...

    def filter(
        self,
        solutions: List[Tuple[float, float, float, float, float, float]],
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Filter a list of joint configurations, keeping only those within limits.

        :param solutions: List of joint configurations (each a tuple of 6 angles).
        :return: List of joint configurations that satisfy all limits.
        """
        ...

    def __repr__(self) -> str: ...


__all__: List[str] = ["CheckMode", "CollisionBody", "EulerConvention", "JointLimits", "KinematicModel", "Robot", "RobotBody", "SafetyDistances"]
