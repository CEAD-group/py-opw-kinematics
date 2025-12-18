from typing import List, Tuple, Literal, Optional
import polars as pl

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
    has_constraints: bool
    axis_limits: Optional[Tuple[Tuple[float, float], ...]]
    relative_constraints: Optional[List[Tuple[int, int, float, float]]]
    """
    Current relative constraints as (axis, reference_axis, min_offset, max_offset) tuples.
    Values are always returned in radians regardless of how they were set.
    """
    sum_constraints: Optional[List[Tuple[int, int, float, float]]]
    """
    Current sum constraints as (axis, reference_axis, min_sum, max_sum) tuples.
    Values are always returned in radians regardless of how they were set.
    """

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
        axis_limits: Optional[List[Tuple[float, float]]] = None,
        relative_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
        sum_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
    ) -> None:
        """
        Initializes a KinematicModel instance.

        :param a1, a2, b, c1, c2, c3, c4: Kinematic parameters.
        :param offsets: Joint offsets.
        :param flip_axes: Boolean flags for flipping axes.
        :param has_parallelogram: Indicates if the model has a parallelogram linkage.
        :param axis_limits: Optional list of (min, max) limits for each axis.
        :param relative_constraints: Optional list of relative constraints as (axis, reference_axis, min_offset, max_offset) tuples.
                                   Constraint offsets are specified in degrees and converted to radians internally.
        :param sum_constraints: Optional list of sum constraints as (axis, reference_axis, min_sum, max_sum) tuples.
                               For parallelogram constraints like J2+J3. Values specified in degrees and converted to radians internally.
        """
        ...

    def set_axis_limits(self, limits: Optional[List[Tuple[float, float]]]) -> None:
        """
        Sets the axis limits for joint angles.

        :param limits: Optional list of (min, max) limits for each axis.
        """
        ...

    def set_absolute_constraint(
        self, axis: int, min: float, max: float, degrees: bool = False
    ) -> None:
        """
        Sets an absolute constraint for a specific axis.

        :param axis: The axis index (0-5).
        :param min: Minimum allowed value (in radians by default, degrees if degrees=True).
        :param max: Maximum allowed value (in radians by default, degrees if degrees=True).
        :param degrees: If True, min and max are in degrees; if False (default), in radians.
        """
        ...

    def set_relative_constraint(
        self,
        axis: int,
        reference_axis: int,
        min_offset: float,
        max_offset: float,
        degrees: bool = False,
    ) -> None:
        """
        Sets a relative constraint for a specific axis.

        :param axis: The axis index to constrain (0-5).
        :param reference_axis: The reference axis index (0-5).
        :param min_offset: Minimum offset from reference axis (in radians by default, degrees if degrees=True).
        :param max_offset: Maximum offset from reference axis (in radians by default, degrees if degrees=True).
        :param degrees: If True, offsets are in degrees; if False (default), in radians.
        """
        ...

    def set_sum_constraint(
        self,
        axis: int,
        reference_axis: int,
        min_sum: float,
        max_sum: float,
        degrees: bool = False,
    ) -> None:
        """
        Sets a sum constraint for a specific axis (for parallelogram constraints).

        :param axis: The axis index to constrain (0-5).
        :param reference_axis: The reference axis index (0-5).
        :param min_sum: Minimum sum of axis + reference_axis (in radians by default, degrees if degrees=True).
        :param max_sum: Maximum sum of axis + reference_axis (in radians by default, degrees if degrees=True).
        :param degrees: If True, values are in degrees; if False (default), in radians.
        """
        ...

    def clear_axis_constraint(self, axis: int) -> None:
        """
        Clears the constraint for a specific axis.

        :param axis: The axis index (0-5).
        """
        ...

    def clear_all_constraints(self) -> None:
        """
        Clears all advanced constraints.
        """
        ...

    def joints_within_limits_vec(
        self, joints: List[float], degrees: Optional[bool] = None
    ) -> bool:
        """
        Check if given joints satisfy all constraints (Python-friendly).

        :param joints: List of 6 joint values.
        :param degrees: Whether joint values are in degrees (default: True).
        :return: True if all constraints are satisfied.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the KinematicModel instance.
        """
        ...

class Robot:
    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Initializes a Robot instance.

        :param kinematic_model: The kinematic model of the robot.
        :param euler_convention: Euler convention used for end-effector rotation.
        :param ee_rotation: Initial rotation of the end-effector.
        :param ee_translation: Initial translation of the end-effector.
        """
        ...

    @property
    def ee_rotation(self) -> Tuple[float, float, float]:
        """
        Gets the current end-effector rotation.

        :return: End-effector rotation as (rx, ry, rz).
        """
        ...

    @property
    def ee_translation(self) -> Tuple[float, float, float]:
        """
        Gets the current end-effector translation.

        :return: End-effector translation as (x, y, z).
        """
        ...

    @property
    def kinematic_model(self) -> KinematicModel:
        """
        Gets the kinematic model used by this robot.

        :return: The kinematic model instance.
        """
        ...

    def forward(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Computes the forward kinematics for the given joint angles.

        :param joints: Joint angles of the robot.
        :return: A tuple containing the position and orientation of the end-effector.
        """
        ...

    def joint_positions(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """
        Computes the 3D positions of all joints using forward kinematics.

        :param joints: Joint angles of the robot.
        :return: List of joint positions from base to end-effector: [base, J1, J2, J3, J4, J5, J6, TCP].
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

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param current_joints: Current joint configuration (optional).
        :return: A list of possible joint configurations that achieve the desired pose.
        """
        ...

    def batch_inverse(
        self,
        poses: pl.DataFrame,
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> pl.DataFrame:
        """
        Computes the inverse kinematics for multiple poses in batch mode.

        :param poses: DataFrame containing desired poses.
        :param current_joints: Current joint configuration (optional).
        :return: DataFrame containing the computed joint configurations.
        """
        ...

    def batch_forward(self, joints: pl.DataFrame) -> pl.DataFrame:
        """
        Computes the forward kinematics for multiple sets of joint angles in batch mode.

        :param joints: DataFrame containing joint configurations.
        :return: DataFrame containing the computed poses.
        """
        ...

    def inverse_with_config(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        target_config: Optional[str] = None,
    ) -> Tuple[
        List[Tuple[float, float, float, float, float, float]],
        List[str],
        Optional[Tuple[Tuple[float, float, float, float, float, float], str, int]],
    ]:
        """
        Computes inverse kinematics with configuration analysis and optional target matching.

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param target_config: Optional target configuration string (e.g., "J3+ J5- OH+").
        :return: Tuple of (joint_solutions, config_strings, best_match).
                 best_match is (joints, config, score) if target_config provided, else None.
        """
        ...

    def inverse_with_target_config(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        target_config: str,
        current_joints: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
    ) -> Optional[Tuple[Tuple[float, float, float, float, float, float], str, int]]:
        """
        Finds the best inverse kinematics solution matching a specific configuration.

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param target_config: Target configuration string (e.g., "J3+ J5- OH+").
        :param current_joints: Current joint configuration (optional, not yet used).
        :return: Tuple of (joints, config_string, score) if match found, else None.
                 score indicates how many criteria matched (0-3).
        """
        ...

    def analyze_configuration(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> str:
        """
        Analyzes the configuration of given joint angles.

        :param joints: Joint angles to analyze.
        :return: Configuration string (e.g., "J3+ J5- OH+").
                 J3+/J3- = elbow up/down, J5+/J5- = wrist normal/flipped,
                 OH+/OH- = overhead/non-overhead.
        """
        ...

    def compare_configurations(
        self, joint_solutions: List[Tuple[float, float, float, float, float, float]]
    ) -> List[str]:
        """
        Compares multiple joint solutions and returns their configuration strings.

        :param joint_solutions: List of joint angle arrays to analyze.
        :return: List of configuration strings corresponding to each solution.
        """
        ...

    def parallelogram_positions(
        self,
        joints: Tuple[float, float, float, float, float, float],
        link_length: float,
        rest_angle: float,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Calculate parallelogram P1 and P2 positions using actual robot geometry.

        :param joints: Joint angles of the robot.
        :return: Tuple of (P1_position, P2_position) if robot has parallelogram, None otherwise.
        """
        ...

    def analyze_configuration_full(
        self,
        joints: Tuple[float, float, float, float, float, float],
        include_turns: bool = False,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Analyze configuration with full STAT/TU information.

        :param joints: Joint angles to analyze.
        :param include_turns: Whether to include turn information in full string.
        :return: Tuple of (stat_tu_string, stat_binary, full_string).
        """
        ...

    def find_stat_matches(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        stat_bits: int,
    ) -> List[Tuple[Tuple[float, float, float, float, float, float], str, int]]:
        """
        Find solutions matching STAT bits (ignoring turn numbers).

        :param pose: Desired pose (position and orientation) of the end-effector.
        :param stat_bits: Target STAT bits (0-7).
        :return: List of (joints, config_string, score) tuples.
        """
        ...

    def create_stat_tu_target(self, stat_bits: int, tu_bits: int) -> str:
        """
        Create target configuration from STAT/TU bits.

        :param stat_bits: STAT bits (0-7).
        :param tu_bits: TU bits (0-63).
        :return: Configuration string like "STAT=101 TU=000011".
        """
        ...

    def get_configuration_details(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[str, int, str, int, str]:
        """
        Get detailed configuration analysis for a solution.

        :param joints: Joint angles to analyze.
        :return: Tuple of (stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary).
        """
        ...

    def get_configuration_details_geometric(
        self,
        joints: Tuple[float, float, float, float, float, float],
        robot_params: "RobotKinematicParams",
    ) -> Tuple[str, int, str, int, str]:
        """
        Get configuration analysis using geometric calculation.
        Uses robot-specific kinematic parameters for accurate shoulder classification.

        :param joints: Joint angles to analyze.
        :param robot_params: Robot kinematic parameters for geometric calculation.
        :return: Tuple of (stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary).
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the Robot instance.
        """
        ...

class RobotKinematicParams:
    """
    Robot kinematic parameters for geometric overhead calculation in STAT bits.
    Used for geometric calculation in configuration analysis.
    """

    def __init__(
        self, a1: float, a2: float, b: float, c1: float, c2: float, c3: float, c4: float
    ) -> None:
        """
        Initialize robot kinematic parameters.

        :param a1: Link length 1 (shoulder to elbow).
        :param a2: Link length 2 (elbow to wrist).
        :param b: Base offset.
        :param c1: Shoulder height offset.
        :param c2, c3, c4: Wrist offsets.
        """
        ...

    @classmethod
    def from_kinematic_model(
        cls, kinematic_model: KinematicModel
    ) -> "RobotKinematicParams":
        """
        Create RobotKinematicParams from a KinematicModel.

        :param kinematic_model: The kinematic model to extract parameters from.
        :return: RobotKinematicParams instance.
        """
        ...

    def __repr__(self) -> str:
        """
        Returns a string representation of the RobotKinematicParams instance.
        """
        ...

__all__: List[str] = [
    "EulerConvention",
    "KinematicModel",
    "Robot",
    "RobotKinematicParams",
]
