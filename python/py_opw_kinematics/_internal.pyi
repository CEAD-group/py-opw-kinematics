from typing import List, Tuple, Literal
import polars as pl

EulerSequence = Literal[
    "XYX", "XYZ", "XZX", "XZY", "YXY", "YXZ", "YZX", "YZY", "ZXY", "ZXZ", "ZYX", "ZYZ"
]

class EulerConvention:
    sequence: EulerSequence
    extrinsic: bool
    degrees: bool

    def __init__(
        self, sequence: EulerSequence, extrinsic: bool, degrees: bool
    ) -> None: ...
    def convert(
        self, other: "EulerConvention", angles: Tuple[float, float, float]
    ) -> Tuple[float, float, float]: ...
    def angles_from_rotation_matrix(
        self, rot: List[List[float]]
    ) -> Tuple[float, float, float]: ...
    def to_rotation_matrix(
        self, angles: Tuple[float, float, float]
    ) -> List[List[float]]: ...

class KinematicModel:
    def __init__(
        self,
        a1: float,
        a2: float,
        b: float,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
        flip_axes: None | Tuple[bool, bool, bool, bool, bool, bool],
        has_parallelogram: bool,
    ) -> None: ...

class Robot:
    # robot: "OPWKinematics"  # Assuming OPWKinematics is defined in the Rust module
    has_parallelogram: bool
    euler_convention: EulerConvention
    ee_rotation: Tuple[float, float, float]
    ee_translation: Tuple[float, float, float]
    _internal_euler_convention: EulerConvention

    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None: ...
    def forward(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]: ...
    def inverse(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Tuple[float, float, float, float, float, float] | None = None,
    ) -> List[Tuple[float, float, float, float, float, float]]: ...
    def batch_inverse(
        self, poses: pl.DataFrame, current_joints: List[float] | None = None
    ) -> pl.DataFrame: ...
    def batch_forward(self, joints: pl.DataFrame) -> pl.DataFrame: ...

__all__: List[str]
