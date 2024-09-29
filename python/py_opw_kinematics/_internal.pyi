from typing import List, Tuple, Literal

Sign = Literal[-1, 1]

class EulerConvention:
    sequence: str
    extrinsic: bool
    degrees: bool

    def __init__(self, sequence: str, extrinsic: bool, degrees: bool) -> None: ...
    def convert(
        self, other: "EulerConvention", angles: Tuple[float, float, float]
    ) -> Tuple[float, float, float]: ...
    def from_rotation_matrix(
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
        offsets: Tuple[float, float, float, float, float, float],
        sign_corrections: Tuple[Sign, Sign, Sign, Sign, Sign, Sign],
        has_parallellogram: bool,
    ) -> None: ...

class Robot:
    # robot: "OPWKinematics"  # Assuming OPWKinematics is defined in the Rust module
    has_parallellogram: bool
    degrees: bool
    euler_convention: EulerConvention
    _internal_euler_convention: EulerConvention

    def __init__(
        self,
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Tuple[float, float, float],
    ) -> None: ...
    def forward(
        self, joints: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]: ...
    def inverse(
        self, pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float, float, float, float]]: ...
    def inverse_continuing(
        self,
        pose: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        current_joints: Tuple[float, float, float, float, float, float],
    ) -> List[Tuple[float, float, float, float, float, float]]: ...

__all__: List[str]
