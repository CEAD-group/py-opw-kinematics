import pytest
from py_opw_kinematics import EulerConvention
import numpy as np
from scipy.spatial.transform import Rotation


@pytest.fixture(
    params=[
        "XYX",
        "XYZ",
        "XZX",
        "XZY",
        "YXY",
        "YXZ",
        "YZX",
        "YZY",
        "ZXY",
        "ZXZ",
        "ZYX",
        "ZYZ",
    ]
)
def seq(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param([10, 20, 30], id="basic_combined"),
        pytest.param([90, 0, 0], id="basic_x_90"),
        pytest.param([0, 90, 0], id="basic_y_90"),
        pytest.param([0, 0, 90], id="basic_z_90"),
        pytest.param([180, 0, 0], id="basic_x_180"),
        pytest.param([0, 180, 0], id="basic_y_180"),
        pytest.param([0, 0, 180], id="basic_z_180"),
        pytest.param([90, 90, 0], id="combined_x_90_y_90"),
        pytest.param([45, 45, 45], id="combined_xyz_45"),
        pytest.param([90, 0, 90], id="combined_x_90_z_90"),
        pytest.param([-90, 90, -90], id="combined_neg_x_90_y_90_z_neg_90"),
        pytest.param([0, 180, 90], id="combined_y_180_z_90"),
        pytest.param([90, 90, 0], id="gimbal_lock_y_90"),
        pytest.param([0, 90, 90], id="gimbal_lock_x_90_z_90"),
        pytest.param([90, 0, 90], id="gimbal_lock_x_90_z_90"),
        pytest.param([0, 0, 360], id="boundary_z_360"),
        pytest.param([-180, 0, 0], id="boundary_neg_x_180"),
        pytest.param([180, 180, 180], id="boundary_xyz_180"),
        pytest.param([0.001, 0, 0], id="boundary_small_x_0_001"),
        pytest.param([0, 89.999, 0], id="boundary_y_89_999"),
        pytest.param([0, 0, 0], id="singularity_identity"),
        pytest.param([90, 0, 90], id="singularity_x_90_z_90"),
        pytest.param([360, 0, 0], id="full_cycle_x_360"),
        pytest.param([0, 360, 360], id="full_cycle_y_360_z_360"),
        pytest.param([720, 0, 0], id="full_cycle_x_720"),
    ]
)
def angles(request):
    return request.param


@pytest.fixture(params=[True, False])
def extrinsic(request):
    return request.param


@pytest.fixture(params=[True, False])
def degrees(request):
    return request.param


def test_euler_to_matrix(extrinsic, seq, angles, degrees):
    # Create EulerConvention object
    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)

    if not degrees:
        angles = np.deg2rad(angles)
    # Get the result from EulerConvention
    res = np.array(convention.euler_to_matrix(angles))

    # Get the expected result using SciPy
    expected = Rotation.from_euler(
        seq.lower() if extrinsic else seq.upper(), angles, degrees=degrees
    ).as_matrix()

    # Rotation.from_matrix(res).as_euler(seq.lower() if extrinsic else seq.upper(), degrees=degrees)
    assert res == pytest.approx(expected)


def test_matrix_to_quaternion(angles, extrinsic=True, seq="XYZ", degrees=True):
    # Convert input angles to a rotation matrix using scipy
    rotation = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=True
    )
    rotation_matrix = rotation.as_matrix()

    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)
    custom_quaternion = np.array(convention.matrix_to_quaternion(rotation_matrix))
    scipy_quaternion = rotation.as_quat(canonical=True, scalar_first=True)
    assert custom_quaternion == pytest.approx(scipy_quaternion, abs=1e-5)


def test_quaternion_to_euler_equal_to_scipy(angles, extrinsic, seq, degrees=True):
    quaternion = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=True
    ).as_quat(canonical=True, scalar_first=True)
    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)

    computed_scipy = Rotation.from_quat(quaternion, scalar_first=True).as_euler(
        seq.lower() if extrinsic else seq, degrees=degrees
    )

    computed_custom = np.array(convention.quaternion_to_euler(quaternion))
    assert computed_custom == pytest.approx(computed_scipy, abs=1e-5)


def test_quaternion_to_euler_equivalent(angles, extrinsic, seq, degrees=True):
    quaternion = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=True
    ).as_quat(canonical=True, scalar_first=True)
    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)
    computed_custom = np.array(convention.quaternion_to_euler(quaternion))

    recomputed_quaternion = Rotation.from_euler(
        seq.lower() if extrinsic else seq, computed_custom, degrees=True
    ).as_quat(canonical=True, scalar_first=True)
    assert recomputed_quaternion == pytest.approx(
        quaternion, abs=1e-5
    ) or recomputed_quaternion == pytest.approx(-quaternion, abs=1e-5)


def test_matrix_to_euler_exact(angles, extrinsic, seq, degrees=True):
    matrix = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=degrees
    ).as_matrix()
    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)
    computed_custom = np.array(convention.matrix_to_euler(matrix))
    computed_scipy = Rotation.from_matrix(matrix).as_euler(
        seq.lower() if extrinsic else seq, degrees=degrees
    )
    # if scipy can compute the angles correctly but the custom implementation can't, mark the test as xfail
    if pytest.approx(computed_scipy, abs=1e-5) != angles:
        pytest.xfail("Scipy also failed to compute the angles correctly.")

    assert computed_custom == pytest.approx(angles, abs=1e-5)


def test_matrix_to_euler_equivalent(degrees, angles, extrinsic, seq="XYZ"):
    matrix = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=degrees
    ).as_matrix()
    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)
    computed_custom = np.array(convention.matrix_to_euler(matrix))

    # If the original angles cannot be reproduced, then check if the recomputed matrix is close to the original matrix
    recomputed_matrix = Rotation.from_euler(
        seq.lower() if extrinsic else seq, computed_custom, degrees
    ).as_matrix()

    # Check if the recomputed matrix is approximately equal to the original matrix
    assert recomputed_matrix == pytest.approx(matrix, abs=1e-3)


@pytest.mark.parametrize("in_angles", [[30, 0, 50], [60, 40, 50]])
@pytest.mark.parametrize("out_extrinsic", [True, False])
@pytest.mark.parametrize(
    "out_seq",
    [
        "XYZ",
        "XZX",
    ],
)
def test_euler_convention_to_other_convention(
    extrinsic,
    in_angles,
    out_extrinsic,
    out_seq,
    seq="XYZ",
):
    # Create EulerConvention object
    in_convention = EulerConvention(seq, extrinsic=extrinsic, degrees=True)
    out_convention = EulerConvention(out_seq, extrinsic=out_extrinsic, degrees=True)

    # Get the result from EulerConvention
    res = in_convention.convert(out_convention, in_angles)

    # Get the expected result using SciPy
    expected = Rotation.from_matrix(
        Rotation.from_euler(
            seq.lower() if extrinsic else seq.upper(), in_angles, degrees=True
        ).as_matrix()
    ).as_euler(out_seq.lower() if out_extrinsic else out_seq.upper(), degrees=True)

    assert res == pytest.approx(expected)

    if (extrinsic == out_extrinsic) and (seq == out_seq):
        assert expected == pytest.approx(in_angles)
