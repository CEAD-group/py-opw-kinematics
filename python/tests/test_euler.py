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
        [0, 0, 0],
        [0, 90, 0],
        [90, 0, 0],
        [0, 0, 90],
        [1, 90, 0],
        [90, 1, 0],
        [1, 0, 90],
        [11, 12, 13],
        [10, 120, 130],
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


def test_matrix_to_quaternion(extrinsic, seq, angles, degrees):
    # Convert input angles to a rotation matrix using scipy
    rotation = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=True
    )
    rotation_matrix = rotation.as_matrix()

    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=degrees)
    # Convert rotation matrix to quaternion using custom implementation
    custom_quaternion = np.array(convention.matrix_to_quaternion(rotation_matrix))

    # Convert rotation matrix to quaternion using scipy for verification
    scipy_quaternion = rotation.as_quat()
    scipy_quaternion2 = np.array(
        [
            scipy_quaternion[3],
            scipy_quaternion[0],
            scipy_quaternion[1],
            scipy_quaternion[2],
        ]
    )  # Convert to [w, x, y, z] format

    # Verify that the quaternions are approximately equal
    assert np.allclose(custom_quaternion, scipy_quaternion2, atol=1e-5)


def test_matrix_to_euler(extrinsic, seq, angles, degrees):
    # Convert angles to rotation matrix
    rotation_matrix = Rotation.from_euler(seq.upper(), angles, degrees=True).as_matrix()

    # Create EulerConvention object
    convention = EulerConvention(seq, extrinsic=not extrinsic, degrees=degrees)
    # Convert back from rotation matrix to Euler angles
    res = np.array(convention.matrix_to_euler(rotation_matrix))

    # Get the expected result using SciPy
    expected = Rotation.from_matrix(rotation_matrix).as_euler(
        seq.upper(), degrees=degrees
    )

    assert expected == pytest.approx(res)
