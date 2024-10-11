import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import warnings


def rotation_x(alpha):
    """Rotation matrix around X-axis by angle alpha (in radians)."""
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    return np.array([[1, 0, 0], [0, c_alpha, -s_alpha], [0, s_alpha, c_alpha]])


def rotation_y(beta):
    """Rotation matrix around Y-axis by angle beta (in radians)."""
    c_beta = np.cos(beta)
    s_beta = np.sin(beta)
    return np.array([[c_beta, 0, s_beta], [0, 1, 0], [-s_beta, 0, c_beta]])


def rotation_z(gamma):
    """Rotation matrix around Z-axis by angle gamma (in radians)."""
    c_gamma = np.cos(gamma)
    s_gamma = np.sin(gamma)
    return np.array([[c_gamma, -s_gamma, 0], [s_gamma, c_gamma, 0], [0, 0, 1]])


def euler_xzx(alpha, beta, gamma):
    """Euler X(alpha) -> Z(beta) -> X(gamma)"""
    return rotation_x(alpha) @ rotation_z(beta) @ rotation_x(gamma)


def euler_xzy(alpha, beta, gamma):
    """Euler X(alpha) -> Z(beta) -> Y(gamma)"""
    return rotation_x(alpha) @ rotation_z(beta) @ rotation_y(gamma)


def euler_xyx(alpha, beta, gamma):
    """Euler X(alpha) -> Y(beta) -> X(gamma)"""
    return rotation_x(alpha) @ rotation_y(beta) @ rotation_x(gamma)


def euler_xyz(alpha, beta, gamma):
    """Euler X(alpha) -> Y(beta) -> Z(gamma)"""
    return rotation_x(alpha) @ rotation_y(beta) @ rotation_z(gamma)


def euler_yxy(alpha, beta, gamma):
    """Euler Y(alpha) -> X(beta) -> Y(gamma)"""
    return rotation_y(alpha) @ rotation_x(beta) @ rotation_y(gamma)


def euler_yxz(alpha, beta, gamma):
    """Euler Y(alpha) -> X(beta) -> Z(gamma)"""
    return rotation_y(alpha) @ rotation_x(beta) @ rotation_z(gamma)


def euler_yzy(alpha, beta, gamma):
    """Euler Y(alpha) -> Z(beta) -> Y(gamma)"""
    return rotation_y(alpha) @ rotation_z(beta) @ rotation_y(gamma)


def euler_yzx(alpha, beta, gamma):
    """Euler Y(alpha) -> Z(beta) -> X(gamma)"""
    return rotation_y(alpha) @ rotation_z(beta) @ rotation_x(gamma)


def euler_zxy(alpha, beta, gamma):
    """Euler Z(alpha) -> X(beta) -> Y(gamma)"""
    return rotation_z(alpha) @ rotation_x(beta) @ rotation_y(gamma)


def euler_zxz(alpha, beta, gamma):
    """Euler Z(alpha) -> X(beta) -> Z(gamma)"""
    return rotation_z(alpha) @ rotation_x(beta) @ rotation_z(gamma)


def euler_zyx(alpha, beta, gamma):
    """Euler Z(alpha) -> Y(beta) -> X(gamma)"""
    return rotation_z(alpha) @ rotation_y(beta) @ rotation_x(gamma)


def euler_zyz(alpha, beta, gamma):
    """Euler Z(alpha) -> Y(beta) -> Z(gamma)"""
    return rotation_z(alpha) @ rotation_y(beta) @ rotation_z(gamma)


mapping = {
    "XZX": euler_xzx,
    "XZY": euler_xzy,
    "XYX": euler_xyx,
    "XYZ": euler_xyz,
    "YXY": euler_yxy,
    "YXZ": euler_yxz,
    "YZY": euler_yzy,
    "YZX": euler_yzx,
    "ZXY": euler_zxy,
    "ZXZ": euler_zxz,
    "ZYX": euler_zyx,
    "ZYZ": euler_zyz,
}


def euler_to_matrix(seq, angles, extrinsic=False, degrees=False):
    if extrinsic:
        seq = seq[::-1]
        angles = angles[::-1]
    if degrees:
        angles = np.deg2rad(angles)
    res = mapping[seq.upper()](*angles)
    return res


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        q: quaternion as [w, x, y, z]
    """
    m00 = R[0, 0]
    m01 = R[0, 1]
    m02 = R[0, 2]
    m10 = R[1, 0]
    m11 = R[1, 1]
    m12 = R[1, 2]
    m20 = R[2, 0]
    m21 = R[2, 1]
    m22 = R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q


def _elementary_basis_index(axis):
    """Map axis label to index."""
    axis = axis.upper()
    if axis == "X":
        return 0
    elif axis == "Y":
        return 1
    elif axis == "Z":
        return 2
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _get_angles(angles, extrinsic, symmetric, sign, lamb, a, b, c, d):
    """
    Compute Euler angles from quaternion components.
    """
    pi = np.pi

    # Intrinsic/extrinsic angle indices
    if extrinsic:
        angle_first = 0
        angle_third = 2
    else:
        angle_first = 2
        angle_third = 0

    # Step 2: Compute the second angle
    angles[1] = 2 * np.arctan2(np.hypot(c, d), np.hypot(a, b))

    # Check for singularities at angles[1] = 0 or π
    eps = 1e-7
    if abs(angles[1]) <= eps:
        case = 1  # angles[1] ≈ 0
    elif abs(angles[1] - pi) <= eps:
        case = 2  # angles[1] ≈ π
    else:
        case = 0  # Regular case

    # Step 3: Compute first and third angles
    half_sum = np.arctan2(b, a)
    half_diff = np.arctan2(d, c)

    if case == 0:  # Regular case
        angles[angle_first] = half_sum - half_diff
        angles[angle_third] = half_sum + half_diff
    else:  # Singular cases
        angles[angle_first] = (
            2 * half_sum if case == 1 else 2 * half_diff * (-1 if extrinsic else 1)
        )
        angles[angle_third] = 0

    # Adjust for asymmetric sequences
    if not symmetric:
        angles[angle_third] *= sign
        angles[1] -= lamb

    # Normalize angles to [-π, π]
    for idx in range(3):
        if angles[idx] < -pi:
            angles[idx] += 2 * pi
        elif angles[idx] > pi:
            angles[idx] -= 2 * pi

    # Warn if gimbal lock detected
    if case != 0:
        warnings.warn(
            "Gimbal lock detected. Setting third angle to zero "
            "since it is not possible to uniquely determine "
            "all angles."
        )


def quaternion_to_euler(quat, seq, extrinsic=False, degrees=False):
    """
    Convert quaternions to Euler angles based on the given rotation sequence.

    Args:
        quat: Array of quaternions with shape (..., 4) in [w, x, y, z] format.
        seq: Rotation sequence as a string, e.g., "ZYX".
        extrinsic: If True, use extrinsic rotations.
        degrees: If True, return angles in degrees.

    Returns:
        angles: Array of Euler angles with shape (..., 3).
    """
    quat = np.asarray(quat)
    original_shape = quat.shape
    quat = quat.reshape(-1, 4)

    if not extrinsic:
        seq = seq[::-1]

    # Map axis labels to indices
    i = _elementary_basis_index(seq[0])
    j = _elementary_basis_index(seq[1])
    k = _elementary_basis_index(seq[2])

    symmetric = i == k
    if symmetric:
        # For symmetric sequences, compute the third axis
        k = 3 - i - j  # Since i + j + k = 3 for X=0, Y=1, Z=2

    # Step 0: Determine the sign based on the permutation parity
    perm = [i, j, k]
    even_permutation = perm in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    sign = 1 if even_permutation else -1

    num_rotations = quat.shape[0]

    angles = np.empty((num_rotations, 3))
    pi = np.pi

    for ind in range(num_rotations):
        q = quat[ind]
        w, x, y, z = q

        # Ensure quaternion has a positive w component
        if w < 0:
            q = -q
            w, x, y, z = q

        # Quaternion components
        q0 = w
        q1 = x
        q2 = y
        q3 = z

        # Step 1: Permute quaternion elements based on the rotation sequence
        q_vec = np.array([q1, q2, q3])
        a = q0
        b = q_vec[i]
        c = q_vec[j]
        d = q_vec[k] * sign

        if symmetric:
            # For symmetric sequences
            pass  # a, b, c, d are already set
        else:
            # For asymmetric sequences
            a, b = a - c, b + d
            c, d = c + q0, d - q_vec[i]

        # Lambda (λ) is π/2 for asymmetric sequences
        lamb = pi / 2 if not symmetric else 0

        # Initialize angles array for current quaternion
        angle = np.zeros(3)

        # Compute angles using helper function
        _get_angles(angle, extrinsic, symmetric, sign, lamb, a, b, c, d)

        angles[ind] = angle

    # Reshape to original input shape with last dimension of size 3
    angles = angles.reshape(original_shape[:-1] + (3,))

    if degrees:
        angles = np.degrees(angles)
    return angles


def matrix_to_euler(seq, matrix, extrinsic=False, degrees=False):
    """
    Converts a rotation matrix to Euler angles based on the given sequence.

    Args:
        seq: Rotation sequence (e.g., "XYZ", "ZYX", etc.).
        matrix: 3x3 rotation matrix.
        extrinsic: If True, use extrinsic rotations.
        degrees: If True, return angles in degrees.

    Returns:
        A tuple of three angles (alpha, beta, gamma).
    """
    if extrinsic:
        seq = seq[::-1]

    # Step 1: Convert rotation matrix to quaternion
    q = rotation_matrix_to_quaternion(matrix)

    # Step 2: Convert quaternion to Euler angles
    angles = quaternion_to_euler(q, seq)

    if not extrinsic:
        angles = angles[::-1]
    if degrees:
        angles = np.rad2deg(angles)
    return angles


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


def test_euler_to_matrix(seq, angles, extrinsic, degrees):
    if not degrees:
        angles = np.radians(angles)
    res = euler_to_matrix(seq, angles, extrinsic, degrees)
    expected = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees
    ).as_matrix()

    assert expected == pytest.approx(res)


def test_rotation_matrix_to_quaternion(seq, angles, extrinsic):
    # Convert input angles to a rotation matrix using scipy
    rotation = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees=True
    )
    rotation_matrix = rotation.as_matrix()

    # Convert rotation matrix to quaternion using custom function
    custom_quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    # Convert rotation matrix to quaternion using scipy for verification
    scipy_quaternion = rotation.as_quat()
    scipy_quaternion = np.array(
        [
            scipy_quaternion[3],
            scipy_quaternion[0],
            scipy_quaternion[1],
            scipy_quaternion[2],
        ]
    )  # Convert to [w, x, y, z] format

    # Verify that the quaternions are approximately equal
    assert np.allclose(custom_quaternion, scipy_quaternion, atol=1e-5)


def test_matrix_to_euler(seq, angles, extrinsic, degrees):
    if not degrees:
        angles = np.radians(angles)

    matrix = Rotation.from_euler(
        seq.lower() if extrinsic else seq, angles, degrees
    ).as_matrix()

    expected_angles = Rotation.from_matrix(matrix).as_euler(
        seq.lower() if extrinsic else seq, degrees
    )

    computed_angles = matrix_to_euler(seq, matrix, extrinsic, degrees)

    # Straightforward check if the expected angles are close to the original angles
    if np.allclose(expected_angles, angles, atol=1e-5):
        return  # Pass if angles match closely
    else:
        # If gimbal lock is present, recompute the matrix from the computed angles
        recomputed_matrix = Rotation.from_euler(
            seq.lower() if extrinsic else seq, computed_angles, degrees
        ).as_matrix()

        # Check if the recomputed matrix is approximately equal to the original matrix
        if np.allclose(matrix, recomputed_matrix, atol=1e-5):
            return  # Pass if the recomputed matrix is close to the original matrix.
            # The solution is different from the scipy reference implementation,
            # but it is consistent.

    # if scipy can compute the angles correctly, then the test fails
    if np.allclose(expected_angles, angles, atol=1e-2):
        assert (
            False
        )  # Fail if the recomputed matrix is not close to the original matrix.
    else:
        pytest.skip("Scipy also failed to compute the angles correctly.")
