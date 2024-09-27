import pytest
from py_opw_kinematics import EulerConvention
import scipy
import numpy as np


@pytest.mark.parametrize("extrinsic", [True, False])
@pytest.mark.parametrize("seq", ["XYZ", "XZY", "ZYX", "ZXY"])
@pytest.mark.parametrize("angles", [[30, 0, 50], [60, 40, 90]])
def test_euler_convention_to_rotation_matrix(extrinsic, seq, angles):
    # Create EulerConvention object
    convention = EulerConvention(seq, extrinsic=extrinsic, degrees=True)

    # Get the result from EulerConvention
    res = convention.to_rotation_matrix(angles)

    # Get the expected result using SciPy
    expected = scipy.spatial.transform.Rotation.from_euler(
        seq.lower() if extrinsic else seq.upper(), angles, degrees=True
    ).as_matrix()

    # Compare the results using np.isclose
    assert np.all(
        np.isclose(res, expected)
    ), f"Failed for {seq=}, {angles=}, {extrinsic=}"


@pytest.mark.parametrize("in_extrinsic", [True, False])
@pytest.mark.parametrize("in_seq", ["XYZ", "XZY", "ZYX", "ZXY"])
@pytest.mark.parametrize("in_angles", [[30, 0, 50], [60, 40, 90]])
@pytest.mark.parametrize("out_extrinsic", [True, False])
@pytest.mark.parametrize("out_seq", ["XYZ", "XZY", "YXZ", "YZX"])
def test_euler_convention_to_other_convention(
    in_extrinsic, in_seq, in_angles, out_extrinsic, out_seq
):
    # Create EulerConvention object
    in_convention = EulerConvention(in_seq, extrinsic=in_extrinsic, degrees=True)
    out_convention = EulerConvention(out_seq, extrinsic=out_extrinsic, degrees=True)

    # Get the result from EulerConvention
    res = in_convention.convert(out_convention, in_angles)

    # Get the expected result using SciPy
    expected = scipy.spatial.transform.Rotation.from_euler(
        in_seq.lower() if in_extrinsic else in_seq.upper(), in_angles, degrees=True
    ).as_matrix()

    expected = scipy.spatial.transform.Rotation.from_matrix(expected).as_euler(
        out_seq.lower() if out_extrinsic else out_seq.upper(), degrees=True
    )

    # Compare the results using np.isclose
    assert np.all(
        np.isclose(res, expected)
    ), f"Failed for {in_seq=}, {in_angles=}, {in_extrinsic=}, {out_seq=}, {out_extrinsic=}"
