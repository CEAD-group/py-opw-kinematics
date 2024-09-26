# %%
from py_opw_kinematics.py_opw_kinematics import Robot, EulerConvention, KinematicModel
from math import pi
import numpy as np
import pytest

# %%

#   <xacro:property name="hoogte_as_1" value="0.830" />
#   <xacro:property name="lengte_as_1_2" value="0.400" />
#   <xacro:property name="lengte_arm_1" value="1.175" />
#   <xacro:property name="hoogte_3_4" value="0.250" />
#   <xacro:property name="lengte_arm_2" value="1.444" />
#   <xacro:property name="afstand_5_flens" value="0.230" />
# (a1: f64, a2: f64, b: f64, c1: f64, c2: f64, c3: f64, c4: f64, offsets: [f64; 6], sign_corrections: [i32; 6]
# robot = PyOPWKinematics(
#     a1=0.4,  # lengte_as_1_2
#     a2=0.25,  # hoogte_3_4
#     b=0,
#     c1=0.83,  # hoogte_as_1
#     c2=1.175,  # lengte_arm_1
#     c3=1.444,  # lengte_arm_2
#     c4=0.23,  # afstand_5_flens
#     offsets=[0, 0, 0, 0, 0, 0],
#     sign_corrections=[1, 1, 1, 1, 1, 1],
# )


@pytest.mark.parametrize("a1_sign", [-1, +1])
@pytest.mark.parametrize("a2_sign", [-1, +1])
# @pytest.mark.parametrize("offset1", [0, pi])
# @pytest.mark.parametrize("offset2", [0, pi])
# @pytest.mark.parametrize("offset3", [0, pi])
# @pytest.mark.parametrize("offset4", [0, pi])
@pytest.mark.parametrize("s1", [1, -1])
@pytest.mark.parametrize("s2", [1, -1])
@pytest.mark.parametrize("s3", [1, -1])
@pytest.mark.parametrize("s4", [1, -1])
@pytest.mark.parametrize("s5", [1, -1])
@pytest.mark.xfail
def test_poses(a1_sign, a2_sign, s1, s2, s3, s4, s5):
    offset1, offset2, offset3, offset4 = 0, 0, 0, 0
    kinematic_model = KinematicModel(
        a1=a1_sign * 400.333,  # lengte_as_1_2 * -1
        a2=a2_sign * 251.449,  # hoogte_3_4
        b=0,
        c1=830,  # hoogte_as_1
        c2=1177.556,  # lengte_arm_1
        c3=1443.593,  # lengte_arm_2
        c4=230,  # afstand_5_flens
        offsets=[offset1, offset2, offset3, offset4, 0, 0],
        sign_corrections=[s1, s2, s3, s4, s5, -1],
        has_parallellogram=True,
    )
    euler_convention = EulerConvention("XYZ", extrinsic=True, degrees=True)
    robot = Robot(kinematic_model, euler_convention)

    t, r_rad = robot.forward(joints=[0, 0, -90, 0, 0, 0])
    assert np.allclose(t, [2073.926, 0.0, 2259.005], atol=1e-3)

    t, r_rad = robot.forward(joints=[10, 0, -90, 0, 0, 0])
    assert np.allclose(t, [2042.418, -360.133, 2259.005], atol=1e-3)

    t, r_rad = robot.forward(joints=[10, 10, -90, 0, 0, 0])
    assert np.allclose(t, [2243.792, -395.641, 2241.115], atol=1e-3)

    t, r_rad = robot.forward(joints=[0, 0, -90, 0, 10, 0])
    assert np.allclose(t, [2070.432, 0.0, 2219.066], atol=1e-3)

    t, r_rad = robot.forward(joints=[0, 0, -90, 10, 10, 0])
    assert np.allclose(t, [2070.432, -6.935, 2219.673], atol=1e-3)

    t, r_rad = robot.forward(joints=[10, 20, -90, 30, 20, 10])
    print(np.allclose(t, [2118.558, -477.396, 2119.864], atol=1e-3))
    print(t)
    print(
        f"{a1_sign=}, {a2_sign=}, {offset1=}, {offset2=}, {offset3=}, {offset4=}, {s1=}, {s2=}, {s3=}, {s4=}, {s5=}"
    )


# a1_sign = -1, offset1 = 3.141592653589793, offset2 = 0, offset3 = 0, offset4 = 3.141592653589793, s1 = -1, s2 = -1, s3 = 1, s4 = -1, s5 = 1
# a1_sign = -1, offset1 = -3.141592653589793, offset2 = 0, offset3 = 0, offset4 = 3.141592653589793, s1 = -1, s2 = -1, s3 = 1, s4 = -1, s5 = 1
# a1_sign = -1, offset1 = 3.141592653589793, offset2 = 0, offset3 = 0, offset4 = -3.141592653589793, s1 = -1, s2 = -1, s3 = 1, s4 = -1, s5 = 1
# a1_sign = -1, offset1 = -3.141592653589793, offset2 = 0, offset3 = 0, offset4 = -3.141592653589793, s1 = -1, s2 = -1, s3 = 1, s4 = -1, s5 = 1
# a1_sign = -1, offset1 = 3.141592653589793, offset2 = 0, offset3 = 0, offset4 = 0, s1 = -1, s2 = -1, s3 = 1, s4 = -1, s5 = -1
# a1_sign = -1, offset1 = -3.141592653589793, offset2 = 0, offset3 = 0, offset4 = 0, s1 = -1, s2 = -1, s3 = 1, s4 = -1, s5 = -1

# # %%

# robot = PyOPWKinematics(
#     a1=-400.333,  # lengte_as_1_2 * -1
#     a2=251.449,  # hoogte_3_4
#     b=0,
#     c1=830,  # hoogte_as_1
#     c2=1177.556,  # lengte_arm_1
#     c3=1443.593,  # lengte_arm_2
#     c4=230,  # afstand_5_flens
#     offsets=[pi, 0, 0, 0, 0, 0],
#     sign_corrections=[-1, -1, 1, -1, -1, -1],
#     # has_parallel=True,
# )


# # %%


# # motion dir: -1, 1, 1,1,1,1
# # robx axes dir -1 1 -1 -1 1 -1
# # options = robot.inverse((t, r))
# # np.rad2deg(options).round(2)
# # %%
# joint_angles = robot.inverse((t, r_rad))
# np.rad2deg(joint_angles).round(2)


# # %%
# angles = [10, 20, -90, 30, 20, 10]

# flange_xyzabc = [2118.558, -477.396, 2119.864, -37.346, 25.987, -4.814]
# tool_offset = [234.096, 132.2, -551.725]
# tool_xyzabc = [2396.467, -743.091, 1572.479, -37.346, 25.987, -4.814]

# %%
