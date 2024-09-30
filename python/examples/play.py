# %%
from py_opw_kinematics import KinematicModel

km = KinematicModel(
    a1=1,
    a2=1,
    b=1,
    c1=1,
    c2=1,
    c3=1,
    c4=1,
    offsets=(1, 1, 1, 1, 1, 1),
    sign_corrections=(1, 1, 1, 1, 1, 1),
    has_parallellogram=False,
)
print(km)
KinematicModel(
    a1=1,
    a2=1,
    b=1,
    c1=1,
    c2=1,
    c3=1,
    c4=1,
    offsets=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    sign_corrections=[1, 1, 1, 1, 1, 1],
    has_parallellogram=False,
)
# %%

# # %%
# import pytest
# from py_opw_kinematics import EulerConvention
# import scipy
# import numpy as np

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

# # %%


# from py_opw_kinematics import KinematicModel, Robot, EulerConvention
