# %%
from py_opw_kinematics.py_opw_kinematics import EulerConvention

xyz_e = EulerConvention("XYZ", extrinsic=False, degrees=True)
xyz_i = EulerConvention("XYZ", extrinsic=True, degrees=True)

xyz_i.convert(xyz_e, [0, 0, 0])


# %%
from py_opw_kinematics.py_opw_kinematics import PyOPWKinematics, euler_angles_ordered
from math import pi
import numpy as np

dir(PyOPWKinematics)
# %%

#   <xacro:property name="hoogte_as_1" value="0.830" />
#   <xacro:property name="lengte_as_1_2" value="0.400" />
#   <xacro:property name="lengte_arm_1" value="1.175" />
#   <xacro:property name="hoogte_3_4" value="0.250" />
#   <xacro:property name="lengte_arm_2" value="1.444" />
#   <xacro:property name="afstand_5_flens" value="0.230" />
# (a1: f64, a2: f64, b: f64, c1: f64, c2: f64, c3: f64, c4: f64, offsets: [f64; 6], sign_corrections: [i32; 6]
robot = PyOPWKinematics(
    a1=0.4,
    a2=0.25,
    b=0,
    c1=0.83,
    c2=1.175,
    c3=1.444,
    c4=0.23,
    offsets=[0, 0, -pi / 2, 0, 0, 0],
    sign_corrections=[1, 1, 1, 1, 1, 1],
    has_parallellogram=True,
)
# %%

print(robot.forward(joints=[0, 0, 0, 0, 0, 0]))
# print(robot.forward(joints=[0, 0, 0, 0, 0.123, 0]))

# # %%
# robot.inverse(pose=([0.65, 0.0, 3.679], [0.0, -0.0, 0.0]))
# # %%
# %%


print(
    np.rad2deg(
        euler_angles_ordered(
            euler=np.deg2rad([10, 20, 30]), seq="XYZ", extrinsic=False
        )[0]
    )
)

# %%
