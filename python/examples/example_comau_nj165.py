# %%
from py_opw_kinematics import KinematicModel, Robot, EulerConvention
import numpy as np


kinematic_model = KinematicModel(
    a1=400,
    a2=-250,
    b=0,
    c1=830,
    c2=1175,
    c3=1444,
    c4=230,
    offsets=[0] * 6,
    flip_axes=[True, False, True, True, False, True],
    has_parallellogram=True,
)

euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)

np.set_printoptions(precision=2, suppress=True)
robot = Robot(kinematic_model, euler_convention, ee_rotation=[0, -90, 0])


# %%
for angles, t_exp, r_exp in [
    ([0, 0, -90, 0, 0, 0], [2074.00, 0.00, 2255.00], [0.00, 0.00, 0.00]),
    ([10, 0, -90, 0, 0, 0], [2042.49, -360.15, 2255.00], [0.00, 0.00, -10.00]),
    ([0, 10, -90, 0, 0, 0], [2278.04, 0.00, 2237.15], [0.00, 0.00, 0.00]),
    ([0, 0, -80, 0, 0, 0], [2005.16, 0.00, 2541.89], [0.00, -10.00, 0.00]),
    ([0, 0, -90, 10, 0, 0], [2074.00, 0.00, 2255.00], [-10.00, 0.00, 0.00]),
    ([0, 0, -90, 0, 10, 0], [2070.51, 0.00, 2215.06], [0.00, 10.00, 0.00]),
    ([0, 0, -90, 0, 0, 10], [2074.00, 0.00, 2255.00], [-10.00, 0.00, 0.00]),
    ([10, 20, -90, 30, 20, 10], [2417.77, -466.26, 2116.01], [-37.35, 25.99, -4.81]),
]:
    t, r = robot.forward(joints=angles)
    pos, rot = (np.allclose(t, t_exp, atol=1e-2), np.allclose(r, r_exp, atol=1e-2))
    if not rot:
        print(angles, np.array(r), np.array(r_exp))
    else:
        print(angles, np.array(r), "ok")

# %%
