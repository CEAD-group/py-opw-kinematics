from __future__ import annotations

from collections import OrderedDict

from ._internal import EulerConvention, KinematicModel, Robot as _Robot


def _compute_link_transforms_py(self: _Robot, joints_deg):
    """
    Wrap Rust output:
      {"robot_l0": [..16..], ...}
    into legacy Python output:
      {"robot_l0": {"matrix": [..16..]}, ...}
    """
    raw = self.compute_link_transforms(joints_deg)

    order = (
        "robot_l0",
        "robot_l1",
        "robot_l2",
        "robot_l3",
        "robot_l4",
        "robot_l5",
        "robot_l6",
        "robot_l7",
        "robot_l8",
    )
    out = OrderedDict()

    for k in order:
        if k in raw:
            out[k] = {"matrix": raw[k]}

    # include any extra links if they ever appear
    for k, v in raw.items():
        if k not in out:
            out[k] = {"matrix": v}

    return out


# Monkey-patch onto the compiled Robot class
_Robot.compute_link_transforms_py = _compute_link_transforms_py

# Re-export under the original name
Robot = _Robot

__all__ = ["EulerConvention", "KinematicModel", "Robot"]
