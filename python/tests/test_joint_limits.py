import pytest
import numpy as np
from py_opw_kinematics import JointLimits


class TestJointLimits:
    """Tests for JointLimits class."""

    def test_create_joint_limits_degrees(self):
        """Test creating joint limits in degrees."""
        limits = JointLimits(
            limits=[
                (-170, 170),
                (-120, 120),
                (-170, 170),
                (-185, 185),
                (-120, 120),
                (-360, 360),
            ],
            degrees=True,
        )
        assert limits.sorting_weight == 0.0
        assert np.allclose(limits.from_limits, [-170, -120, -170, -185, -120, -360])
        assert np.allclose(limits.to_limits, [170, 120, 170, 185, 120, 360])

    def test_create_joint_limits_radians(self):
        """Test creating joint limits in radians."""
        limits = JointLimits(
            limits=[
                (-np.pi, np.pi),
                (-np.pi / 2, np.pi / 2),
                (-np.pi, np.pi),
                (-np.pi, np.pi),
                (-np.pi / 2, np.pi / 2),
                (-2 * np.pi, 2 * np.pi),
            ],
            degrees=False,
        )
        assert np.allclose(limits.from_limits[0], -np.pi)
        assert np.allclose(limits.to_limits[0], np.pi)

    def test_compliant_within_limits(self):
        """Test that joints within limits are compliant."""
        limits = JointLimits(
            limits=[
                (-170, 170),
                (-120, 120),
                (-170, 170),
                (-185, 185),
                (-120, 120),
                (-360, 360),
            ],
        )
        # All zeros should be within limits
        assert limits.compliant((0, 0, 0, 0, 0, 0))
        # Some non-zero values within limits
        assert limits.compliant((10, -30, 60, 90, -45, 180))

    def test_compliant_outside_limits(self):
        """Test that joints outside limits are not compliant."""
        limits = JointLimits(
            limits=[
                (-170, 170),
                (-120, 120),
                (-170, 170),
                (-185, 185),
                (-120, 120),
                (-360, 360),
            ],
        )
        # J1 exceeds limit
        assert not limits.compliant((180, 0, 0, 0, 0, 0))
        # J2 exceeds limit
        assert not limits.compliant((0, 130, 0, 0, 0, 0))

    def test_filter_solutions(self):
        """Test filtering solutions by joint limits."""
        limits = JointLimits(
            limits=[
                (-90, 90),
                (-90, 90),
                (-90, 90),
                (-90, 90),
                (-90, 90),
                (-90, 90),
            ],
        )
        solutions = [
            (0, 0, 0, 0, 0, 0),  # Valid
            (100, 0, 0, 0, 0, 0),  # Invalid: J1 > 90
            (45, -45, 30, -30, 15, -15),  # Valid
            (0, 0, 0, 0, 0, 100),  # Invalid: J6 > 90
        ]
        filtered = limits.filter(solutions)
        assert len(filtered) == 2
        assert np.allclose(filtered[0], (0, 0, 0, 0, 0, 0))
        assert np.allclose(filtered[1], (45, -45, 30, -30, 15, -15))

    def test_sorting_weight(self):
        """Test that sorting weight is stored correctly."""
        limits = JointLimits(
            limits=[(-180, 180)] * 6,
            sorting_weight=0.5,
        )
        assert limits.sorting_weight == 0.5

    def test_centers(self):
        """Test that centers are computed correctly."""
        limits = JointLimits(
            limits=[
                (-180, 180),
                (-90, 90),
                (0, 180),
                (-45, 45),
                (-60, 60),
                (-360, 360),
            ],
        )
        centers = limits.centers
        assert np.allclose(centers[0], 0)  # (-180 + 180) / 2
        assert np.allclose(centers[1], 0)  # (-90 + 90) / 2
        assert np.allclose(centers[2], 90)  # (0 + 180) / 2
        assert np.allclose(centers[3], 0)  # (-45 + 45) / 2
        assert np.allclose(centers[4], 0)  # (-60 + 60) / 2
        assert np.allclose(centers[5], 0)  # (-360 + 360) / 2

    def test_repr(self):
        """Test string representation."""
        limits = JointLimits(
            limits=[(-180, 180)] * 6,
            sorting_weight=0.5,
        )
        repr_str = repr(limits)
        assert "JointLimits" in repr_str
        assert "sorting_weight=0.5" in repr_str

    def test_wrap_around_limits(self):
        """Test wrap-around limits where min > max."""
        # J6 wraps around: from 170 to -170 (going through 180/-180)
        limits = JointLimits(
            limits=[
                (-180, 180),
                (-180, 180),
                (-180, 180),
                (-180, 180),
                (-180, 180),
                (170, -170),  # Wrap-around: 170 -> 180 -> -180 -> -170
            ],
        )
        # 175 is in the wrap-around range (170 to -170 through 180)
        assert limits.compliant((0, 0, 0, 0, 0, 175))
        # -175 is also in the wrap-around range
        assert limits.compliant((0, 0, 0, 0, 0, -175))
        # 0 is NOT in the wrap-around range
        assert not limits.compliant((0, 0, 0, 0, 0, 0))
