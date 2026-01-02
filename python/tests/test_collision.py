"""Tests for collision detection and joint limits integration."""

import numpy as np
import pytest
from py_opw_kinematics import (
    CheckMode,
    EulerConvention,
    JointLimits,
    KinematicModel,
    Robot,
    SafetyDistances,
)


# Fixture for a basic robot without collision geometry
@pytest.fixture
def basic_robot():
    """Create a basic robot without collision geometry."""
    model = KinematicModel(
        a1=0.150, a2=-0.115, b=0.0, c1=0.440, c2=0.455, c3=0.095, c4=0.080
    )
    euler = EulerConvention("ZYX", extrinsic=False, degrees=True)
    return Robot(model, euler)


@pytest.fixture
def robot_with_limits():
    """Create a robot with joint limits but no collision geometry."""
    model = KinematicModel(
        a1=0.150, a2=-0.115, b=0.0, c1=0.440, c2=0.455, c3=0.095, c4=0.080
    )
    euler = EulerConvention("ZYX", extrinsic=False, degrees=True)
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
    return Robot(model, euler, joint_limits=limits)


class TestRobotCollisionProperties:
    """Tests for Robot collision-related properties."""

    def test_has_collision_geometry_false_by_default(self, basic_robot):
        """Robot without collision geometry should return False."""
        assert basic_robot.has_collision_geometry is False

    def test_has_joint_limits_false_by_default(self, basic_robot):
        """Robot without joint limits should return False."""
        assert basic_robot.has_joint_limits is False

    def test_has_joint_limits_true_when_configured(self, robot_with_limits):
        """Robot with joint limits should return True."""
        assert robot_with_limits.has_joint_limits is True
        assert robot_with_limits.has_collision_geometry is False


class TestRobotCollisionMethodsWithoutGeometry:
    """Tests for collision methods when no geometry is configured."""

    def test_collides_returns_false_without_geometry(self, basic_robot):
        """collides() should return False when no collision geometry."""
        joints = (0, 0, 0, 0, 0, 0)
        assert basic_robot.collides(joints) is False

    def test_collision_details_returns_empty_without_geometry(self, basic_robot):
        """collision_details() should return empty list when no collision geometry."""
        joints = (0, 0, 0, 0, 0, 0)
        result = basic_robot.collision_details(joints)
        assert result == []

    def test_near_returns_empty_without_geometry(self, basic_robot):
        """near() should return empty list when no collision geometry."""
        joints = (0, 0, 0, 0, 0, 0)
        safety = SafetyDistances(to_environment=0.01, to_robot_default=0.01)
        result = basic_robot.near(joints, safety)
        assert result == []


class TestRobotJointsCompliant:
    """Tests for joints_compliant method."""

    def test_joints_compliant_true_without_limits(self, basic_robot):
        """joints_compliant() should return True when no limits configured."""
        joints = (0, 0, 0, 0, 0, 0)
        assert basic_robot.joints_compliant(joints) is True

    def test_joints_compliant_within_limits(self, robot_with_limits):
        """joints_compliant() should return True for joints within limits."""
        joints = (0, 0, 0, 0, 0, 0)
        assert robot_with_limits.joints_compliant(joints) is True

        joints = (100, -50, 60, 90, -45, 180)
        assert robot_with_limits.joints_compliant(joints) is True

    def test_joints_compliant_outside_limits(self, robot_with_limits):
        """joints_compliant() should return False for joints outside limits."""
        # J1 exceeds limit of 170
        joints = (180, 0, 0, 0, 0, 0)
        assert robot_with_limits.joints_compliant(joints) is False

        # J2 exceeds limit of 120
        joints = (0, 130, 0, 0, 0, 0)
        assert robot_with_limits.joints_compliant(joints) is False


class TestBatchCollisionMethods:
    """Tests for batch collision checking methods."""

    def test_batch_collides_returns_correct_shape(self, basic_robot):
        """batch_collides() should return array of correct shape."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [10, -20, 30, 0, 45, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        result = basic_robot.batch_collides(joints)
        assert result.shape == (3,)
        assert result.dtype == np.bool_

    def test_batch_collides_all_false_without_geometry(self, basic_robot):
        """batch_collides() should return all False when no collision geometry."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [10, -20, 30, 0, 45, 0],
        ])
        result = basic_robot.batch_collides(joints)
        assert np.all(result == False)

    def test_batch_collides_handles_nan(self, basic_robot):
        """batch_collides() should return False for NaN rows."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        result = basic_robot.batch_collides(joints)
        assert result[1] == False

    def test_batch_joints_compliant_returns_correct_shape(self, robot_with_limits):
        """batch_joints_compliant() should return array of correct shape."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [10, -20, 30, 0, 45, 0],
            [180, 0, 0, 0, 0, 0],  # Outside limits
        ])
        result = robot_with_limits.batch_joints_compliant(joints)
        assert result.shape == (3,)
        assert result.dtype == np.bool_

    def test_batch_joints_compliant_correct_results(self, robot_with_limits):
        """batch_joints_compliant() should correctly identify compliant joints."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],      # Within limits
            [100, -50, 60, 90, -45, 180],  # Within limits
            [180, 0, 0, 0, 0, 0],    # J1 outside limit
            [0, 130, 0, 0, 0, 0],    # J2 outside limit
        ])
        result = robot_with_limits.batch_joints_compliant(joints)
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False
        assert result[3] == False

    def test_batch_joints_compliant_all_true_without_limits(self, basic_robot):
        """batch_joints_compliant() should return all True when no limits."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [180, 130, 200, 200, 150, 400],  # Would be outside typical limits
        ])
        result = basic_robot.batch_joints_compliant(joints)
        assert np.all(result == True)

    def test_batch_joints_compliant_handles_nan(self, robot_with_limits):
        """batch_joints_compliant() should return False for NaN rows."""
        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [np.nan, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        result = robot_with_limits.batch_joints_compliant(joints)
        assert result[0] == True
        assert result[1] == False  # NaN treated as non-compliant
        assert result[2] == True


class TestCheckMode:
    """Tests for CheckMode enum."""

    def test_check_mode_values(self):
        """CheckMode should have expected values."""
        assert CheckMode.FirstCollisionOnly is not None
        assert CheckMode.AllCollisions is not None
        assert CheckMode.NoCheck is not None

    def test_check_mode_comparison(self):
        """CheckMode values should be comparable."""
        assert CheckMode.FirstCollisionOnly == CheckMode.FirstCollisionOnly
        assert CheckMode.FirstCollisionOnly != CheckMode.AllCollisions


class TestSafetyDistances:
    """Tests for SafetyDistances class."""

    def test_default_construction(self):
        """SafetyDistances should construct with defaults."""
        sd = SafetyDistances()
        assert sd.to_environment == 0.0
        assert sd.to_robot_default == 0.0
        assert sd.mode == CheckMode.FirstCollisionOnly

    def test_custom_construction(self):
        """SafetyDistances should accept custom values."""
        sd = SafetyDistances(
            to_environment=0.05,
            to_robot_default=0.02,
            mode=CheckMode.AllCollisions,
        )
        assert sd.to_environment == pytest.approx(0.05)
        assert sd.to_robot_default == pytest.approx(0.02)
        assert sd.mode == CheckMode.AllCollisions

    def test_special_distances(self):
        """SafetyDistances should handle special distances."""
        sd = SafetyDistances(
            special_distances={
                (SafetyDistances.J1, SafetyDistances.J2): SafetyDistances.NEVER_COLLIDES,
                (SafetyDistances.J5, SafetyDistances.J6): 0.01,
            }
        )
        # Check special distance is returned
        assert sd.min_distance(SafetyDistances.J1, SafetyDistances.J2) == SafetyDistances.NEVER_COLLIDES
        # Check symmetric lookup
        assert sd.min_distance(SafetyDistances.J2, SafetyDistances.J1) == SafetyDistances.NEVER_COLLIDES
        # Check default is returned for non-special pairs
        assert sd.min_distance(SafetyDistances.J3, SafetyDistances.J4) == 0.0

    def test_class_constants(self):
        """SafetyDistances should have expected class constants."""
        assert SafetyDistances.J1 == 0
        assert SafetyDistances.J2 == 1
        assert SafetyDistances.J3 == 2
        assert SafetyDistances.J4 == 3
        assert SafetyDistances.J5 == 4
        assert SafetyDistances.J6 == 5
        assert SafetyDistances.J_TOOL == 100
        assert SafetyDistances.J_BASE == 101
        assert SafetyDistances.NEVER_COLLIDES == -1.0
        assert SafetyDistances.TOUCH_ONLY == 0.0

    def test_set_special_distance(self):
        """SafetyDistances.set_special_distance should work."""
        sd = SafetyDistances()
        sd.set_special_distance(SafetyDistances.J1, SafetyDistances.J3, 0.05)
        assert sd.min_distance(SafetyDistances.J1, SafetyDistances.J3) == pytest.approx(0.05)

    def test_repr(self):
        """SafetyDistances should have a string representation."""
        sd = SafetyDistances(to_environment=0.01)
        repr_str = repr(sd)
        assert "SafetyDistances" in repr_str
        assert "0.01" in repr_str


class TestRobotKinematicsWithLimits:
    """Tests that kinematics still work correctly with limits configured."""

    def test_forward_kinematics_unchanged(self, basic_robot, robot_with_limits):
        """Forward kinematics should be the same with or without limits."""
        joints = (0, -30, 60, 0, 45, 0)
        pos1, rot1 = basic_robot.forward(joints)
        pos2, rot2 = robot_with_limits.forward(joints)

        assert np.allclose(pos1, pos2)
        assert np.allclose(rot1, rot2)

    def test_inverse_kinematics_unchanged(self, basic_robot, robot_with_limits):
        """Inverse kinematics should be the same with or without limits."""
        pose = ((0.5, 0.2, 0.3), (0, -45, 0))
        solutions1 = basic_robot.inverse(pose)
        solutions2 = robot_with_limits.inverse(pose)

        # Both should find solutions (limits are wide enough)
        assert len(solutions1) > 0
        assert len(solutions2) > 0
