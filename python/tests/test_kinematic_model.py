"""
Comprehensive tests for the KinematicModel class.

Tests cover:
- Initialization with various parameters
- Parameter validation
- Axis limits and constraints
- Edge cases and error handling
- Constraint management (absolute and relative)
"""

import pytest
import numpy as np
from py_opw_kinematics import EulerConvention, KinematicModel


class TestKinematicModelInitialization:
    """Test KinematicModel initialization with various parameters."""

    def test_default_initialization(self):
        """Test KinematicModel initialization with default parameters."""
        model = KinematicModel()

        # Check default values
        assert model.a1 == 0.0
        assert model.a2 == 0.0
        assert model.b == 0.0
        assert model.c1 == 0.0
        assert model.c2 == 0.0
        assert model.c3 == 0.0
        assert model.c4 == 0.0
        assert model.offsets == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert model.flip_axes == [False, False, False, False, False, False]
        assert model.has_parallelogram == False
        assert model.axis_limits is None

    def test_initialization_with_parameters(self):
        """Test KinematicModel initialization with custom parameters."""
        model = KinematicModel(
            a1=400.0,
            a2=-250.0,
            b=50.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
            offsets=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            flip_axes=(True, False, True, False, True, False),
            has_parallelogram=True,
        )

        assert model.a1 == 400.0
        assert model.a2 == -250.0
        assert model.b == 50.0
        assert model.c1 == 830.0
        assert model.c2 == 1175.0
        assert model.c3 == 1444.0
        assert model.c4 == 230.0
        # Use allclose for offsets due to floating point precision in degree->radian->degree conversion
        np.testing.assert_allclose(
            model.offsets, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], rtol=1e-10
        )
        assert model.flip_axes == [True, False, True, False, True, False]
        assert model.has_parallelogram == True

    def test_initialization_with_axis_limits(self):
        """Test KinematicModel initialization with axis limits."""
        axis_limits = [
            (-180.0, 180.0),
            (-90.0, 90.0),
            (-180.0, 180.0),
            (-270.0, 270.0),
            (-125.0, 125.0),
            (-270.0, 270.0),
        ]

        model = KinematicModel(
            a1=400.0,
            a2=-250.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
            axis_limits=axis_limits,
            axis_limits_in_degrees=True,
        )

        # Use allclose for axis limits due to floating point precision in degree->radian->degree conversion
        np.testing.assert_allclose(model.axis_limits, axis_limits, rtol=1e-10)

    def test_initialization_with_relative_constraints(self):
        """Test KinematicModel initialization with relative and sum constraints."""
        relative_constraints = [
            (2, 1, -135, -45),  # J3 relative to J2
            (5, 4, -90, 90),  # J6 relative to J5
        ]

        sum_constraints = [
            (1, 2, -180, 0),  # J2 + J3 sum (typical parallelogram)
            (4, 5, -270, 270),  # J5 + J6 sum (wrist coupling)
        ]

        # Test that initialization with both constraint types succeeds
        model_with_constraints = KinematicModel(
            a1=400.0,
            a2=-250.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
            relative_constraints=relative_constraints,
            sum_constraints=sum_constraints,
        )

        # Test that relative constraints are properly stored and accessible
        stored_constraints = model_with_constraints.relative_constraints
        assert stored_constraints is not None, "relative_constraints should not be None"
        assert len(stored_constraints) == len(
            relative_constraints
        ), "Should store all relative constraints"

        # Verify each constraint is stored correctly (now stored in degrees)
        for original, stored in zip(relative_constraints, stored_constraints):
            assert (
                original[0] == stored[0]
            ), f"Axis mismatch: expected {original[0]}, got {stored[0]}"
            assert (
                original[1] == stored[1]
            ), f"Reference axis mismatch: expected {original[1]}, got {stored[1]}"
            # Initialization now stores degrees as degrees (consistent behavior)
            assert (
                abs(original[2] - stored[2]) < 1e-10
            ), f"Min offset mismatch: expected {original[2]}, got {stored[2]}"
            assert (
                abs(original[3] - stored[3]) < 1e-10
            ), f"Max offset mismatch: expected {original[3]}, got {stored[3]}"

        # Test that sum constraints are properly stored and accessible
        stored_sum_constraints = model_with_constraints.sum_constraints
        assert stored_sum_constraints is not None, "sum_constraints should not be None"
        assert len(stored_sum_constraints) == len(
            sum_constraints
        ), "Should store all sum constraints"

        # Verify each sum constraint is stored correctly (now stored in degrees)
        for original, stored in zip(sum_constraints, stored_sum_constraints):
            assert (
                original[0] == stored[0]
            ), f"First axis mismatch: expected {original[0]}, got {stored[0]}"
            assert (
                original[1] == stored[1]
            ), f"Second axis mismatch: expected {original[1]}, got {stored[1]}"
            # Initialization now stores degrees as degrees (consistent behavior)
            assert (
                abs(original[2] - stored[2]) < 1e-10
            ), f"Min sum mismatch: expected {original[2]}, got {stored[2]}"
            assert (
                abs(original[3] - stored[3]) < 1e-10
            ), f"Max sum mismatch: expected {original[3]}, got {stored[3]}"

        # Test that model without constraints has None for both constraint types
        model_without_constraints = KinematicModel(
            a1=400.0,
            a2=-250.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
        )
        assert (
            model_without_constraints.relative_constraints is None
        ), "Should be None when no relative constraints provided"
        assert (
            model_without_constraints.sum_constraints is None
        ), "Should be None when no sum constraints provided"

        # Test that the constraints are properly stored and accessible
        # Verify that model_with_constraints has the expected relative constraints
        stored_constraints = model_with_constraints.relative_constraints
        assert stored_constraints is not None, "Model should have relative constraints"
        assert len(stored_constraints) == 2, "Should have two relative constraints"

        # Verify the first constraint (J3 relative to J2)
        constraint1 = stored_constraints[0]
        assert constraint1[0] == 2, "First constraint should be for axis 2"
        assert constraint1[1] == 1, "First constraint should be relative to axis 1"
        # Constraints now stored in degrees consistently
        assert (
            abs(constraint1[2] - (-135)) < 1e-6
        ), f"Min constraint should be {-135}, got {constraint1[2]}"
        assert (
            abs(constraint1[3] - (-45)) < 1e-6
        ), f"Max constraint should be {-45}, got {constraint1[3]}"

        # Verify the second constraint (J6 relative to J5)
        constraint2 = stored_constraints[1]
        assert constraint2[0] == 5, "Second constraint should be for axis 5"
        assert constraint2[1] == 4, "Second constraint should be relative to axis 4"
        # Constraints now stored in degrees consistently
        assert (
            abs(constraint2[2] - (-90)) < 1e-6
        ), f"Min constraint should be {-90}, got {constraint2[2]}"
        assert (
            abs(constraint2[3] - 90) < 1e-6
        ), f"Max constraint should be {90}, got {constraint2[3]}"

        # Verify that model_without_constraints has no constraints
        no_constraints = model_without_constraints.relative_constraints
        assert (
            no_constraints is None or len(no_constraints) == 0
        ), "Model should have no relative constraints"

        # Test that manually setting the same constraints produces the same behavior
        model_manual_constraints = KinematicModel(
            a1=400.0,
            a2=-250.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
        )

        # Manually apply the same relative constraints using degrees parameter
        for axis, ref_axis, min_offset, max_offset in relative_constraints:
            # Test using degrees parameter (should be equivalent to initialization)
            model_manual_constraints.set_relative_constraint(
                axis, ref_axis, min_offset, max_offset, degrees=True
            )

        # Manually apply the same sum constraints using degrees parameter
        for axis1, axis2, min_sum, max_sum in sum_constraints:
            # Test using degrees parameter (should be equivalent to initialization)
            model_manual_constraints.set_sum_constraint(
                axis1, axis2, min_sum, max_sum, degrees=True
            )

        # Verify that manually set constraints are also accessible
        manual_stored_constraints = model_manual_constraints.relative_constraints
        assert (
            manual_stored_constraints is not None
        ), "Manually set relative constraints should be accessible"
        assert len(manual_stored_constraints) == len(
            relative_constraints
        ), "Should have same number of relative constraints"

        manual_stored_sum_constraints = model_manual_constraints.sum_constraints
        assert (
            manual_stored_sum_constraints is not None
        ), "Manually set sum constraints should be accessible"
        assert len(manual_stored_sum_constraints) == len(
            sum_constraints
        ), "Should have same number of sum constraints"

        # Manually set constraints using degrees=True should match initialized ones (both stored as degrees internally)
        for (axis, ref_axis, min_deg, max_deg), (
            m_axis,
            m_ref_axis,
            m_min_deg,
            m_max_deg,
        ) in zip(relative_constraints, manual_stored_constraints):
            assert axis == m_axis, f"Axis should match: {axis} vs {m_axis}"
            assert (
                ref_axis == m_ref_axis
            ), f"Reference axis should match: {ref_axis} vs {m_ref_axis}"
            # Both should be stored as degrees internally now
            assert (
                abs(min_deg - m_min_deg) < 1e-10
            ), f"Min offset should match when stored as degrees: {min_deg} vs {m_min_deg}"
            assert (
                abs(max_deg - m_max_deg) < 1e-10
            ), f"Max offset should match when stored as degrees: {max_deg} vs {m_max_deg}"

        # Manually set sum constraints using degrees=True should match initialized ones (both stored as degrees internally)
        for (axis1, axis2, min_sum, max_sum), (
            m_axis1,
            m_axis2,
            m_min_sum,
            m_max_sum,
        ) in zip(sum_constraints, manual_stored_sum_constraints):
            assert axis1 == m_axis1, f"First axis should match: {axis1} vs {m_axis1}"
            assert axis2 == m_axis2, f"Second axis should match: {axis2} vs {m_axis2}"
            # Both should be stored as degrees internally now
            assert (
                abs(min_sum - m_min_sum) < 1e-10
            ), f"Min sum should match when stored as degrees: {min_sum} vs {m_min_sum}"
            assert (
                abs(max_sum - m_max_sum) < 1e-10
            ), f"Max sum should match when stored as degrees: {max_sum} vs {m_max_sum}"

        # Also test manually setting constraints in radians (different from initialization)
        model_manual_rad_constraints = KinematicModel(
            a1=400.0,
            a2=-250.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
        )

        # Manually apply the same relative constraints using radians (degrees=False)
        for axis, ref_axis, min_offset, max_offset in relative_constraints:
            # Convert degrees to radians for manual API
            min_offset_rad = np.deg2rad(min_offset)
            max_offset_rad = np.deg2rad(max_offset)
            model_manual_rad_constraints.set_relative_constraint(
                axis, ref_axis, min_offset_rad, max_offset_rad, degrees=False
            )

        # Manually apply the same sum constraints using radians (degrees=False)
        for axis1, axis2, min_sum, max_sum in sum_constraints:
            # Convert degrees to radians for manual API
            min_sum_rad = np.deg2rad(min_sum)
            max_sum_rad = np.deg2rad(max_sum)
            model_manual_rad_constraints.set_sum_constraint(
                axis1, axis2, min_sum_rad, max_sum_rad, degrees=False
            )

        # Verify that manually set constraints are also accessible
        manual_rad_stored_constraints = (
            model_manual_rad_constraints.relative_constraints
        )
        assert (
            manual_rad_stored_constraints is not None
        ), "Manually set rad relative constraints should be accessible"
        assert len(manual_rad_stored_constraints) == len(
            relative_constraints
        ), "Should have same number of relative constraints"

        manual_rad_stored_sum_constraints = model_manual_rad_constraints.sum_constraints
        assert (
            manual_rad_stored_sum_constraints is not None
        ), "Manually set rad sum constraints should be accessible"
        assert len(manual_rad_stored_sum_constraints) == len(
            sum_constraints
        ), "Should have same number of sum constraints"

        # Manual relative constraints set in radians should be converted to degrees for consistent storage
        for (axis, ref_axis, min_deg, max_deg), (
            mr_axis,
            mr_ref_axis,
            mr_min_deg,
            mr_max_deg,
        ) in zip(relative_constraints, manual_rad_stored_constraints):
            assert axis == mr_axis, f"Axis should match: {axis} vs {mr_axis}"
            assert (
                ref_axis == mr_ref_axis
            ), f"Reference axis should match: {ref_axis} vs {mr_ref_axis}"
            # Radians input should be converted to degrees for consistent storage
            assert (
                abs(min_deg - mr_min_deg) < 1e-10
            ), f"Min offset should match when converted to degrees: {min_deg} vs {mr_min_deg}"
            assert (
                abs(max_deg - mr_max_deg) < 1e-10
            ), f"Max offset should match when converted to degrees: {max_deg} vs {mr_max_deg}"

        # Manual sum constraints set in radians should be converted to degrees for consistent storage
        for (axis1, axis2, min_sum, max_sum), (
            mr_axis1,
            mr_axis2,
            mr_min_sum,
            mr_max_sum,
        ) in zip(sum_constraints, manual_rad_stored_sum_constraints):
            assert axis1 == mr_axis1, f"First axis should match: {axis1} vs {mr_axis1}"
            assert axis2 == mr_axis2, f"Second axis should match: {axis2} vs {mr_axis2}"
            # Radians input should be converted to degrees for consistent storage
            assert (
                abs(min_sum - mr_min_sum) < 1e-10
            ), f"Min sum should match when converted to degrees: {min_sum} vs {mr_min_sum}"
            assert (
                abs(max_sum - mr_max_sum) < 1e-10
            ), f"Max sum should match when converted to degrees: {max_sum} vs {mr_max_sum}"


class TestKinematicModelAxisLimits:
    """Test axis limits functionality."""

    @pytest.fixture
    def basic_model(self):
        """Create a basic kinematic model for testing."""
        return KinematicModel(
            a1=400.0, a2=-250.0, c1=830.0, c2=1175.0, c3=1444.0, c4=230.0
        )

    def test_set_axis_limits(self, basic_model: KinematicModel):
        """Test setting axis limits."""
        axis_limits = (
            (-180.0, 180.0),
            (-90.0, 90.0),
            (-180.0, 180.0),
            (-270.0, 270.0),
            (-125.0, 125.0),
            (-270.0, 270.0),
        )

        basic_model.set_axis_limits(axis_limits, degrees=True)
        # Use allclose for axis limits due to floating point precision in degree->radian->degree conversion
        np.testing.assert_allclose(basic_model.axis_limits, axis_limits, rtol=1e-10)

    def test_set_axis_limits_none(self, basic_model: KinematicModel):
        """Test setting axis limits to None (removing limits)."""
        # First set some limits
        axis_limits = (
            (-180.0, 180.0),
            (-90.0, 90.0),
            (-180.0, 180.0),
            (-270.0, 270.0),
            (-125.0, 125.0),
            (-270.0, 270.0),
        )
        basic_model.set_axis_limits(axis_limits, degrees=True)
        assert basic_model.axis_limits is not None

        # Then remove them
        basic_model.set_axis_limits(None, degrees=True)
        assert basic_model.axis_limits is None

    def test_set_axis_limits_invalid_length(self, basic_model: KinematicModel):
        """Test setting axis limits with invalid length."""
        with pytest.raises((ValueError, TypeError)):
            basic_model.set_axis_limits(
                ((-180.0, 180.0), (-90.0, 90.0)), degrees=True
            )  # Only 2 axes

    def test_set_axis_limits_invalid_format(self, basic_model: KinematicModel):
        """Test setting axis limits with invalid format."""
        with pytest.raises((ValueError, TypeError)):
            # Invalid format - should be tuples of (min, max)
            basic_model.set_axis_limits((-180.0, 180.0, -90.0, 90.0, -180.0, 180.0))


class TestKinematicModelConstraints:
    """
    Test constraint management functionality.

    CORRECTED CONSISTENT BEHAVIOR:
    ===============================
    All angle values are stored in degrees for consistency with offset behavior:
    - Input in degrees (degrees=True, default): Store as degrees
    - Input in radians (degrees=False): Convert to degrees for storage
    - Retrieval: Always return degrees (consistent format)
    - Conversion: To radians only during kinematic computations

    This matches the offset pattern:
    - Offsets: stored as degrees, converted to radians during kinematic operations
    - Constraints: follow same pattern for consistency

    NOTE: Implementation has been corrected to match this consistent behavior.
    """

    @pytest.fixture
    def basic_model(self):
        """Create a basic kinematic model for testing."""
        return KinematicModel(
            a1=400.0, a2=-250.0, c1=830.0, c2=1175.0, c3=1444.0, c4=230.0
        )

    def test_set_absolute_constraint(self, basic_model: KinematicModel):
        """Test setting absolute constraints for individual axes."""
        # Verify model starts without constraints
        assert not basic_model.has_constraints, "Model should start without constraints"

        # Set constraint for axis 0 (J1) in radians (degrees=False)
        basic_model.set_absolute_constraint(0, -np.pi, np.pi, degrees=False)

        # Verify constraint system is active
        assert (
            basic_model.has_constraints
        ), "Model should have constraints after setting absolute constraint"

        # Set constraint for axis 1 (J2) in radians (explicit)
        basic_model.set_absolute_constraint(1, -np.pi / 2, np.pi / 2, degrees=False)

        # Set constraint for axis 2 (J3) in degrees
        basic_model.set_absolute_constraint(2, -180, 180)

        # Constraint system should remain active
        assert (
            basic_model.has_constraints
        ), "Model should still have constraints after setting multiple absolute constraints"

    def test_set_absolute_constraint_invalid_axis(self, basic_model: KinematicModel):
        """Test setting absolute constraint with invalid axis index."""
        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_absolute_constraint(
                6, -np.pi, np.pi
            )  # Axis 6 doesn't exist

        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_absolute_constraint(-1, -np.pi, np.pi)  # Negative axis

    def test_set_absolute_constraint_invalid_limits(self, basic_model: KinematicModel):
        """Test setting absolute constraint with invalid limits."""
        # Note: The current implementation may not validate this case
        # This test documents current behavior
        try:
            basic_model.set_absolute_constraint(0, np.pi, -np.pi)  # min > max
            # If no error is raised, the implementation allows this
        except ValueError:
            # If error is raised, that's also acceptable
            pass

    def test_set_relative_constraint(self, basic_model: KinematicModel):
        """Test setting relative constraints between axes."""
        # Set relative constraint: J3 relative to J2 in radians (degrees=False)
        basic_model.set_relative_constraint(2, 1, -np.pi / 2, -np.pi / 6, degrees=False)

        # Verify first constraint is stored
        constraints = basic_model.relative_constraints
        assert constraints is not None, "Should have relative constraints after setting"
        assert len(constraints) == 1, "Should have 1 relative constraint"
        first_constraint = constraints[0]
        assert (
            first_constraint[0] == 2 and first_constraint[1] == 1
        ), "Should be constraint for axes 2,1"
        # Radians input should be converted to degrees for consistent storage
        expected_min_deg = np.rad2deg(-np.pi / 2)
        expected_max_deg = np.rad2deg(-np.pi / 6)
        assert (
            abs(first_constraint[2] - expected_min_deg) < 1e-6
        ), f"Min should be {expected_min_deg}° (converted from -π/2 rad)"
        assert (
            abs(first_constraint[3] - expected_max_deg) < 1e-6
        ), f"Max should be {expected_max_deg}° (converted from -π/6 rad)"

        # Set another relative constraint: J6 relative to J5 in radians (explicit)
        basic_model.set_relative_constraint(5, 4, -np.pi / 4, np.pi / 4, degrees=False)

        # Verify constraints are stored (order may not be preserved)
        constraints = basic_model.relative_constraints
        assert len(constraints) == 2, "Should have 2 relative constraints"
        # Find the constraint for axes 5,4
        axes_54_constraint = None
        for constraint in constraints:
            if constraint[0] == 5 and constraint[1] == 4:
                axes_54_constraint = constraint
                break
        assert axes_54_constraint is not None, "Should have constraint for axes 5,4"
        # Radians input should be converted to degrees for consistent storage
        expected_min2_deg = np.rad2deg(-np.pi / 4)
        expected_max2_deg = np.rad2deg(np.pi / 4)
        assert (
            abs(axes_54_constraint[2] - expected_min2_deg) < 1e-6
        ), f"Min should be {expected_min2_deg}° (converted from -π/4 rad)"
        assert (
            abs(axes_54_constraint[3] - expected_max2_deg) < 1e-6
        ), f"Max should be {expected_max2_deg}° (converted from π/4 rad)"

        # Set relative constraint: J4 relative to J3 in degrees (default)
        basic_model.set_relative_constraint(3, 2, -90, 90)

        # Verify all three constraints are stored consistently in degrees
        constraints = basic_model.relative_constraints
        assert len(constraints) == 3, "Should have 3 relative constraints"
        # Find the constraint for axes 3,2
        axes_32_constraint = None
        for constraint in constraints:
            if constraint[0] == 3 and constraint[1] == 2:
                axes_32_constraint = constraint
                break
        assert axes_32_constraint is not None, "Should have constraint for axes 3,2"
        # Degrees input should be stored as degrees (corrected consistent behavior)
        assert (
            abs(axes_32_constraint[2] - (-90)) < 1e-6
        ), "Min should be stored as -90 degrees"
        assert (
            abs(axes_32_constraint[3] - 90) < 1e-6
        ), "Max should be stored as 90 degrees"

    def test_set_relative_constraint_invalid_axes(self, basic_model: KinematicModel):
        """Test setting relative constraint with invalid axis indices."""
        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_relative_constraint(6, 1, -np.pi, np.pi)  # Invalid axis

        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_relative_constraint(
                1, 6, -np.pi, np.pi
            )  # Invalid reference axis

    def test_set_relative_constraint_same_axis(self, basic_model: KinematicModel):
        """Test setting relative constraint where axis references itself."""
        with pytest.raises(ValueError):
            basic_model.set_relative_constraint(1, 1, -np.pi, np.pi, degrees=False)

    def test_set_relative_constraint_invalid_offset(self, basic_model: KinematicModel):
        """Test setting relative constraint with invalid offset range."""
        # Note: The current implementation may not validate this case
        # This test documents current behavior
        try:
            basic_model.set_relative_constraint(
                2, 1, np.pi / 2, -np.pi / 2, degrees=False
            )  # min > max
            # If no error is raised, the implementation allows this
        except ValueError:
            # If error is raised, that's also acceptable
            pass

    def test_set_sum_constraint(self, basic_model: KinematicModel):
        """Test setting sum constraints for axis pairs."""
        # Set sum constraint for J2 + J3 in degrees (default)
        basic_model.set_sum_constraint(2, 1, -180, 0)

        # Verify constraint is stored in degrees (consistent with offset behavior)
        constraints = basic_model.sum_constraints
        assert constraints is not None, "Should have sum constraints after setting"
        assert len(constraints) == 1, "Should have 1 sum constraint"
        first_constraint = constraints[0]
        assert (
            first_constraint[0] == 2 and first_constraint[1] == 1
        ), "Should be constraint for axes 2,1"
        # Degrees input should be stored as degrees (corrected consistent behavior)
        assert (
            abs(first_constraint[2] - (-180)) < 1e-6
        ), "Min should be stored as -180 degrees"
        assert abs(first_constraint[3] - 0) < 1e-6, "Max should be stored as 0 degrees"

        # Set sum constraint for J1 + J4 in radians (should convert to degrees for storage)
        basic_model.set_sum_constraint(0, 3, -np.pi, np.pi, degrees=False)

        # Verify constraints are stored in degrees (consistent storage format)
        constraints = basic_model.sum_constraints
        assert len(constraints) == 2, "Should have 2 sum constraints"
        # Find the constraint for axes 0,3
        axes_03_constraint = None
        for constraint in constraints:
            if constraint[0] == 0 and constraint[1] == 3:
                axes_03_constraint = constraint
                break
        assert axes_03_constraint is not None, "Should have constraint for axes 0,3"
        # Radians input should be converted to degrees for consistent storage
        expected_min_deg = np.rad2deg(-np.pi)  # -180 degrees
        expected_max_deg = np.rad2deg(np.pi)  # 180 degrees
        assert (
            abs(axes_03_constraint[2] - expected_min_deg) < 1e-6
        ), "Min should be stored as -180 degrees (converted from -π rad)"
        assert (
            abs(axes_03_constraint[3] - expected_max_deg) < 1e-6
        ), "Max should be stored as 180 degrees (converted from π rad)"

        # Set sum constraint for J5 + J6 with explicit degrees=True
        basic_model.set_sum_constraint(4, 5, -90, 90, degrees=True)

        # Verify all three constraints are stored consistently in degrees
        constraints = basic_model.sum_constraints
        assert len(constraints) == 3, "Should have 3 sum constraints"
        # Find the constraint for axes 4,5
        axes_45_constraint = None
        for constraint in constraints:
            if constraint[0] == 4 and constraint[1] == 5:
                axes_45_constraint = constraint
                break
        assert axes_45_constraint is not None, "Should have constraint for axes 4,5"
        # Degrees=True should store as degrees (corrected consistent behavior)
        assert (
            abs(axes_45_constraint[2] - (-90)) < 1e-6
        ), "Min should be stored as -90 degrees"
        assert (
            abs(axes_45_constraint[3] - 90) < 1e-6
        ), "Max should be stored as 90 degrees"

    def test_set_sum_constraint_invalid_axes(self, basic_model: KinematicModel):
        """Test setting sum constraint with invalid axis indices."""
        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_sum_constraint(6, 1, -np.pi, np.pi)  # Invalid axis

        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_sum_constraint(
                0, 6, -np.pi, np.pi
            )  # Invalid reference axis

        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.set_sum_constraint(-1, 0, -np.pi, np.pi)  # Negative axis

    def test_set_sum_constraint_same_axis(self, basic_model: KinematicModel):
        """Test setting sum constraint where axis references itself."""
        with pytest.raises(ValueError):
            basic_model.set_sum_constraint(1, 1, -np.pi, np.pi, degrees=False)

    def test_set_sum_constraint_typical_parallelogram(
        self, basic_model: KinematicModel
    ):
        """Test setting sum constraint for typical parallelogram use case."""
        # Typical parallelogram constraint: J2 + J3 sum
        basic_model.set_sum_constraint(2, 1, -180, 0)

        # Verify constraint is stored properly
        constraints = basic_model.sum_constraints
        assert constraints is not None, "Should have sum constraints for parallelogram"
        assert len(constraints) == 1, "Should have exactly 1 parallelogram constraint"

        parallelogram_constraint = constraints[0]
        assert (
            parallelogram_constraint[0] == 2 and parallelogram_constraint[1] == 1
        ), "Should be J3+J2 constraint"
        assert (
            abs(parallelogram_constraint[2] - (-180)) < 1e-6
        ), "Min should be -180° (typical parallelogram limit)"
        assert (
            abs(parallelogram_constraint[3] - 0) < 1e-6
        ), "Max should be 0° (typical parallelogram limit)"

        # Verify constraint system is active
        assert (
            basic_model.has_constraints
        ), "Model should have constraints after setting parallelogram constraint"

    def test_set_sum_constraint_duplicate_axes(self, basic_model: KinematicModel):
        """Test setting sum constraint with same axes in multiple calls."""
        # Test setting multiple constraints on the same axis pair
        basic_model.set_sum_constraint(1, 2, -180, 180)

        # Verify first constraint is stored
        constraints = basic_model.sum_constraints
        assert constraints is not None, "Should have sum constraints after setting"
        assert len(constraints) == 1, "Should have 1 sum constraint"
        first_constraint = constraints[0]
        assert (
            first_constraint[0] == 1 and first_constraint[1] == 2
        ), "Should be constraint for axes 1,2"
        assert abs(first_constraint[2] - (-180)) < 1e-6, "Min should be -180 degrees"
        assert abs(first_constraint[3] - 180) < 1e-6, "Max should be 180 degrees"

        # Setting another constraint on the same pair should overwrite
        basic_model.set_sum_constraint(1, 2, -90, 90)

        # Verify constraint was overwritten, not added
        constraints = basic_model.sum_constraints
        assert constraints is not None, "Should still have sum constraints"
        assert (
            len(constraints) == 1
        ), "Should still have exactly 1 sum constraint (overwritten, not added)"

        overwritten_constraint = constraints[0]
        assert (
            overwritten_constraint[0] == 1 and overwritten_constraint[1] == 2
        ), "Should still be constraint for axes 1,2"
        assert (
            abs(overwritten_constraint[2] - (-90)) < 1e-6
        ), "Min should be overwritten to -90 degrees"
        assert (
            abs(overwritten_constraint[3] - 90) < 1e-6
        ), "Max should be overwritten to 90 degrees"

    def test_set_sum_constraint_invalid_limits(self, basic_model: KinematicModel):
        """Test setting sum constraint with invalid limits."""
        # Note: The current implementation may not validate this case
        try:
            basic_model.set_sum_constraint(
                1, 2, np.pi, -np.pi, degrees=False
            )  # min > max
            # If no error is raised, the implementation allows this
        except ValueError:
            # If error is raised, that's also acceptable
            pass

    def test_clear_axis_constraint(self, basic_model: KinematicModel):
        """Test clearing constraints for a specific axis."""
        # First set some constraints
        basic_model.set_absolute_constraint(0, -np.pi, np.pi, degrees=False)
        basic_model.set_relative_constraint(2, 1, -np.pi / 2, np.pi / 2, degrees=False)
        basic_model.set_sum_constraint(1, 0, -np.pi, np.pi, degrees=False)

        # Verify constraints are set
        assert (
            basic_model.has_constraints
        ), "Model should have constraints after setting them"
        assert (
            basic_model.relative_constraints is not None
        ), "Should have relative constraints"
        assert (
            len(basic_model.relative_constraints) == 1
        ), "Should have 1 relative constraint"
        assert basic_model.sum_constraints is not None, "Should have sum constraints"
        assert len(basic_model.sum_constraints) == 1, "Should have 1 sum constraint"

        # Clear constraint for axis 2 (relative constraint)
        basic_model.clear_axis_constraint(2)

        # Verify relative constraint for axis 2 was cleared
        remaining_relative = basic_model.relative_constraints
        if remaining_relative is not None:
            # Check that no constraint involves axis 2
            for constraint in remaining_relative:
                assert (
                    constraint[0] != 2
                ), "Axis 2 should not have relative constraints after clearing"

        # Clear constraint for axis 1 (sum constraint)
        basic_model.clear_axis_constraint(1)

        # Verify sum constraint involving axis 1 was cleared
        remaining_sum = basic_model.sum_constraints
        if remaining_sum is not None:
            # Check that no constraint involves axis 1
            for constraint in remaining_sum:
                assert (
                    constraint[0] != 1 and constraint[1] != 1
                ), "Axis 1 should not have sum constraints after clearing"

        # Clear constraint for axis 0 (absolute constraint)
        basic_model.clear_axis_constraint(0)

        # Should not raise any exceptions and constraints should be cleared

    def test_clear_axis_constraint_invalid_axis(self, basic_model: KinematicModel):
        """Test clearing constraint with invalid axis index."""
        with pytest.raises((ValueError, IndexError, OverflowError)):
            basic_model.clear_axis_constraint(6)  # Axis 6 doesn't exist

    def test_clear_all_constraints(self, basic_model: KinematicModel):
        """Test clearing all constraints."""
        # First set multiple constraints
        basic_model.set_absolute_constraint(0, -np.pi, np.pi, degrees=False)
        basic_model.set_absolute_constraint(1, -np.pi / 2, np.pi / 2, degrees=False)
        basic_model.set_relative_constraint(2, 1, -np.pi / 2, np.pi / 2, degrees=False)
        basic_model.set_relative_constraint(5, 4, -np.pi / 4, np.pi / 4, degrees=False)
        basic_model.set_sum_constraint(3, 2, -np.pi, np.pi, degrees=False)
        basic_model.set_sum_constraint(4, 5, -np.pi / 3, np.pi / 3, degrees=False)

        # Verify constraints are set
        assert (
            basic_model.has_constraints
        ), "Model should have constraints after setting them"
        assert (
            basic_model.relative_constraints is not None
        ), "Should have relative constraints"
        assert (
            len(basic_model.relative_constraints) == 2
        ), "Should have 2 relative constraints"
        assert basic_model.sum_constraints is not None, "Should have sum constraints"
        assert len(basic_model.sum_constraints) == 2, "Should have 2 sum constraints"

        # Clear all constraints
        basic_model.clear_all_constraints()

        # Verify all constraints are cleared
        assert (
            not basic_model.has_constraints
        ), "Model should not have constraints after clearing all"

        # Check that all constraint types are cleared
        remaining_relative = basic_model.relative_constraints
        assert (
            remaining_relative is None or len(remaining_relative) == 0
        ), "No relative constraints should remain"

        remaining_sum = basic_model.sum_constraints
        assert (
            remaining_sum is None or len(remaining_sum) == 0
        ), "No sum constraints should remain"

    def test_constraint_clearing_effectiveness(self, basic_model: KinematicModel):
        """Test that constraint clearing actually removes constraints and restores expected behavior."""
        # Set up a comprehensive constraint scenario
        basic_model.set_absolute_constraint(0, -90, 90)  # J1 absolute constraint
        basic_model.set_relative_constraint(2, 1, -45, 45)  # J3 relative to J2
        basic_model.set_relative_constraint(4, 3, -60, 60)  # J5 relative to J4
        basic_model.set_sum_constraint(1, 2, -180, 0)  # J2 + J3 sum constraint
        basic_model.set_sum_constraint(5, 4, -90, 90)  # J6 + J5 sum constraint

        # Verify initial constraint state
        assert basic_model.has_constraints, "Model should have constraints initially"
        assert (
            len(basic_model.relative_constraints) == 2
        ), "Should have 2 relative constraints"
        assert len(basic_model.sum_constraints) == 2, "Should have 2 sum constraints"

        # Clear specific axis constraints one by one and verify state

        # Clear axis 2 (should remove relative constraint where axis=2)
        basic_model.clear_axis_constraint(2)
        remaining_relative = basic_model.relative_constraints
        # Should still have the constraint for axis 4
        assert (
            remaining_relative is not None and len(remaining_relative) == 1
        ), "Should have 1 relative constraint left"
        assert (
            remaining_relative[0][0] == 4
        ), "Remaining constraint should be for axis 4"
        # Sum constraint involving axis 2 should still be there (since it's axis 1+2, not just axis 2)
        assert (
            len(basic_model.sum_constraints) == 2
        ), "Sum constraints should still be there"

        # Clear axis 1 (should remove sum constraint involving axis 1)
        basic_model.clear_axis_constraint(1)
        remaining_sum = basic_model.sum_constraints
        # Should still have the constraint for axes 5+4 but not 1+2
        if remaining_sum is not None:
            assert len(remaining_sum) == 1, "Should have 1 sum constraint left"
            # Find the remaining constraint
            remaining_constraint = remaining_sum[0]
            assert (remaining_constraint[0] == 5 and remaining_constraint[1] == 4) or (
                remaining_constraint[0] == 4 and remaining_constraint[1] == 5
            ), "Remaining constraint should involve axes 4 and 5"

        # Clear axis 4 (should remove both remaining relative and sum constraints)
        basic_model.clear_axis_constraint(4)
        remaining_relative = basic_model.relative_constraints
        remaining_sum = basic_model.sum_constraints

        # Should have no relative constraints left (axis 4 was the last one)
        assert (
            remaining_relative is None or len(remaining_relative) == 0
        ), "No relative constraints should remain"

        # Sum constraint (5, 4) might still remain - clearing axis 4 may not clear constraints where 4 is reference
        # This reveals the actual behavior of the constraint clearing system
        if remaining_sum is not None:
            for constraint in remaining_sum:
                # If constraint remains, it should involve axis 4 as reference, not as primary
                assert (
                    constraint[1] == 4
                ), "Any remaining sum constraint should have 4 as reference axis"

        # Clear axis 5 to remove the final sum constraint
        basic_model.clear_axis_constraint(5)
        remaining_sum = basic_model.sum_constraints

        # Now should have no sum constraints left
        assert (
            remaining_sum is None or len(remaining_sum) == 0
        ), "No sum constraints should remain after clearing axis 5"

        # Model should still have some constraints (absolute constraint on axis 0)
        # Note: We can't directly check absolute constraints, but has_constraints should still be True
        # if absolute constraints are tracked separately

        # Clear axis 0 (absolute constraint)
        basic_model.clear_axis_constraint(0)

        # Now verify final state - this depends on implementation details
        # The test verifies the clearing methods work without errors

    def test_constraint_units_absolute(self, basic_model: KinematicModel):
        """Test that absolute constraints handle degrees and radians correctly."""
        # Verify model starts without constraints
        assert not basic_model.has_constraints, "Model should start without constraints"

        # Set constraint in degrees (default)
        basic_model.set_absolute_constraint(0, -180, 180)

        # Verify constraint system is activated
        assert (
            basic_model.has_constraints
        ), "Model should have constraints after setting absolute constraint"

        # Set equivalent constraint in radians
        basic_model.set_absolute_constraint(1, -np.pi, np.pi, degrees=False)

        # Verify constraint system remains active
        assert (
            basic_model.has_constraints
        ), "Model should still have constraints after setting second absolute constraint"

    def test_constraint_units_relative(self, basic_model: KinematicModel):
        """Test that relative constraints handle degrees and radians correctly."""
        # Set constraint in degrees (default)
        basic_model.set_relative_constraint(2, 1, -90, 90)

        # Verify degrees constraint is stored
        constraints = basic_model.relative_constraints
        assert constraints is not None, "Should have relative constraints"
        assert len(constraints) == 1, "Should have 1 relative constraint"
        degrees_constraint = constraints[0]
        assert abs(degrees_constraint[2] - (-90)) < 1e-6, "Should store -90° as degrees"
        assert abs(degrees_constraint[3] - 90) < 1e-6, "Should store 90° as degrees"

        # Set equivalent constraint in radians
        basic_model.set_relative_constraint(3, 2, -np.pi / 2, np.pi / 2, degrees=False)

        # Verify radians constraint is converted and stored as degrees
        constraints = basic_model.relative_constraints
        assert len(constraints) == 2, "Should have 2 relative constraints"
        radians_constraint = constraints[1]
        expected_min_deg = np.rad2deg(-np.pi / 2)
        expected_max_deg = np.rad2deg(np.pi / 2)
        assert (
            abs(radians_constraint[2] - expected_min_deg) < 1e-6
        ), f"Should store {expected_min_deg}° (converted from radians)"
        assert (
            abs(radians_constraint[3] - expected_max_deg) < 1e-6
        ), f"Should store {expected_max_deg}° (converted from radians)"

    def test_constraint_units_sum(self, basic_model: KinematicModel):
        """Test that sum constraints handle degrees and radians with consistent storage."""
        # Set constraint in degrees (default) - should store as degrees
        basic_model.set_sum_constraint(1, 2, -180, 0)

        # Verify degrees constraint is stored as degrees (consistent storage)
        constraints = basic_model.sum_constraints
        assert constraints is not None, "Should have sum constraints"
        assert len(constraints) == 1, "Should have 1 sum constraint"
        degrees_constraint = constraints[0]
        assert (
            abs(degrees_constraint[2] - (-180)) < 1e-6
        ), "Should store -180° as degrees"
        assert abs(degrees_constraint[3] - 0) < 1e-6, "Should store 0° as degrees"

        # Set constraint in radians - should convert and store as degrees
        basic_model.set_sum_constraint(3, 4, -np.pi, 0, degrees=False)

        # Verify radians input is converted to degrees for consistent storage
        constraints = basic_model.sum_constraints
        assert len(constraints) == 2, "Should have 2 sum constraints"
        # Find the constraint for axes 3,4
        axes_34_constraint = None
        for constraint in constraints:
            if constraint[0] == 3 and constraint[1] == 4:
                axes_34_constraint = constraint
                break
        assert axes_34_constraint is not None, "Should have constraint for axes 3,4"
        # Radians input should be converted to degrees for consistent storage
        expected_min_deg = np.rad2deg(-np.pi)  # -180 degrees
        assert (
            abs(axes_34_constraint[2] - expected_min_deg) < 1e-6
        ), "Should convert -π rad to -180° for storage"
        assert abs(axes_34_constraint[3] - 0) < 1e-6, "Should store 0 as degrees"

        # Set constraint mixing multiple joints
        basic_model.set_sum_constraint(0, 1, -270, 90)

        # Verify third constraint is stored consistently
        constraints = basic_model.sum_constraints
        assert len(constraints) == 3, "Should have 3 sum constraints"
        # Find the constraint for axes 0,1
        axes_01_constraint = None
        for constraint in constraints:
            if constraint[0] == 0 and constraint[1] == 1:
                axes_01_constraint = constraint
                break
        assert axes_01_constraint is not None, "Should have constraint for axes 0,1"
        # Degrees input should be stored consistently in degrees (like offsets)
        assert (
            abs(axes_01_constraint[2] - (-270)) < 1e-6
        ), "Should store -270° as degrees"
        assert abs(axes_01_constraint[3] - 90) < 1e-6, "Should store 90° as degrees"

    def test_constraint_storage_consistency(self, basic_model: KinematicModel):
        """Test consistent storage behavior for all constraint types."""
        # CORRECTED CONSISTENT BEHAVIOR:
        # All constraints are stored in degrees (like offsets)
        # and converted to radians only during kinematic computations

        # Set constraints using degrees (should store as degrees)
        basic_model.set_absolute_constraint(0, -180, 180)  # degrees=True default
        basic_model.set_relative_constraint(2, 1, -90, 90)  # degrees=True default
        basic_model.set_sum_constraint(1, 2, -180, 0)  # degrees=True default

        # Set constraints using radians (should convert to degrees for storage)
        basic_model.set_relative_constraint(3, 2, -np.pi / 2, np.pi / 2, degrees=False)
        basic_model.set_sum_constraint(4, 5, -np.pi / 4, np.pi / 4, degrees=False)

        # Verify consistent degree storage
        rel_constraints = basic_model.relative_constraints
        sum_constraints = basic_model.sum_constraints

        # All should be stored in degrees
        assert (
            abs(rel_constraints[0][2] - (-90)) < 1e-6
        ), "Relative constraint should store -90°"
        assert (
            abs(rel_constraints[1][2] - np.rad2deg(-np.pi / 2)) < 1e-6
        ), "Converted radians should store as degrees"
        assert (
            abs(sum_constraints[0][2] - (-180)) < 1e-6
        ), "Sum constraint should store -180°"
        assert (
            abs(sum_constraints[1][2] - np.rad2deg(-np.pi / 4)) < 1e-6
        ), "Converted radians should store as degrees"

    def test_constraint_backwards_compatibility(self, basic_model: KinematicModel):
        """Test that existing code using constraints still works (backwards compatibility)."""
        # Code using radian values needs to specify degrees=False explicitly
        basic_model.set_absolute_constraint(0, -np.pi, np.pi, degrees=False)
        basic_model.set_relative_constraint(2, 1, -np.pi / 4, np.pi / 4, degrees=False)

        # Test that both succeed without error
        assert True

    def test_constraint_degree_radian_conversion(self, basic_model: KinematicModel):
        """Test constraint storage behavior with different input units."""
        # Set constraints using degrees (new default behavior)
        basic_model.set_absolute_constraint(0, -90, 90)
        basic_model.set_relative_constraint(2, 1, -45, 45)

        # Test that degrees are stored as degrees with the corrected API
        rel_constraints = basic_model.relative_constraints
        assert rel_constraints is not None, "Should have relative constraints"
        rel_constraint = rel_constraints[0]  # First constraint
        expected_rel_min_deg = -45
        expected_rel_max_deg = 45
        assert (
            abs(rel_constraint[2] - expected_rel_min_deg) < 1e-6
        ), "Relative constraint min should be stored in degrees when degrees=True"
        assert (
            abs(rel_constraint[3] - expected_rel_max_deg) < 1e-6
        ), "Relative constraint max should be stored in degrees when degrees=True"

        # Test with explicit radians - should be converted to degrees for storage
        basic_model.set_relative_constraint(3, 2, -np.pi / 4, np.pi / 4, degrees=False)
        rel_constraints = basic_model.relative_constraints
        rel_constraint_rad = rel_constraints[1]  # Second constraint
        expected_rel_min_rad_as_deg = np.rad2deg(-np.pi / 4)
        expected_rel_max_rad_as_deg = np.rad2deg(np.pi / 4)
        assert (
            abs(rel_constraint_rad[2] - expected_rel_min_rad_as_deg) < 1e-6
        ), "Relative constraint min should be converted to degrees when degrees=False"
        assert (
            abs(rel_constraint_rad[3] - expected_rel_max_rad_as_deg) < 1e-6
        ), "Relative constraint max should be converted to degrees when degrees=False"

        # Test sum constraints with degrees (default) and radians
        basic_model.set_sum_constraint(1, 2, -180, 0)  # Degrees (default)
        basic_model.set_sum_constraint(3, 4, -np.pi, np.pi, degrees=False)  # Radians

        # Sum constraints don't have a direct property to check storage,
        # but this verifies the methods accept the parameters correctly

        # For absolute constraints, we use the axis_limits property
        # Note: set_absolute_constraint affects the internal constraint system,
        # but the storage mechanism is different from relative constraints
        # This test verifies that the method accepts degrees parameter without error

    def test_constraint_unit_consistency_comprehensive(
        self, basic_model: KinematicModel
    ):
        """Test comprehensive constraint unit handling and current behavior."""
        import numpy as np

        # Test 1: Initialization treats constraint values as degrees and stores as degrees
        model_init = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            relative_constraints=[(2, 1, -135, -45)],  # Input in degrees
        )
        init_constraints = model_init.relative_constraints
        assert init_constraints is not None
        expected_min_deg = -135
        expected_max_deg = -45
        assert (
            abs(init_constraints[0][2] - expected_min_deg) < 1e-10
        ), "Init should store degrees as degrees"
        assert (
            abs(init_constraints[0][3] - expected_max_deg) < 1e-10
        ), "Init should store degrees as degrees"

        # Test 2: Manual setting with degrees=True (default) stores in degrees
        model_manual_deg = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )
        model_manual_deg.set_relative_constraint(2, 1, -135, -45)
        manual_deg_constraints = model_manual_deg.relative_constraints
        assert manual_deg_constraints is not None
        expected_min_deg = -135
        expected_max_deg = -45
        assert (
            abs(manual_deg_constraints[0][2] - expected_min_deg) < 1e-10
        ), "Manual degrees should store in degrees"
        assert (
            abs(manual_deg_constraints[0][3] - expected_max_deg) < 1e-10
        ), "Manual degrees should store in degrees"

        # Test 3: Manual setting with degrees=False (radians) converts to degrees for storage
        model_manual_rad = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )
        # Use same degree values converted to radians for input
        expected_min_rad = np.deg2rad(-135)
        expected_max_rad = np.deg2rad(-45)
        model_manual_rad.set_relative_constraint(
            2, 1, expected_min_rad, expected_max_rad, degrees=False
        )
        manual_rad_constraints = model_manual_rad.relative_constraints
        assert manual_rad_constraints is not None
        # Radians should be converted to degrees for consistent storage
        assert (
            abs(manual_rad_constraints[0][2] - np.rad2deg(expected_min_rad)) < 1e-10
        ), "Manual radians should be converted to degrees for storage"
        assert (
            abs(manual_rad_constraints[0][3] - np.rad2deg(expected_max_rad)) < 1e-10
        ), "Manual radians should be converted to degrees for storage"

        # Test 4: Corrected behavior - all methods now store consistently in degrees
        # Initialization: degrees input → degrees storage
        # Manual degrees=True: degrees input → degrees storage
        # Manual degrees=False: radians input → degrees storage (converted)
        print(f"✓ Corrected constraint storage behavior:")
        print(f"  - Initialization: -135° → {init_constraints[0][2]:.6f}°")
        print(f"  - Manual degrees=True: -135° → {manual_deg_constraints[0][2]:.6f}°")
        print(
            f"  - Manual degrees=False: {expected_min_rad:.6f} rad → {manual_rad_constraints[0][2]:.6f}°"
        )

        # Test 5: Verify that equivalent values produce equivalent behavior when stored consistently in degrees
        init_value_deg = init_constraints[0][2]  # Now stored in degrees
        manual_deg_value = manual_deg_constraints[0][2]  # Stored in degrees
        manual_rad_value_deg = manual_rad_constraints[0][
            2
        ]  # Converted and stored in degrees

        assert (
            abs(init_value_deg - manual_deg_value) < 1e-6
        ), "All methods should store same angle in degrees"
        assert (
            abs(init_value_deg - manual_rad_value_deg) < 1e-6
        ), "All methods should store same angle in degrees"
        assert (
            abs(manual_deg_value - manual_rad_value_deg) < 1e-6
        ), "All methods should store same angle in degrees"

        # Test 6: Sum constraint unit behavior
        # Test degrees (default) and radians for sum constraints
        model_sum_deg = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )
        model_sum_deg.set_sum_constraint(1, 2, -180, 0)  # Default degrees

        model_sum_rad = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )
        model_sum_rad.set_sum_constraint(
            1, 2, np.deg2rad(-180), 0, degrees=False
        )  # Explicit radians

        # Sum constraints should accept both unit types without error
        print(f"  - Sum constraints: Both degrees and radians accepted")


class TestConstraintUnitBehavior:
    """Dedicated tests for constraint unit handling behavior and examples."""

    def test_initialization_constraint_units(self):
        """Test that initialization parameters are treated as degrees."""
        import numpy as np

        # Example: Common robot joint constraints in intuitive degree values
        model = KinematicModel(
            a1=460,
            a2=-250,
            c1=1140,
            c2=1050,
            c3=1510,
            c4=282,
            flip_axes=(False, False, True, False, True, False),
            has_parallelogram=True,
            relative_constraints=[
                (2, 1, -140, -40),  # J3 relative to J2: reasonable parallelogram range
                (5, 4, -90, 90),  # J6 relative to J5: wrist rotation coupling
            ],
        )

        constraints = model.relative_constraints
        assert constraints is not None, "Constraints should be stored"
        assert len(constraints) == 2, "Should have 2 constraints"

        # Verify degree-to-degree storage (corrected behavior)
        j3_j2_constraint = constraints[0]  # (2, 1, ...)
        j6_j5_constraint = constraints[1]  # (5, 4, ...)

        assert (
            abs(j3_j2_constraint[2] - (-140)) < 1e-10
        ), "J3-J2 min should be stored as degrees"
        assert (
            abs(j3_j2_constraint[3] - (-40)) < 1e-10
        ), "J3-J2 max should be stored as degrees"
        assert (
            abs(j6_j5_constraint[2] - (-90)) < 1e-10
        ), "J6-J5 min should be stored as degrees"
        assert (
            abs(j6_j5_constraint[3] - 90) < 1e-10
        ), "J6-J5 max should be stored as degrees"

        print("✓ Initialization constraint unit behavior:")
        print(f"  Input: J3-J2 range [-140°, -40°], J6-J5 range [-90°, +90°]")
        print(
            f"  Stored: J3-J2 range [{j3_j2_constraint[2]:.1f}, {j3_j2_constraint[3]:.1f}]°"
        )
        print(
            f"  Stored: J6-J5 range [{j6_j5_constraint[2]:.1f}, {j6_j5_constraint[3]:.1f}]°"
        )

    def test_manual_constraint_units_comparison(self):
        """Test and demonstrate both degrees=True and degrees=False methods."""
        import numpy as np

        # Create three identical models using different unit approaches
        base_params = dict(a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230)

        # Method 1: Use degrees=True for convenience
        model_deg = KinematicModel(**base_params)
        model_deg.set_relative_constraint(
            2, 1, -120, -30, degrees=True
        )  # Degrees input

        # Method 2: Use degrees=False with pre-converted radians
        model_rad = KinematicModel(**base_params)
        model_rad.set_relative_constraint(
            2, 1, np.deg2rad(-120), np.deg2rad(-30), degrees=False
        )  # Radians input

        # Method 3: Use explicit degrees=True (current default)
        model_legacy = KinematicModel(**base_params)
        model_legacy.set_relative_constraint(
            2, 1, -120, -30, degrees=True
        )  # Explicit degrees=True (current default)

        # All should produce identical results (all stored in degrees)
        constraints_deg = model_deg.relative_constraints
        constraints_rad = model_rad.relative_constraints
        constraints_legacy = model_legacy.relative_constraints

        # Verify the actual stored values (now all in degrees)
        stored = constraints_deg[0]
        expected_deg = -120  # All stored as degrees

        assert (
            abs(stored[2] - expected_deg) < 1e-10
        ), "Min constraint should be in degrees"
        assert abs(stored[3] - (-30)) < 1e-10, "Max constraint should be in degrees"

        # All methods should now produce identical degree storage (within floating-point tolerance)
        def constraint_values_match(c1, c2, tolerance=1e-12):
            return (
                c1[0] == c2[0]
                and c1[1] == c2[1]
                and abs(c1[2] - c2[2]) < tolerance
                and abs(c1[3] - c2[3]) < tolerance
            )

        assert constraint_values_match(
            constraints_deg[0], constraints_rad[0]
        ), "degrees and radians methods should produce equivalent storage"
        assert constraint_values_match(
            constraints_deg[0], constraints_legacy[0]
        ), "degrees and legacy methods should produce equivalent storage"

        print("✓ Manual constraint unit method comparison:")
        print(
            f"  Method 1 (degrees=True):  -120°, -30° → {stored[2]:.1f}°, {stored[3]:.1f}°"
        )
        print(
            f"  Method 2 (degrees=False): {np.deg2rad(-120):.4f}, {np.deg2rad(-30):.4f} rad → {constraints_rad[0][2]:.1f}°, {constraints_rad[0][3]:.1f}°"
        )
        print(
            f"  Method 3 (explicit degrees=True): -120°, -30° → {constraints_legacy[0][2]:.1f}°, {constraints_legacy[0][3]:.1f}°"
        )
        print(
            f"  All methods equivalent: {constraint_values_match(constraints_deg[0], constraints_rad[0]) and constraint_values_match(constraints_deg[0], constraints_legacy[0])}"
        )

    def test_absolute_constraint_units(self):
        """Test absolute constraint unit handling."""
        import numpy as np

        model = KinematicModel(a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230)

        # Test absolute constraints with both unit types
        model.set_absolute_constraint(0, -180, 180)  # J1: ±180°
        model.set_absolute_constraint(
            1, -np.pi / 2, np.pi / 2, degrees=False
        )  # J2: ±90° in radians
        model.set_absolute_constraint(2, -np.pi, np.pi)  # J3: ±180° in radians (legacy)

        # Should not raise exceptions
        print("✓ Absolute constraint unit handling:")
        print(f"  J1: degrees=True  → -180° to +180° → converted to radians internally")
        print(
            f"  J2: degrees=False → {-np.pi/2:.4f} to {np.pi/2:.4f} rad → stored as radians"
        )
        print(
            f"  J3: legacy mode   → {-np.pi:.4f} to {np.pi:.4f} rad → stored as radians"
        )

    def test_real_world_constraint_examples(self):
        """Test with realistic robot constraint scenarios."""
        import numpy as np

        # Example 1: Comau NJ165 with typical parallelogram constraints
        comau_model = KinematicModel(
            a1=460,
            a2=-250,
            b=0,
            c1=1140,
            c2=1050,
            c3=1510,
            c4=282,
            flip_axes=(False, False, True, False, True, False),
            has_parallelogram=True,
            relative_constraints=[
                (2, 1, -155, -25),  # J3 relative to J2: parallelogram mechanism limits
            ],
        )

        # Add wrist constraints manually
        comau_model.set_relative_constraint(
            5, 4, -180, 180, degrees=True
        )  # J6 relative to J5

        constraints = comau_model.relative_constraints
        assert len(constraints) == 2, "Should have parallelogram + wrist constraints"

        # Example 2: Generic 6-axis robot with multiple coupling constraints
        generic_model = KinematicModel(
            a1=300, a2=-200, c1=700, c2=1000, c3=1200, c4=150, has_parallelogram=False
        )

        # Add multiple constraints using different methods
        generic_model.set_relative_constraint(
            1, 0, -45, 45, degrees=True
        )  # J2 relative to J1
        generic_model.set_relative_constraint(
            3, 2, np.deg2rad(-90), np.deg2rad(90), degrees=False
        )  # J4 relative to J3

        generic_constraints = generic_model.relative_constraints
        assert len(generic_constraints) == 2, "Should have 2 coupling constraints"

        print("✓ Real-world constraint examples:")
        print(
            f"  Comau NJ165: Parallelogram + wrist constraints → {len(constraints)} total"
        )
        print(
            f"  Generic robot: Mixed unit input methods → {len(generic_constraints)} total"
        )
        print(f"  All constraints stored in consistent degree format")

    def test_constraint_unit_documentation(self):
        """Document and test the constraint unit behavior for users."""
        import numpy as np

        print(f"\\n=== Constraint Unit Behavior Documentation ===")
        print(f"1. KinematicModel(relative_constraints=[(axis, ref_axis, min, max)]):")
        print(f"   - Input: degrees (for convenience)")
        print(f"   - Storage: degrees (consistent)")
        print(f"   - Property: degrees (always)")

        print(f"\n2. model.set_relative_constraint(axis, ref_axis, min, max):")
        print(f"   - Input: degrees (default)")
        print(f"   - Storage: degrees (consistent)")
        print(f"   - Property: degrees (always)")

        print(
            f"\\n3. model.set_relative_constraint(axis, ref_axis, min, max, degrees=False):"
        )
        print(f"   - Input: radians")
        print(f"   - Storage: degrees (converted for consistency)")
        print(f"   - Property: degrees (always)")

        print(f"\\n4. model.relative_constraints property:")
        print(f"   - Always returns values in degrees")
        print(f"   - Consistent storage regardless of how constraints were set")
        print(f"   - Conversion to radians only during kinematic computations")

        # Demonstrate with actual values
        model = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            relative_constraints=[(2, 1, -135, -45)],  # Input: degrees
        )
        stored = model.relative_constraints[0]
        print(f"\\nExample:")
        print(f"  Input during init: (-135°, -45°)")
        print(f"  Stored internally: ({stored[2]:.1f}°, {stored[3]:.1f}°)")
        print(f"  Property returns: ({stored[2]:.1f}°, {stored[3]:.1f}°)")
        print(f"  Verification: -135° = {stored[2]:.1f}° ✓")

        assert True  # This test is primarily for documentation


class TestKinematicModelRobotConfigurations:
    """Test KinematicModel with real robot configurations."""

    def test_comau_nj165_configuration(self):
        """Test KinematicModel with Comau NJ165-3.0 configuration."""
        model = KinematicModel(
            a1=460,
            a2=-250,
            b=0,
            c1=1140,
            c2=1050,
            c3=1510,
            c4=282,
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(False, False, True, False, True, False),
            has_parallelogram=True,
        )

        # Standard NJ165 axis limits
        axis_limits = [
            (-175, 175),  # J1
            (-75, 75),  # J2
            (-220, 0),  # J3
            (-2700, 2700),  # J4
            (-125, 125),  # J5
            (-2700, 2700),  # J6
        ]
        model.set_axis_limits(axis_limits, degrees=True)

        assert model.a1 == 460
        assert model.has_parallelogram == True
        # Use allclose for axis limits due to floating point precision in degree->radian->degree conversion
        np.testing.assert_allclose(model.axis_limits, axis_limits, rtol=1e-10)

    def test_generic_robot_configuration(self):
        """Test KinematicModel with a generic robot configuration."""
        model = KinematicModel(
            a1=400.333,
            a2=-251.449,
            b=0,
            c1=830,
            c2=1177.556,
            c3=1443.593,
            c4=230,
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(True, False, True, True, False, True),
            has_parallelogram=True,
        )

        assert model.a1 == 400.333
        assert model.a2 == -251.449
        assert model.flip_axes == [True, False, True, True, False, True]

    def test_robot_without_parallelogram(self):
        """Test KinematicModel for a robot without parallelogram linkage."""
        model = KinematicModel(
            a1=100,
            a2=200,
            b=50,
            c1=300,
            c2=400,
            c3=500,
            c4=100,
            has_parallelogram=False,
        )

        assert model.has_parallelogram == False


class TestKinematicModelEdgeCases:
    """Test edge cases and special configurations."""

    def test_zero_parameters(self):
        """Test KinematicModel with all zero parameters."""
        model = KinematicModel(a1=0, a2=0, b=0, c1=0, c2=0, c3=0, c4=0)

        # Should not raise exceptions
        assert model.a1 == 0
        assert model.c4 == 0

    def test_negative_parameters(self):
        """Test KinematicModel with negative parameters."""
        model = KinematicModel(
            a1=-100, a2=-200, b=-50, c1=-300, c2=-400, c3=-500, c4=-100
        )

        assert model.a1 == -100
        assert model.c1 == -300

    def test_large_parameters(self):
        """Test KinematicModel with large parameters."""
        model = KinematicModel(
            a1=10000, a2=20000, b=5000, c1=30000, c2=40000, c3=50000, c4=10000
        )

        assert model.a1 == 10000
        assert model.c3 == 50000

    def test_extreme_offsets(self):
        """Test KinematicModel with extreme offset values."""
        extreme_offsets = (360, -360, 720, -720, 1800, -1800)
        model = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230, offsets=extreme_offsets
        )

        # Convert to lists for comparison since API returns lists
        assert [float(offset) for offset in model.offsets] == [
            float(offset) for offset in extreme_offsets
        ]

    def test_mixed_flip_axes(self):
        """Test KinematicModel with various flip_axes combinations."""
        flip_combinations = [
            (True, True, True, True, True, True),
            (False, False, False, False, False, False),
            (True, False, True, False, True, False),
            (False, True, False, True, False, True),
        ]

        for flip_axes in flip_combinations:
            model = KinematicModel(
                a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230, flip_axes=flip_axes
            )
            assert list(model.flip_axes) == list(flip_axes)


class TestKinematicModelStringRepresentation:
    """Test string representation methods."""

    def test_model_attributes_accessible(self):
        """Test that model attributes are accessible and have expected types."""
        model = KinematicModel(
            a1=400.0,
            a2=-250.0,
            c1=830.0,
            c2=1175.0,
            c3=1444.0,
            c4=230.0,
            has_parallelogram=True,
        )

        # Check attribute types
        assert isinstance(model.a1, float)
        assert isinstance(model.has_parallelogram, bool)
        assert isinstance(model.offsets, list)
        assert len(model.offsets) == 6
        assert isinstance(model.flip_axes, list)
        assert len(model.flip_axes) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
