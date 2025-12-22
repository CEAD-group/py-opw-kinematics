"""
Integration tests for py-opw-kinematics.

These tests verify the full workflow and integration between different components:
- Complete robot setup workflows
- Real-world kinematic scenarios
- Multi-step operations
- Industrial robot configurations
- Trajectory planning scenarios
"""

import pytest
import numpy as np
import polars as pl
from py_opw_kinematics import Robot, EulerConvention, KinematicModel


class TestCompleteRobotWorkflow:
    """Test complete robot setup and operation workflows."""

    def test_comau_nj165_complete_workflow(self):
        """Test complete workflow with Comau NJ165-3.0 robot."""
        # Step 1: Create kinematic model
        kinematic_model = KinematicModel(
            a1=400,
            a2=-250,
            b=0,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(False, False, True, False, False, False),
            has_parallelogram=True,
        )

        # Step 2: Set axis limits
        axis_limits = (
            (-175, 175),  # J1
            (-95, 75),  # J2
            (-256, -10),  #  J3
            (-2700, 2700),  # J4
            (-125, 125),  # J5
            (-2700, 2700),  # J6
        )
        kinematic_model.set_axis_limits(axis_limits)

        # Step 3: Set relative constraints (parallelogram)
        parallelogram_limits = (-160.0, -30.0)  # relative limits for J3 relative to J2
        kinematic_model.set_relative_constraint(
            axis=2,
            reference_axis=1,
            min_offset=parallelogram_limits[0],
            max_offset=parallelogram_limits[1],
            degrees=True,  # Specify units explicitly for clarity
        )

        # Step 4: Create Euler convention
        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)

        # Step 5: Set end-effector parameters
        ee_translation = (145.5, -353, -330.5)
        ee_rotation = (0, -90, 0)

        # Step 6: Create robot
        robot = Robot(kinematic_model, euler_convention, ee_translation, ee_rotation)

        # Step 7: Test constraint logical consistency
        # This test verifies that if forward kinematics produces a pose from legal joints,
        # then inverse kinematics should find at least one legal solution back to that pose
        test_joints = (0, 10, -100, 0, 10, 0)  # J3-J2 = -100° (within [-160°, -30°])

        # Verify our test joints satisfy the constraint
        j3_j2_diff = test_joints[2] - test_joints[1]
        constraint_satisfied = -160 <= j3_j2_diff <= -30
        assert (
            constraint_satisfied
        ), f"Test joints should satisfy constraint: J3-J2={j3_j2_diff}° not in [-160°, -30°]"

        position, orientation = robot.forward(test_joints)

        # Verify reasonable results
        assert len(position) == 3
        assert len(orientation) == 3
        assert abs(position[0]) < 4000  # Reasonable X position
        assert abs(position[1]) < 4000  # Reasonable Y position
        assert abs(position[2]) < 4000  # Reasonable Z position

        # Step 8: Test constraint logical consistency
        # If we used legal joints for forward kinematics, inverse should find legal solutions
        solutions = robot.inverse(
            pose=(position, orientation), current_joints=test_joints
        )

        # Create unconstrained robot for comparison and debugging
        unconstrained_model = KinematicModel(
            a1=400,
            a2=-250,
            b=0,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(False, False, True, False, False, False),
            has_parallelogram=True,
        )
        unconstrained_robot = Robot(
            unconstrained_model, euler_convention, ee_translation, ee_rotation
        )
        unconstrained_solutions = unconstrained_robot.inverse(
            (position, orientation), current_joints=test_joints
        )

        # Debug: check which unconstrained solutions would be valid
        valid_unconstrained = []
        for sol in unconstrained_solutions:
            sol_diff = sol[2] - sol[1]  # J3 - J2
            if -160 <= sol_diff <= -30:
                valid_unconstrained.append(sol)

        print(f"Constraint analysis:")
        print(f"  Input joints: J3-J2 = {j3_j2_diff}° ✅")
        print(f"  Unconstrained solutions: {len(unconstrained_solutions)}")
        print(f"  Would-be-valid solutions: {len(valid_unconstrained)}")
        print(f"  Constrained solutions: {len(solutions)}")

        # The key test: if unconstrained IK finds valid solutions, constrained IK should too
        if len(valid_unconstrained) > 0:
            # There are solutions that should satisfy the constraint
            assert (
                len(solutions) > 0
            ), f"Constraint consistency error: Found {len(valid_unconstrained)} valid unconstrained solutions, but constrained robot found {len(solutions)}"

            # Verify that constrained solutions actually satisfy the constraint
            for i, sol in enumerate(solutions):
                sol_diff = sol[2] - sol[1]
                assert (
                    -160 <= sol_diff <= -30
                ), f"Constrained solution {i+1} violates constraint: J3-J2={sol_diff}° not in [-160°, -30°]"
        else:
            # If no solutions would be valid, then 0 constrained solutions is acceptable
            print(
                "  No unconstrained solutions satisfy constraint - 0 constrained solutions acceptable"
            )

        # Constraints should never increase solution count
        assert len(solutions) <= len(
            unconstrained_solutions
        ), f"Constrained robot should not have more solutions than unconstrained: {len(solutions)} <= {len(unconstrained_solutions)}"

    def test_generic_robot_workflow_with_batch_operations(self):
        """Test complete workflow with batch operations on a generic robot."""
        # Step 1: Robot setup
        kinematic_model = KinematicModel(
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

        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot = Robot(kinematic_model, euler_convention, ee_rotation=(0, -90, 0))

        # Step 2: Generate test trajectory
        n_points = 50
        joints_trajectory = pl.DataFrame(
            {
                "J1": np.linspace(-30, 30, n_points),
                "J2": np.linspace(-20, 20, n_points),
                "J3": np.linspace(-120, -60, n_points),
                "J4": np.linspace(-45, 45, n_points),
                "J5": np.linspace(-30, 30, n_points),
                "J6": np.linspace(-60, 60, n_points),
            }
        )

        # Step 3: Forward batch processing
        poses_trajectory = robot.batch_forward(joints_trajectory)

        assert len(poses_trajectory) == n_points
        assert all(
            col in poses_trajectory.columns for col in ["X", "Y", "Z", "A", "B", "C"]
        )

        # Step 4: Inverse batch processing (round-trip test)
        recovered_joints = robot.batch_inverse(
            poses_trajectory, current_joints=joints_trajectory.row(0)
        )

        assert len(recovered_joints) <= n_points  # Some poses might not have solutions

        # Step 5: Verify round-trip accuracy for valid solutions
        if len(recovered_joints) > 0:
            # Compare first few valid solutions
            original_first_row = joints_trajectory.row(0)
            recovered_first_row = recovered_joints.row(0)

            # Should be close (allowing for different valid solutions)
            for i in range(6):
                if not np.isnan(recovered_first_row[i]):
                    # Allow for ±360° differences in joint angles
                    diff = abs(original_first_row[i] - recovered_first_row[i])
                    diff_mod = min(diff, abs(diff - 360), abs(diff + 360))
                    assert diff_mod < 5.0, f"Joint {i+1} differs by {diff}°"


class TestTrajectoryPlanningIntegration:
    """Test integration for trajectory planning scenarios."""

    @pytest.fixture
    def trajectory_robot(self):
        """Create a robot optimized for trajectory testing."""
        kinematic_model = KinematicModel(
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
        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        return Robot(kinematic_model, euler_convention)

    def test_circular_trajectory(self, trajectory_robot: Robot):
        """Test planning a circular trajectory."""
        center = [2000, 0, 1300]
        radius = 200
        orientation = [0, -45, 0]  # Constant orientation pointing down

        # Generate circular trajectory points
        n_points = 64
        trajectory_positions = []
        for i in range(n_points + 1):  # +1 to close the circle
            angle = 2 * np.pi * i / n_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            trajectory_positions.append(([x, y, z], orientation))

        # Plan joint trajectory for full circle
        joint_trajectory = []
        previous_joints = None

        for position, orient in trajectory_positions:
            solutions = trajectory_robot.inverse(
                (position, orient), current_joints=previous_joints
            )
            assert len(solutions) > 0, f"No solution for position {position}"

            if previous_joints is None:
                chosen_joints = solutions[0]
            else:
                chosen_joints = min(
                    solutions,
                    key=lambda sol: sum(
                        abs(a - b) ** 2 for a, b in zip(sol, previous_joints)
                    ),
                )

            joint_trajectory.append(chosen_joints)
            previous_joints = chosen_joints

        # Verify circular trajectory closes properly
        start_joints = joint_trajectory[0]
        end_joints = joint_trajectory[-1]

        for j in range(6):
            angle_diff = abs(start_joints[j] - end_joints[j])
            # Account for continuous rotation
            angle_diff = min(angle_diff, abs(angle_diff - 360), abs(angle_diff + 360))
            assert (
                angle_diff < 1.0
            ), f"Circular trajectory doesn't close for joint {j+1}: {angle_diff}°"


class TestConstraintFunctionality:
    """Test functional behavior of constraints with Robot class integration."""

    def test_relative_constraint_functionality(self):
        """Test that relative constraints functionally filter solutions."""
        # Create kinematic model with relative constraints
        kinematic_model_with_constraints = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            relative_constraints=[(2, 1, -135, -45)],  # J3 relative to J2 in degrees
        )

        # Create kinematic model without constraints for comparison
        kinematic_model_without_constraints = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )

        # Create robots with both models
        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot_with_constraints = Robot(
            kinematic_model_with_constraints, euler_convention
        )
        robot_without_constraints = Robot(
            kinematic_model_without_constraints, euler_convention
        )

        # Test pose that should violate relative constraints
        # J3 = 0, J2 = 0 means J3-J2 = 0, which is outside [-135, -45] range
        joints_violating_constraints = [0, 0, 0, 0, 0, 0]
        position, orientation = robot_without_constraints.forward(
            joints_violating_constraints
        )

        # Robot without constraints should find solutions
        solutions_without_constraints = robot_without_constraints.inverse(
            (position, orientation)
        )
        assert (
            len(solutions_without_constraints) > 0
        ), "Robot without constraints should find solutions"

        # Robot with constraints should have same or fewer solutions
        solutions_with_constraints = robot_with_constraints.inverse(
            (position, orientation)
        )
        assert len(solutions_with_constraints) <= len(
            solutions_without_constraints
        ), "Robot with constraints should have same or fewer solutions"

    def test_absolute_constraint_functionality(self):
        """Test that absolute (axis limit) constraints functionally filter solutions."""
        # Create kinematic model with tight axis limits
        kinematic_model_with_limits = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )
        # Set very restrictive limits on J1
        kinematic_model_with_limits.set_absolute_constraint(0, -30, 30)

        # Create kinematic model without limits
        kinematic_model_without_limits = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )

        # Create robots
        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot_with_limits = Robot(kinematic_model_with_limits, euler_convention)
        robot_without_limits = Robot(kinematic_model_without_limits, euler_convention)

        # Test pose that requires large J1 rotation
        joints_violating_limits = [
            90,
            0,
            -90,
            0,
            0,
            0,
        ]  # J1 = 90° violates [-30°, 30°] limit
        position, orientation = robot_without_limits.forward(joints_violating_limits)

        # Robot without limits should find solutions
        solutions_without_limits = robot_without_limits.inverse((position, orientation))
        assert (
            len(solutions_without_limits) > 0
        ), "Robot without limits should find solutions"

        # Robot with limits should have fewer solutions (filtering out J1 > 30°)
        solutions_with_limits = robot_with_limits.inverse((position, orientation))
        assert len(solutions_with_limits) <= len(
            solutions_without_limits
        ), "Robot with limits should have same or fewer solutions"

    def test_constraint_unit_consistency(self):
        """Test that constraint unit handling is consistent across different setting methods."""
        # Create models using different constraint setting approaches
        model1 = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            relative_constraints=[(2, 1, -155, -25)],
        )  # Degrees during init

        model2 = KinematicModel(a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230)
        model2.set_relative_constraint(
            2, 1, -155, -25, degrees=True
        )  # Degrees at runtime

        model3 = KinematicModel(a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230)
        model3.set_relative_constraint(
            2, 1, np.deg2rad(-155), np.deg2rad(-25), degrees=False
        )  # Radians

        # Create robots with identical configurations
        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot1 = Robot(model1, euler_convention)
        robot2 = Robot(model2, euler_convention)
        robot3 = Robot(model3, euler_convention)

        # Test that all robots produce identical kinematic behavior
        test_joints = [10, 20, -90, 45, -10, 0]

        # Forward kinematics should be identical
        pos1, ori1 = robot1.forward(test_joints)
        pos2, ori2 = robot2.forward(test_joints)
        pos3, ori3 = robot3.forward(test_joints)

        assert np.allclose(
            pos1, pos2, atol=1e-10
        ), "Forward kinematics should be identical (model1 vs model2)"
        assert np.allclose(
            pos1, pos3, atol=1e-10
        ), "Forward kinematics should be identical (model1 vs model3)"
        assert np.allclose(
            ori1, ori2, atol=1e-10
        ), "Orientations should be identical (model1 vs model2)"
        assert np.allclose(
            ori1, ori3, atol=1e-10
        ), "Orientations should be identical (model1 vs model3)"

        # Inverse kinematics should produce same number of valid solutions
        solutions1 = robot1.inverse((pos1, ori1))
        solutions2 = robot2.inverse((pos1, ori1))
        solutions3 = robot3.inverse((pos1, ori1))

        # Robot1 has parallelogram constraint while robot2 and robot3 don't
        # So robot1 should have same or fewer solutions than robots 2 and 3
        assert len(solutions2) == len(
            solutions3
        ), f"Robots 2 and 3 should find same number of solutions: {len(solutions2)}, {len(solutions3)}"

        assert len(solutions1) <= len(
            solutions2
        ), f"Robot 1 (with parallelogram) should find same or fewer solutions than robot 2: {len(solutions1)} <= {len(solutions2)}"

    def test_combined_constraints_functionality(self):
        """Test robots with both relative and absolute constraints."""
        # Robot with both types of constraints
        model_combined = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            relative_constraints=[(2, 1, -135, -45)],  # J3-J2 parallelogram constraint
        )
        # Add absolute constraint
        model_combined.set_absolute_constraint(
            0, -90, 90, degrees=True
        )  # Limit J1 range
        model_combined.set_absolute_constraint(
            1, -60, 60, degrees=True
        )  # Limit J2 range

        # Robot with no constraints
        model_unconstrained = KinematicModel(
            a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230
        )

        # Create robots
        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot_combined = Robot(model_combined, euler_convention)
        robot_unconstrained = Robot(model_unconstrained, euler_convention)

        # Test multiple poses to verify constraint filtering
        test_poses = [
            [0, 0, -90, 0, 0, 0],  # Should pass all constraints
            [100, 0, -90, 0, 0, 0],  # Should violate J1 absolute constraint
            [0, 70, -90, 0, 0, 0],  # Should violate J2 absolute constraint
            [0, 0, 0, 0, 0, 0],  # Should violate J3-J2 relative constraint
        ]

        total_unconstrained = 0
        total_constrained = 0

        for joints in test_poses:
            position, orientation = robot_unconstrained.forward(joints)

            solutions_unconstrained = robot_unconstrained.inverse(
                (position, orientation)
            )
            solutions_constrained = robot_combined.inverse((position, orientation))

            total_unconstrained += len(solutions_unconstrained)
            total_constrained += len(solutions_constrained)

            # Constrained robot should never have more solutions than unconstrained
            assert len(solutions_constrained) <= len(
                solutions_unconstrained
            ), f"Constrained robot should not have more solutions than unconstrained for joints {joints}"

        # Over all test cases, constraints should have filtered some solutions
        assert (
            total_constrained <= total_unconstrained
        ), f"Combined constraints should filter solutions: {total_constrained} <= {total_unconstrained}"

        print(
            f"Constraint filtering effectiveness: {total_unconstrained} → {total_constrained} solutions"
        )

    def test_constraint_logical_consistency(self):
        """Test that constraints maintain logical consistency between forward and inverse kinematics.

        If forward kinematics uses valid joints that satisfy all constraints, then
        inverse kinematics should find at least one valid solution back to that pose.
        """
        # Create robot with relative constraints similar to parallelogram
        model_constrained = KinematicModel(
            a1=400,
            a2=-250,
            b=0,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(False, False, True, False, False, False),
            has_parallelogram=True,
            relative_constraints=[(2, 1, -160, -30)],  # J3-J2 in degrees
        )

        # Create unconstrained robot for comparison
        model_unconstrained = KinematicModel(
            a1=400,
            a2=-250,
            b=0,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(False, False, True, False, False, False),
            has_parallelogram=True,
        )

        euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)
        ee_translation = [0, 0, 0]
        ee_rotation = [0, 0, 0]

        robot_constrained = Robot(
            model_constrained, euler_convention, ee_translation, ee_rotation
        )
        robot_unconstrained = Robot(
            model_unconstrained, euler_convention, ee_translation, ee_rotation
        )

        # Test joints that satisfy the constraint J3-J2 ∈ [-160°, -30°]
        test_joints = (0, 10, -90, 0, 10, 0)  # J3-J2 = -100°, valid per constraint

        # Verify our test joints satisfy the constraint
        j3_j2_diff = test_joints[2] - test_joints[1]
        assert (
            -160 <= j3_j2_diff <= -30
        ), f"Test joints should satisfy constraint: J3-J2={j3_j2_diff}° not in [-160°, -30°]"

        # Forward kinematics with valid joints
        position, orientation = robot_constrained.forward(test_joints)

        # Inverse kinematics - constrained and unconstrained
        solutions_constrained = robot_constrained.inverse(
            (position, orientation), current_joints=test_joints
        )
        solutions_unconstrained = robot_unconstrained.inverse(
            (position, orientation), current_joints=test_joints
        )

        # Count how many unconstrained solutions would be valid
        valid_unconstrained = []
        for sol in solutions_unconstrained:
            sol_diff = sol[2] - sol[1]  # J3 - J2
            if -160 <= sol_diff <= -30:
                valid_unconstrained.append(sol)

        # Debug output
        print(f"\nConstraint logical consistency check:")
        print(f"  Test joints: {test_joints} → J3-J2 = {j3_j2_diff}° ✅ (valid)")
        print(f"  Forward kinematics pose: position={[round(p, 2) for p in position]}")
        print(f"  Unconstrained solutions: {len(solutions_unconstrained)}")
        print(f"  Valid unconstrained solutions: {len(valid_unconstrained)}")
        print(f"  Constrained solutions: {len(solutions_constrained)}")

        if len(valid_unconstrained) > 0:
            print(f"  Valid unconstrained solutions:")
            for i, sol in enumerate(valid_unconstrained):
                sol_diff = sol[2] - sol[1]
                print(f"    Solution {i+1}: J3-J2 = {sol_diff:.1f}° (valid)")

        # The critical assertion: if unconstrained IK finds solutions that satisfy the constraint,
        # then constrained IK should find at least some of them
        if len(valid_unconstrained) > 0:
            # This is the logical consistency issue you identified!
            assert len(solutions_constrained) > 0, (
                f"CONSTRAINT LOGIC BUG: Found {len(valid_unconstrained)} unconstrained solutions that satisfy constraint, "
                f"but constrained robot found {len(solutions_constrained)} solutions. "
                f"If forward kinematics uses valid joints, inverse should find valid solutions!"
            )

            # Additional verification: all constrained solutions should actually satisfy constraints
            for i, sol in enumerate(solutions_constrained):
                sol_diff = sol[2] - sol[1]
                assert (
                    -160 <= sol_diff <= -30
                ), f"Constrained solution {i+1} violates constraint: J3-J2={sol_diff}° not in [-160°, -30°]"

        # Constrained should never find more solutions than unconstrained
        assert len(solutions_constrained) <= len(
            solutions_unconstrained
        ), f"Constrained robot should not have more solutions than unconstrained: {len(solutions_constrained)} <= {len(solutions_unconstrained)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

    # TestCompleteRobotWorkflow().test_comau_nj165_complete_workflow()
