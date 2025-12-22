"""
Performance benchmarks for py-opw-kinematics.

Tests performance of:
- Single forward/inverse kinematics operations
- Batch operations with various data sizes
- Memory usage and efficiency
- Comparison with target performance metrics
"""

import pytest
import time
import numpy as np
import polars as pl
from py_opw_kinematics import Robot, EulerConvention, KinematicModel


class TestSingleOperationPerformance:
    """Test performance of single kinematic operations."""

    @pytest.fixture
    def benchmark_robot(self) -> Robot:
        """Create a robot for performance benchmarking."""
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
        return Robot(kinematic_model, euler_convention, ee_rotation=(0, -90, 0))

    def test_forward_kinematics_performance(self, benchmark_robot: Robot):
        """Test forward kinematics performance for single operations."""
        test_joints = [10, 20, -90, 30, 40, 50]

        # Warm-up
        for _ in range(100):
            benchmark_robot.forward(test_joints)

        # Benchmark
        n_iterations = 10000
        start_time = time.perf_counter()

        for _ in range(n_iterations):
            position, orientation = benchmark_robot.forward(test_joints)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / n_iterations
        target_time = (
            0.02 / 1000  # Performance target: < 0.1ms per inverse kinematics operation
        )
        # Performance target: < 0.01ms per forward kinematics operation
        assert (
            avg_time < target_time
        ), f"Forward kinematics too slow: {avg_time*1000:.3f}ms per operation (vs target {target_time*1000:.3f}ms)"

        print(
            f"Forward kinematics: {avg_time*1000000:.1f}μs per operation ({n_iterations} iterations)"
        )

    def test_inverse_kinematics_performance(self, benchmark_robot: Robot):
        """Test inverse kinematics performance for single operations."""
        # Get a valid pose first
        test_joints = [10, 20, -90, 30, 40, 50]
        position, orientation = benchmark_robot.forward(test_joints)
        test_pose = (position, orientation)

        # Warm-up
        for _ in range(50):
            benchmark_robot.inverse(test_pose)

        # Benchmark
        n_iterations = 1000
        start_time = time.perf_counter()

        for _ in range(n_iterations):
            solutions = benchmark_robot.inverse(test_pose)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / n_iterations
        target_time = (
            0.1 / 1000  # Performance target: < 0.1ms per inverse kinematics operation
        )

        assert (
            avg_time < target_time
        ), f"Inverse kinematics too slow: {avg_time*1000:.3f}ms per operation (vs target {target_time*1000:.3f}ms)"

        print(
            f"Inverse kinematics: {avg_time*1000:.3f}ms per operation ({n_iterations} iterations)"
        )


class TestBatchPerformance:
    """Test batch operation performance with large datasets."""

    @pytest.fixture
    def large_batch_robot(self):
        """Create a robot for large batch testing."""
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
        return Robot(kinematic_model, euler_convention, ee_rotation=(0, -90, 0))

    def test_batch_forward_large(self, large_batch_robot: Robot):
        """Test batch forward kinematics with large dataset (100K points)."""
        n_points = 100000

        # Generate test data
        joints_df = pl.DataFrame(
            {
                "J1": np.random.uniform(-30, 30, n_points),
                "J2": np.random.uniform(-20, 20, n_points),
                "J3": np.random.uniform(-120, -60, n_points),
                "J4": np.random.uniform(-45, 45, n_points),
                "J5": np.random.uniform(-30, 30, n_points),
                "J6": np.random.uniform(-60, 60, n_points),
            }
        )

        # Warm-up
        large_batch_robot.batch_forward(joints_df[:1000])

        # Benchmark
        start_time = time.perf_counter()
        result = large_batch_robot.batch_forward(joints_df)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / n_points

        # Verify correctness
        assert len(result) == n_points

        # Performance target from README: 100K in 0.4s for inverse, so forward should be faster
        # Target: < 0.2s for 100K forward operations
        target_time = (
            0.02 / 1000  # Performance target: < 0.1ms per inverse kinematics operation
        )
        # Performance target: < 0.01ms per Inverse kinematics operation
        assert (
            avg_time < target_time
        ), f"Inverse kinematics too slow: {avg_time*1000:.3f}ms per operation (vs target {target_time*1000:.3f}ms)"

        print(
            f"Batch forward (100K): {avg_time*1000000:.1f}μs per point, {total_time:.3f}s total"
        )

    def test_batch_inverse_large(self, large_batch_robot: Robot):
        """Test batch inverse kinematics with large dataset (100K points)."""
        n_points = 100_000

        # Generate realistic poses using forward kinematics to ensure reachability
        np.random.seed(42)  # For reproducible results
        test_joints = {
            "J1": np.random.uniform(-30, 30, n_points),
            "J2": np.random.uniform(-20, 20, n_points),
            "J3": np.random.uniform(-120, -60, n_points),
            "J4": np.random.uniform(-45, 45, n_points),
            "J5": np.random.uniform(-30, 30, n_points),
            "J6": np.random.uniform(-60, 60, n_points),
        }

        joints_df = pl.DataFrame(test_joints)
        poses_df = large_batch_robot.batch_forward(joints_df)

        # Warm-up
        large_batch_robot.batch_inverse(poses_df[:100])

        # Benchmark - the main performance test from README
        start_time = time.perf_counter()
        result = large_batch_robot.batch_inverse(poses_df)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / n_points

        # Verify reasonable number of solutions
        assert len(result) > n_points * 0.5, "Too few solutions found"

        target_time = (
            0.3
            / 1000  # Performance target: < 0.3ms per inverse kinematics operation. Batch operations are slower than individual operations because a single solution must be selected for each pose.
        )
        assert (
            avg_time < target_time
        ), f"Inverse kinematics too slow: {avg_time*1000:.3f}ms per operation (vs target {target_time*1000:.3f}ms)"

        print(
            f"Inverse kinematics: {avg_time*1000000:.1f}μs per operation ({n_points} iterations), {total_time:.3f}s total"
        )


class TestPerformanceRegression:
    """Test for performance regressions across different scenarios."""

    @pytest.fixture
    def regression_robot(self) -> Robot:
        """Create a robot for regression testing."""
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
        return Robot(kinematic_model, euler_convention, ee_rotation=(0, -90, 0))

    def test_performance_with_constraints(self):
        """Test that axis constraints don't significantly impact performance."""
        # Robot without constraints
        kinematic_model_free = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            flip_axes=(True, False, True, True, False, True),
            has_parallelogram=True,
        )

        # Robot with sum constraints (parallelogram linkage)
        kinematic_model_sum_constrained = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            flip_axes=(True, False, True, True, False, True),
            has_parallelogram=True,
            sum_constraints=[(2, 1, -160, -30)],  # J2+J3 constraint in degrees
        )

        # Robot with mixed constraints
        kinematic_model_constrained = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            flip_axes=(True, False, True, True, False, True),
            has_parallelogram=True,
        )

        # Set constraints
        axis_limits = (
            (-175, 175),
            (-75, 75),
            (-180, 0),
            (-270, 270),
            (-125, 125),
            (-270, 270),
        )
        kinematic_model_constrained.set_axis_limits(axis_limits)
        kinematic_model_constrained.set_absolute_constraint(
            0, np.deg2rad(-175), np.deg2rad(175), degrees=False
        )
        kinematic_model_constrained.set_relative_constraint(
            2, 1, np.deg2rad(-160), np.deg2rad(-30), degrees=False
        )
        kinematic_model_constrained.set_sum_constraint(
            2, 1, np.deg2rad(-160), np.deg2rad(-30), degrees=False
        )

        euler_conv = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot_free = Robot(kinematic_model_free, euler_conv)
        robot_sum_constrained = Robot(kinematic_model_sum_constrained, euler_conv)
        robot_constrained = Robot(kinematic_model_constrained, euler_conv)

        # Test pose
        test_pose = ([2000, 0, 1200], [0, 0, 0])
        n_iterations = 100

        # Benchmark free robot
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            robot_free.inverse(test_pose)
        time_free = time.perf_counter() - start_time

        # Benchmark sum constrained robot
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            robot_sum_constrained.inverse(test_pose)
        time_sum_constrained = time.perf_counter() - start_time

        # Benchmark mixed constrained robot
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            robot_constrained.inverse(test_pose)
        time_constrained = time.perf_counter() - start_time

        # Constraints should not significantly impact performance (< 2x slower)
        sum_performance_ratio = time_sum_constrained / time_free
        mixed_performance_ratio = time_constrained / time_free

        print(f"Sum constraint overhead: {sum_performance_ratio:.2f}x")
        print(f"Mixed constraint overhead: {mixed_performance_ratio:.2f}x")

        assert (
            sum_performance_ratio < 2.0
        ), f"Sum constraints cause excessive performance degradation: {sum_performance_ratio:.2f}x"

        assert (
            mixed_performance_ratio < 2.0
        ), f"Mixed constraints cause excessive performance degradation: {mixed_performance_ratio:.2f}x"

        # Verify constraint functionality
        assert kinematic_model_sum_constrained.sum_constraints is not None
        assert len(kinematic_model_sum_constrained.sum_constraints) == 1
        assert kinematic_model_constrained.sum_constraints is not None
        assert len(kinematic_model_constrained.sum_constraints) == 1

        # Robot with constraints
        kinematic_model_constrained = KinematicModel(
            a1=400,
            a2=-250,
            c1=830,
            c2=1175,
            c3=1444,
            c4=230,
            flip_axes=(True, False, True, True, False, True),
            has_parallelogram=True,
        )

        # Set constraints
        axis_limits = (
            (-175, 175),
            (-75, 75),
            (-180, 0),
            (-270, 270),
            (-125, 125),
            (-270, 270),
        )
        kinematic_model_constrained.set_axis_limits(axis_limits)
        kinematic_model_constrained.set_absolute_constraint(
            0, np.deg2rad(-175), np.deg2rad(175)
        )
        kinematic_model_constrained.set_relative_constraint(
            2, 1, np.deg2rad(-160), np.deg2rad(-30)
        )

        euler_conv = EulerConvention("XYZ", extrinsic=False, degrees=True)
        robot_free = Robot(kinematic_model_free, euler_conv)
        robot_constrained = Robot(kinematic_model_constrained, euler_conv)

        # Test pose
        test_pose = ([2000, 0, 1200], [0, 0, 0])
        n_iterations = 100

        # Benchmark free robot
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            robot_free.inverse(test_pose)
        time_free = time.perf_counter() - start_time

        # Benchmark constrained robot
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            robot_constrained.inverse(test_pose)
        time_constrained = time.perf_counter() - start_time

        # Constraints should not significantly impact performance (< 2x slower)
        performance_ratio = time_constrained / time_free
        print(f"Constraint overhead: {performance_ratio:.2f}x")

        assert (
            performance_ratio < 2.0
        ), f"Constraints cause excessive performance degradation: {performance_ratio:.2f}x"

    def test_memory_usage_batch_operations(self, regression_robot: Robot):
        """Test memory efficiency of batch operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Large batch operation
        n_points = 50000
        joints_df = pl.DataFrame(
            {
                "J1": np.random.uniform(-30, 30, n_points),
                "J2": np.random.uniform(-20, 20, n_points),
                "J3": np.random.uniform(-120, -60, n_points),
                "J4": np.random.uniform(-45, 45, n_points),
                "J5": np.random.uniform(-30, 30, n_points),
                "J6": np.random.uniform(-60, 60, n_points),
            }
        )

        # Perform batch operation
        result = regression_robot.batch_forward(joints_df)

        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory

        # Memory usage should be reasonable (< 500MB for 50K points)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB"

        print(f"Memory usage for 50K points: {memory_increase:.1f}MB")

        # Cleanup
        del joints_df, result

    def test_performance_new_methods(self, regression_robot: Robot):
        """Test performance of newer analysis methods."""
        # Test pose
        test_joints = [10, 20, -90, 30, 40, 50]
        test_pose = regression_robot.forward(test_joints)

        n_iterations = 1000

        # Test joint_positions performance
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            positions = regression_robot.joint_positions(test_joints)
        joint_positions_time = time.perf_counter() - start_time

        avg_joint_positions_time = joint_positions_time / n_iterations
        print(f"joint_positions: {avg_joint_positions_time*1000:.3f}ms per operation")

        # Test analyze_configuration performance (fewer iterations - this is more complex)
        n_config_iterations = 100
        start_time = time.perf_counter()
        for _ in range(n_config_iterations):
            config_info = regression_robot.analyze_configuration(test_joints)
        analyze_time = time.perf_counter() - start_time

        avg_analyze_time = analyze_time / n_config_iterations
        print(f"analyze_configuration: {avg_analyze_time*1000:.3f}ms per operation")

        # Test inverse_with_config performance
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            solutions = regression_robot.inverse_with_config(test_pose)
        inverse_config_time = time.perf_counter() - start_time

        avg_inverse_config_time = inverse_config_time / n_iterations
        print(
            f"inverse_with_config: {avg_inverse_config_time*1000:.3f}ms per operation"
        )

        # Performance targets - these are more complex operations so allow more time
        assert (
            avg_joint_positions_time < 0.001
        ), f"joint_positions too slow: {avg_joint_positions_time*1000:.3f}ms"
        assert (
            avg_analyze_time < 0.005
        ), f"analyze_configuration too slow: {avg_analyze_time*1000:.3f}ms"
        assert (
            avg_inverse_config_time < 0.5
        ), f"inverse_with_config too slow: {avg_inverse_config_time*1000:.3f}ms"


if __name__ == "__main__":
    # Run with performance markers
    pytest.main([__file__, "-v", "-m", "not slow"])
