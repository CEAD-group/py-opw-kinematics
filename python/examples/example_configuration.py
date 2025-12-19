#!/usr/bin/env python3
"""
Comprehensive Robot Configuration Management Example

This unified example demonstrates all aspects of robot configuration selection:
1. Practical STAT/TU configuration input for path planning
2. SINUMERIK ROBX-compatible STAT/TU bit analysis
3. Configuration selection and comparison methods
4. Industrial applications and use cases

Combines functionality from:
- example_config_input.py: Path planning with target configurations
- example_stat_tu_bits.py: STAT/TU bit analysis and format support
- example_configuration_selection.py: Configuration matching and selection
"""
from py_opw_kinematics import KinematicModel, Robot, EulerConvention
import polars as pl
import numpy as np


class ComprehensiveConfigurationDemo:
    """Comprehensive demonstration of robot configuration management."""

    def __init__(self):
        """Initialize robot models for different demonstration scenarios."""
        # Primary robot for STAT/TU demonstrations (Comau NJ165-3.0 style)
        # Demonstrating constraint initialization with degrees input (stored as radians)
        self.primary_model = KinematicModel(
            a1=460,  # $MC_ROBX_MAIN_LENGTH_AB[0]
            a2=-250,  # - $MC_ROBX_TX3P3_POS[2]
            b=0,
            c1=1140,  # $MC_ROBX_TIRORO_POS[2]
            c2=1050,  # $MC_ROBX_MAIN_LENGTH_AB[1]
            c3=1510,  # $MC_ROBX_TX3P3_POS[0]
            c4=282,  # $MC_ROBX_TFLWP_POS[2]
            offsets=(0, 0, 0, 0, 0, 0),
            flip_axes=(False, False, True, False, True, False),
            has_parallelogram=True,
            relative_constraints=[
                (
                    2,
                    1,
                    -160,
                    -30,
                ),  # J3-J2 parallelogram limits in degrees (stored as radians)
            ],
        )  # Create kinematic model for Comau NJ165-3.0

        euler = EulerConvention(
            "XYZ", extrinsic=False, degrees=True
        )  # Create Euler convention
        ee_translation = (
            145.5,
            -353,
            -330.5,
        )  # set the end-effector translation (TCP offset)
        ee_rotation = (0, -90, 0)  # set the end-effector rotation ($MC_ROBX_TFLWP_RPY)
        start_position_joints = (
            0,
            0,
            -100,
            0,
            10,
            0,
        )  # set the current joint angles so that the robot is not in a singularity

        # Setup robots
        euler = EulerConvention("XYZ", extrinsic=False, degrees=True)
        self.robot = Robot(self.primary_model, euler, ee_translation, ee_rotation)

        # Demonstrate constraint unit handling verification
        self._verify_constraint_units()

    def _verify_constraint_units(self):
        """Verify that constraints are stored correctly in radians."""
        print("\n🔧 Constraint Unit Handling Verification:")
        stored_constraints = self.primary_model.relative_constraints
        if stored_constraints:
            j3_j2_constraint = stored_constraints[0]
            print(f"  Input during init: J3-J2 range [-160°, -30°]")
            print(
                f"  Stored internally: [{j3_j2_constraint[2]:.4f}, {j3_j2_constraint[3]:.4f}] rad"
            )
            print(
                f"  Converted back: [{np.rad2deg(j3_j2_constraint[2]):.1f}°, {np.rad2deg(j3_j2_constraint[3]):.1f}°]"
            )
        else:
            print("  No relative constraints found")

    def print_header(self, title, level=1):
        """Print formatted section headers."""
        if level == 1:
            print("\n" + "=" * 80)
            print(f"=== {title} ===")
            print("=" * 80)
        elif level == 2:
            print(f"\n🔧 {title}")
            print("-" * 60)
        else:
            print("-" * 40)

    def demonstrate_path_planning(self):
        """Demonstrate practical path planning with STAT/TU configurations."""
        self.print_header("PRACTICAL PATH PLANNING WITH STAT/TU", 1)

        print(
            "Demonstrating real-world robot path planning using STAT/TU configuration control"
        )
        print("for maintaining consistent postures and avoiding singularities.")
        print()

        # Define a realistic welding path
        waypoints = [
            ([2000.0, 500.0, 1200.0], [0.0, 0.0, 0.0]),  # Start position
            ([2100.0, 400.0, 1300.0], [15.0, -10.0, 5.0]),  # Waypoint 1
            ([2200.0, 200.0, 1400.0], [30.0, -20.0, 10.0]),  # Waypoint 2
            ([2000.0, 0.0, 1500.0], [45.0, -30.0, 15.0]),  # End position
        ]

        # Choose preferred configuration for the operation
        preferred_config = "STAT=110 TU=000000"  # shoulder_right + elbow_up + handflip

        print(f"🎯 Target Configuration: '{preferred_config}'")
        print("   STAT=110 (binary) = shoulder_right + elbow_up + handflip")
        print("   • Bit 0 (shoulder_left=0): Right shoulder configuration")
        print("   • Bit 1 (elbow_up=1): Elbow up (avoids floor collisions)")
        print("   • Bit 2 (handflip=1): Handflip (wrist flipped for downward tasks)")
        print("   TU=000000: All joints prefer positive angles")
        print()

        self.print_header("WAYPOINT ANALYSIS", 3)

        successful_path = []
        for i, (pos, rot) in enumerate(waypoints):
            print(f"Waypoint {i+1}: Position {pos}, Rotation {rot}")

            result = self.robot.inverse_with_target_config((pos, rot), preferred_config)

            if result:
                joints, config, score = result
                successful_path.append((joints, config, score))

                # Color code based on match quality
                if score == 3:
                    status = "✅ Perfect match"
                elif score == 2:
                    status = "⚠️  Good match"
                else:
                    status = "⚡ Partial match"

                print(f"  Result: {status} ({score}/3 criteria)")
                print(f"  Config: {config}")
                print(
                    f"  Joints: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}, "
                    f"{joints[3]:.1f}, {joints[4]:.1f}, {joints[5]:.1f}]°"
                )
            else:
                print("  Result: ❌ No solution found for this pose")
                successful_path.append(None)
            print()

        # Path analysis summary
        self.print_header("PATH PLANNING RESULTS", 3)
        valid_waypoints = sum(1 for wp in successful_path if wp is not None)
        perfect_matches = sum(1 for wp in successful_path if wp and wp[2] == 3)

        print(f"Valid waypoints: {valid_waypoints}/{len(waypoints)}")
        print(f"Perfect matches: {perfect_matches}/{len(waypoints)}")
        print(f"Configuration consistency: {perfect_matches/len(waypoints)*100:.1f}%")
        print()

        if valid_waypoints == len(waypoints):
            print("✅ Complete path found with consistent configuration!")
            print("🚀 Ready for robot execution")
        else:
            print("⚠️  Some waypoints need alternative configurations")
            print("💡 Consider adjusting poses or using fallback configurations")

        return waypoints[0]  # Return first waypoint for later analysis

    def demonstrate_stat_tu_analysis(self, test_pose):
        """Demonstrate comprehensive STAT/TU bit analysis."""
        self.print_header("SINUMERIK ROBX STAT/TU BIT ANALYSIS", 1)

        print("Demonstrating industrial STAT/TU configuration system compatible")
        print("with SINUMERIK ROBX controllers and professional robot programming.")
        print()

        self.print_header("BASIC CONFIGURATION ANALYSIS", 3)

        # Get all solutions
        solutions = self.robot.inverse(test_pose)
        print(f"Found {len(solutions)} inverse kinematics solutions")
        print()

        # Analyze first few solutions
        print("Configuration analysis for first 3 solutions:")
        for i, joints in enumerate(solutions[:3]):
            config = self.robot.analyze_configuration(joints)
            print(f"  Solution {i+1}:")
            print(
                f"    Joints: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}, "
                f"{joints[3]:.1f}, {joints[4]:.1f}, {joints[5]:.1f}]°"
            )
            print(f"    Config: {config}")
            print()

        self.print_header("TARGET CONFIGURATION MATCHING", 3)

        # Test different STAT/TU format support
        test_targets = [
            "STAT=110 TU=000000",  # Binary STAT, binary TU
            "STAT=5 TU=3",  # Decimal STAT, decimal TU
            "STAT=B000101 TU=B000011",  # B-prefix binary format
        ]

        for target in test_targets:
            print(f"Target: '{target}'")
            result = self.robot.inverse_with_target_config(test_pose, target)
            if result:
                joints, config, score = result
                print(f"  Found: {config} (score: {score}/3)")
                print(
                    f"  Joints: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}, "
                    f"{joints[3]:.1f}, {joints[4]:.1f}, {joints[5]:.1f}]°"
                )
            else:
                print("  No matching solution found")
            print()

        self.print_header("STAT BIT DETAILED ANALYSIS", 3)

        # Test STAT bit matching
        print(
            "Finding solutions matching STAT=110 (shoulder_right + elbow_up + handflip):"
        )
        stat_matches = self.robot.find_stat_matches(test_pose, 6)  # 6 = 110 binary
        for joints, config, stat_score in stat_matches:
            print(f"  {config} (STAT score: {stat_score}/3)")
        print()

        # Show STAT bit meanings
        print("ROBX STAT BIT MEANINGS:")
        print("  Bit 0 (shoulder_left): Shoulder configuration")
        print("    0 = Right shoulder")
        print("    1 = Left shoulder")
        print("  Bit 1 (elbow_up): Elbow configuration")
        print("    0 = Elbow down (virtual J3 < 0°)")
        print("    1 = Elbow up (virtual J3 ≥ 0°)")
        print("  Bit 2 (handflip): Handflip configuration")
        print("    0 = No handflip (J5 ≥ 0°)")
        print("    1 = Handflip (J5 < 0°)")
        print()

        # Configuration comparison table
        self.print_header("CONFIGURATION COMPARISON TABLE", 3)

        print("STAT | Binary | Shoulder | Elbow | Handflip | Solutions")
        print("-----|--------|----------|-------|----------|----------")
        for stat in range(8):
            binary = f"{stat:03b}"
            shoulder = "Left " if (stat & 1) else "Right"
            elbow = "Up  " if (stat & 2) else "Down"
            handflip = "Yes" if (stat & 4) else "No "

            # Find solutions with this STAT
            matches = self.robot.find_stat_matches(test_pose, stat)
            count = len(matches)
            desc = f"{count} found" if count > 0 else "None"

            print(
                f"  {stat}  |  {binary}   | {shoulder}   | {elbow} |    {handflip}    | {desc}"
            )
        print()

        self.print_header("TURN NUMBER (TU) ANALYSIS", 3)

        print("TU bits indicate joint angle sign preferences:")
        print("• Each bit represents one joint (J1-J6)")
        print("• 0 = prefer positive angle, 1 = prefer negative angle")
        print("• Uses [-360°, 360°] normalization (270° = positive, -200° = negative)")
        print()

        # Test with extreme angles
        extreme_joints = [0.0, 0.0, 0.0, 450.0, 0.0, -380.0]
        config = self.robot.analyze_configuration(extreme_joints)
        print(f"Example with extreme angles:")
        print(f"  Input joints: [0°, 0°, 0°, 450°, 0°, -380°]")
        print(f"  Normalized: J4=450°→90° (positive), J6=-380°→-20° (negative)")
        print(f"  Config: {config}")
        print()

    def demonstrate_configuration_selection(self, test_pose):
        """Demonstrate configuration selection and comparison methods."""
        self.print_header("CONFIGURATION SELECTION AND COMPARISON", 1)

        print("Comparing legacy J3+/- J5+/- OH+/- format with modern STAT/TU system")
        print("and demonstrating various configuration selection methods.")
        print()

        # Get all solutions
        solutions = self.robot.inverse(test_pose)
        if not solutions:
            print("❌ No solutions found for test pose")
            return

        self.print_header("LEGACY vs MODERN FORMAT COMPARISON", 2)

        print("All available configurations (legacy format):")
        print(
            f"{'#':<3} {'Legacy Config':<15} {'Modern STAT/TU':<20} {'J1':<8} {'J2':<8} {'J3':<8} {'J4':<8} {'J5':<8} {'J6':<8}"
        )
        print("-" * 95)

        for i, solution in enumerate(solutions):
            modern_config = self.robot.analyze_configuration(solution)
            print(
                f"{i+1:<3} {modern_config:<20} "
                f"{solution[0]:<8.1f} {solution[1]:<8.1f} {solution[2]:<8.1f} "
                f"{solution[3]:<8.1f} {solution[4]:<8.1f} {solution[5]:<8.1f}"
            )

        self.print_header("CONFIGURATION TARGET MATCHING", 2)

        # Test legacy format targets
        modern_targets = [
            "STAT=010 TU=000000",
            "STAT=100 TU=000000",
            "STAT=110 TU=000000",
        ]

        print("\nModern STAT/TU format matching:")
        for target_config in modern_targets:
            result = self.robot.inverse_with_target_config(test_pose, target_config)
            if result:
                solution, config, score = result
                print(
                    f"Target: {target_config:<18} → Found: {config:<18} (Score: {score}/3)"
                )
            else:
                print(f"Target: {target_config:<18} → No matching solution found")

    def demonstrate_alternative_configurations(self, test_pose):
        """Demonstrate alternative configuration testing."""
        self.print_header("ALTERNATIVE CONFIGURATION ANALYSIS", 2)

        print(f"Testing different configurations for pose: {test_pose[0]}")
        print()

        alternative_configs = [
            "STAT=010 TU=000000",  # shoulder_right + elbow_up + no_handflip
            "STAT=110 TU=000000",  # shoulder_right + elbow_up + handflip
            "STAT=100 TU=000000",  # shoulder_right + elbow_down + handflip
            "STAT=111 TU=000000",  # shoulder_left + elbow_up + handflip
        ]

        for alt_config in alternative_configs:
            result = self.robot.inverse_with_target_config(test_pose, alt_config)
            if result:
                joints, config, score = result
                # Decode STAT bits for readable output
                stat_value = int(alt_config.split("=")[1].split()[0], 2)
                shoulder = "left" if (stat_value & 1) else "right"
                elbow = "up" if (stat_value & 2) else "down"
                handflip = "flip" if (stat_value & 4) else "normal"
                description = f"shoulder_{shoulder}, elbow_{elbow}, {handflip}"
                print(f"{alt_config} → {config} (Score: {score}/3) [{description}]")
            else:
                print(f"{alt_config} → No solution")

    def demonstrate_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        self.print_header("BATCH PROCESSING COMPARISON", 1)

        # Test poses for batch processing
        test_poses_data = [
            {
                "X": 2000.0,
                "Y": 500.0,
                "Z": 1200.0,
                "A": 0.0,
                "B": 0.0,
                "C": 0.0,
                "name": "Test Pose 1",
            },
            # {
            #     "X": 1800.0,
            #     "Y": -300.0,
            #     "Z": 1500.0,
            #     "A": 30.0,
            #     "B": 45.0,
            #     "C": -15.0,
            #     "name": "Test Pose 2",
            # },
            # {
            #     "X": 2200.0,
            #     "Y": 0.0,
            #     "Z": 800.0,
            #     "A": 0.0,
            #     "B": 90.0,
            #     "C": 0.0,
            #     "name": "Test Pose 3",
            # },
        ]

        poses_df = pl.DataFrame(test_poses_data)
        print("Comparing batch_inverse with individual configuration selection...")

        # Use batch_inverse
        start_position_joints = (0, 0, -100, 0, 10, 0)
        batch_solutions = self.robot.batch_inverse(
            poses_df, current_joints=start_position_joints
        )

        if batch_solutions is not None:
            print(f"\nBatch processing results:")
            for i, pose_data in enumerate(test_poses_data):
                print(f"\n{pose_data['name']}:")

                if i < len(batch_solutions):
                    row = batch_solutions.row(i, named=True)
                    joint_columns = ["J1", "J2", "J3", "J4", "J5", "J6"]

                    if all(col in row for col in joint_columns):
                        batch_solution = [row[col] for col in joint_columns]

                        if all(val is not None for val in batch_solution):
                            # Analyze batch solution
                            try:
                                batch_config = self.robot.analyze_configuration(
                                    batch_solution
                                )
                                print(f"  Batch solution: {batch_config}")
                                print(
                                    f"  Joints: [{batch_solution[0]:.1f}, {batch_solution[1]:.1f}, "
                                    f"{batch_solution[2]:.1f}, {batch_solution[3]:.1f}, "
                                    f"{batch_solution[4]:.1f}, {batch_solution[5]:.1f}]°"
                                )
                            except Exception as e:
                                print(f"  ❌ Error analyzing batch solution: {e}")
                        else:
                            print(f"  ❌ Invalid batch solution (contains None values)")
                    else:
                        print(f"  ❌ Missing joint columns in batch result")
                else:
                    print(f"  ❌ Pose index out of range")
        else:
            print("❌ Batch inverse failed")

    def demonstrate_axis_limits_impact(self, test_pose):
        """Demonstrate how axis limits affect available configurations."""
        self.print_header("AXIS LIMITS IMPACT ON CONFIGURATION SELECTION", 1)

        print("Demonstrating how joint axis limits significantly reduce available")
        print("configurations and why STAT/TU selection becomes even more critical.")
        print()

        # Define realistic axis limits for industrial robot
        axis_limits_nj290 = (  # lower and upper limits per axis in degrees
            (-175, 175),
            (-75, 75),
            (-220, 0),
            (-2700, 2700),
            (-125, 125),
            (-2700, 2700),
        )
        parallelogram_limits = (-160.0, -30.0)  # relative limits for J3 relative to J2

        print("🔧 AXIS LIMITS CONFIGURATION:")
        print(
            f"  J1 (Base):      {axis_limits_nj290[0][0]:6.1f}° to {axis_limits_nj290[0][1]:6.1f}°"
        )
        print(
            f"  J2 (Shoulder):  {axis_limits_nj290[1][0]:6.1f}° to {axis_limits_nj290[1][1]:6.1f}°"
        )
        print(
            f"  J3 (Elbow):     {axis_limits_nj290[2][0]:6.1f}° to {axis_limits_nj290[2][1]:6.1f}°"
        )
        print(
            f"  J4 (Wrist):     {axis_limits_nj290[3][0]:6.0f}° to {axis_limits_nj290[3][1]:6.0f}°"
        )
        print(
            f"  J5 (Bend):      {axis_limits_nj290[4][0]:6.1f}° to {axis_limits_nj290[4][1]:6.1f}°"
        )
        print(
            f"  J6 (Flange):    {axis_limits_nj290[5][0]:6.0f}° to {axis_limits_nj290[5][1]:6.0f}°"
        )
        print(
            f"  Parallelogram:  J3-J2 ∈ [{parallelogram_limits[0]:.1f}°, {parallelogram_limits[1]:.1f}°]"
        )
        print()

        # Test different constraint scenarios
        test_scenarios = [
            {
                "name": "No Constraints",
                "apply_limits": False,
                "apply_relative": False,
                "description": "All kinematically valid solutions",
            },
            {
                "name": "Simple Axis Limits",
                "apply_limits": True,
                "apply_relative": False,
                "description": "Joint angle limits only",
            },
            {
                "name": "Parallelogram Constraint",
                "apply_limits": False,
                "apply_relative": True,
                "description": "J3-J2 relative constraint only",
            },
            {
                "name": "Full Constraints",
                "apply_limits": True,
                "apply_relative": True,
                "description": "Both axis limits and parallelogram constraint",
            },
        ]

        print("📊 CONSTRAINT IMPACT ANALYSIS:")
        print(f"Test pose: {test_pose[0]}, {test_pose[1]}")
        print()

        scenario_results = []

        for scenario in test_scenarios:
            print(f"🔍 {scenario['name']} ({scenario['description']})")

            # Clear previous constraints
            self.primary_model.clear_all_constraints()

            # Apply constraints as specified
            if scenario["apply_limits"]:
                self.primary_model.set_axis_limits(limits=axis_limits_nj290)

            if scenario["apply_relative"]:
                self.primary_model.set_relative_constraint(
                    axis=2,  # J3 is axis 2
                    reference_axis=1,  # J2 is axis 1
                    min_offset=parallelogram_limits[0],
                    max_offset=parallelogram_limits[1],
                )

            # Create robot with current constraints
            euler = EulerConvention("XYZ", extrinsic=False, degrees=True)
            ee_translation = (145.5, -353, -330.5)
            ee_rotation = (0, -90, 0)
            constrained_robot = Robot(
                self.primary_model, euler, ee_translation, ee_rotation
            )

            # Get solutions
            try:
                solutions = constrained_robot.inverse(test_pose)

                if solutions and len(solutions) > 0:
                    print(f"  ✅ Found {len(solutions)} valid solution(s)")

                    # Analyze configurations of valid solutions
                    configs = []
                    for i, joints in enumerate(solutions):
                        config = constrained_robot.analyze_configuration(joints)
                        configs.append(config)
                        if i < 3:  # Show first 3 solutions
                            print(f"    Solution {i+1}: {config}")
                            print(
                                f"      Joints: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}, {joints[3]:.1f}, {joints[4]:.1f}, {joints[5]:.1f}]°"
                            )

                    if len(solutions) > 3:
                        print(f"    ... and {len(solutions)-3} more solutions")

                    # Count unique configurations
                    unique_configs = set(configs)
                    print(f"  📋 Unique STAT/TU configurations: {len(unique_configs)}")

                    scenario_results.append(
                        {
                            "name": scenario["name"],
                            "solutions": len(solutions),
                            "unique_configs": len(unique_configs),
                            "configs": list(unique_configs),
                        }
                    )
                else:
                    print(f"  ❌ No valid solutions found")
                    scenario_results.append(
                        {
                            "name": scenario["name"],
                            "solutions": 0,
                            "unique_configs": 0,
                            "configs": [],
                        }
                    )

            except Exception as e:
                print(f"  ❌ Error: {e}")
                scenario_results.append(
                    {
                        "name": scenario["name"],
                        "solutions": 0,
                        "unique_configs": 0,
                        "configs": [],
                    }
                )

            print()

        # Summary comparison
        self.print_header("CONSTRAINT IMPACT SUMMARY", 3)

        print("Scenario                    | Solutions | Configurations | Reduction")
        print("-" * 70)

        base_solutions = scenario_results[0]["solutions"] if scenario_results else 0

        for result in scenario_results:
            solutions = result["solutions"]
            configs = result["unique_configs"]

            if base_solutions > 0:
                reduction = f"{(1 - solutions/base_solutions)*100:5.1f}%"
            else:
                reduction = "  N/A"

            print(
                f"{result['name']:<27} | {solutions:^9} | {configs:^14} | {reduction:^9}"
            )

        print()

        # Configuration availability analysis
        if len(scenario_results) >= 4:
            self.print_header("CONFIGURATION AVAILABILITY ANALYSIS", 3)

            no_constraints = set(scenario_results[0]["configs"])
            full_constraints = set(scenario_results[3]["configs"])

            if no_constraints and full_constraints:
                lost_configs = no_constraints - full_constraints
                remaining_configs = no_constraints & full_constraints

                print(f"📉 Lost configurations due to constraints: {len(lost_configs)}")
                if lost_configs:
                    for config in sorted(lost_configs):
                        print(f"    ❌ {config}")

                print(f"📈 Remaining valid configurations: {len(remaining_configs)}")
                if remaining_configs:
                    for config in sorted(remaining_configs):
                        print(f"    ✅ {config}")

                print()
                print("💡 PRACTICAL IMPLICATIONS:")
                print(
                    f"  • Axis limits reduce solution space by {(1-len(remaining_configs)/len(no_constraints))*100:.0f}%"
                )
                print(
                    f"  • Configuration selection becomes critical with limited options"
                )
                print(f"  • STAT/TU targeting essential for consistent robot behavior")
                print(
                    f"  • Path planning must consider constraint-valid configurations"
                )
                print(f"  • Some operational poses may become unreachable")

        # Reset constraints
        self.primary_model.clear_all_constraints()

        print()

    def run_comprehensive_demo(self):
        """Run the complete comprehensive demonstration."""
        print("🤖 COMPREHENSIVE ROBOT CONFIGURATION MANAGEMENT")
        print("=" * 80)
        print("Unified demonstration of STAT/TU configuration systems")
        print("for professional robot programming and industrial applications.")
        print()

        # 1. Path planning demonstration
        test_pose = self.demonstrate_path_planning()

        # 2. STAT/TU bit analysis
        self.demonstrate_stat_tu_analysis(test_pose)

        # 3. Configuration selection comparison
        self.demonstrate_configuration_selection(test_pose)

        # 4. Alternative configurations
        self.demonstrate_alternative_configurations(test_pose)

        # 5. Axis limits impact
        self.demonstrate_axis_limits_impact(test_pose)

        # 6. Batch processing
        self.demonstrate_batch_processing()


def main():
    """Run the comprehensive configuration demonstration."""
    demo = ComprehensiveConfigurationDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()
