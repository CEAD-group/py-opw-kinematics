#!/usr/bin/env python3
"""
Live Interactive Robot Structure Animation

Creates a web-based 3D animation showing the robot structure (joints, links, axes)
with full interactive camera controls and exports JSON data compatible with robot_animation_viewer.html.
"""

import os
import json
import numpy as np
from py_opw_kinematics import KinematicModel, Robot, EulerConvention


def create_robot():
    """Create robot with exact same parameters as robot_structure_viewer.py"""
    params = {
        "a1": 400.0,
        "a2": -250.0,
        "b": 0.0,
        "c1": 830.0,
        "c2": 1175.0,
        "c3": 1444.0,
        "c4": 230.0,
        "offsets": (0, 0, 0, 0, 0, 0),
        "flip_axes": (False, False, True, False, False, False),
        "has_parallelogram": True,
        "sum_constraints": [
            (
                2,  # J3 (axis index 2)
                1,  # J2 (axis index 1)
                -160,  # min: J2 + J3 > -160°
                -30,  # max: J2 + J3 < -30°
            ),  # Parallelogram constraint: -160° < J2+J3 < -30°
        ],
        "axis_limits": [
            (-175, 175),
            (-95, 75),
            (-256, -10),
            (-2700, 2700),
            (-125, 125),
            (-2700, 2700),
        ],
    }
    km = KinematicModel(**params)
    return Robot(
        km,
        EulerConvention("XYZ", extrinsic=False, degrees=True),
        ee_rotation=(0, -90, 0),
        ee_translation=(200, 0, -200),
    )


def calculate_parallelogram_links(robot: Robot, joints):
    """Calculate parallelogram linkage positions for the robot."""
    try:
        # Get joint positions
        positions = robot.joint_positions(joints)

        # For robots with parallelogram constraint, typically link J2 and J3
        # This creates the parallelogram mechanism common in industrial robots
        if len(positions) >= 4:
            parallelogram_result = robot.parallelogram_positions(
                joints, rest_angle=95.0, link_length=350.0
            )

            if parallelogram_result is not None:
                p1_pos, p2_pos = parallelogram_result
                p1_pos = np.array(p1_pos)
                p2_pos = np.array(p2_pos)

                j2_pos = positions[2]  # J2
                j3_pos = positions[3]  # J3

            # Calculate parallelogram bars between J2 (elbow) and J3 (wrist1)
            # This represents the actual parallelogram linkage mechanism

            # For demonstration, create a simple parallelogram offset
            # In a real robot, these would be the actual mechanical linkage positions

            parallelogram_links = [
                {
                    "start": {
                        "x": j2_pos[0] / 1000.0,
                        "y": j2_pos[1] / 1000.0,
                        "z": j2_pos[2] / 1000.0,
                    },
                    "end": {
                        "x": j3_pos[0] / 1000.0,
                        "y": j3_pos[1] / 1000.0,
                        "z": j3_pos[2] / 1000.0,
                    },
                },
                {
                    "start": {
                        "x": j3_pos[0] / 1000.0,
                        "y": j3_pos[1] / 1000.0,
                        "z": j3_pos[2] / 1000.0,
                    },
                    "end": {
                        "x": p2_pos[0] / 1000.0,
                        "y": p2_pos[1] / 1000.0,
                        "z": p2_pos[2] / 1000.0,
                    },
                },
                {
                    "start": {
                        "x": (p2_pos[0]) / 1000.0,
                        "y": (p2_pos[1]) / 1000.0,
                        "z": (p2_pos[2]) / 1000.0,
                    },
                    "end": {
                        "x": (p1_pos[0]) / 1000.0,
                        "y": (p1_pos[1]) / 1000.0,
                        "z": (p1_pos[2]) / 1000.0,
                    },
                },
                {
                    "start": {
                        "x": (p1_pos[0]) / 1000.0,
                        "y": (p1_pos[1]) / 1000.0,
                        "z": (p1_pos[2]) / 1000.0,
                    },
                    "end": {
                        "x": (j2_pos[0]) / 1000.0,
                        "y": (j2_pos[1]) / 1000.0,
                        "z": (j2_pos[2]) / 1000.0,
                    },
                },
            ]

            return parallelogram_links
    except Exception as e:
        print(f"Warning: Could not calculate parallelogram links: {e}")

    return []


def generate_trajectory(robot: Robot, n_points=100):
    """Generate simple circular trajectory that works."""
    # Simple circular path - more conservative and guaranteed to work
    n_points = 200  # Reduced from 500 for faster processing
    center = [2100, 0, 100]  # Closer, more reachable center
    radius_1 = 700  # Smaller, more conservative radius
    radius_2 = radius_1 - 100
    orientation = [0, 0, 45]  # Simple orientation

    p = 0.8
    q = (1 - p) / 2

    # Get initial solution to use as seed
    current_joints = [0, 0, -100, 0, 10, 0]  # Start from a safe default position

    print(f"Generating {n_points} trajectory points...")
    trajectory_joints = []
    trajectory_joints.append([0, 0, -90, 0, 0, 0])

    poses = []
    for i in range(n_points + 1):  # +1 to close the circle
        angle = 2 * np.pi * (i + ((n_points + 1) * q)) / (n_points / p)
        x = center[0] - radius_1 * np.cos(angle)
        y = center[1] + radius_1 * np.sin(angle)
        z = center[2]
        c = np.degrees((angle - np.pi) / 2)
        pose = (
            (x, y, z),
            (orientation[0], orientation[1], c),
        )
        poses.append(pose)

    for i in range(n_points + 1):  # +1 to close the circle
        new_radius = radius_2
        angle = 2 * np.pi * (i + ((n_points + 1) * q)) / (n_points / p)
        x = center[0] - new_radius * np.cos(angle)
        y = center[1] - new_radius * np.sin(angle)
        z = center[2]
        c = -np.degrees((angle - np.pi) / 2)
        pose = (
            (x, y, z),
            (orientation[0], orientation[1], c),
        )
        poses.append(pose)

    ## interpolate between starting joint axes and first pose
    start_joints = trajectory_joints[0]
    first_pose = poses[0]
    try:
        first_solutions = robot.inverse(first_pose, start_joints)
        if first_solutions:
            first_solution = min(
                first_solutions,
                key=lambda sol: sum((a - b) ** 2 for a, b in zip(sol, start_joints)),
            )
            n_interp = 100
            for t in range(1, n_interp + 1):
                interp_joints = [
                    start + (end - start) * t / n_interp
                    for start, end in zip(start_joints, first_solution)
                ]
                trajectory_joints.append(interp_joints)
            current_joints = first_solution
        else:
            print("❌ Could not find IK solution for first pose during interpolation.")
    except Exception as e:
        print(f"❌ Error during interpolation to first pose: {e}")
    for pose in poses:
        x, y, z = pose[0]

        # Use individual inverse kinematics with previous solution as seed
        try:
            solutions = robot.inverse(pose, current_joints)

            if solutions:
                # Find solution closest to previous joints (for continuity)
                best_solution = min(
                    solutions,
                    key=lambda sol: sum(
                        (a - b) ** 2 for a, b in zip(sol, current_joints)
                    ),
                )
                trajectory_joints.append(best_solution)
                current_joints = best_solution
            else:
                # Use previous joints if no solution found
                trajectory_joints.append(current_joints)

        except Exception as e:
            print(f"❌ Error at point {len(trajectory_joints)}: {e}")
            trajectory_joints.append(current_joints)

    # # Select evenly spaced points to get the desired number
    # if len(trajectory_joints) > n_points:
    #     step = len(trajectory_joints) // n_points
    #     trajectory_joints = trajectory_joints[::step][:n_points]

    # Return the trajectory we just generated
    print(f"✅ Generated trajectory with {len(trajectory_joints)} points")
    return trajectory_joints


def calculate_tcp_frame(robot: Robot, joints):
    """Calculate TCP frame orientation based on robot pose."""
    try:
        # Get the forward kinematics result which includes orientation
        pose = robot.forward(joints)
        tcp_pos, tcp_rot = pose

        # Convert position from mm to meters
        tcp_position = {
            "x": tcp_pos[0] / 1000.0,
            "y": tcp_pos[1] / 1000.0,
            "z": tcp_pos[2] / 1000.0,
        }

        # Convert rotation from degrees to orientation matrix
        # tcp_rot is in degrees [rx, ry, rz] in XYZ Euler convention
        rx, ry, rz = np.radians(tcp_rot)

        # Calculate rotation matrix for XYZ Euler angles
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)

        # Rotation matrix for XYZ Euler angles (extrinsic)
        R_x = np.array([[1, 0, 0], [0, cos_rx, -sin_rx], [0, sin_rx, cos_rx]])

        R_y = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])

        R_z = np.array([[cos_rz, -sin_rz, 0], [sin_rz, cos_rz, 0], [0, 0, 1]])

        # Combined rotation matrix (intrinsic XYZ - corrects mirroring issue)
        R = R_x @ R_y @ R_z

        # Extract frame axes (unit vectors)
        x_axis = R[:, 0]  # X-axis direction
        y_axis = R[:, 1]  # Y-axis direction
        z_axis = R[:, 2]  # Z-axis direction

        return {
            "position": tcp_position,
            "orientation": {"x": tcp_rot[0], "y": tcp_rot[1], "z": tcp_rot[2]},
            "x_axis": x_axis.tolist(),  # Convert numpy array to list for JSON serialization
            "y_axis": y_axis.tolist(),  # Convert numpy array to list for JSON serialization
            "z_axis": z_axis.tolist(),  # Convert numpy array to list for JSON serialization
        }

    except Exception as e:
        print(f"Warning: Could not calculate TCP frame: {e}")
        # Return default frame
        positions = robot.joint_positions(joints)
        tcp_pos = positions[-1] if positions else [0, 0, 0]
        return {
            "position": {
                "x": tcp_pos[0] / 1000.0,
                "y": tcp_pos[1] / 1000.0,
                "z": tcp_pos[2] / 1000.0,
            },
            "orientation": {"x": 0, "y": 0, "z": 0},
            "x_axis": [1, 0, 0],  # Use list instead of numpy array
            "y_axis": [0, 1, 0],  # Use list instead of numpy array
            "z_axis": [0, 0, 1],  # Use list instead of numpy array
        }


def create_collision_shapes_config():
    """
    Create collision shapes configuration for the robot.

    OFFSET COORDINATE SYSTEM:

    For shapes attached to joints 0-6: World Coordinates
    - X: Forward/Backward in world frame (+ = forward, - = backward)
    - Y: Left/Right in world frame (+ = left, - = right)
    - Z: Up/Down in world frame (+ = up, - = down)

    For TCP-attached shapes (joint 7): TCP Local Coordinates
    - X: Forward/Backward relative to TCP orientation (+ = TCP X-axis direction)
    - Y: Left/Right relative to TCP orientation (+ = TCP Y-axis direction)
    - Z: Up/Down relative to TCP orientation (+ = TCP Z-axis direction)

    All units are in MILLIMETERS (mm) to match the kinematics coordinate system.

    EXAMPLES:
    - [100, 0, 0]: Move 100mm forward (world X+ or TCP X+)
    - [0, 50, 0]: Move 50mm left (world Y+ or TCP Y+)
    - [0, 0, 30]: Move 30mm up (world Z+ or TCP Z+)
    - [-50, 0, 0]: Move 50mm backward (world X- or TCP X-)

    ROTATION (for cylinders and boxes only):
    - "rotation": [rx, ry, rz] - Euler angles in degrees (XYZ order)
    - Applied AFTER the shape is oriented along the link direction
    - [90, 0, 0]: Rotate 90° around X-axis
    - [0, 45, 0]: Rotate 45° around Y-axis
    - [0, 0, -90]: Rotate -90° around Z-axis

    SIZE CONFIGURATION:
    - For boxes: "size": [length, width, height] - all in mm
    - For cylinders: "radius": value in mm
    - For spheres: "radius": value in mm
    """
    return {
        "link1_cylinder": {
            "type": "cylinder",
            "from_joint": 1,
            "to_joint": 2,
            "radius": 80,  # mm
            "color": "#FF6B6B",
            "opacity": 0.6,
            "description": "Shoulder to elbow link protection",
            # Optional offset: [x, y, z] in mm relative to the shape center
            # "offset": [0, 0, 50]  # Example: 50mm offset in Z direction
            # Optional rotation: [rx, ry, rz] in degrees to rotate from link direction
            # "rotation": [0, 90, 0]  # Example: rotate 90° around Y to make horizontal
        },
        "link2_box": {
            "type": "box",
            "from_joint": 2,
            "to_joint": 3,
            "size": [1000, 120, 120],  # [length, width, height] in mm
            "color": "#4ECDC4",
            "opacity": 0.6,
            "description": "Elbow to wrist link protection",
            # Example offset: move box 30mm up (Z+) and 20mm to the side (Y+)
            # "offset": [0, 20, 30]
            # Example rotation: rotate box 45° around Z-axis to angle it
            "rotation": [0, 0, 45],
        },
        "wrist_sphere": {
            "type": "sphere",
            "at_joint": 4,
            "radius": 90,  # mm
            "color": "#FFE66D",
            "opacity": 0.5,
            "description": "Wrist joint protection",
            # Example: offset sphere 50mm forward (X+) in world coordinates
            "offset": [50, 0, 0],
        },
        "base_box": {
            "type": "box",
            "from_joint": 0,
            "to_joint": 1,
            "size": [700, 200, 200],  # [length, width, height] in mm
            "color": "#9B59B6",
            "opacity": 0.4,
            "description": "Base to shoulder link protection",
            "offset": [0, 0, 100],  # Move 100mm up (Z+) to clear the base
        },
        "end_effector_cylinder": {
            "type": "cylinder",
            "from_joint": 6,  # Last mechanical joint
            "to_joint": 8,  # TCP position (includes ee_translation) - index 8 in positions array
            "radius": 30,  # mm - cylinder from last joint to TCP
            "color": "#E67E22",
            "opacity": 0.5,
            "description": "Joint 6 to TCP protection",
            # Note: length is automatically determined by distance between joint 6 and TCP
        },
        "tcp_sphere": {
            "type": "sphere",
            "at_joint": 8,  # TCP position (includes ee_translation) - index 8 in positions array
            "radius": 40,  # mm
            "color": "#FF4444",
            "opacity": 0.7,
            "description": "TCP position indicator",
            "offset": [100, 0, 0],  # 100mm forward from TCP in TCP local coordinates
        },
    }


def generate_animation_json(robot, trajectory):
    """Generate JSON animation data compatible with robot_animation_viewer.html"""

    frames = []
    all_tcp_positions = []

    print("Generating JSON animation data...")

    # Calculate all TCP positions first for trajectory
    for joints in trajectory:
        try:
            positions = robot.joint_positions(joints)
            tcp_pos = positions[-1]
            all_tcp_positions.append(
                {
                    "x": tcp_pos[0] / 1000.0,  # Convert mm to meters
                    "y": tcp_pos[1] / 1000.0,
                    "z": tcp_pos[2] / 1000.0,
                }
            )
        except Exception:
            continue

    # Generate frames
    for frame_idx, joints in enumerate(trajectory):
        try:
            positions = robot.joint_positions(joints)

            # Convert positions to viewer format (meters)
            robot_positions = []
            for pos in positions:
                robot_positions.append(
                    {
                        "x": pos[0] / 1000.0,  # Convert mm to meters
                        "y": pos[1] / 1000.0,
                        "z": pos[2] / 1000.0,
                    }
                )

            # Add TCP position (includes ee_translation) as the last position
            tcp_pose = robot.forward(joints)
            tcp_pos, _ = tcp_pose
            robot_positions.append(
                {
                    "x": tcp_pos[0] / 1000.0,  # Convert mm to meters
                    "y": tcp_pos[1] / 1000.0,
                    "z": tcp_pos[2] / 1000.0,
                }
            )

            # Calculate parallelogram links
            parallelogram_links = calculate_parallelogram_links(robot, joints)

            # Calculate proper TCP frame with orientation
            tcp_frame = calculate_tcp_frame(robot, joints)

            # Create frame data
            frame_data = {
                "frame": frame_idx,
                "joint_angles": joints,
                "positions": robot_positions,
                "trajectory": all_tcp_positions,  # Always include complete trajectory
                "parallelogram_links": parallelogram_links,
                "end_effector_pose": tcp_frame,
            }

            frames.append(frame_data)

        except Exception as e:
            print(f"Warning: Error generating frame {frame_idx}: {e}")
            continue

    # Create complete animation data structure
    animation_data = {
        "metadata": {
            "robot_type": "comau_nj165",
            "trajectory_type": "circular",
            "total_frames": len(frames),
            "created_at": "2025-12-04T12:00:00Z",
            "has_parallelogram": True,
            "collision_shapes": create_collision_shapes_config(),
        },
        "frames": frames,
    }

    print(f"✅ Generated {len(frames)} frames for JSON export")
    return animation_data


def main():
    """Main function."""
    print("🤖 Live Interactive Robot Structure Animation")
    print("=" * 50)

    robot = create_robot()
    trajectory = generate_trajectory(robot)

    # Generate animation data compatible with robot_animation_viewer.html
    animation_data = generate_animation_json(robot, trajectory)

    # Save JSON file
    json_filename = "live_robot_animation.json"
    with open(json_filename, "w") as f:
        json.dump(animation_data, f, indent=2)

    print(f"✅ Animation data saved to {json_filename}")
    print(f"📁 File size: {os.path.getsize(json_filename) / 1024:.1f} KB")
    print(f"📊 Frames: {len(animation_data['frames'])}")
    print(
        f"🔗 Collision shapes: {len(animation_data['metadata']['collision_shapes'])} configured"
    )
    print(
        "💡 Use robot_animation_viewer_collision_shapes.html to view this animation with collision shapes"
    )

    # Print collision shape summary
    print("\n🛡️  Collision Shape Configuration:")
    for name, config in animation_data["metadata"]["collision_shapes"].items():
        if config["type"] == "cylinder":
            print(
                f"  • {name}: {config['type']} (J{config['from_joint']}-J{config['to_joint']}, r={config['radius']}m)"
            )
        elif config["type"] == "box":
            print(
                f"  • {name}: {config['type']} (J{config['from_joint']}-J{config['to_joint']}, size={config['size']})"
            )
        elif config["type"] == "sphere":
            print(
                f"  • {name}: {config['type']} (J{config['at_joint']}, r={config['radius']}m)"
            )


if __name__ == "__main__":
    main()
