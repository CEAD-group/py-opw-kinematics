# Advanced Features Tutorial

This tutorial covers advanced features of py-opw-kinematics for production robotics applications.

## Table of Contents

1. [Axis Limits and Constraints](#axis-limits-and-constraints)
2. [Batch Processing for High Performance](#batch-processing-for-high-performance)
3. [Trajectory Visualization](#trajectory-visualization)
4. [Handling Singularities](#handling-singularities)
5. [Industrial Robot Configurations](#industrial-robot-configurations)
6. [Trajectory Planning](#trajectory-planning)
7. [Performance Optimization](#performance-optimization)

## Axis Limits and Constraints

### Setting Basic Axis Limits

Most industrial robots have joint angle limits:

```python
from py_opw_kinematics import KinematicModel
import numpy as np

# Create robot model
model = KinematicModel(
    a1=460,  # $MC_ROBX_MAIN_LENGTH_AB[0]
    a2=-250,  # - $MC_ROBX_TX3P3_POS[2]
    b=0,
    c1=1140,  # $MC_ROBX_TIRORO_POS[2]
    c2=1050,  # $MC_ROBX_MAIN_LENGTH_AB[1]
    c3=1510,  # $MC_ROBX_TX3P3_POS[0]
    c4=282,  # $MC_ROBX_TFLWP_POS[2]
    offsets=(0, 0, 0, 0, 0, 0),
    flip_axes=(False, False, True, False, False, False),
    has_parallelogram=True,
)

# Set axis limits (in degrees)
axis_limits = (
    (-175, 175),   # J1: ±175°
    (-75, 75),     # J2: ±75°
    (-220, 0),     # J3: -220° to 0°
    (-2700, 2700), # J4: ±2700° (multiple rotations)
    (-125, 125),   # J5: ±125°
    (-2700, 2700), # J6: ±2700° (multiple rotations)
)
model.set_axis_limits(axis_limits)
```


### Sum Constraints (Parallelogram Linkage Support)

For robots with parallelogram linkages where joint angles must satisfy sum constraints (e.g., J2 + J3 within limits), use sum constraints:

```python
# Method 1: Set sum constraints during model initialization (input in degrees)
model = KinematicModel(
    a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230,
    flip_axes=(False, False, True, False, False, False),
    has_parallelogram=True,
    sum_constraints=[
        (2, 1, -160, -30),  # J3 + J2: -160° < J3+J2 < -30° (strict inequalities)
        (5, 4, -90, 90)     # J6 + J5: -90° < J6+J5 < 90°
    ]  # Input values are in degrees, stored as radians internally
)

# Method 2: Set sum constraints at runtime using degrees
model.set_sum_constraint(2, 1, -160, -30)  # Input in degrees (default)
model.set_sum_constraint(5, 4, -90, 90)   # Input in degrees (default)

# Method 3: Set sum constraints at runtime using radians
min_sum = np.deg2rad(-160)  # Convert to radians
max_sum = np.deg2rad(-30)   # Convert to radians
model.set_sum_constraint(2, 1, min_sum, max_sum, degrees=False)  # Input in radians

# View current sum constraints (always returned in radians)
print(f"Sum constraints: {model.sum_constraints}")
# Output: [(2, 1, -2.792526803190927, -0.5235987755982988)]

# Example: Checking if joint configuration satisfies constraints
joints = [20, 20, -150, 0, 10, 0]  # J2=20°, J3=-150°, sum = -130°
is_valid = model.joints_within_limits(joints)  # Returns True since -160° < -130° < -30°
print(f"Joints {joints} are valid: {is_valid}")
```

**Key Differences between Relative and Sum Constraints:**
- **Relative constraints**: Check `axis_angle - ref_axis_angle` (difference)
- **Sum constraints**: Check `axis_angle + ref_axis_angle` (sum)
- **Boundary behavior**: Sum constraints use strict inequalities (`<` and `>`) rather than inclusive (`<=` and `>=`)

#### Constraint Units Summary

- **Initialization**: `relative_constraints` and `sum_constraints` parameters accept degrees (for convenience) and convert to radians
- **Runtime setting**: Use `degrees=True` for degree inputs, `degrees=False` (default) for radian inputs  
- **Storage**: All constraints stored internally in radians
- **Property access**: `relative_constraints` and `sum_constraints` properties always return radians

### Advanced Constraint Management

```python
# Set individual axis constraints
model.set_absolute_constraint(0, np.deg2rad(-175), np.deg2rad(175))  # J1

# Clear specific constraints
model.clear_axis_constraint(0)  # Remove J1 constraint

# Clear all constraints
model.clear_all_constraints()
```


## Batch Processing for High Performance

### Large-Scale Forward Kinematics

```python
import numpy as np
import polars as pl
import time

# Generate 100,000 joint configurations
n_points = 100000
joints_df = pl.DataFrame({
    "J1": np.random.uniform(-30, 30, n_points),
    "J2": np.random.uniform(-20, 20, n_points), 
    "J3": np.random.uniform(-120, -60, n_points),
    "J4": np.random.uniform(-45, 45, n_points),
    "J5": np.random.uniform(-30, 30, n_points),
    "J6": np.random.uniform(-60, 60, n_points),
})

# Batch forward kinematics
start_time = time.time()
poses_df = robot.batch_forward(joints_df)
elapsed_time = time.time() - start_time

print(f"Processed {n_points} points in {elapsed_time:.3f}s")
print(f"Rate: {n_points/elapsed_time:.0f} points/second")
```

### Efficient Inverse Kinematics

The "batch_inverse" function does not return multiple results per pose. Instead, it outputs only the result that requires the robotic axes to move the least out of the permissible results. This should result in a similar trajectory as the Sinumerik controller in Continuous Path mode.

```python
# Generate reachable poses using forward kinematics
test_joints = pl.DataFrame({
    "J1": np.linspace(-30, 30, 10000),
    "J2": np.linspace(-20, 20, 10000),
    "J3": np.linspace(-120, -60, 10000),
    "J4": np.linspace(-45, 45, 10000),
    "J5": np.linspace(-30, 30, 10000),
    "J6": np.linspace(-60, 60, 10000),
})

reachable_poses = robot.batch_forward(test_joints)

# Batch inverse kinematics with current joint hint
start_joints = test_joints.row(0)  # Use first configuration as hint
start_time = time.time()
recovered_joints = robot.batch_inverse(reachable_poses, current_joints=start_joints)
elapsed_time = time.time() - start_time

print(f"Inverse kinematics: {len(reachable_poses)} poses in {elapsed_time:.3f}s")
print(f"Success rate: {len(recovered_joints)/len(reachable_poses)*100:.1f}%")
```

### Memory-Efficient Processing

```python
# Process large datasets in chunks to manage memory
def process_large_trajectory(robot, poses_df, chunk_size=10000):
    results = []
    
    for i in range(0, len(poses_df), chunk_size):
        chunk = poses_df[i:i+chunk_size]
        chunk_result = robot.batch_inverse(chunk)
        results.append(chunk_result)
        print(f"Processed chunk {i//chunk_size + 1}, found {len(chunk_result)} solutions")
    
    return pl.concat(results, how="vertical")

# Process 1M poses in chunks
large_poses = pl.DataFrame({
    "X": np.random.uniform(1500, 2500, 1000000),
    "Y": np.random.uniform(-1000, 1000, 1000000),
    "Z": np.random.uniform(800, 1800, 1000000),
    "A": np.random.uniform(-45, 45, 1000000),
    "B": np.random.uniform(-45, 45, 1000000),
    "C": np.random.uniform(-180, 180, 1000000),
})

results = process_large_trajectory(robot, large_poses)
```

### Practical Visualization Examples

The `python/examples/` directory contains complete visualization scripts:

1. **`simple_robot_visualization.py`** - Basic 2D/3D plotting with image output
2. **`visualize_realistic_robot.py`** - Interactive plots with animations
3. **`visualize_robot_trajectory.py`** - Advanced trajectory analysis

Run these examples to see different visualization approaches:

```bash
python python/examples/simple_robot_visualization.py
```


## Handling Singularities

### Understanding Singularities

Kinematic singularities occur when the robot loses degrees of freedom:

1. **Wrist singularities**: J5 = 0° or ±180°
2. **Shoulder singularities**: Robot arm fully extended or retracted
3. **Elbow singularities**: J3 aligns with other axes

### Detecting Singularities

```python
def check_wrist_singularity(joints, threshold=5.0):
    """Check if robot is near wrist singularity (J5 ≈ 0° or ±180°)"""
    j5 = joints[4]  # J5 (zero-indexed)
    
    # Check for J5 near 0°
    if abs(j5) < threshold:
        return True, f"Wrist singularity: J5 = {j5:.1f}° (near 0°)"
    
    # Check for J5 near ±180°
    if abs(abs(j5) - 180) < threshold:
        return True, f"Wrist singularity: J5 = {j5:.1f}° (near ±180°)"
    
    return False, "No wrist singularity detected"

# Test joint configurations
test_configs = [
    [0, 30, -90, 45, 0, 60],      # J5 = 0° (singular)
    [0, 30, -90, 45, 180, 60],    # J5 = 180° (singular)
    [0, 30, -90, 45, 90, 60],     # J5 = 90° (non-singular)
]

for i, joints in enumerate(test_configs):
    is_singular, message = check_wrist_singularity(joints)
    print(f"Config {i+1}: {message}")
```

### Avoiding Singularities in Path Planning

```python
def plan_singularity_free_path(robot, start_joints, end_pose, n_steps=20):
    """Plan a path that avoids singularities"""
    
    # Get all solutions for end pose
    end_solutions = robot.inverse(end_pose, current_joints=start_joints)
    
    # Filter out singular solutions
    valid_end_solutions = []
    for solution in end_solutions:
        is_singular, _ = check_wrist_singularity(solution)
        if not is_singular:
            valid_end_solutions.append(solution)
    
    if not valid_end_solutions:
        raise ValueError("No non-singular solution found for end pose")
    
    # Choose solution closest to start configuration
    end_joints = min(valid_end_solutions, 
                     key=lambda sol: sum(abs(a-b)**2 for a, b in zip(sol, start_joints)))
    
    # Generate linear interpolation in joint space
    path_joints = []
    for i in range(n_steps + 1):
        alpha = i / n_steps
        joints = [start_joints[j] + alpha * (end_joints[j] - start_joints[j]) 
                 for j in range(6)]
        path_joints.append(joints)
    
    # Check for singularities along path
    singular_points = []
    for i, joints in enumerate(path_joints):
        is_singular, message = check_wrist_singularity(joints)
        if is_singular:
            singular_points.append((i, message))
    
    return path_joints, end_joints, singular_points

# Example usage
start_config = [10, 20, -90, 30, 45, 50]
target_pose = ([2200, 300, 1400], [30, -15, 10])

try:
    path, final_joints, singularities = plan_singularity_free_path(robot, start_config, target_pose)
    print(f"Planned {len(path)} step path")
    print(f"Final joints: {np.round(final_joints, 1)}")
    if singularities:
        print(f"Warning: {len(singularities)} singular points detected")
    else:
        print("Path is singularity-free")
except ValueError as e:
    print(f"Path planning failed: {e}")
```

## Industrial Robot Configurations

### Configuration String Analysis

Some robotic systems use configuration strings to describe robot posture:

```python
def analyze_robot_configuration(joints):
    """Analyze robot configuration and return descriptive string"""
    j1, j2, j3, j4, j5, j6 = joints
    
    config_parts = []
    
    # Shoulder configuration
    if j1 >= 0:
        config_parts.append("Right shoulder")
    else:
        config_parts.append("Left shoulder")
    
    # Elbow configuration
    if j3 < -90:
        config_parts.append("Elbow down")
    else:
        config_parts.append("Elbow up")
    
    # Wrist configuration
    if j5 >= 0:
        config_parts.append("Wrist normal")
    else:
        config_parts.append("Wrist flipped")
    
    # Tool orientation
    if abs(j4) < 90:
        config_parts.append("No handflip")
    else:
        config_parts.append("Handflip")
    
    return " + ".join(config_parts)

# Analyze different configurations
configurations = [
    [0, 30, -120, 30, 45, 60],
    [0, 30, -60, 30, 45, 60], 
    [0, 30, -120, 150, 45, 60],
    [0, 30, -120, 30, -45, 60],
]

for i, joints in enumerate(configurations):
    config_str = analyze_robot_configuration(joints)
    print(f"Config {i+1}: {config_str}")
    print(f"  Joints: {np.round(joints, 1)}")
```

### Selecting Preferred Configurations

```python
def select_preferred_configuration(robot, pose, preferences):
    """Select robot configuration based on preferences"""
    solutions = robot.inverse(pose)
    
    if not solutions:
        return None, "No solutions found"
    
    # Score each solution based on preferences
    scored_solutions = []
    for solution in solutions:
        score = 0
        config_str = analyze_robot_configuration(solution)
        
        # Score based on preferences
        for preference in preferences:
            if preference in config_str:
                score += 1
        
        scored_solutions.append((solution, config_str, score))
    
    # Return best scoring solution
    best_solution = max(scored_solutions, key=lambda x: x[2]) 
    return best_solution

# Example: Prefer right shoulder, elbow down, normal wrist
preferred_config = ["Right shoulder", "Elbow down", "Wrist normal"]
test_pose = ([2000, 500, 1300], [0, -45, 0])

result = select_preferred_configuration(robot, test_pose, preferred_config)
if result[0] is not None:
    joints, config, score = result
    print(f"Selected configuration: {config}")
    print(f"Joints: {np.round(joints, 1)}")
    print(f"Preference score: {score}/{len(preferred_config)}")
```

## Trajectory Planning

### Linear Cartesian Trajectories

```python
def plan_linear_trajectory(robot, start_pose, end_pose, n_points=50):
    """Plan linear trajectory in Cartesian space"""
    start_pos, start_rot = start_pose
    end_pos, end_rot = end_pose
    
    # Linear interpolation in Cartesian space
    trajectory_poses = []
    for i in range(n_points):
        alpha = i / (n_points - 1)
        
        # Linear interpolation of position
        pos = [start_pos[j] + alpha * (end_pos[j] - start_pos[j]) for j in range(3)]
        
        # Linear interpolation of orientation (simple, not optimal for rotations)
        rot = [start_rot[j] + alpha * (end_rot[j] - start_rot[j]) for j in range(3)]
        
        trajectory_poses.append((pos, rot))
    
    # Convert to joint trajectory
    joint_trajectory = []
    previous_joints = None
    
    for pose in trajectory_poses:
        solutions = robot.inverse(pose, current_joints=previous_joints)
        if not solutions:
            print(f"Warning: No solution for pose {pose}")
            continue
        
        # Select solution with minimum joint motion
        if previous_joints is None:
            chosen_joints = solutions[0]
        else:
            chosen_joints = min(solutions,
                              key=lambda sol: sum(abs(a-b)**2 for a, b in zip(sol, previous_joints)))
        
        joint_trajectory.append(chosen_joints)
        previous_joints = chosen_joints
    
    return joint_trajectory, trajectory_poses

# Example linear trajectory
start_pose = ([1800, 400, 1200], [0, 0, 0])
end_pose = ([2200, -400, 1400], [30, -20, 15])

joint_path, cartesian_path = plan_linear_trajectory(robot, start_pose, end_pose, 30)
print(f"Generated trajectory with {len(joint_path)} points")

# Analyze trajectory smoothness
if len(joint_path) > 1:
    max_joint_velocities = [0] * 6
    for i in range(1, len(joint_path)):
        for j in range(6):
            velocity = abs(joint_path[i][j] - joint_path[i-1][j])
            max_joint_velocities[j] = max(max_joint_velocities[j], velocity)
    
    print("Maximum joint velocities (deg/segment):")
    for j, vel in enumerate(max_joint_velocities):
        print(f"  J{j+1}: {vel:.2f}°")
```