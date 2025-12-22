# py-opw-kinematics API Documentation

## Overview

The `py-opw-kinematics` library provides Python bindings for solving inverse and forward kinematics of six-axis industrial robots with a parallel base and spherical wrist. The library is built on the high-performance Rust implementation `rs-opw-kinematics`.

## Core Classes

### KinematicModel

The `KinematicModel` class defines the physical parameters and constraints of a robot.

#### Constructor

```python
KinematicModel(
    a1: float = 0,
    a2: float = 0,
    b: float = 0,
    c1: float = 0,
    c2: float = 0,
    c3: float = 0,
    c4: float = 0,
    offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    flip_axes: Optional[Tuple[bool, bool, bool, bool, bool, bool]] = (False, False, False, False, False, False),
    has_parallelogram: bool = False,
    axis_limits: Optional[List[Tuple[float, float]]] = None,
    relative_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
    sum_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
)
```

**Parameters:**
- `a1, a2, b, c1, c2, c3, c4`: Kinematic parameters defining the robot geometry (in millimeters)
- `offsets`: Joint offsets in degrees to align with manufacturer's zero position
- `flip_axes`: Boolean flags to flip the direction of each axis
- `has_parallelogram`: Whether the robot has a parallelogram linkage between joints 2 and 3
- `axis_limits`: Optional joint angle limits as (min, max) tuples for each axis
- `relative_constraints`: Optional relative constraints as (axis, reference_axis, min_offset, max_offset) tuples.
  Input values are in **degrees** during initialization and converted to radians internally
- `sum_constraints`: Optional sum constraints as (axis, reference_axis, min_sum, max_sum) tuples for parallelogram constraints.
  Input values are in **degrees** during initialization and converted to radians internally

**Unit Handling:**
- **Initialization**: Constraint parameters accept degrees (for convenience) and are converted to radians internally
- **Storage**: All constraints stored internally in radians for consistency
- **Property access**: `relative_constraints` and `sum_constraints` properties always return values in radians

#### Properties

- `a1`, `a2`, `b`, `c1`, `c2`, `c3`, `c4`: Kinematic parameters (read-only)
- `offsets`: Joint offsets (read-only)
- `flip_axes`: Axis flip flags (read-only)
- `has_parallelogram`: Parallelogram flag (read-only)
- `has_constraints`: Whether any constraints are active (read-only)
- `axis_limits`: Current axis limits (read-only)
- `relative_constraints`: Current relative constraints as (axis, reference_axis, min_offset, max_offset) tuples in radians (read-only)
- `sum_constraints`: Current sum constraints as (axis, reference_axis, min_sum, max_sum) tuples in radians (read-only)

#### Methods

##### set_axis_limits(limits)
Set absolute limits for all joint axes.

```python
axis_limits = [
    (-175, 175),   # J1: ±175°
    (-75, 75),     # J2: ±75°
    (-180, 0),     # J3: -180° to 0°
    (-270, 270),   # J4: ±270°
    (-125, 125),   # J5: ±125°
    (-270, 270)    # J6: ±270°
]
model.set_axis_limits(axis_limits)
```

##### set_absolute_constraint(axis, min, max, degrees=True)
Set absolute constraint for a specific axis.

**Parameters:**
- `axis`: Axis index (0-5)
- `min`, `max`: Constraint limits
- `degrees`: If `True` (default), interpret limits as degrees; if `False`, interpret as radians

**Unit Handling:**
- Input values converted to radians internally regardless of input format
- Stored values always in radians for consistency
- Robotics-friendly: Uses degrees by default

```python
# Degrees input (default, robotics-friendly)
model.set_absolute_constraint(0, -180, 180)        # J1: ±180° (degrees=True default)
model.set_absolute_constraint(1, -90, 90)          # J2: ±90° (degrees=True default)

# Radians input (explicit)
model.set_absolute_constraint(0, -np.pi, np.pi, degrees=False)  # J1 in radians

# Verify storage (always returns radians)
constraints = model.axis_limits
if constraints:
    min_rad, max_rad = constraints[0]
    print(f"Stored: [{min_rad:.4f}, {max_rad:.4f}] rad")
    print(f"Equivalent: [{np.rad2deg(min_rad):.1f}°, {np.rad2deg(max_rad):.1f}°]")
```

##### set_relative_constraint(axis, reference_axis, min_offset, max_offset, degrees=True)
Set relative constraint between two axes (difference constraint: axis - reference_axis).

**Parameters:**
- `axis`: Axis to constrain (0-5)
- `reference_axis`: Reference axis (0-5)
- `min_offset`, `max_offset`: Constraint limits for (axis - reference_axis)
- `degrees`: If `True` (default), interpret offsets as degrees; if `False`, interpret as radians

**Unit Handling:**
- Input values converted to radians internally regardless of input format
- Stored constraints always in radians
- Property `relative_constraints` returns values in radians

```python
# Degrees input (default, robotics-friendly)
model.set_relative_constraint(2, 1, -160, -30)  # J3-J2: [-160°, -30°] (degrees=True default)

# Radians input (explicit)
model.set_relative_constraint(2, 1, np.deg2rad(-160), np.deg2rad(-30), degrees=False)

# Verify storage (always returns radians)
rel_constraints = model.relative_constraints
if rel_constraints:
    axis, ref_axis, min_rad, max_rad = rel_constraints[0]
    print(f"Constraint J{axis}-J{ref_axis}: [{min_rad:.4f}, {max_rad:.4f}] rad")
    print(f"Equivalent: [{np.rad2deg(min_rad):.1f}°, {np.rad2deg(max_rad):.1f}°]")
```

##### set_sum_constraint(axis, reference_axis, min_sum, max_sum, degrees=True)
Set sum constraint between two axes (sum constraint: axis + reference_axis).
Typically used for parallelogram linkages.

**Parameters:**
- `axis`: Axis to constrain (0-5)
- `reference_axis`: Reference axis (0-5)  
- `min_sum`, `max_sum`: Constraint limits for (axis + reference_axis)
- `degrees`: If `True` (default), interpret sums as degrees; if `False`, interpret as radians

**Constraint Behavior:**
- Uses strict inequalities: `min_sum < (axis + reference_axis) < max_sum`
- Boundary values are excluded (unlike relative constraints which use ≤ and ≥)

**Unit Handling:**
- Input values converted to radians internally regardless of input format
- Stored constraints always in radians
- Property `sum_constraints` returns values in radians

```python
# Degrees input (default, robotics-friendly)
model.set_sum_constraint(2, 1, -160, -30)  # J2+J3: (-160°, -30°) (degrees=True default)

# Radians input (explicit)
min_sum = np.deg2rad(-160)  # Convert to radians
max_sum = np.deg2rad(-30)   # Convert to radians
model.set_sum_constraint(2, 1, min_sum, max_sum, degrees=False)

# Verify storage (always returns radians)
sum_constraints = model.sum_constraints
if sum_constraints:
    axis, ref_axis, min_rad, max_rad = sum_constraints[0]
    print(f"Constraint J{axis}+J{ref_axis}: ({min_rad:.4f}, {max_rad:.4f}) rad")
    print(f"Equivalent: ({np.rad2deg(min_rad):.1f}°, {np.rad2deg(max_rad):.1f}°)")
    
# Test constraint (J2=20°, J3=-150°, sum = -130°)
joints = [0, 20, -150, 0, 0, 0]  # J2=20°, J3=-150°, sum = -130°
is_valid = model.joints_within_limits_vec(joints, degrees=True)
print(f"Joints {joints} satisfy constraints: {is_valid}")  # True since -160° < -130° < -30°
```

##### clear_axis_constraint(axis)
Clear constraint for a specific axis.

```python
model.clear_axis_constraint(0)  # Clear J1 constraint
```

##### clear_all_constraints()
Clear all advanced constraints.

```python
model.clear_all_constraints()
```

##### joints_within_limits_vec(joints, degrees=None)
Check if given joints satisfy all constraints.

```python
joints = [20, 20, -150, 0, 10, 0]
is_valid = model.joints_within_limits_vec(joints, degrees=True)
print(f"Joints are valid: {is_valid}")
```

#### Example: Comau NJ165-3.0 Setup

```python
from py_opw_kinematics import KinematicModel
import numpy as np

# Create Comau NJ165-3.0 kinematic model with sum constraints
model = KinematicModel(
    a1=460,       # Base to shoulder
    a2=-250,      # Shoulder offset
    b=0,
    c1=1140,      # Base height
    c2=1050,      # Upper arm length
    c3=1510,      # Forearm length
    c4=282,       # Wrist length
    offsets=(0, 0, 0, 0, 0, 0),
    flip_axes=(False, False, True, False, True, False),
    has_parallelogram=True,
    sum_constraints=[
        (2, 1, -160, -30),  # J3 + J2: -160° < J3+J2 < -30° (parallelogram constraint)
    ]  # Input in degrees, stored as radians internally
)

# Set additional axis limits
axis_limits = [(-175, 175), (-75, 75), (-220, 0), (-270, 270), (-125, 125), (-270, 270)]
model.set_axis_limits(axis_limits)

# Alternative: Set sum constraint at runtime
# model.set_sum_constraint(2, 1, -160, -30)  # Defaults to degrees
```

---

### EulerConvention

The `EulerConvention` class defines how rotations are represented and converted.

#### Constructor

```python
EulerConvention(
    sequence: str,
    extrinsic: bool,
    degrees: bool
)
```

**Parameters:**
- `sequence`: Euler rotation sequence (e.g., "XYZ", "ZYX", "XYX", etc.)
- `extrinsic`: Whether rotations are extrinsic (True) or intrinsic (False)
- `degrees`: Whether to use degrees (True) or radians (False)

#### Methods

##### euler_to_matrix(angles)
Convert Euler angles to rotation matrix.

```python
euler_conv = EulerConvention("XYZ", extrinsic=False, degrees=True)
matrix = euler_conv.euler_to_matrix([30, 45, 60])
```

##### matrix_to_euler(matrix)
Convert rotation matrix to Euler angles.

```python
angles = euler_conv.matrix_to_euler(rotation_matrix)
```

##### matrix_to_quaternion(matrix)
Convert rotation matrix to quaternion.

```python
quaternion = euler_conv.matrix_to_quaternion(rotation_matrix)
# Returns (w, i, j, k) quaternion
```

##### quaternion_to_euler(quaternion)
Convert quaternion to Euler angles.

```python
angles = euler_conv.quaternion_to_euler((w, i, j, k))
```

##### convert(other_convention, angles)
Convert angles to another Euler convention.

```python
xyz_conv = EulerConvention("XYZ", extrinsic=False, degrees=True)
zyx_conv = EulerConvention("ZYX", extrinsic=True, degrees=True)
converted_angles = xyz_conv.convert(zyx_conv, [30, 45, 60])
```

#### Example: Different Euler Conventions

```python
from py_opw_kinematics import EulerConvention

# Intrinsic XYZ (Roll-Pitch-Yaw)
rpy_conv = EulerConvention("XYZ", extrinsic=False, degrees=True)

# Extrinsic ZYX (Yaw-Pitch-Roll)
ypr_conv = EulerConvention("ZYX", extrinsic=True, degrees=True)

# Convert between conventions
rpy_angles = [30, 45, 60]  # Roll, Pitch, Yaw
ypr_angles = rpy_conv.convert(ypr_conv, rpy_angles)
```

---

### Robot

The `Robot` class combines kinematic model and Euler convention to perform forward and inverse kinematics.

#### Constructor

```python
Robot(
    kinematic_model: KinematicModel,
    euler_convention: EulerConvention,
    ee_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
)
```

**Parameters:**
- `kinematic_model`: The robot's kinematic model
- `euler_convention`: Euler convention for rotation representation
- `ee_rotation`: End-effector rotation offset (tool frame)
- `ee_translation`: End-effector translation offset (tool center point)

#### Properties

##### kinematic_model
Get the kinematic model used by this robot.

```python
model = robot.kinematic_model
```

##### ee_rotation
Get/set end-effector rotation offset.

```python
robot.ee_rotation = [0, -90, 0]  # Tool pointing down
current_rotation = robot.ee_rotation
```

##### ee_translation
Get/set end-effector translation offset.

```python
robot.ee_translation = [100, 0, -50]  # TCP offset
current_translation = robot.ee_translation
```

#### Methods

##### forward(joints)
Compute forward kinematics for given joint angles.

```python
joints = [10, 20, -90, 30, 40, 50]  # Joint angles in degrees
position, orientation = robot.forward(joints)
# position: [x, y, z] in mm
# orientation: [rx, ry, rz] in degrees (or radians based on convention)
```

##### joint_positions(joints)
Compute 3D positions of all joints using forward kinematics.

```python
joints = [10, 20, -90, 30, 40, 50]
positions = robot.joint_positions(joints)
# Returns: [base, J1, J2, J3, J4, J5, J6, TCP] positions
for i, pos in enumerate(positions):
    print(f"Joint {i}: {pos}")
```

##### inverse(pose, current_joints=None)
Compute inverse kinematics for a given pose.

```python
pose = ([2000, 500, 1200], [0, 0, 0])  # (position, orientation)
solutions = robot.inverse(pose)
# Returns list of joint angle solutions

# Use current joints for better solution selection
current_joints = [0, 0, -90, 0, 0, 0]
solutions = robot.inverse(pose, current_joints=current_joints)
```

##### inverse_with_config(pose, target_config=None)
Compute inverse kinematics with configuration analysis and optional target matching.

```python
pose = ([2000, 500, 1200], [0, 0, 0])
solutions, config_strings, best_match = robot.inverse_with_config(pose)

# With target configuration
target = "STAT=101 TU=000011"
solutions, config_strings, best_match = robot.inverse_with_config(pose, target)
if best_match:
    joints, config, score = best_match
    print(f"Best match: {joints} with config {config} (score: {score})")
```

##### inverse_with_target_config(pose, target_config, current_joints=None)
Find the best inverse kinematics solution matching a specific configuration.

```python
pose = ([2000, 500, 1200], [0, 0, 0])
target_config = "STAT=101 TU=000011"
result = robot.inverse_with_target_config(pose, target_config)

if result:
    joints, config_string, score = result
    print(f"Found solution: {joints}")
    print(f"Configuration: {config_string}")
    print(f"Match score: {score}/3")
```

##### analyze_configuration(joints)
Analyze the configuration of given joint angles.

```python
joints = [10, 20, -90, 30, 40, 50]
config = robot.analyze_configuration(joints)
print(f"Configuration: {config}")  # e.g., "STAT=101 TU=000011"
```

##### analyze_configuration_full(joints, include_turns=False)
Analyze configuration with full STAT/TU information.

```python
joints = [10, 20, -90, 30, 40, 50]
stat_tu_string, stat_binary, full_string = robot.analyze_configuration_full(joints)
print(f"STAT/TU: {stat_tu_string}")
print(f"STAT bits: {stat_binary}")
```

##### compare_configurations(joint_solutions)
Compare multiple joint solutions and return their configuration strings.

```python
solutions = [
    [0, 30, -120, 30, 45, 60],
    [0, 30, -60, 30, 45, 60], 
]
configs = robot.compare_configurations(solutions)
for i, config in enumerate(configs):
    print(f"Solution {i+1}: {config}")
```

##### get_configuration_details(joints)
Get detailed configuration analysis for a solution.

```python
joints = [10, 20, -90, 30, 40, 50]
stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary = robot.get_configuration_details(joints)
print(f"STAT/TU config: {stat_tu_config}")
print(f"STAT bits: {stat_bits} ({stat_binary})")
print(f"TU bits: {tu_bits} ({tu_binary})")
```

##### get_configuration_details_geometric(joints, robot_params)
Get configuration analysis using geometric calculation with robot-specific parameters.

```python
from py_opw_kinematics import RobotKinematicParams

robot_params = RobotKinematicParams.from_kinematic_model(kinematic_model)
joints = [10, 20, -90, 30, 40, 50]
stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary = \
    robot.get_configuration_details_geometric(joints, robot_params)
```

##### find_stat_matches(pose, stat_bits)
Find solutions matching STAT bits (ignoring turn numbers).

```python
pose = ([2000, 500, 1200], [0, 0, 0])
stat_bits = 5  # Binary 101: shoulder_left=True, elbow_up=False, handflip=True
matches = robot.find_stat_matches(pose, stat_bits)
for joints, config_string, score in matches:
    print(f"Match: {joints} - {config_string} (score: {score})")
```

##### create_stat_tu_target(stat_bits, tu_bits)
Create target configuration string from STAT/TU bits.

```python
stat_bits = 5  # Binary 101
tu_bits = 3    # Binary 000011
target = robot.create_stat_tu_target(stat_bits, tu_bits)
print(f"Target: {target}")  # "STAT=101 TU=000011"
```

##### parallelogram_positions(joints, link_length, rest_angle)
Calculate parallelogram P1 and P2 positions using actual robot geometry.

```python
joints = [0, 30, -120, 30, 45, 60]
link_length = 400.0  # Length of parallelogram links
rest_angle = 95.0    # Rest angle for parallelogram
positions = robot.parallelogram_positions(joints, link_length, rest_angle)
if positions:
    p1_pos, p2_pos = positions
    print(f"P1 position: {p1_pos}")
    print(f"P2 position: {p2_pos}")
```

##### batch_forward(joints_df)
Compute forward kinematics for multiple joint configurations.

```python
import polars as pl

joints_df = pl.DataFrame({
    "J1": [0, 10, 20],
    "J2": [0, 0, 0],
    "J3": [-90, -90, -90],
    "J4": [0, 30, 60],
    "J5": [0, 0, 0],
    "J6": [0, 0, 0]
})

poses_df = robot.batch_forward(joints_df)
# Returns DataFrame with columns: X, Y, Z, A, B, C
```

##### batch_inverse(poses_df, current_joints=None)
Compute inverse kinematics for multiple poses.

```python
poses_df = pl.DataFrame({
    "X": [2000, 2100, 2200],
    "Y": [0, 100, 200],
    "Z": [1200, 1300, 1400],
    "A": [0, 0, 0],
    "B": [0, 0, 0],
    "C": [0, 0, 0]
})

joints_df = robot.batch_inverse(poses_df)
# Returns DataFrame with columns: J1, J2, J3, J4, J5, J6
```

---

### RobotKinematicParams

The `RobotKinematicParams` class provides robot kinematic parameters for geometric configuration analysis.

#### Constructor

```python
RobotKinematicParams(
    a1: float,
    a2: float, 
    b: float,
    c1: float,
    c2: float,
    c3: float,
    c4: float
)
```

#### Class Methods

##### from_kinematic_model(kinematic_model)
Create RobotKinematicParams from a KinematicModel.

```python
from py_opw_kinematics import RobotKinematicParams, KinematicModel

model = KinematicModel(a1=400, a2=-250, c1=830, c2=1175, c3=1444, c4=230)
robot_params = RobotKinematicParams.from_kinematic_model(model)
```

---

## Complete Robot Setup Example

```python
from py_opw_kinematics import Robot, EulerConvention, KinematicModel, RobotKinematicParams
import numpy as np

# 1. Create kinematic model with sum constraints
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
    sum_constraints=[
        (2, 1, -160, -30),  # J2 + J3 constraint for parallelogram
    ]
)

# 2. Set additional axis limits
axis_limits = [(-175, 175), (-75, 75), (-180, 0), (-270, 270), (-125, 125), (-270, 270)]
kinematic_model.set_axis_limits(axis_limits)

# 3. Create Euler convention
euler_convention = EulerConvention("XYZ", extrinsic=False, degrees=True)

# 4. Create robot with end-effector offset
robot = Robot(
    kinematic_model, 
    euler_convention,
    ee_translation=(100, 0, -200),  # Tool center point offset
    ee_rotation=(0, -90, 0)         # Tool pointing down
)

# 5. Use robot for kinematics
joints = [10, 20, -90, 30, 40, 50]
position, orientation = robot.forward(joints)
print(f"Position: {position}")
print(f"Orientation: {orientation}")

# 6. Analyze configuration
config = robot.analyze_configuration(joints)
print(f"Configuration: {config}")

# 7. Get detailed configuration analysis
stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary = robot.get_configuration_details(joints)
print(f"STAT bits: {stat_bits} ({stat_binary})")
print(f"TU bits: {tu_bits} ({tu_binary})")

# 8. Inverse kinematics with configuration matching
pose = (position, orientation)
solutions, config_strings, best_match = robot.inverse_with_config(pose)
print(f"Found {len(solutions)} solutions")
for i, (solution, config_str) in enumerate(zip(solutions, config_strings)):
    print(f"Solution {i+1}: {solution} - {config_str}")

# 9. Target specific configuration
target_config = "STAT=101 TU=000011"
result = robot.inverse_with_target_config(pose, target_config)
if result:
    target_joints, target_config_str, score = result
    print(f"Target solution: {target_joints}")
    print(f"Configuration: {target_config_str}")
    print(f"Match score: {score}/3")
```

## Configuration Analysis

### Overview

The library provides comprehensive robot configuration analysis compatible with SINUMERIK ROBX systems. This includes STAT/TU (Status/Turn) bit analysis for industrial robot path planning and configuration management.

### STAT Bits
- **Bit 0**: Shoulder configuration (0 = right, 1 = left)
- **Bit 1**: Elbow configuration (0 = down, 1 = up)  
- **Bit 2**: Handflip configuration (0 = no handflip, 1 = handflip)

### TU Bits
Define joint angle sign preference for each axis (6 bits total), indicating whether joint angles should be positive or negative after normalization.

### Configuration String Formats

#### STAT/TU Binary Format
```python
config = "STAT=101 TU=000011"
# STAT=101: shoulder_left=True, elbow_up=False, handflip=True
# TU=000011: J1,J2,J3,J4 prefer positive, J5,J6 prefer negative
```

#### Simplified Format
```python
config = "J3+ J5- OH+"
# J3+: elbow up, J5-: wrist flipped, OH+: overhead
```

### Configuration Analysis Example

```python
# Analyze different robot configurations
configurations = [
    [0, 30, -120, 30, 45, 60],   # Elbow down, normal wrist
    [0, 30, -60, 30, 45, 60],    # Elbow up, normal wrist
    [0, 30, -120, 150, 45, 60],  # Elbow down, handflip
    [0, 30, -120, 30, -45, 60],  # Elbow down, flipped wrist
]

for i, joints in enumerate(configurations):
    # Basic configuration string
    config_str = robot.analyze_configuration(joints)
    
    # Detailed analysis
    stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary = \
        robot.get_configuration_details(joints)
    
    print(f"Config {i+1}:")
    print(f"  Joints: {joints}")
    print(f"  STAT/TU: {config_str}")
    print(f"  STAT bits: {stat_bits} ({stat_binary})")
    print(f"  TU bits: {tu_bits} ({tu_binary})")
```

### Target Configuration Matching

```python
# Define target configuration for path planning
target_config = "STAT=101 TU=000011"

# Find solutions matching the target
pose = ([2000, 500, 1200], [0, 0, 0])
result = robot.inverse_with_target_config(pose, target_config)

if result:
    joints, config_string, score = result
    print(f"Found matching solution: {joints}")
    print(f"Configuration: {config_string}")
    print(f"Match quality: {score}/3")
else:
    print("No solution matches the target configuration")

# Alternative: Get all solutions with analysis
solutions, config_strings, best_match = robot.inverse_with_config(pose, target_config)
for solution, config in zip(solutions, config_strings):
    print(f"Solution: {solution} - Config: {config}")
```

### Parallelogram Robot Configuration

For robots with parallelogram linkages, the library automatically handles virtual joint calculations for accurate elbow configuration analysis:

```python
# Parallelogram robot uses virtual J3 = Physical_J3 + J2 for elbow analysis
robot_with_parallelogram = Robot(kinematic_model_with_parallelogram, euler_convention)

joints = [0, 45, -30, 0, 60, 0]  # J2=45°, Physical J3=-30°
# Virtual J3 = 45° + (-30°) = 15° > 0 → Elbow Up

config = robot_with_parallelogram.analyze_configuration(joints)
print(f"Parallelogram config: {config}")  # Shows elbow up due to virtual J3
```

## Performance Considerations

### Single Operations
- Forward kinematics: ~1-10 microseconds per operation
- Inverse kinematics: ~0.01-0.1 milliseconds per operation

### Batch Operations
- Batch forward: ~1-5 microseconds per point
- Batch inverse: ~10-50 microseconds per point
- 100,000 inverse kinematic solutions: ~0.4 seconds

### Memory Usage
- Minimal memory overhead for single operations
- Efficient batch processing with Polars DataFrames
- Memory usage scales linearly with batch size

## Common Use Cases

### 1. Robot Simulation
```python
# Generate smooth trajectory
import numpy as np

n_points = 100
t = np.linspace(0, 2*np.pi, n_points)

# Circular path
center = [2000, 0, 1300]
radius = 200
x = center[0] + radius * np.cos(t)
y = center[1] + radius * np.sin(t)
z = center[2] * np.ones(n_points)

poses_df = pl.DataFrame({
    "X": x, "Y": y, "Z": z,
    "A": np.zeros(n_points),
    "B": np.ones(n_points) * -45,  # Tool angled down
    "C": np.zeros(n_points)
})

# Plan joint trajectory
joints_df = robot.batch_inverse(poses_df)
```

### 2. Path Planning with Configuration Control
```python
# Define waypoints with target configurations
waypoints = [
    (([1800, 500, 1200], [0, 0, 0]), "STAT=101 TU=000000"),   # Start with specific config
    (([2000, 300, 1300], [10, -5, 0]), "STAT=101 TU=000000"), # Maintain configuration
    (([2200, 100, 1400], [20, -10, 0]), "STAT=101 TU=000000") # End with same config
]

# Plan path maintaining configuration
path_joints = []
previous_joints = None

for (position, orientation), target_config in waypoints:
    result = robot.inverse_with_target_config((position, orientation), target_config, previous_joints)
    if result:
        joints, config_str, score = result
        path_joints.append(joints)
        previous_joints = joints
        print(f"Waypoint: {joints} - Config: {config_str} (score: {score})")
    else:
        print(f"Could not find solution for target configuration: {target_config}")
```

### 3. Configuration Analysis and Comparison
```python
# Analyze multiple solutions for the same pose
pose = ([2000, 500, 1200], [0, 0, 0])
solutions, config_strings, _ = robot.inverse_with_config(pose)

print(f"Found {len(solutions)} solutions for pose {pose}")
for i, (solution, config) in enumerate(zip(solutions, config_strings)):
    # Get detailed configuration analysis
    stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary = \
        robot.get_configuration_details(solution)
    
    print(f"Solution {i+1}:")
    print(f"  Joints: {solution}")
    print(f"  Config: {config}")
    print(f"  STAT: {stat_bits} ({stat_binary})")
    print(f"  TU: {tu_bits} ({tu_binary})")
```

### 4. Workspace Analysis with Configuration Info
```python
# Define waypoints
waypoints = [
    ([1800, 500, 1200], [0, 0, 0]),    # Start
    ([2000, 300, 1300], [10, -5, 0]),  # Via point
    ([2200, 100, 1400], [20, -10, 0]), # End
]

# Plan path
path_joints = []
previous_joints = None

for position, orientation in waypoints:
    solutions = robot.inverse((position, orientation), current_joints=previous_joints)
    chosen_joints = solutions[0]  # Or implement better selection logic
    path_joints.append(chosen_joints)
    previous_joints = chosen_joints
```

### 3. Workspace Analysis
```python
# Generate workspace points with configuration analysis
n_samples = 1000
joints_samples = pl.DataFrame({
    "J1": np.random.uniform(-175, 175, n_samples),
    "J2": np.random.uniform(-75, 75, n_samples),
    "J3": np.random.uniform(-180, 0, n_samples),
    "J4": np.random.uniform(-270, 270, n_samples),
    "J5": np.random.uniform(-125, 125, n_samples),
    "J6": np.random.uniform(-270, 270, n_samples),
})

# Check constraints and analyze configurations
valid_joints = []
configurations = []

for joints_row in joints_samples.iter_rows():
    joints = list(joints_row)
    
    # Check if joints satisfy constraints
    if robot.kinematic_model.joints_within_limits_vec(joints, degrees=True):
        valid_joints.append(joints)
        config = robot.analyze_configuration(joints)
        configurations.append(config)

# Compute reachable positions for valid configurations
if valid_joints:
    valid_df = pl.DataFrame({
        "J1": [j[0] for j in valid_joints],
        "J2": [j[1] for j in valid_joints], 
        "J3": [j[2] for j in valid_joints],
        "J4": [j[3] for j in valid_joints],
        "J5": [j[4] for j in valid_joints],
        "J6": [j[5] for j in valid_joints],
    })
    
    workspace = robot.batch_forward(valid_df)
    
    # Analyze workspace by configuration
    config_df = pl.DataFrame({"Configuration": configurations})
    workspace_with_config = pl.concat([workspace, config_df], how="horizontal")
    
    # Analyze workspace statistics
    print(f"Valid configurations: {len(valid_joints)}/{n_samples}")
    print(f"X range: {workspace['X'].min():.1f} to {workspace['X'].max():.1f} mm")
    print(f"Y range: {workspace['Y'].min():.1f} to {workspace['Y'].max():.1f} mm")
    print(f"Z range: {workspace['Z'].min():.1f} to {workspace['Z'].max():.1f} mm")
    
    # Configuration distribution
    config_counts = workspace_with_config.group_by("Configuration").count()
    print("\nConfiguration distribution:")
    for row in config_counts.iter_rows(named=True):
        print(f"  {row['Configuration']}: {row['count']} points")
```

### 5. Joint Position Visualization
```python
# Get joint positions for visualization
joints = [10, 20, -90, 30, 40, 50]
joint_positions = robot.joint_positions(joints)

print("Robot joint positions:")
joint_names = ["Base", "J1", "J2", "J3", "J4", "J5", "J6", "TCP"]
for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
    print(f"{name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

# For parallelogram robots, also get parallelogram positions
if robot.kinematic_model.has_parallelogram:
    para_positions = robot.parallelogram_positions(joints, 400.0, 95.0)
    if para_positions:
        p1_pos, p2_pos = para_positions
        print(f"P1 position: [{p1_pos[0]:.2f}, {p1_pos[1]:.2f}, {p1_pos[2]:.2f}]")
        print(f"P2 position: [{p2_pos[0]:.2f}, {p2_pos[1]:.2f}, {p2_pos[2]:.2f}]")
```

## Advanced Constraints

### Constraint Types

The library supports multiple types of joint constraints with flexible unit handling:

#### 1. Absolute Constraints (Axis Limits)
Define individual joint angle limits.

```python
# Set limits for individual axes (explicit degrees)
model.set_absolute_constraint(0, -180, 180)  # J1: ±180° (default)
model.set_absolute_constraint(1, -90, 90)    # J2: ±90° (default)

# Set limits for all axes at once
axis_limits = [(-180, 180), (-90, 90), (-180, 0), (-270, 270), (-125, 125), (-270, 270)]
model.set_axis_limits(axis_limits)
```

#### 2. Relative Constraints (Difference)
Constrain the difference between two joint angles: `min_offset ≤ (axis - reference_axis) ≤ max_offset`

**Use cases**: Joint coupling, mechanical limits between joints

```python
# J3 relative to J2: -160° ≤ (J3 - J2) ≤ -30°
model.set_relative_constraint(2, 1, -160, -30, degrees=True)

# Example: J2=30°, J3=-120° → difference = -120° - 30° = -150°
# This satisfies the constraint since -160° ≤ -150° ≤ -30°
```

#### 3. Sum Constraints (Parallelogram)
Constrain the sum of two joint angles: `min_sum < (axis + reference_axis) < max_sum`

**Use cases**: Parallelogram linkages, mechanical coupling where angles add up
**Note**: Uses strict inequalities (`<` and `>`) rather than inclusive (`≤` and `≥`)

```python
# J2 + J3 constraint: -160° < (J2 + J3) < -30° (strict inequalities)
model.set_sum_constraint(2, 1, -160, -30, degrees=True)

# Example: J2=20°, J3=-150° → sum = 20° + (-150°) = -130°
# This satisfies the constraint since -160° < -130° < -30°
```

#### Key Differences: Relative vs Sum Constraints

| Aspect | Relative Constraints | Sum Constraints |
|--------|---------------------|-----------------|
| **Formula** | `axis - reference_axis` | `axis + reference_axis` |
| **Boundaries** | Inclusive (`≤`, `≥`) | Strict (`<`, `>`) |
| **Typical Use** | Joint coupling limits | Parallelogram linkages |
| **Example** | Elbow relative to shoulder | Parallelogram mechanism |

```python
# Relative constraint example (difference)
model.set_relative_constraint(2, 1, -160, -30)  # -160° ≤ (J3-J2) ≤ -30°
joints_rel = [0, 30, -120, 0, 0, 0]  # J2=30°, J3=-120° → diff=-150° ✓

# Sum constraint example (parallelogram)
model.set_sum_constraint(2, 1, -160, -30)      # -160° < (J2+J3) < -30°
joints_sum = [0, 20, -150, 0, 0, 0]  # J2=20°, J3=-150° → sum=-130° ✓

# Test constraints
is_rel_valid = model.joints_within_limits_vec(joints_rel, degrees=True)
is_sum_valid = model.joints_within_limits_vec(joints_sum, degrees=True)
```

### Constraint Units and Best Practices

#### Unit Handling Principles

1. **Flexible Input**: All constraint methods accept both degrees and radians
2. **Consistent Storage**: All constraints are stored internally in radians
3. **Clear Interface**: The `degrees` parameter controls input interpretation
4. **Backward Compatibility**: Default behavior matches original API expectations

#### Unit Conversion Rules

- **Initialization**: `relative_constraints` and `sum_constraints` parameters accept degrees (converted to radians internally)
- **Runtime setting**: Use `degrees=True` for degree inputs, `degrees=False` (default) for radian inputs
- **Storage**: All constraints stored internally in radians  
- **Property access**: `relative_constraints` and `sum_constraints` properties always return radians

#### Best Practices

1. **Use Default Degrees**: The API now defaults to degrees for robotics-friendly usage
   ```python
   # Recommended: Simple and clear (degrees=True is default)
   model.set_absolute_constraint(0, -180, 180)
   model.set_sum_constraint(2, 1, -160, -30)
   
   # Only specify degrees parameter when using radians
   model.set_absolute_constraint(0, -np.pi, np.pi, degrees=False)
   ```

2. **Verify Storage**: Check constraints with property getters during development
   ```python
   # Set constraint
   model.set_sum_constraint(2, 1, -160, -30, degrees=True)
   
   # Verify storage (always in radians)
   constraints = model.sum_constraints
   if constraints:
       axis, ref_axis, min_rad, max_rad = constraints[0]
       print(f"Stored: ({min_rad:.4f}, {max_rad:.4f}) rad")
       print(f"Input equivalent: ({np.rad2deg(min_rad):.1f}°, {np.rad2deg(max_rad):.1f}°)")
   ```

3. **Document Units**: Comment your code to indicate input units
   ```python
   # Parallelogram constraint: J2 + J3 must be between -160° and -30°
   model.set_sum_constraint(2, 1, -160, -30, degrees=True)
   
   # Joint limits in degrees
   axis_limits = [(-180, 180), (-90, 90), (-180, 0), (-270, 270), (-125, 125), (-270, 270)]
   model.set_axis_limits(axis_limits)
   ```

4. **Test Consistency**: Verify that different setting methods produce identical results
   ```python
   import numpy as np
   
   # Method 1: Degrees
   model1 = KinematicModel(sum_constraints=[(2, 1, -160, -30)])
   
   # Method 2: Runtime with degrees
   model2 = KinematicModel()
   model2.set_sum_constraint(2, 1, -160, -30)
   
   # Method 3: Runtime with radians
   model3 = KinematicModel()
   model3.set_sum_constraint(2, 1, np.deg2rad(-160), np.deg2rad(-30), degrees=False)
   
   # Verify all methods produce identical storage
   constraints1 = model1.sum_constraints[0] if model1.sum_constraints else None
   constraints2 = model2.sum_constraints[0] if model2.sum_constraints else None
   constraints3 = model3.sum_constraints[0] if model3.sum_constraints else None
   
   assert constraints1 == constraints2 == constraints3
   print("All methods produce identical constraint storage")
   ```

### Constraint Validation

```python
# Check if joints satisfy all constraints
joints = [20, 20, -150, 0, 10, 0]  # degrees
is_valid = model.joints_within_limits_vec(joints, degrees=True)

if is_valid:
    print("Joints satisfy all constraints")
else:
    print("Joints violate one or more constraints")

# Check constraint status
print(f"Has constraints: {model.has_constraints}")
print(f"Axis limits: {model.axis_limits}")
print(f"Relative constraints: {model.relative_constraints}")  # in radians
print(f"Sum constraints: {model.sum_constraints}")          # in radians
```

## Error Handling

The library handles various error conditions gracefully:

- **Unreachable poses**: Returns empty solution list for impossible poses
- **Constraint violations**: Only returns solutions that satisfy all constraints  
- **Singular configurations**: Handles kinematic singularities appropriately
- **Invalid parameters**: Raises appropriate exceptions for malformed inputs
- **Invalid constraint configurations**: Raises `PyValueError` for invalid axis indices or self-referencing constraints
- **Configuration parsing errors**: Invalid configuration strings raise descriptive errors
- **Axis limit violations**: Constraints outside valid ranges (0-5 for axis indices) raise errors

### Common Error Examples

```python
try:
    # Invalid axis index
    model.set_absolute_constraint(6, -180, 180)  # Axis 6 doesn't exist
except ValueError as e:
    print(f"Error: {e}")

try:
    # Self-referencing constraint
    model.set_relative_constraint(2, 2, -30, 30)  # J3 relative to J3
except ValueError as e:
    print(f"Error: {e}")

try:
    # Invalid configuration string
    robot.inverse_with_target_config(pose, "INVALID_CONFIG")
except ValueError as e:
    print(f"Configuration error: {e}")

# Handle empty solutions gracefully
solutions = robot.inverse(impossible_pose)
if not solutions:
    print("No kinematic solutions found for this pose")
```

## Integration with Other Libraries

### Available Imports

```python
from py_opw_kinematics import (
    KinematicModel,
    EulerConvention, 
    Robot,
    RobotKinematicParams
)
```

### NumPy Integration
```python
import numpy as np

# Convert between numpy arrays and tuples
joints_array = np.array([10, 20, -90, 30, 40, 50])
joints_tuple = tuple(joints_array)

position, orientation = robot.forward(joints_tuple)
position_array = np.array(position)
orientation_array = np.array(orientation)
```

### Polars DataFrame Operations
```python
import polars as pl

# Advanced DataFrame operations
joints_df = pl.DataFrame({
    "J1": np.linspace(-30, 30, 100),
    "J2": np.linspace(-20, 20, 100),
    "J3": np.full(100, -90),
    "J4": np.linspace(-45, 45, 100),
    "J5": np.linspace(-30, 30, 100),
    "J6": np.linspace(-60, 60, 100),
})

# Add computed poses
poses_df = robot.batch_forward(joints_df)

# Combine joints and poses
trajectory_df = pl.concat([joints_df, poses_df], how="horizontal")

# Filter and analyze
filtered_df = trajectory_df.filter(
    (pl.col("Z") > 1000) & (pl.col("Z") < 2000)
)
```