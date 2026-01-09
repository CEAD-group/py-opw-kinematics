# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

py-opw-kinematics is a Python wrapper for the rs-opw-kinematics Rust library, providing forward and inverse kinematics for six-axis industrial robots with a parallel base and spherical wrist (OPW-type robots).

## Goals

- Inverse kinematics (single and batch)
- Batch IK considers previous state to find closest solution
- Forward kinematics outputs all link frames via `forward_frames()`
- Optional `ee_transform` parameter for TCP calculations
- Stateless API - pure kinematics library

## Non-Goals

- External axes handling (handled by higher-level coordination layer)
- Collision detection (separate concern, e.g. py-parry3d)
- Parallelogram linkage mechanics (handled by higher-level coordination layer)
- Euler angle / rotation conversions (use scipy.spatial.transform.Rotation)

## Terminology

### Core Terms

| Term | Type | Description |
|------|------|-------------|
| **`pose`** | `RigidTransform` | SE(3) rigid body pose (position + orientation). The main spatial unit. |
| **`pose_A_B`** | `RigidTransform` | Pose of frame B expressed in frame A |
| **`poses`** | `RigidTransform` (len > 1) | Multiple poses in one object (indexable, iterable) |
| **`joints`** | `tuple[float, ...]` | Joint angles J1-J6, in configured unit (deg/rad) |
| **`cartesian`** | `((x,y,z), (a,b,c))` | Position + Euler angles tuple (for G-code I/O) |

### SE(3) and RigidTransform

SE(3) = **S**pecial **E**uclidean group in **3**D — the mathematical group of rigid transformations (rotation + translation). Represented as a 4x4 homogeneous matrix:

```
┌             ┐
│  R    t  │   R = 3x3 rotation matrix (SO(3))
│  0    1  │   t = 3x1 translation vector
└             ┘
```

`scipy.spatial.transform.RigidTransform` is the Python class representing SE(3). Internally stores 4x4 matrix. Convert to/from Euler, quaternion, etc. via `.rotation` property.

**Single vs batch**: A `RigidTransform` can hold one pose or many (like a list). Use `len(pose)` to check count, index with `pose[i]`, iterate with `for p in pose`. Shape is `(4, 4)` for single, `(N, 4, 4)` for batch.

```python
# Single pose
pose = RigidTransform.from_matrix(matrix_4x4)
len(pose)  # 1

# Batch of poses (e.g., from forward_frames for N timesteps)
poses = RigidTransform.from_matrix(matrices_Nx4x4)
len(poses)  # N
poses[0]    # first pose
poses.as_matrix()  # shape (N, 4, 4)
```

### Variable Naming Convention

Use `pose_A_B` pattern (destination_source):

```python
pose_world_base = ...
pose_base_flange = ...
pose_flange_tool = ...

# Composition (right to left)
pose_world_tool = pose_world_base * pose_base_flange * pose_flange_tool
```

### Terms to Avoid

| Avoid | Use | Reason |
|-------|-----|--------|
| `transform` (noun) | `pose` | "transform" better as verb |
| `tf_A_B` | `pose_A_B` | `pose` more readable |
| `matrix` | `pose` | Implementation detail |
| `frame` (for data) | `pose` | "frame" is the coordinate system, "pose" is its location |
| `position` (for 6DOF) | `pose` or `cartesian` | "position" implies 3DOF only |
| `angles` | `joints` | Ambiguous (Euler vs joint) |

## Design Decision: Rotation Handling Delegated to SciPy

Rotation conversions (Euler angles, quaternions, SLERP, etc.) are delegated to `scipy.spatial.transform.Rotation` rather than implemented in Rust.

**Reasoning:**
- Benchmarks show only ~10% speedup for Rust-side Euler conversion (1M poses: 5.7s vs 6.3s)
- SciPy provides comprehensive rotation API: all Euler conventions, quaternions, rotation vectors, SLERP interpolation, composition
- Reduces Rust code complexity and maintenance burden
- Users working with robotics typically need scipy anyway for trajectory planning

**API Approach:**
- Rust accepts/returns 4x4 transformation matrices (RigidTransform-compatible)
- Python layer converts between user-friendly formats (Euler, quaternions) and matrices using scipy
- Joint angles can be in degrees or radians (configured via `degrees` parameter)

## Build & Development Commands

```bash
# Build the Rust extension and install in development mode
maturin develop

# Build release wheel
maturin build --release

# Run tests
pytest

# Run a single test
pytest python/tests/test_robot.py::test_forward_kinematics -v

# Type checking
mypy python

# Lint
ruff check .
```

## Architecture

### Rust Core (src/)

- **lib.rs**: Main PyO3 module exposing `Robot` and `KinematicModel` classes to Python. Contains forward/inverse kinematics logic and NumPy batch operations.
- **kinematic_model.rs**: Robot kinematic parameters (a1, a2, b, c1-c4, offsets, flip_axes). Converts to rs-opw-kinematics Parameters.

### Python Layer (python/)

- **py_opw_kinematics/**: Package re-exports from `_internal` (the compiled Rust module)
- **py_opw_kinematics/_internal.pyi**: Type stubs for the Rust extension
- **tests/**: pytest tests for robot kinematics
- **examples/**: Usage examples including batch operations

### Key Dependencies

- **rs-opw-kinematics**: Rust library providing the core OPW kinematics solver
- **pyo3**: Rust-Python bindings
- **numpy**: NumPy array integration for batch operations
- **scipy >= 1.16**: `RigidTransform` for SE(3) poses, `Rotation` for conversions, `Slerp` for interpolation
- **nalgebra**: Linear algebra for rotation matrices (Rust side)

## Batch Operations

The `Robot` class supports high-performance batch operations:
- `batch_forward(joints)` - Input: numpy array (N,6) or DataFrame with J1-J6 columns; Output: `RigidTransform` containing N poses
- `batch_inverse(poses)` - Input: `RigidTransform` containing N poses; Output: numpy array (N,6) or DataFrame matching input type of `current_joints`

DataFrame support (polars/pandas) is optional - install with `pip install py-opw-kinematics[polars]` or `[pandas]`.

## Interpolation

`interpolate_poses(x, poses, xn)` - SLERP for rotation, linear for translation. API follows `scipy.interpolate.interp1d(x, y)` pattern:
- `x`: array of N keyframe positions (e.g., times)
- `poses`: `RigidTransform` with N keyframe poses
- `xn`: array of M desired interpolation positions
- Returns: `RigidTransform` with M interpolated poses

## CI/CD

GitHub Actions workflow builds wheels for Linux (x86_64, aarch64), Windows (x64), and macOS (x86_64, aarch64). Releases are triggered by tags matching `v*` and publish to PyPI.
