# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

py-opw-kinematics is a Python wrapper for the rs-opw-kinematics Rust library, providing forward and inverse kinematics for six-axis industrial robots with a parallel base and spherical wrist (OPW-type robots).

## Goals

- Inverse kinematics (single and batch)
- Batch IK considers previous state to find closest solution
- Forward kinematics outputs all link frames, not just end effector
- Optional tool_offset parameter for TCP calculations
- Stateless API - pure kinematics library

## Non-Goals

- External axes handling (handled by higher-level coordination layer)
- Collision detection (separate concern, e.g. py-parry3d)
- Parallelogram linkage mechanics (handled by higher-level coordination layer)
- Euler angle / rotation conversions (use scipy.spatial.transform.Rotation)

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
- **tests/**: pytest tests for Euler conventions and robot kinematics
- **examples/**: Usage examples including batch operations

### Key Dependencies

- **rs-opw-kinematics**: Rust library providing the core OPW kinematics solver
- **pyo3**: Rust-Python bindings
- **pyo3-polars**: Polars DataFrame integration for batch operations
- **nalgebra**: Linear algebra for rotation matrices and quaternions

## Batch Operations

The `Robot` class supports high-performance batch operations via Polars DataFrames:
- `batch_inverse(poses: pl.DataFrame)` - Input columns: X, Y, Z, A, B, C; Output: J1-J6
- `batch_forward(joints: pl.DataFrame)` - Input columns: J1-J6; Output: X, Y, Z, A, B, C

## CI/CD

GitHub Actions workflow builds wheels for Linux (x86_64, aarch64), Windows (x64), and macOS (x86_64, aarch64). Releases are triggered by tags matching `v*` and publish to PyPI.
