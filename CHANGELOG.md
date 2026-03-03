# Changelog

All notable changes to this project will be documented in this file.

## [v1.1.1] - 2026-02-19

### Added
- `joint_poses` and `batch_joint_poses` methods for computing forward kinematics joint frames using the rs-opw-kinematics FK chain

### Deprecated
- `forward_frames` is deprecated in favour of `joint_poses`

---

## [v1.1.0] - 2026-02-19

### Removed
- **Breaking:** Parallelogram kinematics support removed entirely. Robots with a parallelogram linkage (where joint 3 compensates for joint 2) must now apply this compensation themselves before passing joint angles.

### Fixed
- Corrected frames output from forward kinematics frame chain

---

## [v1.0.0] - 2026-01-09

Major rewrite of the Python API.

### Added
- `scipy.spatial.transform.Rotation` / `RigidTransform` as the primary input/output type for all poses
- Vectorized batch operations — no Python-level loops
- `interpolate_poses()` for SLERP trajectory interpolation between poses
- Forward kinematics joint frame chain (`forward_frames`)
- `NumpyOrDataFrame` type alias for batch operation signatures
- Polars and Pandas are now optional dependencies, detected at runtime
- mypy type checking added to CI
- `py.typed` marker for downstream type checking
- SLERP trajectory example (`example_slerp_trajectory.py`)

### Changed
- Rotation handling fully delegated to `scipy.spatial.transform` — the custom Euler implementation is removed
- End-effector transforms are now stateless parameters rather than object state
- Rust side replaced `polars`/`pyo3-polars` with `numpy` crate, reducing build time from ~39s to ~7s
- Public batch API now accepts NumPy 2D arrays directly (Polars/Pandas DataFrames still supported via Python wrapper)
- Minimum supported Python version is 3.11

### Removed
- `euler.rs` and the custom Euler-convention implementation
- `test_euler.py` (superseded by delegating to scipy)

---

## [v0.3.1] - 2026-02-04

### Changed
- Made the Python interface more generic (broader type support)
- Updated PyO3 to a newer version

---

## [v0.3.0] - 2025-07-22

### Removed
- Polars DataFrame support removed from the public API (moved to pure NumPy arrays on the Rust side)

### Fixed
- Linter issues

---

## [v0.2.1] - 2025-07-22

### Changed
- Updated Python interface with minor improvements
- Updated lockfile

---

## [v0.2.0] - 2025-07-22

### Added
- `axis_configuration` method
- Axis configuration filtering support

### Changed
- Refactored pose representation to use quaternions (`feat: refactoring for quaternions`)
- Updated tool and base frame representation
- Axis normalization improvements

### Fixed
- Quaternion output for Polars DataFrames
- Type signature fixes
- Naive batch inverse based on axis configuration
- Axis configuration and normalization bugs

---

## [v0.1.10] - 2025-12-10

### Changed
- Updated Polars to 0.51 (fixes compatibility with Polars 1.x Python API)

---

## [v0.1.9] - 2025-08-26

### Changed
- Updated dependencies

---

## [v0.1.8] - 2024-10-21

### Changed
- Relaxed NumPy version requirements

---

## [v0.1.7] - 2024-10-17

### Changed
- Tag-based versioning for automated releases
- Updated CI/CD pipeline

---

## [v0.1.6] - 2024-10-15

### Fixed
- Fixed a typo in documentation

---

## [v0.1.5] - 2024-10-15

### Fixed
- Fixed type hints

---

## [v0.1.4] - 2024-10-14

### Changed
- Refactored `EulerConvention` handling and fixed related bugs

---

## [v0.1.3] - 2024-10-08

### Changed
- `offsets` and `flip_axes` arguments now use explicit tuples; defaults added and type hints updated

### Fixed
- Fixed examples in README and in the example scripts
- Fixed parallelogram spelling
- Specified `pyarrow` and `numpy` as explicit dependencies

---

## [v0.1.2] - 2024-10-07

### Added
- Batch forward kinematics support
- Updated `rs-opw-kinematics` to 1.5.0

### Changed
- Improved type hints for Euler sequence parameters

### Fixed
- Bugs in batch forward and inverse methods

---

## [v0.1.1] - 2024-10-03

### Added
- Contributing guidelines (`CONTRIBUTING.md`)

---

## [v0.1.0] / [v0.1] - 2024-10-03

Initial public release.

### Features
- Forward and inverse kinematics for OPW robots via Rust bindings (`rs-opw-kinematics`)
- Batch forward and inverse kinematics
- Euler angle convention support
- PyPI packaging and CI/CD pipeline
