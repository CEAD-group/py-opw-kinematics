# Developer Guide

This guide provides information for developers who want to contribute to py-opw-kinematics or understand its internal architecture.

## Architecture Overview

py-opw-kinematics is a Python wrapper around the high-performance Rust library `rs-opw-kinematics`. The architecture consists of several layers:

```
┌─────────────────────────────────────┐
│           Python API                │
│  (Robot, KinematicModel, Euler)     │
├─────────────────────────────────────┤
│         PyO3 Bindings               │
│      (Python ↔ Rust bridge)         │
├─────────────────────────────────────┤
│         Rust Core                   │
│    (rs-opw-kinematics library)      │
├─────────────────────────────────────┤
│       External Dependencies         │
│  (nalgebra, polars, rayon, etc.)    │
└─────────────────────────────────────┘
```

## Project Structure

```
py-opw-kinematics/
├── python/
│   ├── py_opw_kinematics/
│   │   ├── __init__.py           # Python package entry point
│   │   ├── _internal.pyi         # Type stubs for Rust bindings
│   │   └── py.typed              # PEP 561 marker
│   ├── examples/                 # Usage examples
│   └── tests/                    # Test suite
│       ├── test_euler.py         # Euler convention tests
│       ├── test_robot.py         # Basic robot tests
│       ├── test_kinematic_model.py    # KinematicModel tests
│       ├── test_robot_advanced.py    # Advanced robot tests
│       ├── test_integration.py        # Integration tests
│       └── test_performance.py        # Performance benchmarks
├── src/                          # Rust source code
│   ├── lib.rs                    # Library entry point
│   ├── kinematic_model.rs        # KinematicModel implementation
│   ├── euler.rs                  # Euler convention handling
│   └── configuration.rs          # Robot configuration logic
├── docs/                         # Documentation
├── Cargo.toml                    # Rust dependencies
├── pyproject.toml               # Python build configuration
└── README.md
```

## Development Setup

### Prerequisites

1. **Rust**: Install from [rustup.rs](https://rustup.rs/)
2. **Python 3.11+**: With development headers
3. **maturin**: Python-Rust build tool

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Install development dependencies
pip install -e ".[test,dev]"
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/CEAD-group/py-opw-kinematics.git
cd py-opw-kinematics

# Build in development mode
maturin develop

# Or build wheel
maturin build --release
```

### Running Tests

```bash
# Run all tests
pytest python/tests/ -v

# Run specific test categories
pytest python/tests/ -m "not slow" -v      # Skip slow tests
pytest python/tests/ -m "performance" -v   # Only performance tests
pytest python/tests/ -k "kinematic" -v     # Tests matching pattern

# Run with coverage
pytest python/tests/ --cov=py_opw_kinematics --cov-report=html
```

## Code Style and Standards

### Python Code Style

- **PEP 8**: Follow Python's style guide
- **Type hints**: Use type annotations for all public APIs
- **Docstrings**: NumPy/SciPy style for all functions and classes
- **Testing**: pytest with comprehensive coverage

```python
def example_function(param1: float, param2: Optional[List[float]] = None) -> Tuple[float, float]:
    """
    Brief description of the function.

    Parameters
    ----------
    param1 : float
        Description of parameter 1.
    param2 : Optional[List[float]], default=None
        Description of optional parameter 2.

    Returns
    -------
    Tuple[float, float]
        Description of return values.

    Examples
    --------
    >>> result = example_function(1.0, [2.0, 3.0])
    >>> print(result)
    (4.0, 5.0)
    """
    # Implementation
    pass
```

### Rust Code Style

- **rustfmt**: Use standard Rust formatting
- **clippy**: Address all clippy warnings
- **Documentation**: Comprehensive doc comments
- **Error handling**: Proper error types and propagation

```rust
/// Brief description of the function.
///
/// # Arguments
///
/// * `param1` - Description of parameter 1
/// * `param2` - Description of parameter 2
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// This function will return an error if...
pub fn example_function(param1: f64, param2: &[f64]) -> Result<(f64, f64), KinematicsError> {
    // Implementation
    Ok((0.0, 0.0))
}
```

## Testing Strategy

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Ensure performance requirements
4. **Property Tests**: Test mathematical properties
5. **Regression Tests**: Prevent performance/functionality regressions

### Test Markers

```python
import pytest

@pytest.mark.slow
def test_large_dataset():
    """Tests that take more than a few seconds"""
    pass

@pytest.mark.performance
def test_benchmark():
    """Performance benchmark tests"""
    pass

@pytest.mark.integration
def test_full_workflow():
    """End-to-end integration tests"""
    pass
```

### Test Data Management

```python
# Use fixtures for complex test data
@pytest.fixture
def industrial_robot():
    """Standard industrial robot for testing"""
    model = KinematicModel(...)
    euler = EulerConvention(...)
    return Robot(model, euler)

@pytest.fixture(params=[
    "comau_nj165",
    "generic_robot",
    "robot_without_parallelogram"
])
def robot_configurations(request):
    """Parametrized robot configurations"""
    if request.param == "comau_nj165":
        return create_comau_nj165()
    # ... other configurations
```

## Performance Considerations

### Profiling Tools

```python
# Use cProfile for detailed profiling
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    result = robot.batch_forward(large_dataset)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)

# Use timeit for micro-benchmarks
import timeit

time_per_op = timeit.timeit(
    lambda: robot.forward(joints),
    number=10000
) / 10000
print(f"Time per operation: {time_per_op*1e6:.2f} μs")
```

### Memory Profiling

```python
import tracemalloc
import psutil
import os

def profile_memory():
    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Code to profile
    large_result = robot.batch_forward(huge_dataset)
    
    # Check memory usage
    current, peak = tracemalloc.get_traced_memory()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory increase: {end_memory - start_memory:.1f} MB")
    print(f"Peak traced memory: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

## API Design Principles

### Consistency

- **Naming**: Use consistent naming conventions across Python and Rust
- **Parameters**: Similar functions should have similar parameter orders
- **Return types**: Use consistent return type patterns

```python
# Good: Consistent naming and parameters
robot.forward(joints) -> (position, orientation)
robot.inverse(pose, current_joints=None) -> List[joints]
robot.batch_forward(joints_df) -> poses_df
robot.batch_inverse(poses_df, current_joints=None) -> joints_df
```

### Error Handling

- **Graceful degradation**: Return empty lists rather than crashing
- **Informative errors**: Provide clear error messages
- **Type safety**: Use type hints and validation

```python
def validate_joints(joints: Tuple[float, ...]) -> None:
    """Validate joint configuration"""
    if len(joints) != 6:
        raise ValueError(f"Expected 6 joints, got {len(joints)}")
    
    if any(not isinstance(j, (int, float)) for j in joints):
        raise TypeError("All joint values must be numeric")
```

### Performance

- **Batch operations**: Prefer batch operations for multiple items
- **Memory efficiency**: Use views and lazy evaluation where possible
- **Rust optimization**: Leverage Rust's performance for heavy computation

## Adding New Features

### 1. Define the API

```python
# First, define the Python API you want
def new_feature(self, param1: float, param2: Optional[List[float]] = None) -> ResultType:
    """
    New feature description.
    
    Parameters
    ----------
    param1 : float
        Parameter description.
    param2 : Optional[List[float]], default=None
        Optional parameter description.
    
    Returns
    -------
    ResultType
        Result description.
    """
    pass
```

### 2. Implement in Rust

```rust
// src/lib.rs or appropriate module
use pyo3::prelude::*;

#[pymethods]
impl Robot {
    fn new_feature(&self, param1: f64, param2: Option<Vec<f64>>) -> PyResult<ResultType> {
        // Implementation
        Ok(result)
    }
}
```

### 3. Add Type Stubs

```python
# python/py_opw_kinematics/_internal.pyi
class Robot:
    def new_feature(
        self, 
        param1: float, 
        param2: Optional[List[float]] = None
    ) -> ResultType: ...
```

### 4. Write Tests

```python
# python/tests/test_new_feature.py
import pytest
from py_opw_kinematics import Robot

class TestNewFeature:
    def test_basic_functionality(self, basic_robot):
        result = basic_robot.new_feature(1.0)
        assert isinstance(result, ExpectedType)
    
    def test_edge_cases(self, basic_robot):
        # Test edge cases
        pass
    
    @pytest.mark.parametrize("param1", [0.0, 1.0, -1.0, 100.0])
    def test_parameter_validation(self, basic_robot, param1):
        result = basic_robot.new_feature(param1)
        # Assertions
```

### 5. Add Documentation

```markdown
# docs/api.md (or appropriate doc)

#### new_feature(param1, param2=None)

Description of the new feature.

**Parameters:**
- `param1`: Description
- `param2`: Optional description

**Returns:**
- Description of return value

**Example:**
```python
result = robot.new_feature(1.0, [2.0, 3.0])
```
```

### 6. Update Examples

```python
# python/examples/example_new_feature.py
"""
Example demonstrating the new feature.
"""
from py_opw_kinematics import Robot, KinematicModel, EulerConvention

# Setup
robot = create_example_robot()

# Use new feature
result = robot.new_feature(1.0)
print(f"Result: {result}")
```

## Release Process

### 1. Version Management

```toml
# pyproject.toml
[project]
dynamic = ["version"]  # Version comes from Cargo.toml

# Cargo.toml
[package]
version = "0.1.0"  # Update this for releases
```

### 2. Pre-release Checklist

```bash
# 1. Run full test suite
pytest python/tests/ -v

# 2. Run performance tests
pytest python/tests/ -m performance -v

# 3. Check code formatting
black python/
isort python/
rustfmt src/

# 4. Run linters
flake8 python/
cargo clippy

# 5. Update documentation
# Update CHANGELOG.md
# Update version numbers
# Update examples if needed

# 6. Build and test wheel
maturin build --release
pip install target/wheels/py_opw_kinematics-*.whl
pytest python/tests/ -v  # Test installed package
```

### 3. Release

```bash
# Create release tag
git tag v0.1.0
git push origin v0.1.0

# Build wheels for multiple platforms
maturin build --release --target x86_64-unknown-linux-gnu
maturin build --release --target x86_64-pc-windows-msvc
maturin build --release --target x86_64-apple-darwin

# Upload to PyPI
maturin publish
```

## Debugging Tips

### Python-Rust Boundary Issues

```python
# Check types at boundary
def debug_rust_call(robot, joints):
    print(f"Python joints type: {type(joints)}")
    print(f"Python joints value: {joints}")
    
    try:
        result = robot.forward(joints)
        print(f"Rust result type: {type(result)}")
        print(f"Rust result value: {result}")
        return result
    except Exception as e:
        print(f"Error at Python-Rust boundary: {e}")
        print(f"Error type: {type(e)}")
        raise
```

### Memory Issues

```python
import gc
import tracemalloc

def debug_memory_leak():
    tracemalloc.start()
    
    # Potentially leaky code
    for i in range(1000):
        result = robot.batch_forward(large_dataset)
        if i % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i}: {current / 1024 / 1024:.1f} MB")
            gc.collect()  # Force garbage collection
    
    tracemalloc.stop()
```

### Performance Issues

```python
import cProfile
import time

def debug_performance():
    # Profile specific operations
    profiler = cProfile.Profile()
    
    # Time overall operation
    start_time = time.perf_counter()
    
    profiler.enable()
    result = expensive_operation()
    profiler.disable()
    
    end_time = time.perf_counter()
    
    print(f"Total time: {end_time - start_time:.3f}s")
    profiler.print_stats(sort='cumulative')
```

## Contributing Guidelines

### Before Contributing

1. **Check existing issues**: Look for related issues or feature requests
2. **Discuss large changes**: Open an issue to discuss significant changes
3. **Read the code**: Understand the existing codebase structure

### Pull Request Process

1. **Fork and branch**: Create a feature branch from main
2. **Implement changes**: Follow coding standards and add tests
3. **Test thoroughly**: Ensure all tests pass
4. **Update documentation**: Add/update relevant documentation
5. **Submit PR**: Provide clear description of changes

### Code Review Criteria

- **Functionality**: Does the code work as intended?
- **Performance**: Does it meet performance requirements?
- **Testing**: Are there comprehensive tests?
- **Documentation**: Is it properly documented?
- **Style**: Does it follow project conventions?

## Common Pitfalls

### 1. Precision Issues

```python
# Be careful with floating point comparisons
# Bad
assert position[0] == expected_x

# Good
import numpy as np
assert np.isclose(position[0], expected_x, atol=1e-6)
```

### 2. Coordinate System Confusion

```python
# Always document coordinate system assumptions
def test_position_accuracy():
    """Test assumes standard OPW coordinate system with Z-up"""
    joints = [0, 0, -90, 0, 0, 0]  # Robot arm pointing forward
    position, _ = robot.forward(joints)
    
    # X should be positive (forward)
    assert position[0] > 0
    # Z should be positive (up)
    assert position[2] > 0
```

### 3. Angle Unit Confusion

```python
# Always be explicit about angle units
def create_robot_degrees():
    """Create robot with degree-based Euler convention"""
    euler_conv = EulerConvention("XYZ", extrinsic=False, degrees=True)
    return Robot(model, euler_conv)

def create_robot_radians():
    """Create robot with radian-based Euler convention"""
    euler_conv = EulerConvention("XYZ", extrinsic=False, degrees=False)
    return Robot(model, euler_conv)
```

This developer guide should help you understand the project structure, contribute effectively, and maintain high code quality.