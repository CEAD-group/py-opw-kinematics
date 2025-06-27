# py-opw-kinematics

⚠️ Forked from [py-opw-kinematics](https://github.com/bourumir-wyngs/py-opw-kinematics) to have a simplified version with quaternions instead of Euler angles.

**py-opw-kinematics** is a Python wrapper for the [rs-opw-kinematics](https://github.com/bourumir-wyngs/rs-opw-kinematics) library, providing an interface for solving inverse and forward kinematics of six-axis industrial robots with a parallel base and spherical wrist. Designed for convenience and performance, this wrapper is suitable for robotics simulation, control, and trajectory planning directly from Python.

## Key Features

- **Ease of Use**: Fully customizable end-effector rotation using Euler angles. Configurable to use degrees or radians.
- **High Performance**: Capable of batch operations using Polars DataFrames for maximum efficiency. For example, 100,000 inverse kinematic solutions can be computed in just 0.4 seconds.
- **Full Rust Integration**: Uses Rust for the core kinematic calculations, offering speed and robustness while allowing access through Python.
- **Singularity Handling**: Manages kinematic singularities such as J5 = 0° or ±180°.

## Installation

Install using pip:

```sh
uv pip install https://github.com/TETMET/py-opw-kinematics
```

Note: Rust is required to compile the underlying Rust library if not using pre-built binaries.

## Usage Example

### Parameters


![OPW Diagram](https://bourumir-wyngs.github.io/rs-opw-kinematics/documentation/opw.gif)
<!-- ![OPW Diagram](documentation/opw.gif) -->

Single Operation Example
    
```python
    # Initialize Kinematic Model with known parameters and inlined signs
    kinematic_model = KinematicModel(
        a1=0.150,
        a2=-0.110,
        b=0.0,
        c1=0.4865,
        c2=0.700,
        c3=0.678,
        c4=0.135,
        offsets=(0, 0, -np.pi / 2, 0, 0, 0),
        sign_corrections=(1, 1, 1, 1, 1, 1),
    )

    base_config = BaseConfig(translation=[0, 0, 2.3], rotation=[0, 1, 0, 0])
    tool_config = ToolConfig(translation=[0, 0, 0.095], rotation=[-0.00012991440873552217, -0.968154906938256, -0.0004965996111545046, 0.2503407964804168])

    robot = Robot(kinematic_model, base_config, tool_config)
    pose = robot.forward([-103.1, -85.03, 19.06, -70.19, -35.87, 185.01])
    print(f"Position: {pose[0]}")
    print(f"Rotation: {pose[1]}")
```
This example prints:
    
```
Position: [0.200, -0.3, 0.9], Rotation: [0.8518, 0.13766, -0.46472, -0.19852]
```

## Acknowledgements

This project builds on the Rust library rs-opw-kinematics by Bourumir Wyngs, which itself draws inspiration from:

- The 2014 research paper: An Analytical Solution of the Inverse Kinematics Problem of Industrial Serial Manipulators with an Ortho-parallel Basis and a Spherical Wrist, authored by Mathias Brandstötter, Arthur Angerer, and Michael Hofbaur (ResearchGate link).
- The C++ project opw_kinematics, which provided valuable insights for validation and testing.


## Licensing

The `py-opw-kinematics` library itself is licensed under MIT.

The image `opw.png`, used for documentation purposes, is sourced from [opw_kinematics](https://github.com/Jmeyer1292/opw_kinematics) and is licensed under the Apache License 2.0.

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details on how to get started.
