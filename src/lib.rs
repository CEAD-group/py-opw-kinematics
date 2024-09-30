use nalgebra::{Isometry3, Matrix3, Rotation, Rotation3, Unit, Vector3};
use polars::frame::DataFrame;
use polars::prelude::*;
use polars::series::Series;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose};
use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;
#[pyclass]
#[derive(Clone)]
struct EulerConvention {
    sequence: String,
    extrinsic: bool,
    degrees: bool,
    _seq: [Unit<Vector3<f64>>; 3],
}

impl EulerConvention {
    fn _from_rotation_matrix_radians(&self, rot: Rotation<f64, 3>) -> [f64; 3] {
        let (angles, _observable) = rot.euler_angles_ordered(self._seq, self.extrinsic);
        angles
    }
    fn _from_rotation_matrix(&self, rot: Rotation<f64, 3>) -> [f64; 3] {
        let angles = self._from_rotation_matrix_radians(rot);
        if self.degrees {
            angles.map(|angle| angle.to_degrees())
        } else {
            angles
        }
    }
    fn _to_rotation_matrix_radians(&self, angles: [f64; 3]) -> Rotation3<f64> {
        let [a1, a2, a3] = angles;
        let r1 = Rotation3::from_axis_angle(&self._seq[0], a1);
        let r2 = Rotation3::from_axis_angle(&self._seq[1], a2);
        let r3 = Rotation3::from_axis_angle(&self._seq[2], a3);
        if self.extrinsic {
            r3 * r2 * r1
        } else {
            r1 * r2 * r3
        }
    }
    fn _to_rotation_matrix(&self, angles: [f64; 3]) -> Rotation3<f64> {
        let mut angles = angles;
        if self.degrees {
            angles = angles.map(|angle| angle.to_radians());
        }
        self._to_rotation_matrix_radians(angles)
    }
}

#[pymethods]
impl EulerConvention {
    #[new]
    fn new(sequence: String, extrinsic: bool, degrees: bool) -> PyResult<Self> {
        if sequence.len() != 3 {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Expected a 3-character sequence, but got {} characters",
                sequence.len()
            )));
        }

        let _seq: [Unit<Vector3<f64>>; 3] = sequence
            .chars()
            .map(|c| match c {
                'X' => Ok(Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0))),
                'Y' => Ok(Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0))),
                'Z' => Ok(Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0))),
                _ => Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid character '{}'. Expected only 'X', 'Y', or 'Z'.",
                    c
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| {
                PyErr::new::<PyValueError, _>("Invalid sequence. Must be exactly 3 characters.")
            })?;

        Ok(EulerConvention {
            sequence,
            extrinsic,
            degrees,
            _seq,
        })
    }

    fn convert(&self, other: &EulerConvention, angles: [f64; 3]) -> PyResult<[f64; 3]> {
        let rot_matrix = self._to_rotation_matrix(angles);
        let result = other._from_rotation_matrix(rot_matrix);

        Ok(result)
    }

    fn from_rotation_matrix(&self, rot: [[f64; 3]; 3]) -> [f64; 3] {
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        self._from_rotation_matrix(rotation)
    }

    fn to_rotation_matrix(&self, angles: [f64; 3]) -> [[f64; 3]; 3] {
        let matrix = self._to_rotation_matrix(angles).into_inner();
        [
            [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
            [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
            [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "EulerConvention(sequence='{}', extrinsic={}, degrees={})",
            self.sequence,
            if self.extrinsic { "True" } else { "False" },
            if self.degrees { "True" } else { "False" }
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[derive(Clone)]
struct KinematicModel {
    parameters: Parameters,
    has_parallellogram: bool,
}

#[pymethods]
impl KinematicModel {
    #[new]
    fn new(
        a1: f64,
        a2: f64,
        b: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        offsets: [f64; 6],
        sign_corrections: [i32; 6],
        has_parallellogram: bool,
    ) -> PyResult<Self> {
        let mut parameters = Parameters::new();
        parameters.a1 = a1;
        parameters.a2 = a2;
        parameters.b = b;
        parameters.c1 = c1;
        parameters.c2 = c2;
        parameters.c3 = c3;
        parameters.c4 = c4;
        parameters.offsets = offsets;
        parameters.sign_corrections = sign_corrections.map(|x| x as i8);

        // check that all signs are either -1 or 1
        for sign in parameters.sign_corrections.iter() {
            if *sign != 1 && *sign != -1 {
                return Err(PyErr::new::<PyValueError, _>(
                    "Sign correction must be either 1 or -1",
                ));
            }
        }
        Ok(KinematicModel {
            parameters,
            has_parallellogram,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "KinematicModel(a1={}, a2={}, b={}, c1={}, c2={}, c3={}, c4={}, offsets={:?}, sign_corrections={:?}, has_parallellogram={})",
            self.parameters.a1,
            self.parameters.a2,
            self.parameters.b,
            self.parameters.c1,
            self.parameters.c2,
            self.parameters.c3,
            self.parameters.c4,
            self.parameters.offsets,
            self.parameters.sign_corrections,
            if self.has_parallellogram { "True" } else { "False" }
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
struct Robot {
    robot: OPWKinematics,
    has_parallellogram: bool,
    euler_convention: EulerConvention,
    _ee_rotation_matrix: Rotation3<f64>, // Store the ee_rotation as a private field
    _internal_euler_convention: EulerConvention,
}

#[pymethods]
impl Robot {
    #[new]
    #[pyo3(signature = (kinematic_model, euler_convention, ee_rotation=None))]
    fn new(
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let robot = OPWKinematics::new(kinematic_model.parameters);
        let has_parallellogram = kinematic_model.has_parallellogram;
        let degrees = euler_convention.degrees;

        // Initialize the internal rotation matrix to identity as a placeholder
        let _ee_rotation_matrix = Rotation3::identity(); // Assuming Rotation3::identity() is a valid way to initialize to identity

        let _internal_euler_convention = EulerConvention::new("XYZ".to_string(), false, degrees)?;

        // Create an instance with initial values
        let mut robot_instance = Robot {
            robot,
            has_parallellogram,
            euler_convention,
            _ee_rotation_matrix,
            _internal_euler_convention,
        };

        // Use the setter to assign ee_rotation if provided
        robot_instance.set_ee_rotation(ee_rotation.unwrap_or([0.0, 0.0, 0.0]))?;

        Ok(robot_instance)
    }

    /// Getter for ee_rotation
    #[getter]
    fn get_ee_rotation(&self) -> PyResult<[f64; 3]> {
        let euler_angles = self
            .euler_convention
            ._from_rotation_matrix(self._ee_rotation_matrix);
        Ok(euler_angles)
    }

    /// Setter for ee_rotation
    #[setter]
    fn set_ee_rotation(&mut self, ee_rotation: [f64; 3]) -> PyResult<()> {
        let ee_rotation = ee_rotation;
        self._ee_rotation_matrix = self.euler_convention._to_rotation_matrix(ee_rotation);
        Ok(())
    }

    /// Forward kinematics: calculates the pose for given joints
    fn forward(&self, mut joints: [f64; 6]) -> ([f64; 3], [f64; 3]) {
        if self.has_parallellogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints = joints.map(|x| x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let translation = pose.translation.vector.into();

        let robot_rotation_matrix = pose.rotation.to_rotation_matrix();
        let combined_rotation = robot_rotation_matrix * self._ee_rotation_matrix;

        let rotation: [f64; 3] = self
            .euler_convention
            ._from_rotation_matrix(combined_rotation);
        (translation, rotation)
    }

    /// Inverse kinematics: calculates the joint solutions for a given pose
    fn inverse(&self, pose: ([f64; 3], [f64; 3])) -> Vec<[f64; 6]> {
        let translation = nalgebra::Translation3::from(pose.0);
        let rotation =
            self.euler_convention._to_rotation_matrix(pose.1) * self._ee_rotation_matrix.inverse();

        let iso_pose = Isometry3::from_parts(translation, rotation.into());
        let mut solutions = self.robot.inverse(&iso_pose);

        // Handle special case for parallelogram configuration
        if self.has_parallellogram {
            solutions.iter_mut().for_each(|x| x[2] -= x[1]);
        }

        if self.euler_convention.degrees {
            solutions
                .iter_mut()
                .for_each(|x| *x = x.map(|angle| angle.to_degrees()));
        }
        solutions
    }

    /// Inverse kinematics with continuation from close joints
    fn inverse_continuing(
        &self,
        pose: ([f64; 3], [f64; 3]),
        current_joints: [f64; 6],
    ) -> Vec<[f64; 6]> {
        let translation = nalgebra::Translation3::from(pose.0);
        let rotation = self.euler_convention._to_rotation_matrix(pose.1);
        let iso_pose = Isometry3::from_parts(translation, rotation.into());

        self.robot.inverse_continuing(&iso_pose, &current_joints)
    }

    #[pyo3(signature = (poses, current_joints=None))]
    fn batch_inverse(
        &self,
        py: Python,
        poses: PyDataFrame,
        current_joints: Option<Vec<f64>>,
    ) -> PyResult<PyDataFrame> {
        // Convert PyDataFrame to a Polars DataFrame
        let df: DataFrame = poses.into();

        // Extract x, y, z, a, b, c columns from the DataFrame safely
        let x = df
            .column("X")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let y = df
            .column("Y")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let z = df
            .column("Z")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let a = df
            .column("A")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let b = df
            .column("B")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let c = df
            .column("C")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        // Check if current_joints is provided, otherwise use default values
        let mut current_joints: [f64; 6] = if let Some(joints) = current_joints {
            if joints.len() != 6 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "current_joints must contain 6 values representing the current joint angles.",
                ));
            }
            joints.try_into().expect("Invalid joint length") // Convert Vec<f64> to [f64; 6]
        } else {
            [0.0; 6] // Default starting joints if not provided
        };

        // Prepare to store joint solutions for each pose
        let mut pose_indices = Vec::with_capacity(df.height());
        let mut j0 = Vec::with_capacity(df.height());
        let mut j1 = Vec::with_capacity(df.height());
        let mut j2 = Vec::with_capacity(df.height());
        let mut j3 = Vec::with_capacity(df.height());
        let mut j4 = Vec::with_capacity(df.height());
        let mut j5 = Vec::with_capacity(df.height());

        // Iterate through all poses and calculate inverse kinematics using continuation
        for i in 0..x.len() {
            // Get elements from the ChunkedArray safely
            let pose = (
                [
                    x.get(i).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing value in x")
                    })?,
                    y.get(i).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing value in y")
                    })?,
                    z.get(i).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing value in z")
                    })?,
                ],
                [
                    a.get(i).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing value in a")
                    })?,
                    b.get(i).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing value in b")
                    })?,
                    c.get(i).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing value in c")
                    })?,
                ],
            );

            let solutions = self.inverse_continuing(pose, current_joints);

            // Assuming the first solution is the best continuation
            if let Some(best_solution) = solutions.first() {
                pose_indices.push(i as f64);
                j0.push(best_solution[0]);
                j1.push(best_solution[1]);
                j2.push(best_solution[2]);
                j3.push(best_solution[3]);
                j4.push(best_solution[4]);
                j5.push(best_solution[5]);

                // Update current joints for the next iteration
                current_joints = *best_solution;
            }
        }

        // Create the DataFrame from the column vectors
        let df_result = DataFrame::new(vec![
            Series::new("pose_index".into(), pose_indices),
            Series::new("j0".into(), j0),
            Series::new("j1".into(), j1),
            Series::new("j2".into(), j2),
            Series::new("j3".into(), j3),
            Series::new("j4".into(), j4),
            Series::new("j5".into(), j5),
        ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        // Wrap the DataFrame into a PyDataFrame and return it to Python
        Ok(PyDataFrame(df_result))
    }
}

/// Module initialization for Python
#[pymodule(name = "_internal")]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EulerConvention>()?;
    m.add_class::<KinematicModel>()?;
    m.add_class::<Robot>()?;
    Ok(())
}
