use nalgebra::{Isometry3, Matrix3, Rotation, Rotation3, Translation3, Unit, Vector3};
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
    fn _angles_from_rotation_matrix_radians(&self, rot: Rotation<f64, 3>) -> [f64; 3] {
        let (angles, _observable) = rot.euler_angles_ordered(self._seq, self.extrinsic);
        angles
    }
    fn _angles_from_rotation_matrix(&self, rot: Rotation<f64, 3>) -> [f64; 3] {
        let angles = self._angles_from_rotation_matrix_radians(rot);
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
        let result = other._angles_from_rotation_matrix(rot_matrix);

        Ok(result)
    }

    fn angles_from_rotation_matrix(&self, rot: [[f64; 3]; 3]) -> [f64; 3] {
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        self._angles_from_rotation_matrix(rotation)
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

#[pyclass(frozen)] // Declare the class as frozen to provide immutability.
#[derive(Clone)]
struct KinematicModel {
    a1: f64,
    a2: f64,
    b: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    offsets: [f64; 6],
    flip_axes: [bool; 6], // Renamed and changed to boolean array
    has_parallellogram: bool,
}

impl KinematicModel {
    fn to_opw_kinematics(&self, degrees: bool) -> OPWKinematics {
        let sign_corrections = self.flip_axes.map(|x| if x { -1 } else { 1 });
        OPWKinematics::new(Parameters {
            a1: self.a1,
            a2: self.a2,
            b: self.b,
            c1: self.c1,
            c2: self.c2,
            c3: self.c3,
            c4: self.c4,
            offsets: if degrees {
                self.offsets
                    .iter()
                    .map(|&x| x.to_radians())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            } else {
                self.offsets
            },
            sign_corrections,
        })
    }
}

#[pymethods]
impl KinematicModel {
    #[new]
    #[pyo3(signature = (
        a1,
        a2,
        b,
        c1,
        c2,
        c3,
        c4,
        offsets = [0.0; 6],
        flip_axes = [false; 6],
        has_parallellogram = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a1: f64,
        a2: f64,
        b: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        offsets: [f64; 6],
        flip_axes: [bool; 6],
        has_parallellogram: bool,
    ) -> PyResult<Self> {
        Ok(KinematicModel {
            a1,
            a2,
            b,
            c1,
            c2,
            c3,
            c4,
            offsets,
            flip_axes,
            has_parallellogram,
        })
    }

    // Getter methods to provide access to attributes since the class is frozen.
    #[getter]
    fn a1(&self) -> f64 {
        self.a1
    }

    #[getter]
    fn a2(&self) -> f64 {
        self.a2
    }

    #[getter]
    fn b(&self) -> f64 {
        self.b
    }

    #[getter]
    fn c1(&self) -> f64 {
        self.c1
    }

    #[getter]
    fn c2(&self) -> f64 {
        self.c2
    }

    #[getter]
    fn c3(&self) -> f64 {
        self.c3
    }

    #[getter]
    fn c4(&self) -> f64 {
        self.c4
    }

    #[getter]
    fn offsets(&self) -> Vec<f64> {
        self.offsets.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    fn flip_axes(&self) -> Vec<bool> {
        self.flip_axes.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    fn has_parallellogram(&self) -> bool {
        self.has_parallellogram
    }

    fn __repr__(&self) -> String {
        format!(
            "KinematicModel(\n    a1={},\n    a2={},\n    b={},\n    c1={},\n    c2={},\n    c3={},\n    c4={},\n    offsets={:?},\n    flip_axes={:?},\n    has_parallellogram={}\n)",
            self.a1, self.a2, self.b, self.c1, self.c2, self.c3, self.c4,
            self.offsets, self.flip_axes, self.has_parallellogram
        )
    }
}

#[pyclass]
struct Robot {
    robot: OPWKinematics,
    has_parallellogram: bool,
    euler_convention: EulerConvention,
    ee_rotation: [f64; 3],
    ee_translation: Vector3<f64>,
    _ee_rotation_matrix: Rotation3<f64>,
    _internal_euler_convention: EulerConvention,
    _kinematic_model: KinematicModel,
}

#[pymethods]
impl Robot {
    #[new]
    #[pyo3(signature = (kinematic_model, euler_convention, ee_rotation=None, ee_translation=None))]
    fn new(
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Option<[f64; 3]>,
        ee_translation: Option<[f64; 3]>,
    ) -> PyResult<Self> {
        let robot = kinematic_model.to_opw_kinematics(euler_convention.degrees);
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
            ee_rotation: ee_rotation.unwrap_or([0.0, 0.0, 0.0]),
            ee_translation: ee_translation.unwrap_or([0.0, 0.0, 0.0]).into(),
            _ee_rotation_matrix,
            _internal_euler_convention,
            _kinematic_model: kinematic_model,
        };

        // Use the setter to assign ee_rotation if provided
        robot_instance.set_ee_rotation(robot_instance.ee_rotation)?;

        Ok(robot_instance)
    }

    fn __repr__(&self) -> String {
        let km_repr = self
            ._kinematic_model
            .__repr__()
            .lines()
            .map(|line| format!("    {}", line)) // Indent each line of KinematicModel's repr with 4 spaces
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "Robot(\n    kinematic_model=\n{},\n    euler_convention={},\n    ee_rotation={:?},\n    ee_translation={:?}\n)",
            km_repr,
            self.euler_convention.__repr__(),
            self.ee_rotation,
            self.ee_translation
        )
    }

    #[getter]
    fn get_ee_rotation(&self) -> PyResult<[f64; 3]> {
        let euler_angles = self
            .euler_convention
            ._angles_from_rotation_matrix(self._ee_rotation_matrix);
        Ok(euler_angles)
    }

    #[setter]
    fn set_ee_rotation(&mut self, ee_rotation: [f64; 3]) -> PyResult<()> {
        self._ee_rotation_matrix = self.euler_convention._to_rotation_matrix(ee_rotation);
        Ok(())
    }

    #[getter]
    fn get_ee_translation(&self) -> [f64; 3] {
        self.ee_translation.into()
    }
    #[setter]
    fn set_ee_translation(&mut self, ee_translation: [f64; 3]) {
        self.ee_translation = ee_translation.into();
    }

    /// Forward kinematics: calculates the pose for given joints
    fn forward(&self, mut joints: [f64; 6]) -> ([f64; 3], [f64; 3]) {
        if self.has_parallellogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let combined_rotation = pose.rotation.to_rotation_matrix() * self._ee_rotation_matrix;
        let translation = pose.translation.vector + combined_rotation * self.ee_translation;
        let rotation = self
            .euler_convention
            ._angles_from_rotation_matrix(combined_rotation);

        (translation.into(), rotation)
    }

    #[pyo3(signature = (pose, current_joints=None))]
    fn inverse(
        &self,
        pose: ([f64; 3], [f64; 3]),
        current_joints: Option<[f64; 6]>,
    ) -> Vec<[f64; 6]> {
        let rotation_matrix = self.euler_convention._to_rotation_matrix(pose.1);
        let rotated_ee_translation = rotation_matrix * Vector3::from(self.ee_translation);
        let translation = Translation3::from(Vector3::from(pose.0) - rotated_ee_translation);
        let rotation = rotation_matrix * self._ee_rotation_matrix.inverse();
        let iso_pose = Isometry3::from_parts(translation, rotation.into());
        let mut solutions = match current_joints {
            Some(joints) => self.robot.inverse_continuing(&iso_pose, &joints),
            None => self.robot.inverse(&iso_pose),
        };

        if self.has_parallellogram {
            solutions.iter_mut().for_each(|x| x[2] -= x[1]);
        }

        if self.euler_convention.degrees {
            solutions.iter_mut().for_each(|x| {
                for angle in x.iter_mut() {
                    *angle = angle.to_degrees();
                }
            });
        }

        solutions
    }

    #[pyo3(signature = (poses, current_joints=None))]
    fn batch_inverse(
        &self,
        poses: PyDataFrame,
        mut current_joints: Option<[f64; 6]>,
    ) -> PyResult<PyDataFrame> {
        // Convert PyDataFrame to a Polars DataFrame
        let df: DataFrame = poses.into();

        let x = extract_column_f64(&df, "X")?;
        let y = extract_column_f64(&df, "Y")?;
        let z = extract_column_f64(&df, "Z")?;
        let a = extract_column_f64(&df, "A")?;
        let b = extract_column_f64(&df, "B")?;
        let c = extract_column_f64(&df, "C")?;

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

            let solutions = self.inverse(pose, current_joints);

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
                current_joints = Some(*best_solution);
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

// Define a function that extracts a column, casting it to Float64Chunked.
fn extract_column_f64(df: &DataFrame, column_name: &str) -> PyResult<Float64Chunked> {
    let column = df.column(column_name).map_err(|e| {
        PyErr::new::<PyValueError, _>(format!("Error extracting column '{}': {}", column_name, e))
    })?;

    // Attempt to cast the column to Float64 data type.
    let casted_column = column.cast(&DataType::Float64).map_err(|e| {
        PyErr::new::<PyValueError, _>(format!(
            "Error casting column '{}' to f64: {}",
            column_name, e
        ))
    })?;

    // Convert the casted Series to Float64Chunked.
    let chunked = casted_column.f64().map_err(|e| {
        PyErr::new::<PyValueError, _>(format!(
            "Error converting column '{}' to Float64Chunked: {}",
            column_name, e
        ))
    })?;

    Ok(chunked.clone()) // Return an owned clone to satisfy ownership requirements.
}
