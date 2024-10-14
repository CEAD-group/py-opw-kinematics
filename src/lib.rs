mod euler; // Add this line to import the new module
mod kinematic_model; // Add this line to import the new module
use crate::euler::EulerConvention;
use crate::kinematic_model::KinematicModel;

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use polars::frame::DataFrame;
use polars::prelude::*;
use polars::series::Series;

use pyo3_polars::PyDataFrame;
use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose};
use rs_opw_kinematics::kinematics_impl::OPWKinematics;

#[pyclass]
struct Robot {
    robot: OPWKinematics,
    has_parallelogram: bool,
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
        let has_parallelogram = kinematic_model.has_parallelogram;
        let degrees = euler_convention.degrees;

        // Initialize the internal rotation matrix to identity as a placeholder
        let _ee_rotation_matrix = Rotation3::identity(); // Assuming Rotation3::identity() is a valid way to initialize to identity

        let _internal_euler_convention = EulerConvention::new("XYZ".to_string(), false, degrees)?;

        // Create an instance with initial values
        let mut robot_instance = Robot {
            robot,
            has_parallelogram,
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
            ._matrix_to_euler(self._ee_rotation_matrix);
        Ok(euler_angles)
    }

    #[setter]
    fn set_ee_rotation(&mut self, ee_rotation: [f64; 3]) -> PyResult<()> {
        self._ee_rotation_matrix = self.euler_convention._euler_to_matrix(ee_rotation);
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
        if self.has_parallelogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let combined_rotation = pose.rotation.to_rotation_matrix() * self._ee_rotation_matrix;
        let translation = pose.translation.vector + combined_rotation * self.ee_translation;
        let rotation = self.euler_convention._matrix_to_euler(combined_rotation);

        (translation.into(), rotation)
    }

    #[pyo3(signature = (pose, current_joints=None))]
    fn inverse(
        &self,
        pose: ([f64; 3], [f64; 3]),
        current_joints: Option<[f64; 6]>,
    ) -> Vec<[f64; 6]> {
        let rotation_matrix = self.euler_convention._euler_to_matrix(pose.1);
        let rotated_ee_translation = rotation_matrix * Vector3::from(self.ee_translation);
        let translation = Translation3::from(Vector3::from(pose.0) - rotated_ee_translation);
        let rotation = rotation_matrix * self._ee_rotation_matrix.inverse();
        let iso_pose = Isometry3::from_parts(translation, rotation.into());
        let mut solutions = match current_joints {
            Some(mut joints) => {
                if self.has_parallelogram {
                    joints[2] += joints[1];
                }
                if self.euler_convention.degrees {
                    joints.iter_mut().for_each(|x| *x = x.to_radians());
                }
                self.robot.inverse_continuing(&iso_pose, &joints)
            }
            None => self.robot.inverse(&iso_pose),
        };

        if self.has_parallelogram {
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
        let df: DataFrame = poses.into();

        let x = extract_column_f64(&df, "X")?;
        let y = extract_column_f64(&df, "Y")?;
        let z = extract_column_f64(&df, "Z")?;
        let a = extract_column_f64(&df, "A")?;
        let b = extract_column_f64(&df, "B")?;
        let c = extract_column_f64(&df, "C")?;

        // Use Vec<Option<f64>> to allow for None (Null) values
        let mut j1: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut j2: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut j3: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut j4: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut j5: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut j6: Vec<Option<f64>> = Vec::with_capacity(df.height());

        for i in 0..df.height() {
            // Safely extract pose components, handling missing values
            let x_i = x.get(i);
            let y_i = y.get(i);
            let z_i = z.get(i);
            let a_i = a.get(i);
            let b_i = b.get(i);
            let c_i = c.get(i);

            if let (Some(x), Some(y), Some(z), Some(a), Some(b), Some(c)) =
                (x_i, y_i, z_i, a_i, b_i, c_i)
            {
                let pose = ([x, y, z], [a, b, c]);

                let solutions = self.inverse(pose, current_joints);
                if let Some(best_solution) = solutions.first() {
                    j1.push(Some(best_solution[0]));
                    j2.push(Some(best_solution[1]));
                    j3.push(Some(best_solution[2]));
                    j4.push(Some(best_solution[3]));
                    j5.push(Some(best_solution[4]));
                    j6.push(Some(best_solution[5]));
                    current_joints = Some(*best_solution);
                } else {
                    // No solution found, push None values
                    j1.push(None);
                    j2.push(None);
                    j3.push(None);
                    j4.push(None);
                    j5.push(None);
                    j6.push(None);
                }
            } else {
                // Missing pose components, push None values
                j1.push(None);
                j2.push(None);
                j3.push(None);
                j4.push(None);
                j5.push(None);
                j6.push(None);
            }
        }

        // Create Series with optional values to allow Nulls
        let df_result = DataFrame::new(vec![
            Series::new("J1".into(), j1),
            Series::new("J2".into(), j2),
            Series::new("J3".into(), j3),
            Series::new("J4".into(), j4),
            Series::new("J5".into(), j5),
            Series::new("J6".into(), j6),
        ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyDataFrame(df_result))
    }

    #[pyo3(signature = (joints))]
    fn batch_forward(&self, joints: PyDataFrame) -> PyResult<PyDataFrame> {
        let df: DataFrame = joints.into();

        let j1 = extract_column_f64(&df, "J1")?;
        let j2 = extract_column_f64(&df, "J2")?;
        let j3 = extract_column_f64(&df, "J3")?;
        let j4 = extract_column_f64(&df, "J4")?;
        let j5 = extract_column_f64(&df, "J5")?;
        let j6 = extract_column_f64(&df, "J6")?;

        // Use Vec<Option<f64>> to allow for None (Null) values
        let mut x: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut y: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut z: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut a: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut b: Vec<Option<f64>> = Vec::with_capacity(df.height());
        let mut c: Vec<Option<f64>> = Vec::with_capacity(df.height());

        for i in 0..df.height() {
            // Safely extract joint values, handling missing values
            let j1_i = j1.get(i);
            let j2_i = j2.get(i);
            let j3_i = j3.get(i);
            let j4_i = j4.get(i);
            let j5_i = j5.get(i);
            let j6_i = j6.get(i);

            if let (Some(j1), Some(j2), Some(j3), Some(j4), Some(j5), Some(j6)) =
                (j1_i, j2_i, j3_i, j4_i, j5_i, j6_i)
            {
                let joints_array = [j1, j2, j3, j4, j5, j6];
                let (translation, rotation) = self.forward(joints_array);

                x.push(Some(translation[0]));
                y.push(Some(translation[1]));
                z.push(Some(translation[2]));
                a.push(Some(rotation[0]));
                b.push(Some(rotation[1]));
                c.push(Some(rotation[2]));
            } else {
                // Missing joint values, push None values
                x.push(None);
                y.push(None);
                z.push(None);
                a.push(None);
                b.push(None);
                c.push(None);
            }
        }

        // Create Series with optional values to allow Nulls
        let df_result = DataFrame::new(vec![
            Series::new("X".into(), x),
            Series::new("Y".into(), y),
            Series::new("Z".into(), z),
            Series::new("A".into(), a),
            Series::new("B".into(), b),
            Series::new("C".into(), c),
        ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
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
