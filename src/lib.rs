mod euler; // Add this line to import the new module
mod kinematic_model; // Add this line to import the new module
use crate::euler::EulerConvention;
use crate::kinematic_model::KinematicModel;

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use pyo3::prelude::*;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
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

    /// Batch inverse kinematics using NumPy arrays.
    /// Input: poses array of shape (n, 6) with columns [X, Y, Z, A, B, C]
    /// Output: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Returns NaN for rows where no solution is found.
    #[pyo3(signature = (poses, current_joints=None))]
    fn batch_inverse<'py>(
        &self,
        py: Python<'py>,
        poses: PyReadonlyArray2<'py, f64>,
        mut current_joints: Option<[f64; 6]>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let poses_array = poses.as_array();
        let n = poses_array.nrows();

        let mut results: Vec<f64> = Vec::with_capacity(n * 6);

        for i in 0..n {
            let row = poses_array.row(i);

            // Check for NaN values in input (treat as missing)
            if row.iter().any(|v| v.is_nan()) {
                results.extend_from_slice(&[f64::NAN; 6]);
                continue;
            }

            let pose = ([row[0], row[1], row[2]], [row[3], row[4], row[5]]);

            let solutions = self.inverse(pose, current_joints);
            if let Some(best_solution) = solutions.first() {
                results.extend_from_slice(best_solution);
                current_joints = Some(*best_solution);
            } else {
                // No solution found
                results.extend_from_slice(&[f64::NAN; 6]);
            }
        }

        // Convert flat Vec to 2D array (n, 6)
        let result_array = Array2::from_shape_vec((n, 6), results)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok(result_array.into_pyarray(py).into())
    }

    /// Batch forward kinematics using NumPy arrays.
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Output: poses array of shape (n, 6) with columns [X, Y, Z, A, B, C]
    /// Returns NaN for rows with NaN input values.
    #[pyo3(signature = (joints,))]
    fn batch_forward<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let joints_array = joints.as_array();
        let n = joints_array.nrows();

        let mut results: Vec<f64> = Vec::with_capacity(n * 6);

        for i in 0..n {
            let row = joints_array.row(i);

            // Check for NaN values in input (treat as missing)
            if row.iter().any(|v| v.is_nan()) {
                results.extend_from_slice(&[f64::NAN; 6]);
                continue;
            }

            let joints_input = [row[0], row[1], row[2], row[3], row[4], row[5]];
            let (translation, rotation) = self.forward(joints_input);

            results.push(translation[0]);
            results.push(translation[1]);
            results.push(translation[2]);
            results.push(rotation[0]);
            results.push(rotation[1]);
            results.push(rotation[2]);
        }

        // Convert flat Vec to 2D array (n, 6)
        let result_array = Array2::from_shape_vec((n, 6), results)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok(result_array.into_pyarray(py).into())
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
