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
    _internal_euler_convention: EulerConvention,
    _kinematic_model: KinematicModel,
}

#[pymethods]
impl Robot {
    #[new]
    fn new(
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
    ) -> PyResult<Self> {
        let robot = kinematic_model.to_opw_kinematics(euler_convention.degrees);
        let has_parallelogram = kinematic_model.has_parallelogram;
        let degrees = euler_convention.degrees;

        let _internal_euler_convention = EulerConvention::new("XYZ".to_string(), false, degrees)?;

        Ok(Robot {
            robot,
            has_parallelogram,
            euler_convention,
            _internal_euler_convention,
            _kinematic_model: kinematic_model,
        })
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
            "Robot(\n    kinematic_model=\n{},\n    euler_convention={}\n)",
            km_repr,
            self.euler_convention.__repr__()
        )
    }

    /// Forward kinematics with RigidTransform input (4x4 matrix)
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (joints, ee_transform=None))]
    fn forward(
        &self, 
        mut joints: [f64; 6],
        ee_transform: Option<[[f64; 4]; 4]>
    ) -> ([f64; 3], [f64; 3]) {
        if self.has_parallelogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        
        if let Some(ee_matrix) = ee_transform {
            // Convert 4x4 matrix to nalgebra components
            let matrix = nalgebra::Matrix4::from_row_slice(&[
                ee_matrix[0][0], ee_matrix[0][1], ee_matrix[0][2], ee_matrix[0][3],
                ee_matrix[1][0], ee_matrix[1][1], ee_matrix[1][2], ee_matrix[1][3], 
                ee_matrix[2][0], ee_matrix[2][1], ee_matrix[2][2], ee_matrix[2][3],
                ee_matrix[3][0], ee_matrix[3][1], ee_matrix[3][2], ee_matrix[3][3]
            ]);
            
            let ee_rotation_matrix = Rotation3::from_matrix_unchecked(matrix.fixed_view::<3, 3>(0, 0).into());
            let ee_translation: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
            
            let combined_rotation = pose.rotation.to_rotation_matrix() * ee_rotation_matrix;
            let translation: Vector3<f64> = pose.translation.vector + combined_rotation * &ee_translation;
            let rotation = self.euler_convention._matrix_to_euler(combined_rotation);
            
            (translation.into(), rotation)
        } else {
            // No end effector transformation (identity)
            let rotation = self.euler_convention._matrix_to_euler(pose.rotation.to_rotation_matrix());
            (pose.translation.vector.into(), rotation)
        }
    }

    /// Inverse kinematics with RigidTransform input
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (pose, current_joints=None, ee_transform=None))]
    fn inverse(
        &self,
        pose: ([f64; 3], [f64; 3]),
        current_joints: Option<[f64; 6]>,
        ee_transform: Option<[[f64; 4]; 4]>
    ) -> Vec<[f64; 6]> {
        let (ee_rotation_matrix, ee_translation_vec) = if let Some(ee_matrix) = ee_transform {
            let matrix = nalgebra::Matrix4::from_row_slice(&[
                ee_matrix[0][0], ee_matrix[0][1], ee_matrix[0][2], ee_matrix[0][3],
                ee_matrix[1][0], ee_matrix[1][1], ee_matrix[1][2], ee_matrix[1][3], 
                ee_matrix[2][0], ee_matrix[2][1], ee_matrix[2][2], ee_matrix[2][3],
                ee_matrix[3][0], ee_matrix[3][1], ee_matrix[3][2], ee_matrix[3][3]
            ]);
            let rot = Rotation3::from_matrix_unchecked(matrix.fixed_view::<3, 3>(0, 0).into());
            let trans: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
            (rot, trans)
        } else {
            (Rotation3::identity(), Vector3::zeros())
        };
        
        let rotation_matrix = self.euler_convention._euler_to_matrix(pose.1);
        let rotated_ee_translation = rotation_matrix * &ee_translation_vec;
        let translation = Translation3::from(Vector3::from(pose.0) - rotated_ee_translation);
        let rotation = rotation_matrix * ee_rotation_matrix.inverse();
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

    /// Batch inverse kinematics with RigidTransform input
    /// Input: poses array of shape (n, 6) with columns [X, Y, Z, A, B, C]
    /// Output: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Returns NaN for rows where no solution is found.
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (poses, current_joints=None, ee_transform=None))]
    fn batch_inverse<'py>(
        &self,
        py: Python<'py>,
        poses: PyReadonlyArray2<'py, f64>,
        mut current_joints: Option<[f64; 6]>,
        ee_transform: Option<[[f64; 4]; 4]>,
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

            let solutions = self.inverse(pose, current_joints, ee_transform);
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

    /// Batch forward kinematics with RigidTransform input
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Output: poses array of shape (n, 6) with columns [X, Y, Z, A, B, C]
    /// Returns NaN for rows with NaN input values.
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (joints, ee_transform=None))]
    fn batch_forward<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
        ee_transform: Option<[[f64; 4]; 4]>,
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
            let (translation, rotation) = self.forward(joints_input, ee_transform);

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

    /// Compute 4x4 transform matrices for all robot links
    /// Returns: Vec<[[f64; 4]; 4]> where each element is a 4x4 matrix (row-major)
    /// Order: [l0, l1, l2, l3, l4, l5, l6, tcp]
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (joints, ee_transform=None))]
    fn forward_frames(
        &self, 
        joints: [f64; 6],
        ee_transform: Option<[[f64; 4]; 4]>
    ) -> Vec<[[f64; 4]; 4]> {
        use std::f64::consts::PI;

        let (ee_rotation_matrix, ee_translation_vec) = if let Some(ee_matrix) = ee_transform {
            let matrix = nalgebra::Matrix4::from_row_slice(&[
                ee_matrix[0][0], ee_matrix[0][1], ee_matrix[0][2], ee_matrix[0][3],
                ee_matrix[1][0], ee_matrix[1][1], ee_matrix[1][2], ee_matrix[1][3], 
                ee_matrix[2][0], ee_matrix[2][1], ee_matrix[2][2], ee_matrix[2][3],
                ee_matrix[3][0], ee_matrix[3][1], ee_matrix[3][2], ee_matrix[3][3]
            ]);
            let rot = Rotation3::from_matrix_unchecked(matrix.fixed_view::<3, 3>(0, 0).into());
            let trans: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
            (rot, trans)
        } else {
            (Rotation3::identity(), Vector3::zeros())
        };

        // Create a mutable copy for processing
        let mut j = joints;
        let params = &self._kinematic_model;

        // Convert degrees to radians if needed
        if self.euler_convention.degrees {
            j.iter_mut().for_each(|x| *x = x.to_radians());
        }

        // Apply flip_axes (sign corrections) to joint angles
        for i in 0..6 {
            if params.flip_axes[i] {
                j[i] = -j[i];
            }
        }

        let mut out: Vec<[[f64; 4]; 4]> = Vec::with_capacity(8);
        let mut t: nalgebra::Isometry3<f64> = nalgebra::Isometry3::identity();

        // Helper function to convert Isometry3 to 4x4 matrix (row-major)
        let to_matrix_4x4 = |iso: &nalgebra::Isometry3<f64>| -> [[f64; 4]; 4] {
            let matrix = iso.to_matrix();
            [
                [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)], matrix[(0, 3)]],
                [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)], matrix[(1, 3)]],
                [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)], matrix[(2, 3)]],
                [0.0,             0.0,             0.0,             1.0],
            ]
        };

        // l0: Base at origin (identity transform)
        out.push(to_matrix_4x4(&t));

        // l1: rotation around Z axis (negated to match original)
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), -j[0]);
        out.push(to_matrix_4x4(&t));

        // l2: translate by (a1, 0, c1), then rotate around Y
        t *= nalgebra::Translation3::new(params.a1, 0.0, params.c1);
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), j[1]);
        out.push(to_matrix_4x4(&t));

        // l3: translate by (0, 0, c2), apply fixed rotation (-pi/2 around Y), then rotate around Y
        t *= nalgebra::Translation3::new(0.0, 0.0, params.c2);
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), -PI / 2.0);
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), j[2]);
        out.push(to_matrix_4x4(&t));

        // l4: translate by (0, 0, |a2|), then rotate around X (negated)
        t *= nalgebra::Translation3::new(0.0, 0.0, params.a2.abs());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), -j[3]);
        out.push(to_matrix_4x4(&t));

        // l5: translate by (c3, 0, 0), then rotate around Y
        t *= nalgebra::Translation3::new(params.c3, 0.0, 0.0);
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), j[4]);
        out.push(to_matrix_4x4(&t));

        // l6: translate by (c4, 0, 0), then rotate around X (negated)
        t *= nalgebra::Translation3::new(params.c4, 0.0, 0.0);
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), -j[5]);
        out.push(to_matrix_4x4(&t));

        // TCP: Apply end effector transformation
        let combined_rotation = t.rotation.to_rotation_matrix() * ee_rotation_matrix;
        let final_translation = t.translation.vector + combined_rotation * &ee_translation_vec;
        let tcp_transform = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(final_translation),
            nalgebra::UnitQuaternion::from_rotation_matrix(&combined_rotation)
        );
        out.push(to_matrix_4x4(&tcp_transform));

        out
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
