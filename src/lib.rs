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

    /// Forward kinematics with 4x4 matrix output
    /// joints: Joint angles in radians
    /// Returns: 4x4 transformation matrix in row-major format
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (joints, ee_transform=None))]
    fn forward(
        &self, 
        mut joints: [f64; 6],
        ee_transform: Option<[[f64; 4]; 4]>
    ) -> [[f64; 4]; 4] {
        if self.has_parallelogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        
        // Apply end effector transformation if provided (matching original logic)
        let (final_rotation, final_translation) = if let Some(ee_matrix) = ee_transform {
            let flattened: Vec<f64> = ee_matrix.into_iter().flatten().collect();
            let ee_transform = nalgebra::Matrix4::from_row_slice(&flattened);
            let ee_rotation = Rotation3::from_matrix_unchecked(ee_transform.fixed_view::<3, 3>(0, 0).into());
            let ee_translation: Vector3<f64> = ee_transform.fixed_view::<3, 1>(0, 3).into();
            
            // Correct transformation: T_final = T_robot * T_ee  
            // But using original working logic: combined_rotation first, then transform translation
            let robot_rotation_matrix = pose.rotation.to_rotation_matrix();
            let robot_rotation = robot_rotation_matrix.matrix();
            let combined_rotation = robot_rotation * ee_rotation;
            let final_translation = pose.translation.vector + combined_rotation * ee_translation;
            
            (combined_rotation, final_translation)
        } else {
            (*pose.rotation.to_rotation_matrix().matrix(), pose.translation.vector)
        };

        // Convert to 4x4 matrix (row-major)
        let matrix = nalgebra::Matrix4::new(
            final_rotation[(0, 0)], final_rotation[(0, 1)], final_rotation[(0, 2)], final_translation[0],
            final_rotation[(1, 0)], final_rotation[(1, 1)], final_rotation[(1, 2)], final_translation[1],
            final_rotation[(2, 0)], final_rotation[(2, 1)], final_rotation[(2, 2)], final_translation[2],
            0.0, 0.0, 0.0, 1.0,
        );
        [
            [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)], matrix[(0, 3)]],
            [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)], matrix[(1, 3)]],
            [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)], matrix[(2, 3)]],
            [matrix[(3, 0)], matrix[(3, 1)], matrix[(3, 2)], matrix[(3, 3)]],
        ]
    }

    /// Inverse kinematics with 4x4 matrix input
    /// pose: 4x4 transformation matrix in row-major format
    /// Returns: Vector of possible joint angle solutions (radians)
    /// ee_transform: 4x4 transformation matrix in row-major format (optional, identity if None)
    #[pyo3(signature = (pose, current_joints=None, ee_transform=None))]
    fn inverse(
        &self,
        pose: [[f64; 4]; 4],
        current_joints: Option<[f64; 6]>,
        ee_transform: Option<[[f64; 4]; 4]>
    ) -> Vec<[f64; 6]> {
        // Convert input 4x4 matrix to nalgebra components
        let flattened: Vec<f64> = pose.into_iter().flatten().collect();
        let pose_matrix = nalgebra::Matrix4::from_row_slice(&flattened);
        
        let target_rotation = Rotation3::from_matrix_unchecked(pose_matrix.fixed_view::<3, 3>(0, 0).into());
        let target_translation: Vector3<f64> = pose_matrix.fixed_view::<3, 1>(0, 3).into();

        // Handle end effector transformation (matching original logic)
        let (final_rotation, final_translation) = if let Some(ee_matrix) = ee_transform {
            let flattened: Vec<f64> = ee_matrix.into_iter().flatten().collect();
            let matrix = nalgebra::Matrix4::from_row_slice(&flattened);
            let ee_rotation = Rotation3::from_matrix_unchecked(matrix.fixed_view::<3, 3>(0, 0).into());
            let ee_translation: Vector3<f64> = matrix.fixed_view::<3, 1>(0, 3).into();
            
            // Inverse transformation matching old logic:
            // rotated_ee_translation = target_rotation * ee_translation
            // robot_translation = target_translation - rotated_ee_translation  
            // robot_rotation = target_rotation * ee_rotation^(-1)
            let ee_rotation_inv = ee_rotation.transpose(); // Inverse of rotation matrix is transpose
            let target_rotation_matrix = target_rotation.matrix();
            let rotated_ee_translation = target_rotation_matrix * ee_translation;
            let final_translation = Translation3::from(target_translation - rotated_ee_translation);
            let final_rotation = target_rotation_matrix * ee_rotation_inv;
            
            (final_rotation, final_translation)
        } else {
            (*target_rotation.matrix(), Translation3::from(target_translation))
        };
        
        let iso_pose = Isometry3::from_parts(final_translation, nalgebra::UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(final_rotation)));
        
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

    /// Batch inverse kinematics with 4x4 matrix input
    /// Input: poses array of shape (n, 16) with rows containing 4x4 matrices in row-major format
    /// Output: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6] (radians)
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

            // Convert row to 4x4 matrix
            let pose_matrix = [
                [row[0], row[1], row[2], row[3]],
                [row[4], row[5], row[6], row[7]],
                [row[8], row[9], row[10], row[11]],
                [row[12], row[13], row[14], row[15]]
            ];

            let solutions = self.inverse(pose_matrix, current_joints, ee_transform);
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

    /// Batch forward kinematics with 4x4 matrix output
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6] (radians)
    /// Output: poses array of shape (n, 16) with rows containing 4x4 matrices in row-major format
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

        let mut results: Vec<f64> = Vec::with_capacity(n * 16);

        for i in 0..n {
            let row = joints_array.row(i);

            // Check for NaN values in input (treat as missing)
            if row.iter().any(|v| v.is_nan()) {
                results.extend_from_slice(&[f64::NAN; 16]);
                continue;
            }

            let joints_input = [row[0], row[1], row[2], row[3], row[4], row[5]];
            let transform_matrix = self.forward(joints_input, ee_transform);

            // Flatten 4x4 matrix to row-major format
            for i in 0..4 {
                for j in 0..4 {
                    results.push(transform_matrix[i][j]);
                }
            }
        }

        // Convert flat Vec to 2D array (n, 16)
        let result_array = Array2::from_shape_vec((n, 16), results)
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
            let flattened: Vec<f64> = ee_matrix.into_iter().flatten().collect();
            let matrix = nalgebra::Matrix4::from_row_slice(&flattened);
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
