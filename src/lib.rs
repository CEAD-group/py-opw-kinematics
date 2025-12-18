mod euler; // Add this line to import the new module
mod kinematic_model; // Add this line to import the new module
mod configuration; // Add configuration module
use crate::euler::EulerConvention;
use crate::kinematic_model::KinematicModel;
use crate::configuration::{RobotConfiguration, ConfigurationSelector, RobotKinematicParams};

use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

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

    #[getter]
    fn get_kinematic_model(&self) -> KinematicModel {
        self._kinematic_model.clone()
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

    /// Compute joint positions using forward kinematics chain
    fn joint_positions(&self, mut joints: [f64; 6]) -> Vec<[f64; 3]> {
        // Apply parallelogram constraint if needed
        if self.has_parallelogram {
            joints[2] += joints[1];
        }
        
        // Convert degrees to radians if needed
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }

        let params = &self._kinematic_model;
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut t: nalgebra::Isometry3<f64> = nalgebra::Isometry3::identity();

        // Apply flip_axes (sign corrections) to joint angles
        let corrected_joints = [
            joints[0] * if params.flip_axes[0] { -1.0 } else { 1.0 },   // J1 - removed hardcoded negation
            joints[1] * if params.flip_axes[1] { -1.0 } else { 1.0 },
            joints[2] * if params.flip_axes[2] { -1.0 } else { 1.0 },
            joints[3] * if params.flip_axes[3] { -1.0 } else { 1.0 },
            joints[4] * if params.flip_axes[4] { -1.0 } else { 1.0 },
            joints[5] * if params.flip_axes[5] { -1.0 } else { 1.0 },
        ];

        // Base position
        positions.push([0.0, 0.0, 0.0]);

        // Joint 1: Translation along Z by c1, then rotation around Z
        t *= nalgebra::Translation3::new(0.0, 0.0, params.c1);
        positions.push(t.translation.vector.into());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), corrected_joints[0]);

        // Joint 2: Translation by a1 along X, rotation around Y  
        t *= nalgebra::Translation3::new(params.a1, 0.0, 0.0);
        positions.push(t.translation.vector.into());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), corrected_joints[1]);

        // Joint 3: Translation by c2 along Z, rotation around Y
        t *= nalgebra::Translation3::new(0.0, 0.0, params.c2);
        positions.push(t.translation.vector.into());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), corrected_joints[2]);

        // Joint 4: Translation by a2 along X and b along Y, rotation around Z (inverted to match standard flip_axes)
        t *= nalgebra::Translation3::new(params.a2, params.b, 0.0);
        positions.push(t.translation.vector.into());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), corrected_joints[3]);

        // Joint 5: Translation by c3 along Z, rotation around Y
        t *= nalgebra::Translation3::new(0.0, 0.0, params.c3);
        positions.push(t.translation.vector.into());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), corrected_joints[4]);

        // Joint 6: Translation by c4 along Z, rotation around Z (inverted to match standard flip_axes)
        t *= nalgebra::Translation3::new(0.0, 0.0, params.c4);
        positions.push(t.translation.vector.into());
        t *= nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), corrected_joints[5]);

        // TCP: Apply end effector transformation
        let combined_rotation = t.rotation.to_rotation_matrix() * self._ee_rotation_matrix;
        let final_translation = t.translation.vector + combined_rotation * self.ee_translation;
        positions.push(final_translation.into());

        positions
    }

    /// Calculate parallelogram P1 and P2 positions using actual robot geometry
    fn parallelogram_positions(&self, joints: [f64; 6], link_length: f64, rest_angle: f64) -> Option<([f64; 3], [f64; 3])> {
        if !self.has_parallelogram {
            return None;
        }

        let positions = self.joint_positions(joints);
        if positions.len() < 4 {
            return None;
        }

        let j1_pos = Vector3::from(positions[1]);  // J1
        let j2_pos = Vector3::from(positions[2]);  // J2
        let j3_pos = Vector3::from(positions[3]);  // J3

        // Parallelogram constraint parameters
        let link_length = link_length; // Distance from P1 to J2 (and P2 to J3)

        // Calculate the constraint angle: 95° + J2_angle + (J3_angle + 90°)
        let j2_angle_deg = if self.euler_convention.degrees {
            joints[1]
        } else {
            joints[1].to_degrees()
        };
        let j3_angle_deg = if self.euler_convention.degrees {
            joints[2]
        } else {
            joints[2].to_degrees()
        };
        
        let constraint_angle_deg = rest_angle + j2_angle_deg + (j3_angle_deg + 90.0);
        let constraint_angle_rad = constraint_angle_deg.to_radians();

        // Vector from J2 to J3
        let j2_to_j3 = j3_pos - j2_pos;
        let j2_to_j3_length = j2_to_j3.norm();

        if j2_to_j3_length < 1e-6 {
            return Some((j2_pos.into(), j3_pos.into()));
        }

        let j2_to_j3_normalized = j2_to_j3 / j2_to_j3_length;

        // Vector from J1 to J2 (gives us reference direction)
        let j1_to_j2 = j2_pos - j1_pos;
        let j1_to_j2_length = j1_to_j2.norm();

        let reference_dir = if j1_to_j2_length < 1e-6 {
            Vector3::new(1.0, 0.0, 0.0) // Default reference
        } else {
            j1_to_j2 / j1_to_j2_length
        };

        // Calculate the normal to the plane containing J1, J2, J3
        let cross_product = j1_to_j2.cross(&j2_to_j3);
        let cross_length = cross_product.norm();

        let plane_normal = if cross_length < 1e-6 {
            Vector3::new(0.0, 0.0, 1.0) // Default to vertical
        } else {
            cross_product / cross_length
        };

        // Create coordinate system in the plane
        let plane_x = j2_to_j3_normalized;
        let plane_y = plane_normal.cross(&plane_x).normalize();

        // Calculate P1 position based on the constraint angle
        let p1_direction_x = link_length * constraint_angle_rad.cos();
        let p1_direction_y = link_length * constraint_angle_rad.sin();

        // Transform to world coordinates
        let p1_offset = p1_direction_x * plane_x + p1_direction_y * plane_y;
        let p1_pos = j2_pos + p1_offset;

        // For a perfect parallelogram: P2 = P1 + (J3 - J2)
        let p2_pos = p1_pos + j2_to_j3;

        Some((p1_pos.into(), p2_pos.into()))
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

        // Optimized constraint checking: skip filtering if no constraints are defined
        if self._kinematic_model.has_constraints {
            solutions.retain(|solution| {
                self._kinematic_model.joints_within_limits(solution, self.euler_convention.degrees)
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
            Series::new("J1".into(), j1).into(),
            Series::new("J2".into(), j2).into(),
            Series::new("J3".into(), j3).into(),
            Series::new("J4".into(), j4).into(),
            Series::new("J5".into(), j5).into(),
            Series::new("J6".into(), j6).into(),
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
                
                // Optimized: only check constraints if they exist
                let joints_valid = if self._kinematic_model.has_constraints {
                    self._kinematic_model.joints_within_limits(&joints_array, self.euler_convention.degrees)
                } else {
                    true // No constraints means all joints are valid
                };
                
                if joints_valid {
                    let (translation, rotation) = self.forward(joints_array);

                    x.push(Some(translation[0]));
                    y.push(Some(translation[1]));
                    z.push(Some(translation[2]));
                    a.push(Some(rotation[0]));
                    b.push(Some(rotation[1]));
                    c.push(Some(rotation[2]));
                } else {
                    // Joints outside limits, push None values
                    x.push(None);
                    y.push(None);
                    z.push(None);
                    a.push(None);
                    b.push(None);
                    c.push(None);
                }
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
            Series::new("X".into(), x).into(),
            Series::new("Y".into(), y).into(),
            Series::new("Z".into(), z).into(),
            Series::new("A".into(), a).into(),
            Series::new("B".into(), b).into(),
            Series::new("C".into(), c).into(),
        ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyDataFrame(df_result))
    }

    /// Analyze joint configurations and return configuration information
    #[pyo3(signature = (pose, target_config=None))]
    fn inverse_with_config(
        &self,
        pose: ([f64; 3], [f64; 3]),
        target_config: Option<String>,
    ) -> PyResult<(Vec<[f64; 6]>, Vec<String>, Option<([f64; 6], String, u8)>)> {
        // Get all inverse kinematics solutions
        let solutions = self.inverse(pose, None);
        
        if solutions.is_empty() {
            return Ok((vec![], vec![], None));
        }
        
        // Analyze configurations for all solutions
        let selector = ConfigurationSelector::new_with_parallelogram(&solutions, self.has_parallelogram);
        let configs = selector.get_all_configurations();
        let config_strings: Vec<String> = configs.iter()
            .map(|config| config.stat_tu_string.clone())
            .collect();
        
        // Find best match if target configuration is specified
        let best_match = if let Some(target_str) = target_config {
            match RobotConfiguration::from_string(&target_str) {
                Ok(target) => {
                    selector.find_best_match(&target)
                        .map(|(index, config, scores)| (solutions[index], config.stat_tu_string.clone(), scores.0))
                }
                Err(_) => None
            }
        } else {
            None
        };
        
        Ok((solutions, config_strings, best_match))
    }
    
    /// Find the best inverse solution matching a specific configuration
    #[pyo3(signature = (pose, target_config, current_joints=None))]
    fn inverse_with_target_config(
        &self,
        pose: ([f64; 3], [f64; 3]),
        target_config: String,
        current_joints: Option<[f64; 6]>,
    ) -> PyResult<Option<([f64; 6], String, u8)>> {
        // Get all inverse kinematics solutions
        let solutions = self.inverse(pose, current_joints);
        
        if solutions.is_empty() {
            return Ok(None);
        }
        
        // Parse target configuration
        let target = match RobotConfiguration::from_string(&target_config) {
            Ok(target) => target,
            Err(e) => return Err(PyErr::new::<PyValueError, _>(format!("Invalid target configuration: {}", e))),
        };
        
        // Find best matching solution
        let selector = ConfigurationSelector::new_with_parallelogram(&solutions, self.has_parallelogram);
        match selector.find_best_match(&target) {
            Some((index, config, scores)) => {
                Ok(Some((solutions[index], config.stat_tu_string.clone(), scores.0)))
            }
            None => Ok(None)
        }
    }
    
    /// Get configuration analysis for given joint values
    fn analyze_configuration(&self, joints: [f64; 6]) -> String {
        let config = RobotConfiguration::from_joints_with_parallelogram(joints, self.has_parallelogram);
        config.stat_tu_string
    }
    
    /// Compare multiple joint solutions and return their configurations
    fn compare_configurations(&self, joint_solutions: Vec<[f64; 6]>) -> Vec<String> {
        joint_solutions.iter()
            .map(|joints| {
                let config = RobotConfiguration::from_joints_with_parallelogram(*joints, self.has_parallelogram);
                config.stat_tu_string
            })
            .collect()
    }
    
    /// Analyze configuration with full STAT/TU information
    #[pyo3(signature = (joints, include_turns=false))]   
    fn analyze_configuration_full(&self, joints: [f64; 6], include_turns: bool) -> PyResult<(String, String, Option<String>)> {
        let config = RobotConfiguration::from_joints_with_parallelogram(joints, self.has_parallelogram);
        
        let stat_tu_string = config.stat_tu_string.clone();
        let stat_string = config.stat.to_binary_string();
        let full_string = if include_turns {
            Some(config.stat_tu_string.clone())
        } else {
            None
        };
        
        Ok((stat_tu_string, stat_string, full_string))
    }    /// Find solutions matching STAT bits (ignoring turn numbers)
    fn find_stat_matches(&self, pose: ([f64; 3], [f64; 3]), stat_bits: u8) -> PyResult<Vec<([f64; 6], String, u8)>> {
        // Get all inverse kinematics solutions
        let solutions = self.inverse(pose, None);
        
        if solutions.is_empty() {
            return Ok(vec![]);
        }
        
        // Find STAT matches
        let selector = ConfigurationSelector::new_with_parallelogram(&solutions, self.has_parallelogram);
        let target_stat = crate::configuration::StatBits::from_bits(stat_bits);
        let matches = selector.find_stat_matches(&target_stat);
        
        let results: Vec<([f64; 6], String, u8)> = matches.iter()
            .map(|(index, config, score)| {
                (solutions[*index], config.stat_tu_string.clone(), *score)
            })
            .collect();
            
        Ok(results)
    }
    
    /// Create target configuration from STAT/TU bits  
    fn create_stat_tu_target(&self, stat_bits: u8, tu_bits: u8) -> String {
        let target = crate::configuration::TargetConfiguration::from_bits(stat_bits, tu_bits);
        target.to_string()
    }
    
    /// Get detailed configuration analysis for a solution
    fn get_configuration_details(&self, joints: [f64; 6]) -> PyResult<(String, u8, String, u8, String)> {
        let config = RobotConfiguration::from_joints_with_parallelogram(joints, self.has_parallelogram);
        
        let stat_tu_config = config.stat_tu_string;
        let stat_bits = config.stat.to_bits();
        let stat_binary = config.stat.to_binary_string();
        let tu_bits = config.tu.to_bits();
        let tu_binary = config.tu.to_binary_string();
        
        Ok((stat_tu_config, stat_bits, stat_binary, tu_bits, tu_binary))
    }
    
    /// Get configuration analysis using geometric calculation
    /// Uses robot-specific kinematic parameters for accurate shoulder classification
    fn get_configuration_details_geometric(&self, joints: [f64; 6], robot_params: &PyRobotKinematicParams) -> PyResult<(String, u8, String, u8, String)> {
        use crate::configuration::StatBits;
        
        // Use geometric calculation
        let stat_bits = StatBits::from_joints_geometric(joints, &robot_params.params, self.has_parallelogram);
        let tu_bits = crate::configuration::TurnBits::from_joints(joints);
        
        let config = RobotConfiguration {
            stat: stat_bits,
            tu: tu_bits,
            joints,
            stat_tu_string: format!("STAT={} TU={}", stat_bits.to_binary_string(), tu_bits.to_binary_string()),
        };
        
        Ok((config.stat_tu_string, stat_bits.to_bits(), stat_bits.to_binary_string(), tu_bits.to_bits(), tu_bits.to_binary_string()))
    }
}

/// Python wrapper for robot kinematic parameters
/// Used for geometric overhead calculation in STAT bits
#[pyclass(name = "RobotKinematicParams")]
#[derive(Debug, Clone)]
pub struct PyRobotKinematicParams {
    pub params: RobotKinematicParams,
}

#[pymethods]
impl PyRobotKinematicParams {
    #[new]
    #[pyo3(signature = (a1, a2, b, c1, c2, c3, c4))]
    pub fn new(a1: f64, a2: f64, b: f64, c1: f64, c2: f64, c3: f64, c4: f64) -> Self {
        Self {
            params: RobotKinematicParams { a1, a2, b, c1, c2, c3, c4 }
        }
    }
    
    /// Create from KinematicModel
    #[classmethod]
    pub fn from_kinematic_model(_cls: &Bound<'_, PyType>, kinematic_model: &KinematicModel) -> Self {
        Self {
            params: RobotKinematicParams {
                a1: kinematic_model.a1,
                a2: kinematic_model.a2, 
                b: kinematic_model.b,
                c1: kinematic_model.c1,
                c2: kinematic_model.c2,
                c3: kinematic_model.c3,
                c4: kinematic_model.c4,
            }
        }
    }
    
    fn __repr__(&self) -> String {
        format!("RobotKinematicParams(a1={}, a2={}, b={}, c1={}, c2={}, c3={}, c4={})",
                self.params.a1, self.params.a2, self.params.b, 
                self.params.c1, self.params.c2, self.params.c3, self.params.c4)
    }
}



/// Module initialization for Python
#[pymodule(name = "_internal")]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EulerConvention>()?;
    m.add_class::<KinematicModel>()?;
    m.add_class::<Robot>()?;
    m.add_class::<PyRobotKinematicParams>()?;
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
