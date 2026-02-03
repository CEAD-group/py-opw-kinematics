mod kinematic_model;
use crate::kinematic_model::KinematicModel;

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use pyo3::prelude::*;

use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose, CONSTRAINT_CENTERED};
use rs_opw_kinematics::tool::{Base, Tool};
use std::sync::Arc;

#[pyclass]
struct Robot {
    base_config: BaseConfig,
    tool_config: ToolConfig,
    _tool: Tool,
    _kinematic_model: KinematicModel,
}

#[pyclass]
#[pyo3(from_py_object)]
#[derive(Clone, Debug)]
struct BaseConfig {
    /// The translation of the base in the world frame
    translation: [f64; 3],
    /// The rotation of the base in quaternion (w, x, y, z)
    rotation: [f64; 4],
}

#[pymethods]
impl BaseConfig {
    #[new]
    fn new(translation: [f64; 3], rotation: [f64; 4]) -> Self {
        BaseConfig {
            translation,
            rotation,
        }
    }
}

#[pyclass]
#[pyo3(from_py_object)]
#[derive(Clone, Debug)]
struct ToolConfig {
    /// The translation of the tool in the base frame
    translation: [f64; 3],
    /// The rotation of the tool in quaternion (w, x, y, z)
    rotation: [f64; 4],
}

#[pymethods]
impl ToolConfig {
    #[new]
    fn new(translation: [f64; 3], rotation: [f64; 4]) -> Self {
        ToolConfig {
            translation,
            rotation,
        }
    }
}

#[pymethods]
impl Robot {
    #[new]
    #[pyo3(signature = (kinematic_model, base_config, tool_config))]
    fn new(
        kinematic_model: KinematicModel,
        base_config: BaseConfig,
        tool_config: ToolConfig,
    ) -> PyResult<Self> {
        let robot = kinematic_model.to_opw_kinematics();

        let base = Isometry3::from_parts(
            Translation3::from(base_config.translation),
            UnitQuaternion::from_quaternion(Quaternion::new(
                base_config.rotation[0],
                base_config.rotation[1],
                base_config.rotation[2],
                base_config.rotation[3],
            )),
        );

        let tool = Isometry3::from_parts(
            Translation3::from(tool_config.translation),
            UnitQuaternion::from_quaternion(Quaternion::new(
                tool_config.rotation[0],
                tool_config.rotation[1],
                tool_config.rotation[2],
                tool_config.rotation[3],
            )),
        );

        let robot_with_base = Base {
            robot: Arc::new(robot),
            base,
        };

        let robot_on_base_with_tool = Tool {
            robot: Arc::new(robot_with_base),
            tool,
        };

        // Create an instance with initial values
        let robot_instance = Robot {
            base_config,
            tool_config,
            _tool: robot_on_base_with_tool,
            _kinematic_model: kinematic_model,
        };

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
            "Robot(\n    kinematic_model=\n{},\n    base_config={:?},\n    tool_config={:?}\n)",
            km_repr, self.base_config, self.tool_config
        )
    }

    /// Forward kinematics: calculates the pose for given joints in degrees
    fn forward(&self, joints: [f64; 6]) -> ([f64; 3], [f64; 4]) {
        let joints = joints.map(|x| x.to_radians());
        let pose: Pose = self._tool.forward(&joints);
        // Storage order is (x, y, z, w)
        let quat = [
            pose.rotation.coords[3],
            pose.rotation.coords[0],
            pose.rotation.coords[1],
            pose.rotation.coords[2],
        ];
        (pose.translation.vector.into(), quat)
    }

    fn convert_to_degrees(&self, joints: [f64; 6]) -> [f64; 6] {
        joints
            .iter()
            .map(|x| x.to_degrees())
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap()
    }

    /// Inverse kinematics: calculates the joint angles for a given pose.
    ///
    /// # Arguments
    /// * `pose` - The target pose as a tuple: ([x, y, z], [w, x, y, z]), where the translation is in meters and the rotation is a quaternion.
    /// * `current_joints` - (Optional) The current joint angles as an array of 6 elements (in degrees). Used as a seed for solution selection. If not provided, a default centered configuration is used.
    ///
    /// # Returns
    /// * `Vec<[f64; 6]>` - A vector of all possible joint solutions (in degrees).
    ///
    /// # Notes
    /// All solutions are returned without filtering or sorting.
    #[pyo3(signature = (pose, current_joints=None))]
    fn inverse(
        &self,
        pose: ([f64; 3], [f64; 4]),
        current_joints: Option<[f64; 6]>,
    ) -> Vec<[f64; 6]> {
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            pose.1[0], pose.1[1], pose.1[2], pose.1[3],
        ));
        let iso_pose = Isometry3::from_parts(Translation3::from(pose.0), quat);

        let joints = if let Some(joints) = current_joints {
            joints.map(|x| x.to_radians())
        } else {
            CONSTRAINT_CENTERED
        };
        let solutions = self._tool.inverse_continuing(&iso_pose, &joints);

        // Convert all solutions to degrees without filtering
        solutions
            .iter()
            .map(|x| self.convert_to_degrees(*x))
            .collect::<Vec<_>>()
    }

    #[pyo3(signature = (poses))]
    fn batch_inverse(&self, poses: Vec<([f64; 3], [f64; 4])>) -> Vec<Vec<[f64; 6]>> {
        poses
            .iter()
            .map(|&pose| self.inverse(pose, None))
            .collect()
    }

    #[pyo3(signature = (joints))]
    fn batch_forward(&self, joints: Vec<[f64; 6]>) -> Vec<([f64; 3], [f64; 4])> {
        joints
            .iter()
            .map(|&joint_set| self.forward(joint_set))
            .collect()
    }
}

/// Module initialization for Python
#[pymodule(name = "_internal")]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KinematicModel>()?;
    m.add_class::<Robot>()?;
    m.add_class::<BaseConfig>()?;
    m.add_class::<ToolConfig>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABB_1660: KinematicModel = KinematicModel {
        a1: 0.150,  // Distance from base to J1 axis
        a2: -0.110, // Distance from J1 to J2 axis (parallel offset)
        b: 0.0,     // Distance from J2 to J3 axis (perpendicular offset)
        c1: 0.4865, // Distance from base to J2 axis (height)
        c2: 0.700,  // Distance from J2 to J3 axis (upper arm length)
        c3: 0.678,  // Distance from J3 to J4 axis (forearm length)
        c4: 0.135,  // Distance from J4 to J6 axis (wrist length)
        offsets: [0.0, 0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0],
        sign_corrections: [1, 1, 1, 1, 1, 1],
    };

    #[test]
    fn test_simple_forward() {
        let kinematic_model = ABB_1660;
        let base_config = BaseConfig {
            translation: [0.0, 0.0, 2.3],
            rotation: [0.0, 1.0, 0.0, 0.0],
        };
        let tool_config = ToolConfig {
            translation: [0.0, 0.0, 0.095],
            rotation: [
                -0.00012991440873552217,
                -0.968154906938256,
                -0.0004965996111545046,
                0.2503407964804168,
            ],
        };
        let robot = Robot::new(kinematic_model, base_config, tool_config).unwrap();
        let joints = [-103.1, -85.03, 19.06, -70.19, -35.87, 185.01];
        let (translation, rotation) = robot.forward(joints);
        assert_eq!(
            translation,
            [0.2000017014027134, -0.30003856402112994, 0.8999972858765594]
        );
        assert_eq!(
            rotation,
            [
                0.8518484534487618,
                0.13765321623120808,
                -0.46476827163476586,
                -0.19848490647852607
            ]
        );
    }

    #[test]
    fn test_simple_inverse() {
        let kinematic_model = ABB_1660;
        let base_config = BaseConfig {
            translation: [0.0, 0.0, 2.3],
            rotation: [0.0, 1.0, 0.0, 0.0],
        };
        let tool_config = ToolConfig {
            translation: [0.0, 0.0, 0.095],
            rotation: [
                -0.00012991440873552217,
                -0.968154906938256,
                -0.0004965996111545046,
                0.2503407964804168,
            ],
        };
        let robot = Robot::new(kinematic_model, base_config, tool_config).unwrap();
        let pose = (
            [0.2000017014027134, -0.30003856402112994, 0.8999972858765594],
            [
                0.8518484534487618,
                0.13765321623120808,
                -0.46476827163476586,
                -0.19848490647852607,
            ],
        );
        let solutions = robot.inverse(pose, None);
        // Check that we get solutions (exact number may vary)
        assert!(!solutions.is_empty());
        // Check that the first solution is reasonable
        assert_eq!(solutions.len(), 8);
    }

    #[test]
    fn test_batch_inverse() {
        let kinematic_model = ABB_1660;
        let base_config = BaseConfig {
            translation: [0.0, 0.0, 2.3],
            rotation: [0.0, 1.0, 0.0, 0.0],
        };
        let tool_config = ToolConfig {
            translation: [0.0, 0.0, 0.095],
            rotation: [
                -0.00012991440873552217,
                -0.968154906938256,
                -0.0004965996111545046,
                0.2503407964804168,
            ],
        };
        let robot = Robot::new(kinematic_model, base_config, tool_config).unwrap();
        
        // Test with multiple poses
        let poses = vec![
            (
                [0.2000017014027134, -0.30003856402112994, 0.8999972858765594],
                [
                    0.8518484534487618,
                    0.13765321623120808,
                    -0.46476827163476586,
                    -0.19848490647852607,
                ],
            ),
            (
                [0.5, 0.0, 1.2],
                [1.0, 0.0, 0.0, 0.0],
            ),
        ];
        
        let batch_solutions = robot.batch_inverse(poses.clone());
        
        // Check that we get solutions for each pose
        assert_eq!(batch_solutions.len(), 2);
        
        // Check that each pose has multiple solutions
        for (i, solutions) in batch_solutions.iter().enumerate() {
            assert!(!solutions.is_empty(), "Pose {} should have solutions", i);
            
            // Verify that solutions are valid by doing forward kinematics
            for solution in solutions {
                let (computed_translation, computed_rotation) = robot.forward(*solution);
                let original_pose = &poses[i];
                
                // Check translation (with some tolerance for numerical precision)
                for j in 0..3 {
                    assert!(
                        (computed_translation[j] - original_pose.0[j]).abs() < 1e-10,
                        "Translation mismatch at pose {}, axis {}: expected {}, got {}",
                        i, j, original_pose.0[j], computed_translation[j]
                    );
                }
            }
        }
        
        // Test that batch_inverse gives same results as individual inverse calls
        for (i, pose) in poses.iter().enumerate() {
            let individual_solutions = robot.inverse(*pose, None);
            let batch_solution = &batch_solutions[i];
            assert_eq!(
                individual_solutions.len(),
                batch_solution.len(),
                "Solution count mismatch for pose {}", i
            );
            
            // Check that all solutions match (order should be the same)
            for (j, (individual, batch)) in individual_solutions.iter().zip(batch_solution.iter()).enumerate() {
                for k in 0..6 {
                    assert!(
                        (individual[k] - batch[k]).abs() < 1e-10,
                        "Joint angle mismatch at pose {}, solution {}, joint {}: individual={}, batch={}",
                        i, j, k, individual[k], batch[k]
                    );
                }
            }
        }
    }
}
