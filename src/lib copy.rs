use std::f32::consts::E;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose, JOINTS_AT_ZERO};
use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;
use nalgebra::Isometry3;
use nalgebra::Vector3;
use nalgebra::Unit;
/// Struct to hold the robot kinematics in Python
#[pyclass]
#[derive(Clone)]
struct EulerConvention {
    sequence: String,
    extrinsic: bool,
    _seq: [Unit<Vector3<T>>; 3]
    
    }

#[pymethods]
impl EulerConvention {
    #[new]
    fn new(sequence: String, extrinsic: bool) -> Self {
        // convert the sequence from "XYZ" or "ZYX" into the corresponding array of vectors
        // chek that the input is exactly thre chars long and 
    // Convert the string into corresponding axis vectors
    let _seq: [nalgebra::Unit<nalgebra::Vector3<f64>>; 3] = sequence.chars().map(|c| {
        match c {
            'X' => nalgebra::Unit::new_normalize(nalgebra::Vector3::new(1.0, 0.0, 0.0)),
            'Y' => nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.0, 1.0, 0.0)),
            'Z' => nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.0, 0.0, 1.0)),
            _ => panic!("The sequence must be a 3-character string like 'XYZ' or 'ZXZ'")
        }
    }).collect::<Vec<_>>().try_into().unwrap();
        EulerConvention {
            sequence,
            extrinsic,
            _seq,
        }
    }

    fn convert(&self, other: &EulerConvention, angles: [f64; 3]) -> [f64; 3] {
        other.from_rotation_matrix(self.to_rotation_matrix(angles))
    }

    fn from_rotation_matrix(&self, rot: nalgebra::Rotation3<f64>) -> [f64; 3] {
        rot.euler_angles_ordered(self._seq, self.extrinsic)[0]
    }

    fn to_rotation_matrix(&self, angles: [f64; 3]) -> nalgebra::Rotation3<f64> {
            let (a1, a2, a3) = angles;
            let seq = self.sequence.clone();

            // Reverse the sequence for extrinsic rotation
            if self.extrinsic {
                seq.reverse();
            }
        
            // Create rotation matrices for each of the angles around the given axes.
            let r1 = Rotation3::from_axis_angle(&seq[0], a1);
            let r2 = Rotation3::from_axis_angle(&seq[1], a2);
            let r3 = Rotation3::from_axis_angle(&seq[2], a3);
        
            // Combine the rotations according to the specified sequence and type of rotation.
            if self.extrinsic {
                // Extrinsic rotation: Rotate in reverse order (global frame of reference)
                r1 * r2 * r3
            } else {
                // Intrinsic rotation: Rotate in the given order (local frame of reference)
                r3 * r2 * r1
            }
        }
}

#[pyclass]
struct PyOPWKinematics {
    robot: OPWKinematics,
    has_parallellogram: bool,
    unit_degrees: bool,
    euler_convention: EulerConvention
}

#[pymethods]
impl PyOPWKinematics {
    #[new]
    fn new(a1: f64, a2: f64, b: f64, c1: f64, c2: f64, c3: f64, c4: f64, offsets: [f64; 6], sign_corrections: [i32; 6], has_parallellogram: bool) -> Self {
        let mut params = Parameters::new();
        params.a1 = a1;
        params.a2 = a2;
        params.b = b;
        params.c1 = c1;
        params.c2 = c2;
        params.c3 = c3;
        params.c4 = c4;
        params.offsets = offsets;
        params.sign_corrections = sign_corrections.map(|x| x as i8);

        // check that all signs are either -1 or 1
        for sign in params.sign_corrections.iter() {
            if *sign != 1 && *sign != -1 {
                panic!("Sign correction must be either 1 or -1");
            }
        }

        let robot = OPWKinematics::new(params);
        PyOPWKinematics { robot , has_parallellogram }
    }

    /// Forward kinematics: calculates the pose for given joints
    fn forward(&self, mut joints: [f64; 6]) -> ([f64; 3], [f64; 3]) {
        if self.has_parallellogram {
            joints[2] += joints[1];
        }
        if self.unit_degrees {
            joints = joints.map(|x| x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let translation = pose.translation.vector.into();
        let rotation:[f64; 3] = pose.rotation.euler_angles().into();
        if self.unit_degrees {
            (translation, rotation.map(|x| x.to_degrees()))
        } else {
            (translation, rotation)
        }
    }


    /// Inverse kinematics: calculates the joint solutions for a given pose
    fn inverse(&self, pose: ([f64; 3], [f64; 3])) -> Vec<[f64; 6]> {
        let translation = nalgebra::Translation3::from(pose.0);
        let rotation = nalgebra::Rotation3::from_euler_angles(pose.1[0], pose.1[1], pose.1[2]);
        let iso_pose = Isometry3::from_parts(translation, rotation.into());
        let solutions = self.robot.inverse(&iso_pose);
        if self.has_parallellogram {
            solutions.into_iter().clone().map(|mut x| {
                x[2] -= x[1];
                x
            }).collect()
        } else {
            solutions
        }
    }

    /// Inverse kinematics with continuation from close joints
    fn inverse_continuing(&self, pose: ([f64; 3], [f64; 3]), current_joints: [f64; 6]) -> Vec<[f64; 6]> {
        let translation = nalgebra::Translation3::from(pose.0);
        let rotation = nalgebra::Rotation3::from_euler_angles(pose.1[0], pose.1[1], pose.1[2]);
        let iso_pose = Isometry3::from_parts(translation, rotation.into());
        let solutions = self.robot.inverse_continuing(&iso_pose, &current_joints);
        solutions
    }
}

#[pyfunction]
fn joints_at_zero() -> [f64; 6] {
    JOINTS_AT_ZERO
}



// pub fn euler_angles_ordered(
//     &self,
//     seq: [Unit<Vector3<T>>; 3],
//     extrinsic: bool,
// ) -> ([T; 3], bool)
// where
//     T: RealField + Copy,
//algebra::Rotation3::euler_angles_ordered

#[pyfunction]
fn euler_angles_ordered(euler: [f64; 3], seq: &str, extrinsic: bool) -> ([f64; 3], bool) {
    // Ensure that `seq` is a 3-character string like "XYZ" or "ZXZ"
    if seq.len() != 3 {
        panic!("The sequence must be a 3-character string like 'XYZ' or 'ZXZ'");
    }

    // Convert the string into corresponding axis vectors
    let seq: [nalgebra::Unit<nalgebra::Vector3<f64>>; 3] = seq.chars().map(|c| {
        match c {
            'X' => nalgebra::Unit::new_normalize(nalgebra::Vector3::new(1.0, 0.0, 0.0)),
            'Y' => nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.0, 1.0, 0.0)),
            'Z' => nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.0, 0.0, 1.0)),
            _ => Err(()).unwrap() // Return an error if the character is invalid
        }
    }).collect::<Vec<_>>().try_into().unwrap();


    // Create the rotation using the provided Euler angles
    let rot = nalgebra::Rotation3::from_euler_angles(euler[0], euler[1], euler[2]);

    // Apply the Euler angles in the specified order
    let new_euler = rot.euler_angles_ordered(seq, extrinsic);

    // Return the ordered Euler angles and true for success
    new_euler
}


/// Module initialization for Python
#[pymodule]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOPWKinematics>()?;
    m.add_function(wrap_pyfunction!(joints_at_zero, m)?)?;
    m.add_function(wrap_pyfunction!(euler_angles_ordered, m)?)?;
    Ok(())
}
