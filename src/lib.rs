use nalgebra::{Isometry3, Matrix3, Rotation, Rotation3, Unit, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose};
use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;

#[pyclass]
#[derive(Clone)]
struct EulerConvention {
    #[allow(dead_code)]
    sequence: String,
    extrinsic: bool,
    degrees: bool,
    _seq: [Unit<Vector3<f64>>; 3],
}

impl EulerConvention {
    fn _from_rotation_matrix(&self, rot: Rotation<f64, 3>) -> [f64; 3] {
        let (angles, _observable) = rot.euler_angles_ordered(self._seq, self.extrinsic);
        angles
    }

    fn _to_rotation_matrix(&self, angles: [f64; 3]) -> Rotation3<f64> {
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

    fn convert(&self, other: &EulerConvention, mut angles: [f64; 3]) -> PyResult<[f64; 3]> {
        // Convert from degrees to radians if necessary
        if self.degrees {
            angles = angles.map(|angle| angle.to_radians());
        }

        // Perform the conversion using the internal logic (in radians)
        let rot_matrix = self._to_rotation_matrix(angles);
        let mut result = other._from_rotation_matrix(rot_matrix);

        // Convert back to degrees if necessary
        if self.degrees {
            result = result.map(|angle| angle.to_degrees());
        }

        Ok(result)
    }

    fn from_rotation_matrix(&self, rot: [[f64; 3]; 3]) -> [f64; 3] {
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        let mut angles = self._from_rotation_matrix(rotation);

        // Convert back to degrees if necessary
        if self.degrees {
            angles = angles.map(|angle| angle.to_degrees());
        }

        angles
    }

    fn to_rotation_matrix(&self, mut angles: [f64; 3]) -> [[f64; 3]; 3] {
        // Convert from degrees to radians if necessary
        if self.degrees {
            angles = angles.map(|angle| angle.to_radians());
        }

        // Perform the conversion using the internal logic (in radians)
        let matrix = self._to_rotation_matrix(angles).into_inner();
        [
            [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
            [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
            [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
        ]
    }

    /// __repr__ method for EulerConvention
    fn __repr__(&self) -> String {
        format!(
            "EulerConvention(sequence='{}', extrinsic={}, degrees={})",
            self.sequence,
            if self.extrinsic { "True" } else { "False" },
            if self.degrees { "True" } else { "False" }
        )
    }

    /// __str__ method for EulerConvention
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
    degrees: bool,
    euler_convention: EulerConvention,
    _internal_euler_convention: EulerConvention,
}

#[pymethods]
impl Robot {
    #[new]
    fn new(kinematic_model: KinematicModel, euler_convention: EulerConvention) -> PyResult<Self> {
        let robot = OPWKinematics::new(kinematic_model.parameters);
        let has_parallellogram = kinematic_model.has_parallellogram;
        let degrees = euler_convention.degrees;
        let _internal_euler_convention = EulerConvention::new("XYZ".to_string(), false, degrees)?;
        Ok(Robot {
            robot,
            has_parallellogram,
            degrees,
            euler_convention,
            _internal_euler_convention,
        })
    }

    /// Forward kinematics: calculates the pose for given joints
    fn forward(&self, mut joints: [f64; 6]) -> ([f64; 3], [f64; 3]) {
        if self.has_parallellogram {
            joints[2] += joints[1];
        }
        if self.degrees {
            joints = joints.map(|x| x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let translation = pose.translation.vector.into();
        let rotation: [f64; 3] = self
            .euler_convention
            ._from_rotation_matrix(pose.rotation.to_rotation_matrix());

        (translation, rotation)
    }

    /// Inverse kinematics: calculates the joint solutions for a given pose
    fn inverse(&self, pose: ([f64; 3], [f64; 3])) -> Vec<[f64; 6]> {
        let translation = nalgebra::Translation3::from(pose.0);

        let rotation = self.euler_convention._to_rotation_matrix(pose.1);
        let iso_pose = Isometry3::from_parts(translation, rotation.into());
        let solutions = self.robot.inverse(&iso_pose);
        if self.has_parallellogram {
            solutions
                .into_iter()
                .map(|mut x| {
                    x[2] -= x[1];
                    x
                })
                .collect()
        } else {
            solutions
        }
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
}

/// Module initialization for Python
#[pymodule(name = "_internal")]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EulerConvention>()?;
    m.add_class::<KinematicModel>()?;
    m.add_class::<Robot>()?;
    Ok(())
}
