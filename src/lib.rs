use nalgebra::{Matrix3, Quaternion, Rotation3, Unit, UnitQuaternion, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyclass]
#[derive(Clone)]
struct EulerConvention {
    sequence: String,
    extrinsic: bool,
    degrees: bool,
    _seq: [Unit<Vector3<f64>>; 3],
}

impl EulerConvention {
    fn _euler_to_matrix_radians(&self, angles: [f64; 3]) -> Matrix3<f64> {
        // If extrinsic, reverse both sequence and angles
        let (seq, angles) = if self.extrinsic {
            (
                self.sequence.chars().rev().collect::<String>(),
                [angles[2], angles[1], angles[0]],
            )
        } else {
            (self.sequence.clone(), angles)
        };
        // Define the individual axis rotations
        let rot_x = |angle: f64| {
            Matrix3::new(
                1.0,
                0.0,
                0.0,
                0.0,
                angle.cos(),
                -angle.sin(),
                0.0,
                angle.sin(),
                angle.cos(),
            )
        };

        let rot_y = |angle: f64| {
            Matrix3::new(
                angle.cos(),
                0.0,
                angle.sin(),
                0.0,
                1.0,
                0.0,
                -angle.sin(),
                0.0,
                angle.cos(),
            )
        };

        let rot_z = |angle: f64| {
            Matrix3::new(
                angle.cos(),
                -angle.sin(),
                0.0,
                angle.sin(),
                angle.cos(),
                0.0,
                0.0,
                0.0,
                1.0,
            )
        };

        // Apply the correct multiplication order based on sequence
        let (alpha, beta, gamma) = (angles[0], angles[1], angles[2]);
        let rotation_matrix = match seq.as_str() {
            "XYZ" => rot_x(alpha) * rot_y(beta) * rot_z(gamma),
            "ZYX" => rot_z(alpha) * rot_y(beta) * rot_x(gamma),
            "XZX" => rot_x(alpha) * rot_z(beta) * rot_x(gamma),
            "XZY" => rot_x(alpha) * rot_z(beta) * rot_y(gamma),
            "XYX" => rot_x(alpha) * rot_y(beta) * rot_x(gamma),
            "YXY" => rot_y(alpha) * rot_x(beta) * rot_y(gamma),
            "YXZ" => rot_y(alpha) * rot_x(beta) * rot_z(gamma),
            "YZX" => rot_y(alpha) * rot_z(beta) * rot_x(gamma),
            "YZY" => rot_y(alpha) * rot_z(beta) * rot_y(gamma),
            "ZXY" => rot_z(alpha) * rot_x(beta) * rot_y(gamma),
            "ZXZ" => rot_z(alpha) * rot_x(beta) * rot_z(gamma),
            "ZYZ" => rot_z(alpha) * rot_y(beta) * rot_z(gamma),
            _ => panic!("Unsupported sequence"),
        };

        rotation_matrix
    }

    fn _euler_to_matrix(&self, angles: [f64; 3]) -> Matrix3<f64> {
        let mut angles = angles;
        if self.degrees {
            angles = angles.map(|angle| angle.to_radians());
        }
        self._euler_to_matrix_radians(angles)
    }

    fn _matrix_to_quaternion(&self, matrix: &Matrix3<f64>) -> UnitQuaternion<f64> {
        // Create a UnitQuaternion from the inverse of the rotation matrix
        let q = UnitQuaternion::from_rotation_matrix(
            &Rotation3::from_matrix_unchecked(*matrix).inverse(),
        );

        if q.w < 0.0 {
            UnitQuaternion::from_quaternion(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
        } else {
            q
        }
    }

    fn _quaternion_to_euler(&self, quat: &Quaternion<f64>) -> [f64; 3] {
        // Convert a quaternion to Euler angles based on the specified sequence
        let (w, x, y, z) = (quat.w, quat.i, quat.j, quat.k);
        let (a, b, c) = match (self.sequence.as_str(), self.extrinsic) {
            ("XYZ", true) | ("ZYX", false) => {
                // Extrinsic XYZ or intrinsic ZYX
                let beta = (2.0 * (w * y - z * x)).asin();
                if beta.abs() < (PI / 2.0 - 1e-6) {
                    (
                        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
                        beta,
                        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
                    )
                } else {
                    (
                        0.0,
                        beta,
                        (2.0 * (x * z - w * y)).atan2(1.0 - 2.0 * (x * x + z * z)),
                    )
                }
            }
            ("XYZ", false) | ("ZYX", true) => {
                // Intrinsic XYZ or extrinsic ZYX
                let beta = (-2.0 * (x * z - w * y)).asin();
                if beta.abs() < (PI / 2.0 - 1e-6) {
                    (
                        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
                        beta,
                        (2.0 * (w * z - x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
                    )
                } else {
                    (
                        (2.0 * (x * y + w * z)).atan2(1.0 - 2.0 * (y * y + z * z)),
                        beta,
                        0.0,
                    )
                }
            }
            _ => panic!("Unsupported sequence"),
        };
        [a, b, c]
    }

    fn _matrix_to_euler_radians(&self, rot: &Matrix3<f64>) -> [f64; 3] {
        let quat = self._matrix_to_quaternion(rot);
        self._quaternion_to_euler(&quat)
    }

    fn _matrix_to_euler(&self, rot: &Matrix3<f64>) -> [f64; 3] {
        let mut result = self._matrix_to_euler_radians(rot);
        if self.degrees {
            result = result.map(|angle| angle.to_degrees());
            result = [
                result[0].to_degrees(),
                result[1].to_degrees(),
                result[2].to_degrees(),
            ];
        }
        result
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
                'X' => Ok(Unit::new_normalize(Vector3::x_axis().into_inner())),
                'Y' => Ok(Unit::new_normalize(Vector3::y_axis().into_inner())),
                'Z' => Ok(Unit::new_normalize(Vector3::z_axis().into_inner())),
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
        let rot_matrix = self._euler_to_matrix(angles);
        let result = other._matrix_to_euler(&rot_matrix);
        Ok(result)
    }

    fn matrix_to_euler(&self, rot: [[f64; 3]; 3]) -> [f64; 3] {
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        self._matrix_to_euler(rotation.matrix())
    }

    fn euler_to_matrix(&self, angles: [f64; 3]) -> [[f64; 3]; 3] {
        let matrix = self._euler_to_matrix(angles);
        [
            [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
            [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
            [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
        ]
    }
    fn matrix_to_quaternion(&self, rot: [[f64; 3]; 3]) -> [f64; 4] {
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        let quaternion = self._matrix_to_quaternion(rotation.matrix());
        [quaternion.i, quaternion.j, quaternion.k, quaternion.w]
    }

    fn quaternion_to_euler(&self, quat: [f64; 4]) -> [f64; 3] {
        let quaternion =
            UnitQuaternion::from_quaternion(Quaternion::new(quat[0], quat[1], quat[2], quat[3]));
        let mut result = self._quaternion_to_euler(&quaternion);
        if self.degrees {
            result = result.map(|angle| angle.to_degrees());
        }
        result
    }

    fn __repr__(&self) -> String {
        format!(
            "EulerConvention(sequence='{}', extrinsic={}, degrees={})",
            self.sequence, self.extrinsic, self.degrees
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Module initialization for Python
#[pymodule(name = "_internal")]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EulerConvention>()?;
    Ok(())
}
