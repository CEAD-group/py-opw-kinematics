use nalgebra::{Matrix3, Quaternion, Rotation3, Unit, UnitQuaternion, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

#[pyclass]
#[derive(Clone)]
pub struct EulerConvention {
    pub sequence: String,
    pub extrinsic: bool,
    pub degrees: bool,
    pub _seq: [Unit<Vector3<f64>>; 3],
}

impl EulerConvention {
    pub fn _euler_to_matrix_radians(&self, angles: [f64; 3]) -> Rotation3<f64> {
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

        Rotation3::from_matrix_unchecked(rotation_matrix)
    }

    pub fn _euler_to_matrix(&self, angles: [f64; 3]) -> Rotation3<f64> {
        let mut angles = angles;
        if self.degrees {
            angles = angles.map(|angle| angle.to_radians());
        }
        self._euler_to_matrix_radians(angles)
    }

    pub fn _matrix_to_quaternion(&self, rotmat: &Rotation3<f64>) -> UnitQuaternion<f64> {
        let q = UnitQuaternion::from_rotation_matrix(&rotmat);

        // Ensure quaternion's w-component is non-negative
        // This is done to avoid discontinuities in representation, similar to SciPy's approach

        if q.w < 0.0 {
            UnitQuaternion::from_quaternion(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
        } else {
            q
        }
    }

    pub fn _elementary_basis_index(&self, axis: char) -> usize {
        // Maps axis labels ('X', 'Y', 'Z') to corresponding indices (0, 1, 2)
        match axis {
            'X' => 0,
            'Y' => 1,
            'Z' => 2,
            _ => panic!("Invalid axis: {}", axis), // Panic if an invalid axis is provided
        }
    }
    pub fn _get_angles(
        &self,
        extrinsic: bool,
        symmetric: bool,
        sign: f64,
        lamb: f64,
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    ) -> [f64; 3] {
        // Ported from SciPy's `scipy.spatial.transform._rotation._get_angles` function
        // https://github.com/scipy/scipy/blob/485b043a6be8edf89d6a54369098aa941201b485/scipy/spatial/transform/_rotation.pyx#L120
        let mut angles = [0.0; 3];

        // Intrinsic/Extrinsic conversion helpers
        let (angle_first, angle_third) = if extrinsic { (0, 2) } else { (2, 0) };

        // Step 2: Compute the second angle
        angles[1] = 2.0 * (c.hypot(d)).atan2(a.hypot(b));

        // Determine if there's a singularity (if the second angle is 0 or pi)
        let case = if angles[1].abs() <= 1e-7 {
            1
        } else if (angles[1] - PI).abs() <= 1e-7 {
            2
        } else {
            0
        };

        // Step 3: Compute first and third angles, based on the case
        let half_sum = b.atan2(a);
        let half_diff = d.atan2(c);

        if case == 0 {
            // No singularities
            angles[angle_first] = half_sum - half_diff;
            angles[angle_third] = half_sum + half_diff;
        } else {
            // Degenerate case
            angles[2] = 0.0;
            if case == 1 {
                angles[0] = 2.0 * half_sum;
            } else {
                angles[0] = 2.0 * half_diff * if extrinsic { -1.0 } else { 1.0 };
            }
        }

        // For Tait-Bryan/asymmetric sequences
        if !symmetric {
            angles[angle_third] *= sign;
            angles[1] -= lamb;
        }

        // Normalize angles to be within [-pi, pi]
        for angle in &mut angles {
            if *angle < -PI {
                *angle += 2.0 * PI;
            } else if *angle > PI {
                *angle -= 2.0 * PI;
            }
        }

        if case != 0 {
            // TODO: Implement a proper python warning
            // eprintln!(
            //     "Warning: Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles."
            // );
        }

        angles
    }

    pub fn _quaternion_to_euler(&self, quat: &Quaternion<f64>) -> [f64; 3] {
        // Ported from SciPy's `scipy.spatial.transform._rotation._compute_euler_from_quat` function
        //    https://github.com/scipy/scipy/blob/485b043a6be8edf89d6a54369098aa941201b485/scipy/spatial/transform/_rotation.pyx#L325

        // Reverse sequence if intrinsic rotation is required
        let mut seq: Vec<char> = self.sequence.chars().collect();
        if !self.extrinsic {
            seq.reverse();
        }

        // Map axis labels to indices
        let i = self._elementary_basis_index(seq[0]);
        let j = self._elementary_basis_index(seq[1]);
        let mut k = self._elementary_basis_index(seq[2]);

        let symmetric = i == k;
        if symmetric {
            k = 3 - i - j;
        }

        // Step 0
        // Check if permutation is even (+1) or odd (-1)
        // Determine the sign based on the permutation parity of the axis sequence
        let permutation = ((i as i8 - j as i8) * (j as i8 - k as i8) * (k as i8 - i as i8)) / 2;
        if permutation != 1 && permutation != -1 {
            panic!("Invalid permutation parity: {}", permutation);
        }
        let even_permutation = permutation == 1;
        let sign = if even_permutation { 1.0 } else { -1.0 };

        let q = *quat;

        // Step 1: Permute quaternion elements based on the rotation sequence
        let (a, b, c, d) = if symmetric {
            (q.w, q[i], q[j], q[k] * sign)
        } else {
            (
                q.w - q[j],
                q[i] + q[k] * sign,
                q[j] + q.w,
                q[k] * sign - q[i],
            )
        };

        // Compute Euler angles using helper function
        self._get_angles(self.extrinsic, symmetric, sign, PI / 2.0, a, b, c, d)
    }

    pub fn _matrix_to_euler_radians(&self, rotmat: Rotation3<f64>) -> [f64; 3] {
        let quat = self._matrix_to_quaternion(&rotmat);
        self._quaternion_to_euler(&quat)
    }

    pub fn _matrix_to_euler(&self, rotmat: Rotation3<f64>) -> [f64; 3] {
        let mut result = self._matrix_to_euler_radians(rotmat);
        if self.degrees {
            result = result.map(|angle| angle.to_degrees());
        }
        result
    }
}

#[pymethods]
impl EulerConvention {
    #[new]
    pub fn new(sequence: String, extrinsic: bool, degrees: bool) -> PyResult<Self> {
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

    pub fn convert(&self, other: &EulerConvention, angles: [f64; 3]) -> PyResult<[f64; 3]> {
        let rotmat = self._euler_to_matrix(angles);
        let result = other._matrix_to_euler(rotmat);
        Ok(result)
    }

    pub fn matrix_to_euler(&self, rot: [[f64; 3]; 3]) -> [f64; 3] {
        let rotmat = Rotation3::from_matrix_unchecked(Matrix3::from(rot).transpose());
        self._matrix_to_euler(rotmat)
    }

    pub fn euler_to_matrix(&self, angles: [f64; 3]) -> [[f64; 3]; 3] {
        let matrix = self._euler_to_matrix(angles);
        [
            [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
            [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
            [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
        ]
    }
    pub fn matrix_to_quaternion(&self, rot: [[f64; 3]; 3]) -> [f64; 4] {
        let rotmat = Rotation3::from_matrix_unchecked(Matrix3::from(rot).transpose());
        let quaternion = self._matrix_to_quaternion(&rotmat);
        [quaternion.w, quaternion.i, quaternion.j, quaternion.k]
    }

    pub fn quaternion_to_euler(&self, quat: [f64; 4]) -> [f64; 3] {
        // w i j k
        let quaternion =
            UnitQuaternion::from_quaternion(Quaternion::new(quat[0], quat[1], quat[2], quat[3]));
        let mut result = self._quaternion_to_euler(&quaternion);
        if self.degrees {
            result = result.map(|angle| angle.to_degrees());
        }
        result
    }

    pub fn __repr__(&self) -> String {
        format!(
            "EulerConvention(sequence='{}', extrinsic={}, degrees={})",
            self.sequence, self.extrinsic, self.degrees
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
