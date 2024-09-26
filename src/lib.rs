use nalgebra::{Matrix3, Rotation, Rotation3, Unit, Vector3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
struct EulerConvention {
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
}

/// Module initialization for Python
#[pymodule]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EulerConvention>()?;
    Ok(())
}
