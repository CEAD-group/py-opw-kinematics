use nalgebra::{Matrix3, Quaternion, Rotation3, Unit, UnitQuaternion, Vector3, Translation3, Isometry3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

use polars::frame::DataFrame;
use polars::prelude::*;
use polars::series::Series;

use pyo3_polars::PyDataFrame;
use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose};
use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;
#[pyclass]
#[derive(Clone)]
struct EulerConvention {
    sequence: String,
    extrinsic: bool,
    degrees: bool,
    _seq: [Unit<Vector3<f64>>; 3],
}

impl EulerConvention {
    fn _euler_to_matrix_radians(&self, angles: [f64; 3]) -> Rotation3<f64> {
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

    fn _euler_to_matrix(&self, angles: [f64; 3]) -> Rotation3<f64> {
        let mut angles = angles;
        if self.degrees {
            angles = angles.map(|angle| angle.to_radians());
        }
        self._euler_to_matrix_radians(angles)
    }

    fn _matrix_to_quaternion(&self, rotmat: &Rotation3<f64>) -> UnitQuaternion<f64> {
        // Create a UnitQuaternion from the inverse of the rotation matrix
        // The inverse of the rotation matrix ensures proper quaternion conversion
        // This is similar to the logic in SciPy's `Rotation` class.
        let q = UnitQuaternion::from_rotation_matrix(
            &rotmat.inverse(),
        );

        // Ensure quaternion's w-component is non-negative
        // This is done to avoid discontinuities in representation, similar to SciPy's approach
        
        if q.w < 0.0 {
            UnitQuaternion::from_quaternion(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
        } else {
            q
        }
    }

    fn _elementary_basis_index(&self, axis: char) -> usize {
        // Maps axis labels ('X', 'Y', 'Z') to corresponding indices (0, 1, 2)
        match axis {
            'X' => 0,
            'Y' => 1,
            'Z' => 2,
            _ => panic!("Invalid axis: {}", axis), // Panic if an invalid axis is provided
        }
    }
    fn _get_angles(
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
        let eps = 1e-7; // Small threshold for floating-point comparison to detect singularities
        let pi = PI;

        let (angle_first, angle_third) = if extrinsic { (0, 2) } else { (2, 0) };

        let mut angles = [0.0; 3];

        // Step 2: Compute the second angle
        angles[1] = 2.0 * (c.hypot(d)).atan2(a.hypot(b));

        // Check for singularities where angles[1] = 0 or π (gimbal lock conditions)
        let case = if (angles[1]).abs() <= eps {
            1 // angles[1] ≈ 0
        } else if (angles[1] - pi).abs() <= eps {
            2 // angles[1] ≈ π
        } else {
            0 // Regular case
        };

        // Step 3: Compute the first and third angles
        let half_sum = b.atan2(a);
        let half_diff = d.atan2(c);

        if case == 0 {
            // Regular case where no gimbal lock occurs
            angles[angle_first] = half_sum - half_diff;
            angles[angle_third] = half_sum + half_diff;
        } else {
            // Singular cases (gimbal lock detected)
            if case == 1 {
                angles[angle_first] = 2.0 * half_sum;
            } else {
                angles[angle_first] = 2.0 * half_diff * if extrinsic { -1.0 } else { 1.0 };
            }
            angles[angle_third] = 0.0; // Set third angle to zero, as it cannot be uniquely determined
        }

        // Adjust for asymmetric sequences
        if !symmetric {
            angles[angle_third] *= sign;
            angles[1] -= lamb;
        }

        // Normalize angles to the range [-π, π]
        for angle in angles.iter_mut() {
            if *angle < -pi {
                *angle += 2.0 * pi;
            } else if *angle > pi {
                *angle -= 2.0 * pi;
            }
        }

        // Print warning for gimbal lock detection
        if case != 0 {
            println!("Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.");
        }

        angles
    }

    fn _quaternion_to_euler(&self, quat: &Quaternion<f64>) -> [f64; 3] {
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

        // Determine the sign based on the permutation parity of the axis sequence
        let perm = vec![i, j, k];
        let even_permutation =
            perm == vec![0, 1, 2] || perm == vec![1, 2, 0] || perm == vec![2, 0, 1];
        let sign = if even_permutation { 1.0 } else { -1.0 };

        // Ensure quaternion has a positive w-component to avoid discontinuities
        let mut q = *quat;
        if q.w < 0.0 {
            q = Quaternion::new(-q.w, -q.i, -q.j, -q.k);
        }

        // Step 1: Permute quaternion elements based on the rotation sequence
        let (a, b, c, d) = if symmetric {
            (q.w, q[i], q[j], q[k] * sign)
        } else {
            // Adjust quaternion elements for asymmetric sequences
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

    fn _matrix_to_euler_radians(&self, rotmat: Rotation3<f64>) -> [f64; 3] {
        let quat = self._matrix_to_quaternion(&rotmat);
        self._quaternion_to_euler(&quat)
    }

    fn _matrix_to_euler(&self, rotmat: Rotation3<f64>) -> [f64; 3] {
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
        let rotmat = self._euler_to_matrix(angles);
        let result = other._matrix_to_euler(rotmat);
        Ok(result)
    }

    fn matrix_to_euler(&self, rot: [[f64; 3]; 3]) -> [f64; 3] {
        let rotmat = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        self._matrix_to_euler(rotmat)
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
        let rotmat = Rotation3::from_matrix_unchecked(Matrix3::from(rot));
        let quaternion = self._matrix_to_quaternion(&rotmat);
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

#[pyclass(frozen)] // Declare the class as frozen to provide immutability.
#[derive(Clone)]
struct KinematicModel {
    a1: f64,
    a2: f64,
    b: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    offsets: [f64; 6],
    flip_axes: [bool; 6], // Renamed and changed to boolean array
    has_parallelogram: bool,
}

impl KinematicModel {
    fn to_opw_kinematics(&self, degrees: bool) -> OPWKinematics {
        let sign_corrections = self.flip_axes.map(|x| if x { -1 } else { 1 });
        OPWKinematics::new(Parameters {
            a1: self.a1,
            a2: self.a2,
            b: self.b,
            c1: self.c1,
            c2: self.c2,
            c3: self.c3,
            c4: self.c4,
            offsets: if degrees {
                self.offsets
                    .iter()
                    .map(|&x| x.to_radians())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            } else {
                self.offsets
            },
            sign_corrections,
            dof: 6,
        })
    }
}

#[pymethods]
impl KinematicModel {
    #[new]
    #[pyo3(signature = (
        a1 = 0.0, 
        a2 = 0.0, 
        b = 0.0, 
        c1 = 0.0, 
        c2 = 0.0, 
        c3 = 0.0, 
        c4 = 0.0, 
        offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 
        flip_axes = (false, false, false, false, false, false), 
        has_parallelogram = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        a1: f64,
        a2: f64,
        b: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        offsets: (f64, f64, f64, f64, f64, f64), 
        flip_axes: (bool, bool, bool, bool, bool, bool),  
        has_parallelogram: bool,
    ) -> PyResult<Self> {
        Ok(KinematicModel {
            a1,
            a2,
            b,
            c1,
            c2,
            c3,
            c4,
            offsets: offsets.try_into().unwrap(),  
            flip_axes: flip_axes.try_into().unwrap(),  
            has_parallelogram,
        })
    }

    // Getter methods to provide access to attributes since the class is frozen.
    #[getter]
    fn a1(&self) -> f64 {
        self.a1
    }

    #[getter]
    fn a2(&self) -> f64 {
        self.a2
    }

    #[getter]
    fn b(&self) -> f64 {
        self.b
    }

    #[getter]
    fn c1(&self) -> f64 {
        self.c1
    }

    #[getter]
    fn c2(&self) -> f64 {
        self.c2
    }

    #[getter]
    fn c3(&self) -> f64 {
        self.c3
    }

    #[getter]
    fn c4(&self) -> f64 {
        self.c4
    }

    #[getter]
    fn offsets(&self) -> Vec<f64> {
        self.offsets.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    fn flip_axes(&self) -> Vec<bool> {
        self.flip_axes.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    fn has_parallelogram(&self) -> bool {
        self.has_parallelogram
    }

    fn __repr__(&self) -> String {
        format!(
            "KinematicModel(\n    a1={},\n    a2={},\n    b={},\n    c1={},\n    c2={},\n    c3={},\n    c4={},\n    offsets={:?},\n    flip_axes={:?},\n    has_parallelogram={}\n)",
            self.a1, self.a2, self.b, self.c1, self.c2, self.c3, self.c4,
            self.offsets, self.flip_axes, self.has_parallelogram
        )
    }
}

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
        let rotation = self
            .euler_convention
            ._matrix_to_euler(combined_rotation);

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
            Series::new("J1".into(), j1),
            Series::new("J2".into(), j2),
            Series::new("J3".into(), j3),
            Series::new("J4".into(), j4),
            Series::new("J5".into(), j5),
            Series::new("J6".into(), j6),
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
                let (translation, rotation) = self.forward(joints_array);

                x.push(Some(translation[0]));
                y.push(Some(translation[1]));
                z.push(Some(translation[2]));
                a.push(Some(rotation[0]));
                b.push(Some(rotation[1]));
                c.push(Some(rotation[2]));
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
            Series::new("X".into(), x),
            Series::new("Y".into(), y),
            Series::new("Z".into(), z),
            Series::new("A".into(), a),
            Series::new("B".into(), b),
            Series::new("C".into(), c),
        ])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyDataFrame(df_result))
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
