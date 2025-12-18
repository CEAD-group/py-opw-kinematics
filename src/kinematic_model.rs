use pyo3::prelude::*;

use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;

#[pyclass]
#[derive(Clone)]
pub enum AxisConstraint {
    Absolute { min: f64, max: f64 },
}

#[pyclass]
#[derive(Clone)]
pub struct KinematicModel {
    pub a1: f64,
    pub a2: f64,
    pub b: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    pub c4: f64,
    pub offsets: [f64; 6],
    pub flip_axes: [bool; 6], // Renamed and changed to boolean array
    pub has_parallelogram: bool,
    pub axis_limits: Option<[(f64, f64); 6]>, // Optional axis limits: (min, max) for each joint
    pub axis_constraints: Option<Vec<Option<AxisConstraint>>>, // More flexible constraint system
    pub has_constraints: bool, // Cached flag for performance optimization
}

impl KinematicModel {
    pub fn to_opw_kinematics(&self, degrees: bool) -> OPWKinematics {
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

    #[inline]
    pub fn joints_within_limits(&self, joints: &[f64; 6], degrees: bool) -> bool {
        // Fast path: if no constraints are defined, return true immediately
        if !self.has_constraints {
            return true;
        }
        
        // Check simple axis limits first
        if let Some(limits) = self.axis_limits {
            for (i, &joint_value) in joints.iter().enumerate() {
                let (min_limit, max_limit) = if degrees {
                    limits[i]
                } else {
                    (limits[i].0.to_radians(), limits[i].1.to_radians())
                };
                
                if joint_value < min_limit || joint_value > max_limit {
                    return false;
                }
            }
        }
        
        // Check advanced constraints (including relative ones)
        if let Some(constraints) = &self.axis_constraints {
            for (i, constraint) in constraints.iter().enumerate() {
                if let Some(constraint) = constraint {
                    let joint_value = if degrees {
                        joints[i].to_radians()
                    } else {
                        joints[i]
                    };
                    
                    match constraint {
                        AxisConstraint::Absolute { min, max } => {
                            if joint_value < *min || joint_value > *max {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        
        true
    }
    
    /// Determine if this model has any constraints defined
    fn has_any_constraints(&self) -> bool {
        if self.axis_limits.is_some() {
            return true;
        }
        
        if let Some(constraints) = &self.axis_constraints {
            for constraint in constraints {
                if constraint.is_some() {
                    return true;
                }
            }
        }
        
        false
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
        has_parallelogram = false,
        axis_limits = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        axis_limits: Option<Vec<(f64, f64)>>,
    ) -> PyResult<Self> {
        let axis_limits = axis_limits.map(|limits| {
            if limits.len() != 6 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "axis_limits must contain exactly 6 tuples of (min, max) values"
                ));
            }
            Ok(limits.try_into().unwrap())
        }).transpose()?;
        
        
        // Create the model first to calculate has_constraints
        let mut model = KinematicModel {
            a1,
            a2,
            b,
            c1,
            c2,
            c3,
            c4,
            offsets: offsets.into(),
            flip_axes: flip_axes.into(),
            has_parallelogram,
            axis_limits,
            axis_constraints, // Use the processed constraints
            has_constraints: false, // Will be set below
        };
        
        // Now calculate and set has_constraints
        model.has_constraints = model.has_any_constraints();
        
        Ok(model)
    }

    // Getter methods to provide access to attributes since the class is frozen.
    #[getter]
    pub fn a1(&self) -> f64 {
        self.a1
    }

    #[getter]
    pub fn a2(&self) -> f64 {
        self.a2
    }

    #[getter]
    pub fn b(&self) -> f64 {
        self.b
    }

    #[getter]
    pub fn c1(&self) -> f64 {
        self.c1
    }

    #[getter]
    pub fn c2(&self) -> f64 {
        self.c2
    }

    #[getter]
    pub fn c3(&self) -> f64 {
        self.c3
    }

    #[getter]
    pub fn c4(&self) -> f64 {
        self.c4
    }

    #[getter]
    pub fn offsets(&self) -> Vec<f64> {
        self.offsets.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    pub fn flip_axes(&self) -> Vec<bool> {
        self.flip_axes.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    pub fn has_parallelogram(&self) -> bool {
        self.has_parallelogram
    }

    #[getter]
    pub fn has_constraints(&self) -> bool {
        self.has_constraints
    }

    #[getter]
    pub fn axis_limits(&self) -> Option<Vec<(f64, f64)>> {
        self.axis_limits.map(|limits| limits.to_vec())
    }
    pub fn set_axis_limits(&mut self, limits: Option<Vec<(f64, f64)>>) -> PyResult<()> {
        if let Some(ref limits_vec) = limits {
            if limits_vec.len() != 6 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "axis_limits must contain exactly 6 tuples of (min, max) values"
                ));
            }
            self.axis_limits = Some(limits_vec.clone().try_into().unwrap());
        } else {
            self.axis_limits = None;
        }
        // Update the cached has_constraints flag
        self.has_constraints = self.has_any_constraints();
        Ok(())
    }

    /// Set an absolute constraint for a specific axis
    #[pyo3(signature = (axis, min, max, degrees=false))]
    pub fn set_absolute_constraint(&mut self, axis: usize, min: f64, max: f64, degrees: bool) -> PyResult<()> {
        if axis >= 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Axis index must be between 0 and 5"
            ));
        }

        // Convert degrees to radians if needed
        let (min_rad, max_rad) = if degrees {
            (min.to_radians(), max.to_radians())
        } else {
            (min, max)
        };

        if self.axis_constraints.is_none() {
            self.axis_constraints = Some(vec![None; 6]);
        }

        if let Some(ref mut constraints) = self.axis_constraints {
            constraints[axis] = Some(AxisConstraint::Absolute { min: min_rad, max: max_rad });
        }
        
        // Update the cached has_constraints flag
        self.has_constraints = self.has_any_constraints();
        Ok(())
    }
    /// Clear all constraints for a specific axis
    pub fn clear_axis_constraint(&mut self, axis: usize) -> PyResult<()> {
        if axis >= 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Axis index must be between 0 and 5"
            ));
        }

        if let Some(ref mut constraints) = self.axis_constraints {
            constraints[axis] = None;
        }
        
        // Update the cached has_constraints flag
        self.has_constraints = self.has_any_constraints();
        Ok(())
    }

    /// Clear all advanced constraints
    pub fn clear_all_constraints(&mut self) {
        self.axis_constraints = None;
        // Update the cached has_constraints flag
        self.has_constraints = self.has_any_constraints();
    }

    /// Check if given joints satisfy all constraints (Python-friendly)
    pub fn joints_within_limits_vec(&self, joints: Vec<f64>, degrees: Option<bool>) -> bool {
        if joints.len() != 6 {
            return false;
        }
        let joints_array: [f64; 6] = joints.try_into().unwrap();
        self.joints_within_limits(&joints_array, degrees.unwrap_or(true))
    }

    pub fn __repr__(&self) -> String {
        format!(
            "KinematicModel(\n    a1={},\n    a2={},\n    b={},\n    c1={},\n    c2={},\n    c3={},\n    c4={},\n    offsets={:?},\n    flip_axes={:?},\n    has_parallelogram={},\n    axis_limits={:?}\n)",
            self.a1, self.a2, self.b, self.c1, self.c2, self.c3, self.c4,
            self.offsets, self.flip_axes, self.has_parallelogram, self.axis_limits
        )
    }
}
