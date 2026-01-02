mod euler; // Add this line to import the new module
mod kinematic_model; // Add this line to import the new module
use crate::euler::EulerConvention;
use crate::kinematic_model::KinematicModel;

use nalgebra::{Isometry3, Quaternion, Rotation3, Translation3, UnitQuaternion, Vector3};
use parry3d::shape::TriMesh;
use pyo3::prelude::*;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use rs_opw_kinematics::collisions::CheckMode as RsCheckMode;
use rs_opw_kinematics::collisions::CollisionBody as RsCollisionBody;
use rs_opw_kinematics::constraints::Constraints;
use rs_opw_kinematics::kinematic_traits::{Kinematics, Pose};
use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_read_trimesh::load_trimesh;
use std::collections::HashMap;

/// Collision checking mode.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckMode {
    /// Stop after finding the first collision (faster)
    FirstCollisionOnly,
    /// Find all collisions (slower but complete)
    AllCollisions,
    /// Disable collision checking entirely
    NoCheck,
}

impl From<CheckMode> for RsCheckMode {
    fn from(mode: CheckMode) -> Self {
        match mode {
            CheckMode::FirstCollisionOnly => RsCheckMode::FirstCollisionOnly,
            CheckMode::AllCollisions => RsCheckMode::AllCollsions, // Note: typo in rs-opw-kinematics
            CheckMode::NoCheck => RsCheckMode::NoCheck,
        }
    }
}

impl From<RsCheckMode> for CheckMode {
    fn from(mode: RsCheckMode) -> Self {
        match mode {
            RsCheckMode::FirstCollisionOnly => CheckMode::FirstCollisionOnly,
            RsCheckMode::AllCollsions => CheckMode::AllCollisions,
            RsCheckMode::NoCheck => CheckMode::NoCheck,
        }
    }
}

/// Constants for special distance values
pub const NEVER_COLLIDES: f32 = -1.0;
pub const TOUCH_ONLY: f32 = 0.0;

/// Joint/body indices for special_distances
pub const J1: u16 = 0;
pub const J2: u16 = 1;
pub const J3: u16 = 2;
pub const J4: u16 = 3;
pub const J5: u16 = 4;
pub const J6: u16 = 5;
pub const J_TOOL: u16 = 100;
pub const J_BASE: u16 = 101;

/// Safety distances configuration for collision detection.
///
/// Defines tolerance bounds between robot parts and environmental objects.
#[pyclass]
#[derive(Clone)]
pub struct SafetyDistances {
    /// Allowed distance between robot and environment objects
    #[pyo3(get, set)]
    pub to_environment: f32,
    /// Default allowed distance between any two parts of the robot
    #[pyo3(get, set)]
    pub to_robot_default: f32,
    /// Special distances for specific joint pairs (e.g., adjacent joints that naturally come close)
    special_distances: HashMap<(u16, u16), f32>,
    /// Collision checking mode
    #[pyo3(get, set)]
    pub mode: CheckMode,
}

#[pymethods]
impl SafetyDistances {
    /// Create a new SafetyDistances configuration.
    ///
    /// Args:
    ///     to_environment: Minimum distance to environment objects. Default: 0.0 (touch only)
    ///     to_robot_default: Default minimum distance between robot parts. Default: 0.0 (touch only)
    ///     special_distances: Optional dict mapping (joint_id, joint_id) pairs to specific distances.
    ///                        Use the J1-J6, J_TOOL, J_BASE constants for joint indices.
    ///                        Use NEVER_COLLIDES (-1.0) to skip checking between specific parts.
    ///     mode: Collision checking mode. Default: FirstCollisionOnly
    #[new]
    #[pyo3(signature = (to_environment=0.0, to_robot_default=0.0, special_distances=None, mode=CheckMode::FirstCollisionOnly))]
    fn new(
        to_environment: f32,
        to_robot_default: f32,
        special_distances: Option<HashMap<(u16, u16), f32>>,
        mode: CheckMode,
    ) -> Self {
        SafetyDistances {
            to_environment,
            to_robot_default,
            special_distances: special_distances.unwrap_or_default(),
            mode,
        }
    }

    /// Get the minimum allowed distance between two objects.
    ///
    /// Returns the special distance if defined, otherwise the default robot distance.
    /// Order of indices doesn't matter (symmetric lookup).
    fn min_distance(&self, id1: u16, id2: u16) -> f32 {
        // Check both orderings since the map may have either
        if let Some(&dist) = self.special_distances.get(&(id1, id2)) {
            dist
        } else if let Some(&dist) = self.special_distances.get(&(id2, id1)) {
            dist
        } else {
            self.to_robot_default
        }
    }

    /// Set a special distance for a specific joint pair.
    fn set_special_distance(&mut self, id1: u16, id2: u16, distance: f32) {
        // Normalize the key ordering (smaller id first)
        let key = if id1 <= id2 { (id1, id2) } else { (id2, id1) };
        self.special_distances.insert(key, distance);
    }

    /// Get all special distances as a dict.
    #[getter]
    fn get_special_distances(&self) -> HashMap<(u16, u16), f32> {
        self.special_distances.clone()
    }

    /// Class constants for joint indices
    #[classattr]
    const J1: u16 = J1;
    #[classattr]
    const J2: u16 = J2;
    #[classattr]
    const J3: u16 = J3;
    #[classattr]
    const J4: u16 = J4;
    #[classattr]
    const J5: u16 = J5;
    #[classattr]
    const J6: u16 = J6;
    #[classattr]
    const J_TOOL: u16 = J_TOOL;
    #[classattr]
    const J_BASE: u16 = J_BASE;
    #[classattr]
    const NEVER_COLLIDES: f32 = NEVER_COLLIDES;
    #[classattr]
    const TOUCH_ONLY: f32 = TOUCH_ONLY;

    fn __repr__(&self) -> String {
        format!(
            "SafetyDistances(to_environment={}, to_robot_default={}, mode={:?}, special_distances={:?})",
            self.to_environment, self.to_robot_default, self.mode, self.special_distances
        )
    }
}

impl SafetyDistances {
    /// Convert to rs-opw-kinematics SafetyDistances
    pub fn to_rs_safety_distances(&self) -> rs_opw_kinematics::collisions::SafetyDistances {
        rs_opw_kinematics::collisions::SafetyDistances {
            to_environment: self.to_environment,
            to_robot_default: self.to_robot_default,
            special_distances: self.special_distances.clone(),
            mode: self.mode.into(),
        }
    }
}

/// A collision body representing an environment obstacle or robot part.
///
/// Loaded from STL/PLY/OBJ mesh files with optional position and orientation.
#[pyclass]
#[derive(Clone)]
pub struct CollisionBody {
    mesh: TriMesh,
    /// Position (x, y, z) in meters
    position: [f64; 3],
    /// Orientation as quaternion (w, x, y, z) - scalar first
    orientation: [f64; 4],
    /// Scale factor applied to the mesh
    scale: f64,
}

#[pymethods]
impl CollisionBody {
    /// Create a CollisionBody from a mesh file.
    ///
    /// Args:
    ///     mesh_path: Path to mesh file (STL, PLY, or OBJ format)
    ///     position: Position (x, y, z) in meters. Default: (0, 0, 0)
    ///     orientation: Orientation as quaternion (w, x, y, z) - scalar first. Default: (1, 0, 0, 0)
    ///     scale: Scale factor to apply to the mesh. Default: 1.0
    #[new]
    #[pyo3(signature = (mesh_path, position=None, orientation=None, scale=1.0))]
    fn new(
        mesh_path: &str,
        position: Option<[f64; 3]>,
        orientation: Option<[f64; 4]>,
        scale: f64,
    ) -> PyResult<Self> {
        let mesh = load_trimesh(mesh_path, scale as f32)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load mesh: {}", e)))?;

        Ok(CollisionBody {
            mesh,
            position: position.unwrap_or([0.0, 0.0, 0.0]),
            orientation: orientation.unwrap_or([1.0, 0.0, 0.0, 0.0]),
            scale,
        })
    }

    /// Get the position (x, y, z).
    #[getter]
    fn get_position(&self) -> [f64; 3] {
        self.position
    }

    /// Set the position (x, y, z).
    #[setter]
    fn set_position(&mut self, position: [f64; 3]) {
        self.position = position;
    }

    /// Get the orientation as quaternion (w, x, y, z).
    #[getter]
    fn get_orientation(&self) -> [f64; 4] {
        self.orientation
    }

    /// Set the orientation as quaternion (w, x, y, z).
    #[setter]
    fn set_orientation(&mut self, orientation: [f64; 4]) {
        self.orientation = orientation;
    }

    /// Get the scale factor.
    #[getter]
    fn get_scale(&self) -> f64 {
        self.scale
    }

    /// Get the number of triangles in the mesh.
    #[getter]
    fn num_triangles(&self) -> usize {
        self.mesh.num_triangles()
    }

    fn __repr__(&self) -> String {
        format!(
            "CollisionBody(position={:?}, orientation={:?}, scale={}, triangles={})",
            self.position, self.orientation, self.scale, self.mesh.num_triangles()
        )
    }
}

impl CollisionBody {
    /// Convert to rs-opw-kinematics CollisionBody
    pub fn to_rs_collision_body(&self) -> RsCollisionBody {
        let translation = Translation3::new(
            self.position[0] as f32,
            self.position[1] as f32,
            self.position[2] as f32,
        );
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
            self.orientation[0] as f32,
            self.orientation[1] as f32,
            self.orientation[2] as f32,
            self.orientation[3] as f32,
        ));
        let pose = Isometry3::from_parts(translation, rotation);

        RsCollisionBody {
            mesh: self.mesh.clone(),
            pose,
        }
    }

    /// Get a reference to the internal mesh
    pub fn get_mesh(&self) -> &TriMesh {
        &self.mesh
    }
}

/// Robot body configuration for collision detection.
///
/// Represents the geometry of a 6-axis robot including joint meshes,
/// optional tool and base, environment obstacles, and safety distances.
#[pyclass]
#[derive(Clone)]
pub struct RobotBody {
    /// Meshes for the 6 robot joints
    joint_meshes: [TriMesh; 6],
    /// Optional tool mesh
    tool_mesh: Option<TriMesh>,
    /// Optional base mesh with its pose
    base_mesh: Option<TriMesh>,
    base_position: [f64; 3],
    base_orientation: [f64; 4],
    /// Environment collision bodies
    environment: Vec<CollisionBody>,
    /// Safety distances configuration
    safety: SafetyDistances,
}

#[pymethods]
impl RobotBody {
    /// Create a RobotBody from individual mesh files for each joint.
    ///
    /// Args:
    ///     joint_meshes: Tuple of 6 paths to mesh files (STL/PLY/OBJ), one per joint
    ///     scale: Scale factor to apply to all meshes. Default: 1.0
    ///     safety: Safety distances configuration. Default: touch-only
    #[new]
    #[pyo3(signature = (joint_meshes, scale=1.0, safety=None))]
    fn new(
        joint_meshes: [String; 6],
        scale: f64,
        safety: Option<SafetyDistances>,
    ) -> PyResult<Self> {
        let scale_f32 = scale as f32;
        let meshes: Result<Vec<TriMesh>, _> = joint_meshes
            .iter()
            .map(|path| load_trimesh(path, scale_f32))
            .collect();

        let meshes = meshes.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load joint mesh: {}", e))
        })?;

        // Convert Vec to array
        let joint_meshes_array: [TriMesh; 6] = meshes.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("Expected exactly 6 joint meshes")
        })?;

        Ok(RobotBody {
            joint_meshes: joint_meshes_array,
            tool_mesh: None,
            base_mesh: None,
            base_position: [0.0, 0.0, 0.0],
            base_orientation: [1.0, 0.0, 0.0, 0.0],
            environment: Vec::new(),
            safety: safety.unwrap_or_else(|| SafetyDistances::new(0.0, 0.0, None, CheckMode::FirstCollisionOnly)),
        })
    }

    /// Add a tool mesh to the robot.
    ///
    /// Args:
    ///     mesh_path: Path to the tool mesh file
    ///     scale: Scale factor for the mesh. Default: 1.0
    ///
    /// Returns:
    ///     Self for method chaining
    #[pyo3(signature = (mesh_path, scale=1.0))]
    fn with_tool(&mut self, mesh_path: &str, scale: f64) -> PyResult<Self> {
        let mesh = load_trimesh(mesh_path, scale as f32)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load tool mesh: {}", e)))?;
        self.tool_mesh = Some(mesh);
        Ok(self.clone())
    }

    /// Add a base mesh to the robot.
    ///
    /// Args:
    ///     mesh_path: Path to the base mesh file
    ///     position: Position (x, y, z) of the base. Default: (0, 0, 0)
    ///     orientation: Orientation as quaternion (w, x, y, z). Default: (1, 0, 0, 0)
    ///     scale: Scale factor for the mesh. Default: 1.0
    ///
    /// Returns:
    ///     Self for method chaining
    #[pyo3(signature = (mesh_path, position=None, orientation=None, scale=1.0))]
    fn with_base(
        &mut self,
        mesh_path: &str,
        position: Option<[f64; 3]>,
        orientation: Option<[f64; 4]>,
        scale: f64,
    ) -> PyResult<Self> {
        let mesh = load_trimesh(mesh_path, scale as f32)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load base mesh: {}", e)))?;
        self.base_mesh = Some(mesh);
        self.base_position = position.unwrap_or([0.0, 0.0, 0.0]);
        self.base_orientation = orientation.unwrap_or([1.0, 0.0, 0.0, 0.0]);
        Ok(self.clone())
    }

    /// Add an environment collision body.
    ///
    /// Args:
    ///     body: CollisionBody to add to the environment
    ///
    /// Returns:
    ///     Self for method chaining
    fn add_environment(&mut self, body: CollisionBody) -> Self {
        self.environment.push(body);
        self.clone()
    }

    /// Set the safety distances configuration.
    ///
    /// Args:
    ///     safety: SafetyDistances configuration
    ///
    /// Returns:
    ///     Self for method chaining
    fn with_safety(&mut self, safety: SafetyDistances) -> Self {
        self.safety = safety;
        self.clone()
    }

    /// Check if the robot has a tool configured.
    #[getter]
    fn has_tool(&self) -> bool {
        self.tool_mesh.is_some()
    }

    /// Check if the robot has a base configured.
    #[getter]
    fn has_base(&self) -> bool {
        self.base_mesh.is_some()
    }

    /// Get the number of environment bodies.
    #[getter]
    fn num_environment_bodies(&self) -> usize {
        self.environment.len()
    }

    /// Get the current safety configuration.
    #[getter]
    fn get_safety(&self) -> SafetyDistances {
        self.safety.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RobotBody(has_tool={}, has_base={}, environment_bodies={})",
            self.has_tool(),
            self.has_base(),
            self.environment.len()
        )
    }
}

impl RobotBody {
    /// Convert to rs-opw-kinematics RobotBody
    pub fn to_rs_robot_body(&self) -> rs_opw_kinematics::collisions::RobotBody {
        use rs_opw_kinematics::collisions::{BaseBody, RobotBody as RsRobotBody};

        let base = self.base_mesh.as_ref().map(|mesh| {
            let translation = Translation3::new(
                self.base_position[0] as f32,
                self.base_position[1] as f32,
                self.base_position[2] as f32,
            );
            let rotation = UnitQuaternion::from_quaternion(Quaternion::new(
                self.base_orientation[0] as f32,
                self.base_orientation[1] as f32,
                self.base_orientation[2] as f32,
                self.base_orientation[3] as f32,
            ));
            BaseBody {
                mesh: mesh.clone(),
                base_pose: Isometry3::from_parts(translation, rotation),
            }
        });

        let collision_environment: Vec<RsCollisionBody> = self
            .environment
            .iter()
            .map(|cb| cb.to_rs_collision_body())
            .collect();

        RsRobotBody {
            joint_meshes: self.joint_meshes.clone(),
            tool: self.tool_mesh.clone(),
            base,
            collision_environment,
            safety: self.safety.to_rs_safety_distances(),
        }
    }

    /// Get the joint meshes
    pub fn get_joint_meshes(&self) -> &[TriMesh; 6] {
        &self.joint_meshes
    }
}

/// Joint limits wrapper around rs-opw-kinematics Constraints.
/// Supports wrap-around ranges for continuous joints.
#[pyclass]
#[derive(Clone)]
pub struct JointLimits {
    constraints: Constraints,
    /// Whether angles are in degrees (for input/output conversion)
    degrees: bool,
}

#[pymethods]
impl JointLimits {
    /// Create joint limits from (min, max) pairs for each of the 6 joints.
    ///
    /// Args:
    ///     limits: Tuple of 6 (min, max) pairs defining the range for each joint.
    ///             Wrap-around is supported: if min > max, the range wraps through 0.
    ///     sorting_weight: Weight for sorting IK solutions (0.0 = prefer previous joints,
    ///                     1.0 = prefer center of constraints). Default: 0.0
    ///     degrees: Whether the limits are specified in degrees. Default: true
    #[new]
    #[pyo3(signature = (limits, sorting_weight=0.0, degrees=true))]
    fn new(limits: [[f64; 2]; 6], sorting_weight: f64, degrees: bool) -> Self {
        let (from, to) = if degrees {
            // Convert degrees to radians for internal storage
            let from: [f64; 6] = [
                limits[0][0].to_radians(),
                limits[1][0].to_radians(),
                limits[2][0].to_radians(),
                limits[3][0].to_radians(),
                limits[4][0].to_radians(),
                limits[5][0].to_radians(),
            ];
            let to: [f64; 6] = [
                limits[0][1].to_radians(),
                limits[1][1].to_radians(),
                limits[2][1].to_radians(),
                limits[3][1].to_radians(),
                limits[4][1].to_radians(),
                limits[5][1].to_radians(),
            ];
            (from, to)
        } else {
            let from: [f64; 6] = [
                limits[0][0],
                limits[1][0],
                limits[2][0],
                limits[3][0],
                limits[4][0],
                limits[5][0],
            ];
            let to: [f64; 6] = [
                limits[0][1],
                limits[1][1],
                limits[2][1],
                limits[3][1],
                limits[4][1],
                limits[5][1],
            ];
            (from, to)
        };

        let constraints = Constraints::new(from, to, sorting_weight);

        JointLimits { constraints, degrees }
    }

    /// Check if the given joint configuration satisfies all joint limits.
    ///
    /// Args:
    ///     joints: Tuple of 6 joint angles.
    ///
    /// Returns:
    ///     True if all joints are within their limits, False otherwise.
    fn compliant(&self, joints: [f64; 6]) -> bool {
        let joints_rad = if self.degrees {
            [
                joints[0].to_radians(),
                joints[1].to_radians(),
                joints[2].to_radians(),
                joints[3].to_radians(),
                joints[4].to_radians(),
                joints[5].to_radians(),
            ]
        } else {
            joints
        };
        self.constraints.compliant(&joints_rad)
    }

    /// Filter a list of joint configurations, keeping only those within limits.
    ///
    /// Args:
    ///     solutions: List of joint configurations (each a tuple of 6 angles).
    ///
    /// Returns:
    ///     List of joint configurations that satisfy all limits.
    fn filter(&self, solutions: Vec<[f64; 6]>) -> Vec<[f64; 6]> {
        solutions
            .into_iter()
            .filter(|joints| self.compliant(*joints))
            .collect()
    }

    /// Get the lower limits for all joints.
    #[getter]
    fn from_limits(&self) -> [f64; 6] {
        if self.degrees {
            [
                self.constraints.from[0].to_degrees(),
                self.constraints.from[1].to_degrees(),
                self.constraints.from[2].to_degrees(),
                self.constraints.from[3].to_degrees(),
                self.constraints.from[4].to_degrees(),
                self.constraints.from[5].to_degrees(),
            ]
        } else {
            self.constraints.from
        }
    }

    /// Get the upper limits for all joints.
    #[getter]
    fn to_limits(&self) -> [f64; 6] {
        if self.degrees {
            [
                self.constraints.to[0].to_degrees(),
                self.constraints.to[1].to_degrees(),
                self.constraints.to[2].to_degrees(),
                self.constraints.to[3].to_degrees(),
                self.constraints.to[4].to_degrees(),
                self.constraints.to[5].to_degrees(),
            ]
        } else {
            self.constraints.to
        }
    }

    /// Get the center values for all joints (used for sorting).
    #[getter]
    fn centers(&self) -> [f64; 6] {
        if self.degrees {
            [
                self.constraints.centers[0].to_degrees(),
                self.constraints.centers[1].to_degrees(),
                self.constraints.centers[2].to_degrees(),
                self.constraints.centers[3].to_degrees(),
                self.constraints.centers[4].to_degrees(),
                self.constraints.centers[5].to_degrees(),
            ]
        } else {
            self.constraints.centers
        }
    }

    /// Get the sorting weight.
    #[getter]
    fn sorting_weight(&self) -> f64 {
        self.constraints.sorting_weight
    }

    fn __repr__(&self) -> String {
        let limits_str = (0..6)
            .map(|i| {
                if self.degrees {
                    format!(
                        "({:.1}, {:.1})",
                        self.constraints.from[i].to_degrees(),
                        self.constraints.to[i].to_degrees()
                    )
                } else {
                    format!(
                        "({:.4}, {:.4})",
                        self.constraints.from[i], self.constraints.to[i]
                    )
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "JointLimits(limits=[{}], sorting_weight={}, degrees={})",
            limits_str, self.constraints.sorting_weight, self.degrees
        )
    }
}

impl JointLimits {
    /// Get a reference to the internal Constraints for use by Robot
    pub fn get_constraints(&self) -> &Constraints {
        &self.constraints
    }

    /// Check if degrees mode is enabled
    pub fn uses_degrees(&self) -> bool {
        self.degrees
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
    // Collision detection support
    robot_body: Option<rs_opw_kinematics::collisions::RobotBody>,
    joint_limits: Option<JointLimits>,
}

#[pymethods]
impl Robot {
    #[new]
    #[pyo3(signature = (kinematic_model, euler_convention, ee_rotation=None, ee_translation=None, robot_body=None, joint_limits=None))]
    fn new(
        kinematic_model: KinematicModel,
        euler_convention: EulerConvention,
        ee_rotation: Option<[f64; 3]>,
        ee_translation: Option<[f64; 3]>,
        robot_body: Option<RobotBody>,
        joint_limits: Option<JointLimits>,
    ) -> PyResult<Self> {
        let robot = kinematic_model.to_opw_kinematics(euler_convention.degrees);
        let has_parallelogram = kinematic_model.has_parallelogram;
        let degrees = euler_convention.degrees;

        // Initialize the internal rotation matrix to identity as a placeholder
        let _ee_rotation_matrix = Rotation3::identity();

        let _internal_euler_convention = EulerConvention::new("XYZ".to_string(), false, degrees)?;

        // Convert RobotBody to rs-opw-kinematics RobotBody if provided
        let rs_robot_body = robot_body.map(|rb| rb.to_rs_robot_body());

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
            robot_body: rs_robot_body,
            joint_limits,
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

    /// Get end-effector rotation as quaternion (w, x, y, z) - scalar first
    #[getter]
    fn get_ee_rotation_quat(&self) -> [f64; 4] {
        let quat = UnitQuaternion::from_rotation_matrix(&self._ee_rotation_matrix);
        // Ensure w is non-negative for consistency
        if quat.w < 0.0 {
            [-quat.w, -quat.i, -quat.j, -quat.k]
        } else {
            [quat.w, quat.i, quat.j, quat.k]
        }
    }

    /// Set end-effector rotation from quaternion (w, x, y, z) - scalar first
    #[setter]
    fn set_ee_rotation_quat(&mut self, quat: [f64; 4]) {
        let unit_quat = UnitQuaternion::from_quaternion(Quaternion::new(quat[0], quat[1], quat[2], quat[3]));
        self._ee_rotation_matrix = unit_quat.to_rotation_matrix();
    }

    #[getter]
    fn get_ee_translation(&self) -> [f64; 3] {
        self.ee_translation.into()
    }
    #[setter]
    fn set_ee_translation(&mut self, ee_translation: [f64; 3]) {
        self.ee_translation = ee_translation.into();
    }

    /// Forward kinematics: calculates the pose for given joints (returns Euler angles)
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

    /// Forward kinematics with quaternion output: calculates the pose for given joints
    /// Returns: ((x, y, z), (w, qx, qy, qz)) - position and quaternion (scalar first)
    fn forward_quat(&self, mut joints: [f64; 6]) -> ([f64; 3], [f64; 4]) {
        if self.has_parallelogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let combined_rotation = pose.rotation.to_rotation_matrix() * self._ee_rotation_matrix;
        let translation = pose.translation.vector + combined_rotation * self.ee_translation;

        // Convert rotation matrix to quaternion
        let quat = UnitQuaternion::from_rotation_matrix(&combined_rotation);
        // Ensure w is non-negative for consistency
        let quat = if quat.w < 0.0 {
            UnitQuaternion::from_quaternion(Quaternion::new(-quat.w, -quat.i, -quat.j, -quat.k))
        } else {
            quat
        };

        (translation.into(), [quat.w, quat.i, quat.j, quat.k])
    }

    /// Forward kinematics with rotation matrix output
    /// Returns: ((x, y, z), [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])
    fn forward_matrix(&self, mut joints: [f64; 6]) -> ([f64; 3], [[f64; 3]; 3]) {
        if self.has_parallelogram {
            joints[2] += joints[1];
        }
        if self.euler_convention.degrees {
            joints.iter_mut().for_each(|x| *x = x.to_radians());
        }
        let pose: Pose = self.robot.forward(&joints);
        let combined_rotation = pose.rotation.to_rotation_matrix() * self._ee_rotation_matrix;
        let translation = pose.translation.vector + combined_rotation * self.ee_translation;

        let matrix = [
            [
                combined_rotation[(0, 0)],
                combined_rotation[(0, 1)],
                combined_rotation[(0, 2)],
            ],
            [
                combined_rotation[(1, 0)],
                combined_rotation[(1, 1)],
                combined_rotation[(1, 2)],
            ],
            [
                combined_rotation[(2, 0)],
                combined_rotation[(2, 1)],
                combined_rotation[(2, 2)],
            ],
        ];

        (translation.into(), matrix)
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

    /// Inverse kinematics with quaternion input
    /// Takes: position (x, y, z) and quaternion (w, qx, qy, qz) - scalar first
    #[pyo3(signature = (position, quaternion, current_joints=None))]
    fn inverse_quat(
        &self,
        position: [f64; 3],
        quaternion: [f64; 4],
        current_joints: Option<[f64; 6]>,
    ) -> Vec<[f64; 6]> {
        // Create rotation from quaternion (w, x, y, z)
        let quat =
            UnitQuaternion::from_quaternion(Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3]));
        let rotation_matrix = quat.to_rotation_matrix();

        let rotated_ee_translation = rotation_matrix * Vector3::from(self.ee_translation);
        let translation = Translation3::from(Vector3::from(position) - rotated_ee_translation);
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

    /// Inverse kinematics with rotation matrix input
    /// Takes: position (x, y, z) and 3x3 rotation matrix
    #[pyo3(signature = (position, rotation_matrix, current_joints=None))]
    fn inverse_matrix(
        &self,
        position: [f64; 3],
        rotation_matrix: [[f64; 3]; 3],
        current_joints: Option<[f64; 6]>,
    ) -> Vec<[f64; 6]> {
        // Create rotation from matrix (row-major input)
        let rot = Rotation3::from_matrix_unchecked(nalgebra::Matrix3::from([
            [rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]],
            [rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1]],
            [rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]],
        ]));

        let rotated_ee_translation = rot * Vector3::from(self.ee_translation);
        let translation = Translation3::from(Vector3::from(position) - rotated_ee_translation);
        let rotation = rot * self._ee_rotation_matrix.inverse();
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

    /// Batch inverse kinematics using NumPy arrays.
    /// Input: poses array of shape (n, 6) with columns [X, Y, Z, A, B, C]
    /// Output: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Returns NaN for rows where no solution is found.
    #[pyo3(signature = (poses, current_joints=None))]
    fn batch_inverse<'py>(
        &self,
        py: Python<'py>,
        poses: PyReadonlyArray2<'py, f64>,
        mut current_joints: Option<[f64; 6]>,
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

            let pose = ([row[0], row[1], row[2]], [row[3], row[4], row[5]]);

            let solutions = self.inverse(pose, current_joints);
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

    /// Batch forward kinematics using NumPy arrays.
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Output: poses array of shape (n, 6) with columns [X, Y, Z, A, B, C]
    /// Returns NaN for rows with NaN input values.
    #[pyo3(signature = (joints,))]
    fn batch_forward<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let joints_array = joints.as_array();
        let n = joints_array.nrows();

        let mut results: Vec<f64> = Vec::with_capacity(n * 6);

        for i in 0..n {
            let row = joints_array.row(i);

            // Check for NaN values in input (treat as missing)
            if row.iter().any(|v| v.is_nan()) {
                results.extend_from_slice(&[f64::NAN; 6]);
                continue;
            }

            let joints_input = [row[0], row[1], row[2], row[3], row[4], row[5]];
            let (translation, rotation) = self.forward(joints_input);

            results.push(translation[0]);
            results.push(translation[1]);
            results.push(translation[2]);
            results.push(rotation[0]);
            results.push(rotation[1]);
            results.push(rotation[2]);
        }

        // Convert flat Vec to 2D array (n, 6)
        let result_array = Array2::from_shape_vec((n, 6), results)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok(result_array.into_pyarray(py).into())
    }

    /// Batch forward kinematics with quaternion output using NumPy arrays.
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Output: poses array of shape (n, 7) with columns [X, Y, Z, W, QX, QY, QZ]
    /// Returns NaN for rows with NaN input values.
    #[pyo3(signature = (joints,))]
    fn batch_forward_quat<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let joints_array = joints.as_array();
        let n = joints_array.nrows();

        let mut results: Vec<f64> = Vec::with_capacity(n * 7);

        for i in 0..n {
            let row = joints_array.row(i);

            // Check for NaN values in input (treat as missing)
            if row.iter().any(|v| v.is_nan()) {
                results.extend_from_slice(&[f64::NAN; 7]);
                continue;
            }

            let joints_input = [row[0], row[1], row[2], row[3], row[4], row[5]];
            let (translation, quat) = self.forward_quat(joints_input);

            results.push(translation[0]);
            results.push(translation[1]);
            results.push(translation[2]);
            results.push(quat[0]); // w
            results.push(quat[1]); // x
            results.push(quat[2]); // y
            results.push(quat[3]); // z
        }

        // Convert flat Vec to 2D array (n, 7)
        let result_array = Array2::from_shape_vec((n, 7), results)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok(result_array.into_pyarray(py).into())
    }

    /// Batch inverse kinematics with quaternion input using NumPy arrays.
    /// Input: poses array of shape (n, 7) with columns [X, Y, Z, W, QX, QY, QZ]
    /// Output: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Returns NaN for rows where no solution is found.
    #[pyo3(signature = (poses, current_joints=None))]
    fn batch_inverse_quat<'py>(
        &self,
        py: Python<'py>,
        poses: PyReadonlyArray2<'py, f64>,
        mut current_joints: Option<[f64; 6]>,
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

            let position = [row[0], row[1], row[2]];
            let quaternion = [row[3], row[4], row[5], row[6]]; // w, x, y, z

            let solutions = self.inverse_quat(position, quaternion, current_joints);
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

    // ==================== Collision Detection Methods ====================

    /// Check if the robot has collision geometry configured.
    #[getter]
    fn has_collision_geometry(&self) -> bool {
        self.robot_body.is_some()
    }

    /// Check if the robot has joint limits configured.
    #[getter]
    fn has_joint_limits(&self) -> bool {
        self.joint_limits.is_some()
    }

    /// Check if a joint configuration collides with itself or the environment.
    ///
    /// Requires robot_body to be configured. Returns false if no collision geometry.
    ///
    /// Args:
    ///     joints: Tuple of 6 joint angles (in the EulerConvention's units)
    ///
    /// Returns:
    ///     True if collision detected, False otherwise
    fn collides(&self, joints: [f64; 6]) -> bool {
        match &self.robot_body {
            Some(body) => {
                let joints_rad = if self.euler_convention.degrees {
                    [
                        joints[0].to_radians(),
                        joints[1].to_radians(),
                        joints[2].to_radians(),
                        joints[3].to_radians(),
                        joints[4].to_radians(),
                        joints[5].to_radians(),
                    ]
                } else {
                    joints
                };
                body.collides(&joints_rad, &self.robot)
            }
            None => false,
        }
    }

    /// Get detailed collision information for a joint configuration.
    ///
    /// Requires robot_body to be configured. Returns empty list if no collision geometry.
    ///
    /// Args:
    ///     joints: Tuple of 6 joint angles (in the EulerConvention's units)
    ///
    /// Returns:
    ///     List of (i, j) tuples where i and j are indices of colliding bodies.
    ///     Joint indices: 0-5 for J1-J6, 100 for tool, 101 for base, 102+ for environment.
    fn collision_details(&self, joints: [f64; 6]) -> Vec<(usize, usize)> {
        match &self.robot_body {
            Some(body) => {
                let joints_rad = if self.euler_convention.degrees {
                    [
                        joints[0].to_radians(),
                        joints[1].to_radians(),
                        joints[2].to_radians(),
                        joints[3].to_radians(),
                        joints[4].to_radians(),
                        joints[5].to_radians(),
                    ]
                } else {
                    joints
                };
                body.collision_details(&joints_rad, &self.robot)
            }
            None => Vec::new(),
        }
    }

    /// Check if a joint configuration is within the configured joint limits.
    ///
    /// Requires joint_limits to be configured. Returns true if no limits configured.
    ///
    /// Args:
    ///     joints: Tuple of 6 joint angles (in the EulerConvention's units)
    ///
    /// Returns:
    ///     True if all joints are within limits, False otherwise
    fn joints_compliant(&self, joints: [f64; 6]) -> bool {
        match &self.joint_limits {
            Some(limits) => limits.compliant(joints),
            None => true,
        }
    }

    /// Check for objects within a specified safety distance.
    ///
    /// Requires robot_body to be configured. Returns empty list if no collision geometry.
    ///
    /// Args:
    ///     joints: Tuple of 6 joint angles (in the EulerConvention's units)
    ///     safety: SafetyDistances configuration to use for distance checking
    ///
    /// Returns:
    ///     List of (i, j) tuples where i and j are indices of bodies within the safety distance.
    fn near(&self, joints: [f64; 6], safety: SafetyDistances) -> Vec<(usize, usize)> {
        match &self.robot_body {
            Some(body) => {
                let joints_rad = if self.euler_convention.degrees {
                    [
                        joints[0].to_radians(),
                        joints[1].to_radians(),
                        joints[2].to_radians(),
                        joints[3].to_radians(),
                        joints[4].to_radians(),
                        joints[5].to_radians(),
                    ]
                } else {
                    joints
                };
                body.near(&joints_rad, &self.robot, &safety.to_rs_safety_distances())
            }
            None => Vec::new(),
        }
    }

    /// Batch collision checking using NumPy arrays.
    ///
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Output: boolean array of shape (n,) where True indicates collision
    ///
    /// Requires robot_body to be configured. Returns all False if no collision geometry.
    /// NaN input rows are treated as non-colliding (returns False).
    #[pyo3(signature = (joints,))]
    fn batch_collides<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<numpy::PyArray1<bool>>> {
        let joints_array = joints.as_array();
        let n = joints_array.nrows();

        let results: Vec<bool> = match &self.robot_body {
            Some(body) => {
                (0..n)
                    .map(|i| {
                        let row = joints_array.row(i);

                        // Check for NaN values - treat as non-colliding
                        if row.iter().any(|v| v.is_nan()) {
                            return false;
                        }

                        let joints_input = if self.euler_convention.degrees {
                            [
                                row[0].to_radians(),
                                row[1].to_radians(),
                                row[2].to_radians(),
                                row[3].to_radians(),
                                row[4].to_radians(),
                                row[5].to_radians(),
                            ]
                        } else {
                            [row[0], row[1], row[2], row[3], row[4], row[5]]
                        };

                        body.collides(&joints_input, &self.robot)
                    })
                    .collect()
            }
            None => vec![false; n],
        };

        Ok(numpy::PyArray1::from_vec(py, results).into())
    }

    /// Batch joint limits checking using NumPy arrays.
    ///
    /// Input: joints array of shape (n, 6) with columns [J1, J2, J3, J4, J5, J6]
    /// Output: boolean array of shape (n,) where True indicates within limits
    ///
    /// Requires joint_limits to be configured. Returns all True if no limits configured.
    /// NaN input rows are treated as non-compliant (returns False).
    #[pyo3(signature = (joints,))]
    fn batch_joints_compliant<'py>(
        &self,
        py: Python<'py>,
        joints: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<numpy::PyArray1<bool>>> {
        let joints_array = joints.as_array();
        let n = joints_array.nrows();

        let results: Vec<bool> = match &self.joint_limits {
            Some(limits) => {
                (0..n)
                    .map(|i| {
                        let row = joints_array.row(i);

                        // Check for NaN values - treat as non-compliant
                        if row.iter().any(|v| v.is_nan()) {
                            return false;
                        }

                        let joints_input = [row[0], row[1], row[2], row[3], row[4], row[5]];
                        limits.compliant(joints_input)
                    })
                    .collect()
            }
            None => vec![true; n],
        };

        Ok(numpy::PyArray1::from_vec(py, results).into())
    }
}

/// Module initialization for Python
#[pymodule(name = "_internal")]
fn py_opw_kinematics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EulerConvention>()?;
    m.add_class::<KinematicModel>()?;
    m.add_class::<Robot>()?;
    m.add_class::<JointLimits>()?;
    m.add_class::<CheckMode>()?;
    m.add_class::<SafetyDistances>()?;
    m.add_class::<CollisionBody>()?;
    m.add_class::<RobotBody>()?;
    Ok(())
}
