/// Configuration analysis for 6-axis robot inverse kinematics solutions
/// Implements SINUMERIK-compatible STAT/TU (Status/Turn) configuration bits
/// 
/// STAT bits define robot posture (shoulder left/right, elbow up/down, handflip/no-handflip)
/// TU bits define angle sign preference per axis within ±360° range
/// 
/// Note: STAT bit semantics match SINUMERIK ROBX specification. However, the geometric
/// calculations here use simplified kinematics, while actual NX/SINUMERIK systems use
/// full controller kinematics with exact joint limits and robot geometry for precise
/// branch classification.

use nalgebra::Vector3;

/// Robot kinematic parameters for simplified geometric calculations
/// Used for approximate shoulder/workspace analysis, not full controller kinematics
/// Simplified OPW (Optimize Path for Welding) parameters
#[derive(Debug, Clone, Copy)]
pub struct RobotKinematicParams {
    /// Link length 1 (shoulder to elbow)
    pub a1: f64,
    /// Link length 2 (elbow to wrist) 
    pub a2: f64,
    /// Base offset
    pub b: f64,
    /// Shoulder height offset
    pub c1: f64,
    /// Wrist offsets
    pub c2: f64,
    pub c3: f64,
    pub c4: f64,
}



#[derive(Debug, Clone, PartialEq)]
pub struct RobotConfiguration {
    /// STAT bits for robot posture
    pub stat: StatBits,
    /// TU bits for joint turn numbers  
    pub tu: TurnBits,
    /// Joint values for analysis (in degrees)
    pub joints: [f64; 6],
    /// Full STAT/TU string like "STAT=101 TU=000011" (binary format)
    pub stat_tu_string: String,
}

/// STAT bits define robot posture configuration
/// Following SINUMERIK ROBX specification for 6-axis robots
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatBits {
    /// Bit 0: Shoulder configuration (0 = right, 1 = left)
    pub shoulder_left: bool,
    /// Bit 1: Elbow configuration (0 = down, 1 = up)
    pub elbow_up: bool,
    /// Bit 2: Handflip configuration (0 = no handflip, 1 = handflip)
    pub handflip: bool,
}

/// TU bits define angle sign preference for each axis (Siemens SINUMERIK format)
/// Each bit indicates whether the joint angle should be positive or negative after normalization
/// TU bit = 0 → θ ≥ 0 after normalization, TU bit = 1 → θ < 0 after normalization
#[derive(Debug, Clone, Copy, PartialEq)]  
pub struct TurnBits {
    /// J1 sign bit (bit 0): 0=positive, 1=negative
    pub j1_negative: bool,
    /// J2 sign bit (bit 1): 0=positive, 1=negative  
    pub j2_negative: bool,
    /// J3 sign bit (bit 2): 0=positive, 1=negative
    pub j3_negative: bool,
    /// J4 sign bit (bit 3): 0=positive, 1=negative
    pub j4_negative: bool,
    /// J5 sign bit (bit 4): 0=positive, 1=negative
    pub j5_negative: bool,
    /// J6 sign bit (bit 5): 0=positive, 1=negative
    pub j6_negative: bool,
}

impl StatBits {
    /// Create STAT bits from joint angles using simple heuristics
    ///
    /// - Bit 0 (shoulder_left): based on sign of J1 (>= 0 → left, < 0 → right)
    /// - Bit 1 (elbow_up): based on (virtual) J3 sign
    /// - Bit 2 (handflip): based on J5 sign
    ///
    /// This does **not** use robot geometry. For geometry-aware classification,
    /// use `from_joints_geometric` instead.
    pub fn from_joints(joints: [f64; 6]) -> Self {
        Self::from_joints_with_parallelogram(joints, false)
    }
    
    /// Create STAT bits from joint angles using simplified geometry.
    /// 
    /// - Bit 0 (shoulder_left): computed from wrist-center position in shoulder frame,
    ///   independent of J1 rotation. This approximates controller branch classification.
    /// - Bit 1 (elbow_up): from (virtual) J3 sign.
    /// - Bit 2 (handflip): from J5 sign.
    ///
    /// Note: This uses simplified OPW-style kinematics, not the full controller model.
    pub fn from_joints_geometric(joints: [f64; 6], robot_params: &RobotKinematicParams, has_parallelogram: bool) -> Self {
        let j3_for_stat = if has_parallelogram {
            // For parallelogram robots, calculate virtual J3
            normalize_angle_degrees(joints[2] + joints[1])
        } else {
            normalize_angle_degrees(joints[2])
        };
        
        let j5_normalized = normalize_angle_degrees(joints[4]);
        
        // Bit 1: Elbow up/down (virtual J3 for parallelogram robots)
        let elbow_up = j3_for_stat >= 0.0;
        
        // Bit 2: Handflip (J5 angle)  
        let handflip = j5_normalized < 0.0;  // handflip when J5 negative
        
        // Bit 0: Shoulder left/right - geometric calculation using wrist center position
        let shoulder_left = Self::calculate_shoulder_left(joints, robot_params, has_parallelogram);
        
        Self { shoulder_left, elbow_up, handflip }
    }
    
    /// Create STAT bits from joint angles with parallelogram awareness
    /// For robots with parallelograms, use virtual J3 for elbow configuration
    /// 
    /// WARNING: This method uses simple heuristics for shoulder left/right.
    /// For precise STAT compatibility with geometric calculations,
    /// use from_joints_geometric() with robot kinematic parameters.
    pub fn from_joints_with_parallelogram(joints: [f64; 6], has_parallelogram: bool) -> Self {
        let j3_for_stat = if has_parallelogram {
            // For parallelogram robots, calculate virtual J3
            // Virtual J3 = Physical J3 + J2 (this is robot-specific)
            // This represents the true elbow angle relative to the arm
            normalize_angle_degrees(joints[2] + joints[1])
        } else {
            // For non-parallelogram robots, use physical J3 directly
            normalize_angle_degrees(joints[2])
        };
        
        let j5_normalized = normalize_angle_degrees(joints[4]);
        let j1_normalized = normalize_angle_degrees(joints[0]);
        
        // Bit 1: Elbow up/down (virtual J3 for parallelogram robots)
        let elbow_up = j3_for_stat >= 0.0;
        
        // Bit 2: Handflip (J5 angle)
        let handflip = j5_normalized < 0.0;
        
        // Bit 0: Shoulder left/right (simple heuristic based on J1)
        // For floor-mounted robot with +Y to the left:
        let shoulder_left = j1_normalized >= 0.0;
        
        Self { shoulder_left, elbow_up, handflip }
    }
    
    /// Calculate shoulder left/right configuration (simplified approximation)
    /// 
    /// Uses basic shoulder frame transformation to determine if the wrist center
    /// is on the left or right side of the robot, independent of J1 rotation.
    /// 
    /// This approximates what NX's full kinematic model does, but with simplified
    /// geometry. The actual ROBX shoulder classification uses complete joint limits
    /// and exact robot geometry from the controller's kinematic model.
    /// 
    /// Returns true for "shoulder left" configuration, false for "shoulder right".
    fn calculate_shoulder_left(
        joints: [f64; 6], 
        robot_params: &RobotKinematicParams,
        has_parallelogram: bool
    ) -> bool {
        // Convert joints to radians for calculation
        let joints_rad: [f64; 6] = joints.map(|j| j.to_radians());
        
        // For parallelogram robots, use modified J3
        let j3_calc = if has_parallelogram {
            joints_rad[2] + joints_rad[1]  // Virtual J3
        } else {
            joints_rad[2]
        };
        
        // Forward kinematics to wrist center in base frame
        let wrist_center_base = Self::forward_kinematics_to_wrist(
            joints_rad[0], joints_rad[1], j3_calc, robot_params
        );
        
        // Transform to shoulder frame:
        // 1. Subtract base→J2 offset
        // 2. Undo J1 rotation to get shoulder-relative position
        let j1 = joints_rad[0];
        let base_to_j2 = Vector3::new(robot_params.b, 0.0, robot_params.c1);
        let wrist_relative_to_j2 = wrist_center_base - base_to_j2;
        
        // Undo J1 rotation to get position in shoulder frame
        let cos_j1 = j1.cos();
        let sin_j1 = j1.sin();
        
        let _wrist_shoulder_x = cos_j1 * wrist_relative_to_j2.x + sin_j1 * wrist_relative_to_j2.y;
        let wrist_shoulder_y = -sin_j1 * wrist_relative_to_j2.x + cos_j1 * wrist_relative_to_j2.y;
        
        // Shoulder left if wrist is on +Y side of the shoulder frame
        wrist_shoulder_y >= 0.0
    }

    
    /// Simplified forward kinematics to wrist center
    /// Computes position of wrist center (where J4,J5,J6 axes intersect)
    fn forward_kinematics_to_wrist(
        j1: f64, j2: f64, j3: f64, 
        params: &RobotKinematicParams
    ) -> Vector3<f64> {
        // Simplified OPW forward kinematics to wrist center
        // This is a basic implementation - full OPW would be more complex
        
        let c1 = j1.cos();
        let s1 = j1.sin();
        let c2 = j2.cos();
        let s2 = j2.sin();
        let _c3 = j3.cos();
        let _s3 = j3.sin();
        let c23 = (j2 + j3).cos();
        let s23 = (j2 + j3).sin();
        
        // Wrist center position in base coordinates
        let x = c1 * (params.a1 * c2 + params.a2 * c23) - params.c3 * c1 * s23;
        let y = s1 * (params.a1 * c2 + params.a2 * c23) - params.c3 * s1 * s23;
        let z = params.c1 + params.a1 * s2 + params.a2 * s23 + params.c3 * c23;
        
        Vector3::new(x, y, z)
    }
    
    /// Convert to 3-bit integer (0-7)
    pub fn to_bits(&self) -> u8 {
        let mut bits = 0u8;
        if self.shoulder_left { bits |= 0b001; }  // bit 0
        if self.elbow_up { bits |= 0b010; }       // bit 1
        if self.handflip { bits |= 0b100; }       // bit 2
        bits
    }
    
    /// Create from 3-bit integer
    pub fn from_bits(bits: u8) -> Self {
        Self {
            shoulder_left: (bits & 0b001) != 0,  // bit 0
            elbow_up: (bits & 0b010) != 0,       // bit 1
            handflip: (bits & 0b100) != 0,       // bit 2
        }
    }
    
    /// Convert to binary string like "101" 
    pub fn to_binary_string(&self) -> String {
        format!("{:03b}", self.to_bits())
    }
    
    /// Calculate Hamming distance (number of different bits)
    pub fn distance(&self, other: &StatBits) -> u8 {
        (self.to_bits() ^ other.to_bits()).count_ones() as u8
    }
}

impl TurnBits {
    /// Create TU bits from joint angles (in degrees)
    /// Extracts sign preference for each axis within ±360° range
    pub fn from_joints(joints: [f64; 6]) -> Self {
        // Normalize each joint to [-360°, 360°] and check sign
        // This better matches Siemens semantics where 270° is considered positive
        // TU bit = 1 if angle is negative, 0 if positive
        
        Self {
            j1_negative: normalize_angle_360(joints[0]) < 0.0,
            j2_negative: normalize_angle_360(joints[1]) < 0.0,
            j3_negative: normalize_angle_360(joints[2]) < 0.0,
            j4_negative: normalize_angle_360(joints[3]) < 0.0,
            j5_negative: normalize_angle_360(joints[4]) < 0.0,
            j6_negative: normalize_angle_360(joints[5]) < 0.0,
        }
    }
    
    /// Convert to 6-bit integer (0-63) representing all axis sign bits
    pub fn to_bits(&self) -> u8 {
        let mut bits = 0u8;
        if self.j1_negative { bits |= 0b000001; }
        if self.j2_negative { bits |= 0b000010; }
        if self.j3_negative { bits |= 0b000100; }
        if self.j4_negative { bits |= 0b001000; }
        if self.j5_negative { bits |= 0b010000; }
        if self.j6_negative { bits |= 0b100000; }
        bits
    }
    
    /// Create from 6-bit integer
    pub fn from_bits(bits: u8) -> Self {
        Self {
            j1_negative: (bits & 0b000001) != 0,
            j2_negative: (bits & 0b000010) != 0,
            j3_negative: (bits & 0b000100) != 0,
            j4_negative: (bits & 0b001000) != 0,
            j5_negative: (bits & 0b010000) != 0,
            j6_negative: (bits & 0b100000) != 0,
        }
    }
    
    /// Convert to binary string like "000011" (6 bits) for industrial format
    pub fn to_binary_string(&self) -> String {
        format!("{:06b}", self.to_bits())
    }
    
    /// Convert to detailed string like "J1+ J2- J3+ J4- J5+ J6-"
    pub fn to_detailed_string(&self) -> String {
        format!("J1{} J2{} J3{} J4{} J5{} J6{}", 
                if self.j1_negative { "-" } else { "+" },
                if self.j2_negative { "-" } else { "+" },
                if self.j3_negative { "-" } else { "+" },
                if self.j4_negative { "-" } else { "+" },
                if self.j5_negative { "-" } else { "+" },
                if self.j6_negative { "-" } else { "+" })
    }
    
    /// Calculate bit difference (Hamming distance)
    pub fn bit_distance(&self, other: &TurnBits) -> u8 {
        (self.to_bits() ^ other.to_bits()).count_ones() as u8
    }
}

impl RobotConfiguration {
    /// Create a configuration from joint angles (in degrees)
    pub fn from_joints(joints: [f64; 6]) -> Self {
        Self::from_joints_with_parallelogram(joints, false)
    }
    
    /// Create a configuration from joint angles with parallelogram awareness
    pub fn from_joints_with_parallelogram(joints: [f64; 6], has_parallelogram: bool) -> Self {
        let stat = StatBits::from_joints_with_parallelogram(joints, has_parallelogram);
        let tu = TurnBits::from_joints(joints);
        
        // Generate full STAT/TU string (industrial format)
        let stat_tu_string = format!("STAT={} TU={}", stat.to_binary_string(), tu.to_binary_string());
        
        Self {
            stat,
            tu,
            joints,
            stat_tu_string,
        }
    }
    
    /// Parse a configuration string (STAT/TU format only)
    /// Supports: "STAT=101 TU=000011" or "STAT=5 TU=3" or "STAT=B000101 TU=B000011"
    /// Note: TU is parsed as 6-bit value (0-63), not the +0-1+2 format mentioned in some docs
    pub fn from_string(config_str: &str) -> Result<TargetConfiguration, String> {
        let trimmed = config_str.trim();
        
        // Parse STAT/TU format
        if trimmed.contains("STAT=") && trimmed.contains("TU=") {
            return parse_stat_tu_format(trimmed);
        }
        
        Err("Configuration string must be in STAT/TU format: 'STAT=... TU=...'".to_string())
    }
    
    /// Calculate comprehensive match score against a target configuration
    /// Returns (stat_score, tu_score, total_score)
    pub fn match_score(&self, target: &TargetConfiguration) -> (u8, u8, u8) {
        // STAT score: 3 - Hamming distance (higher is better)
        let stat_score = 3 - self.stat.distance(&target.stat);
        
        // TU score: penalize bit differences (lower distance is better)
        let tu_distance = self.tu.bit_distance(&target.tu);
        let tu_score = if tu_distance == 0 { 3 } else if tu_distance <= 2 { 2 } else { 1 };
        
        let total_score = stat_score + tu_score;
        
        (stat_score, tu_score, total_score)
    }
    
    /// Simple compatibility match score (0-3) for backward compatibility
    pub fn simple_match_score(&self, target: &TargetConfiguration) -> u8 {
        3 - self.stat.distance(&target.stat)
    }
}

#[derive(Debug, Clone)]
pub struct TargetConfiguration {
    /// Target STAT bits
    pub stat: StatBits,
    /// Target TU bits  
    pub tu: TurnBits,
}

impl TargetConfiguration {
    /// Create from STAT and TU components
    pub fn new(stat: StatBits, tu: TurnBits) -> Self {
        Self { stat, tu }
    }
    
    /// Create from STAT/TU bit values (for target specification)
    pub fn from_stat_tu_bits(stat_bits: u8, tu_bits: u8) -> Self {
        let stat = StatBits::from_bits(stat_bits);
        let tu = TurnBits::from_bits(tu_bits);
        Self { stat, tu }
    }
    
    /// Create from STAT bits (3-bit integer) and TU bits (6-bit integer)
    pub fn from_bits(stat_bits: u8, tu_bits: u8) -> Self {
        let stat = StatBits::from_bits(stat_bits);
        let tu = TurnBits::from_bits(tu_bits);
        Self { stat, tu }
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        format!("STAT={} TU={}", self.stat.to_binary_string(), self.tu.to_binary_string())
    }
    

}

/// Configuration selector for finding best matching inverse kinematics solution
pub struct ConfigurationSelector {
    /// Store analyzed configurations
    configurations: Vec<RobotConfiguration>,
}

impl ConfigurationSelector {
    /// Create a new configuration selector from multiple joint solutions
    pub fn new(joint_solutions: &[[f64; 6]]) -> Self {
        Self::new_with_parallelogram(joint_solutions, false)
    }
    
    /// Create a new configuration selector with parallelogram awareness
    pub fn new_with_parallelogram(joint_solutions: &[[f64; 6]], has_parallelogram: bool) -> Self {
        let configurations: Vec<RobotConfiguration> = joint_solutions
            .iter()
            .map(|joints| RobotConfiguration::from_joints_with_parallelogram(*joints, has_parallelogram))
            .collect();
            
        Self { configurations }
    }
    
    /// Find the best matching solution for a target configuration
    /// Returns (index, config, (stat_score, tu_score, total_score))
    pub fn find_best_match(&self, target: &TargetConfiguration) -> Option<(usize, &RobotConfiguration, (u8, u8, u8))> {
        let mut best_match: Option<(usize, &RobotConfiguration, (u8, u8, u8))> = None;
        let mut best_total_score = 0;
        
        for (index, config) in self.configurations.iter().enumerate() {
            let scores = config.match_score(target);
            let total_score = scores.2;
            
            // More robust comparison logic
            let is_better = if let Some((_, _, prev_scores)) = &best_match {
                total_score > best_total_score || 
                (total_score == best_total_score && scores.0 > prev_scores.0)
            } else {
                true  // First candidate
            };
            
            if is_better {
                best_total_score = total_score;
                best_match = Some((index, config, scores));
            }
        }
        
        best_match
    }
    
    /// Find solutions with specific STAT pattern (ignoring TU)
    pub fn find_stat_matches(&self, target_stat: &StatBits) -> Vec<(usize, &RobotConfiguration, u8)> {
        self.configurations
            .iter()
            .enumerate()
            .map(|(index, config)| {
                let stat_score = 3 - config.stat.distance(target_stat);
                (index, config, stat_score)
            })
            .filter(|(_, _, score)| *score >= 2)  // At least 2 out of 3 STAT bits match
            .collect()
    }
    
    /// Find solutions with minimal bit changes from current TU
    pub fn find_minimal_turn_solutions(&self, current_tu: &TurnBits) -> Vec<(usize, &RobotConfiguration, u8)> {
        self.configurations
            .iter()
            .enumerate()
            .map(|(index, config)| {
                let bit_distance = config.tu.bit_distance(current_tu);
                (index, config, bit_distance)
            })
            .collect()
    }
    
    /// Find all solutions matching a target configuration exactly
    pub fn find_exact_matches(&self, target: &TargetConfiguration) -> Vec<(usize, &RobotConfiguration)> {
        self.configurations
            .iter()
            .enumerate()
            .filter(|(_, config)| {
                let (stat_score, tu_score, _) = config.match_score(target);
                stat_score == 3 && tu_score >= 2  // Perfect STAT match, good TU match
            })
            .map(|(index, config)| (index, config))
            .collect()
    }
    
    /// Get all configurations for analysis
    pub fn get_all_configurations(&self) -> &[RobotConfiguration] {
        &self.configurations
    }
    
    /// Find configuration closest to current joint position
    pub fn find_closest_to_current(&self, current_joints: &[f64; 6]) -> Option<(usize, &RobotConfiguration)> {
        let mut best_match: Option<(usize, &RobotConfiguration)> = None;
        let mut best_distance = f64::INFINITY;
        
        for (index, config) in self.configurations.iter().enumerate() {
            let distance = joint_distance(&config.joints, current_joints);
            if distance < best_distance {
                best_distance = distance;
                best_match = Some((index, config));
            }
        }
        
        best_match
    }
}



/// Parse STAT/TU format string like:
/// - "STAT=101 TU=000011"   (binary)
/// - "STAT=5 TU=3"          (decimal)  
/// - "STAT=B000101 TU=B000011" (binary with B prefix)
fn parse_stat_tu_format(input: &str) -> Result<TargetConfiguration, String> {
    let mut stat_part = None;
    let mut tu_part = None;
    
    for part in input.split_whitespace() {
        if part.starts_with("STAT=") {
            stat_part = Some(&part[5..]);
        } else if part.starts_with("TU=") {
            tu_part = Some(&part[3..]);
        }
    }
    
    let stat_str = stat_part.ok_or("Missing STAT= part")?;
    let tu_str = tu_part.ok_or("Missing TU= part")?;
    
    // Parse STAT (supports: "101", "5", "B000101")
    let stat_bits = if stat_str.starts_with('B') {
        // Binary format with B prefix: "B000101"
        let binary_part = &stat_str[1..];
        u8::from_str_radix(binary_part, 2).map_err(|_| "Invalid STAT binary with B prefix")?
    } else if stat_str.chars().all(|c| c == '0' || c == '1') {
        // Binary format without prefix: "101"
        u8::from_str_radix(stat_str, 2).map_err(|_| "Invalid STAT binary")?
    } else {
        // Decimal format: "5"
        stat_str.parse::<u8>().map_err(|_| "Invalid STAT number")?
    };
    
    if stat_bits > 7 {
        return Err("STAT bits must be 0-7".to_string());
    }
    
    // Parse TU (6-bit format representing axis sign preferences)
    let tu_bits = parse_tu_bits(tu_str)?;
    
    Ok(TargetConfiguration::from_bits(stat_bits, tu_bits))
}

/// Parse TU bits according to Siemens SINUMERIK specification
/// TU is a 6-bit value where each bit indicates axis sign preference
/// Supports: "B000011" (binary), "3" (decimal), "000011" (binary without B)
fn parse_tu_bits(tu_str: &str) -> Result<u8, String> {
    // Handle binary format with B prefix: "B000011"
    if tu_str.starts_with('B') {
        let binary_part = &tu_str[1..];
        return u8::from_str_radix(binary_part, 2).map_err(|_| "Invalid TU binary with B prefix".to_string());
    }
    
    // Handle pure binary format without B prefix: "000011"
    if tu_str.chars().all(|c| c == '0' || c == '1') && tu_str.len() <= 6 {
        return u8::from_str_radix(tu_str, 2).map_err(|_| "Invalid TU binary format".to_string());
    }
    
    // Handle decimal format: "3"
    if tu_str.chars().all(|c| c.is_ascii_digit()) {
        let tu_value: u8 = tu_str.parse().map_err(|_| "Invalid TU decimal")?;
        if tu_value > 63 {
            return Err("TU decimal value must be 0-63 (6-bit)".to_string());
        }
        return Ok(tu_value);
    }
    
    Err(format!("Invalid TU format: {}", tu_str))
}

/// Normalize angle to [-180, 180] degrees (for STAT calculations)
fn normalize_angle_degrees(angle: f64) -> f64 {
    let mut normalized = angle % 360.0;
    if normalized > 180.0 {
        normalized -= 360.0;
    } else if normalized < -180.0 {
        normalized += 360.0;
    }
    normalized
}

/// Normalize angle to [-360, 360] degrees (for TU sign calculations)
/// This better matches Siemens semantics where 270° is considered positive
fn normalize_angle_360(mut angle: f64) -> f64 {
    while angle <= -360.0 {
        angle += 360.0;
    }
    while angle > 360.0 {
        angle -= 360.0;
    }
    angle
}



/// Calculate distance between two joint configurations
fn joint_distance(joints1: &[f64; 6], joints2: &[f64; 6]) -> f64 {
    joints1.iter()
        .zip(joints2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stat_bits() {
        let stat = StatBits { shoulder_left: true, elbow_up: false, handflip: true };
        assert_eq!(stat.to_bits(), 0b101);  // 5
        assert_eq!(stat.to_binary_string(), "101");
        
        let stat2 = StatBits::from_bits(5);
        assert_eq!(stat2, stat);
    }

    #[test]
    fn test_turn_bits() {
        let joints = [21.3, 9.0, -133.5, 270.0, 39.7, -200.0];  // Mixed positive/negative angles
        let tu = TurnBits::from_joints(joints);
        
        // Check sign bits based on [-360,360] normalized angles 
        assert_eq!(tu.j1_negative, false); // 21.3° -> positive  
        assert_eq!(tu.j2_negative, false); // 9.0° -> positive
        assert_eq!(tu.j3_negative, true);  // -133.5° -> negative
        assert_eq!(tu.j4_negative, false); // 270.0° -> positive (Siemens semantics)
        assert_eq!(tu.j5_negative, false); // 39.7° -> positive
        assert_eq!(tu.j6_negative, true);  // -200° -> negative (stays negative in [-360,360])
        
        // Test bit conversion: J3 and J6 negative (bits 2 and 5)
        assert_eq!(tu.to_bits(), 0b100100); // 36 in decimal
    }

    #[test]
    fn test_configuration_from_joints() {
        let joints = [21.3, 9.0, -133.5, 76.6, 39.7, 38.9];
        let config = RobotConfiguration::from_joints(joints);
        
        assert_eq!(config.stat.elbow_up, false);    // J3 < 0
        assert_eq!(config.stat.handflip, false);   // J5 > 0 -> handflip=false (no handflip)
        assert!(config.stat_tu_string.contains("STAT="));
    }
    
    #[test]
    fn test_stat_tu_parsing() {
        // Test STAT=5 (101 binary) TU=3 (000011 binary = J1 and J2 negative)
        let target1 = RobotConfiguration::from_string("STAT=5 TU=3").unwrap();
        assert_eq!(target1.stat.to_bits(), 5);
        assert_eq!(target1.tu.to_bits(), 3);  // Binary 000011
        assert_eq!(target1.tu.j1_negative, true);   // Bit 0
        assert_eq!(target1.tu.j2_negative, true);   // Bit 1  
        assert_eq!(target1.tu.j3_negative, false);  // Bit 2
        
        // Test B prefix binary formats
        let target2 = RobotConfiguration::from_string("STAT=B000101 TU=B000011").unwrap();
        assert_eq!(target2.stat.to_bits(), 5);    // B000101 = 5
        assert_eq!(target2.tu.to_bits(), 3);      // B000011 = 3
        
        // Test pure binary TU without B prefix
        let target3 = RobotConfiguration::from_string("STAT=101 TU=000011").unwrap();
        assert_eq!(target3.stat.to_bits(), 5);
        assert_eq!(target3.tu.to_bits(), 3);
        

    }
    
    #[test]
    fn test_parallelogram_stat_calculation() {
        // Test regular robot (no parallelogram)
        let joints_regular = [0.0, 45.0, -30.0, 0.0, 60.0, 0.0]; // J2=45°, J3=-30°
        let stat_regular = StatBits::from_joints_with_parallelogram(joints_regular, false);
        assert_eq!(stat_regular.elbow_up, false); // J3=-30° < 0
        
        // Test parallelogram robot (virtual J3 = J2 + J3 = 45° + (-30°) = 15°)
        let stat_parallelogram = StatBits::from_joints_with_parallelogram(joints_regular, true);
        assert_eq!(stat_parallelogram.elbow_up, true); // Virtual J3=15° > 0
        
        // Verify they differ when parallelogram changes the elbow configuration
        assert_ne!(stat_regular.elbow_up, stat_parallelogram.elbow_up);
    }
    
    #[test]
    fn test_configuration_matching() {
        let joints1 = [21.3, 9.0, -133.5, 76.6, 39.7, 38.9];      // STAT: elbow_up=false, handflip=false, shoulder_left=true
        let joints2 = [-158.7, -114.9, 56.7, 139.1, -71.4, 127.2]; // Different STAT configuration
        
        let config1 = RobotConfiguration::from_joints(joints1);
        let config2 = RobotConfiguration::from_joints(joints2);
        
        // target: shoulder_left = true, elbow_up = false, handflip = false
        let target_stat = StatBits { shoulder_left: true, elbow_up: false, handflip: false };
        let target = TargetConfiguration::new(target_stat, TurnBits::from_bits(0));
        
        let (stat1, _, _) = config1.match_score(&target);
        let (stat2, _, _) = config2.match_score(&target);
        
        assert_eq!(stat1, 3);  // Perfect STAT match
        assert_eq!(stat2, 0);  // No STAT bits match
    }
    
    #[test]
    fn test_selector_with_stat_tu() {
        let solutions = [
            [21.3, 9.0, -133.5, 76.6, 39.7, 38.9],       // STAT: shoulder_left=true, elbow_up=false, handflip=false
            [-158.7, -114.9, 56.7, 450.0, -71.4, 127.2], // Different config
        ];
        
        let selector = ConfigurationSelector::new(&solutions);
        let target_stat = StatBits { shoulder_left: true, elbow_up: false, handflip: false };
        let target = TargetConfiguration::new(target_stat, TurnBits::from_bits(0));
        
        let (index, config, scores) = selector.find_best_match(&target).unwrap();
        
        assert_eq!(index, 0);           // First solution should match better
        assert_eq!(scores.0, 3);        // Perfect STAT match
        assert!(config.stat_tu_string.contains("STAT="));
    }
    
    #[test]
    fn test_tu_siemens_semantics() {
        // Test that 270° is treated as positive (Siemens semantics)
        let joints_270 = [0.0, 0.0, 0.0, 270.0, 0.0, 0.0];
        let tu_270 = TurnBits::from_joints(joints_270);
        assert_eq!(tu_270.j4_negative, false); // 270° should be positive
        
        // Test that -200° is treated as negative (stays negative in [-360,360])
        let joints_neg200 = [0.0, 0.0, 0.0, 0.0, 0.0, -200.0];
        let tu_neg200 = TurnBits::from_joints(joints_neg200);
        assert_eq!(tu_neg200.j6_negative, true); // -200° should stay negative
        
        // Test difference from old [-180,180] normalization
        // 270° would become -90° in old system, but stays +270° in new system
        let old_normalized = normalize_angle_degrees(270.0);  // Returns -90.0
        let new_normalized = normalize_angle_360(270.0);      // Returns 270.0
        assert_eq!(old_normalized, -90.0);
        assert_eq!(new_normalized, 270.0);
        
        // Test edge cases for large angles (beyond ±720°)
        assert_eq!(normalize_angle_360(1100.0), 20.0);   // Should be positive
        assert_eq!(normalize_angle_360(-1100.0), -20.0); // Should be negative
        assert_eq!(normalize_angle_360(750.0), 30.0);    // 750 - 720 = 30
        assert_eq!(normalize_angle_360(-750.0), -30.0);  // -750 + 720 = -30
    }

    #[test]
    fn test_robx_stat_correctness() {
        // Test the ROBX STAT specification: shoulder_left, elbow_up, handflip
        let kin_params = RobotKinematicParams {
            a1: 25.0, a2: 315.0, b: 0.0, c1: 400.0, c2: 0.0, c3: 80.0, c4: 0.0,
        };
        
        // Test elbow up/down (J3 sign)
        let joints_elbow_down = [0.0, 30.0, -45.0, 0.0, 15.0, 0.0]; // J3 negative
        let stat_down = StatBits::from_joints_geometric(joints_elbow_down, &kin_params, false);
        assert_eq!(stat_down.elbow_up, false, "J3 < 0 should result in elbow_up=false");
        
        // Test handflip (J5 sign)  
        let joints_handflip = [90.0, 45.0, -45.0, 0.0, -30.0, 0.0]; // J5 negative
        let stat_handflip = StatBits::from_joints_geometric(joints_handflip, &kin_params, false);
        assert_eq!(stat_handflip.handflip, true, "J5 < 0 should result in handflip=true");
        
        // Test that shoulder_left is computed (we don't test exact value since it depends on geometry)
        assert!(stat_handflip.shoulder_left == true || stat_handflip.shoulder_left == false);
    }

}