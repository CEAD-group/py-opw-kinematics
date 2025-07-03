use crate::kinematic_model::KinematicModel;

pub struct AxisConfiguration {
    kinematic_model: KinematicModel,
}

impl AxisConfiguration {
    pub fn new(kinematic_model: &KinematicModel) -> Self {
        Self {
            kinematic_model: kinematic_model.clone(),
        }
    }

    fn quadrant(&self, angle: f64) -> i32 {
        (angle / 90.0).floor() as i32
    }

    fn shoulder_singularity(&self, joints: [f64; 6]) -> f64 {
        -self.kinematic_model.c2 * (joints[1].to_radians()).sin()
            - self.kinematic_model.c3 * (joints[2] + joints[1]).to_radians().cos()
            - self.kinematic_model.a2 * (joints[2] + joints[1]).to_radians().sin()
            - self.kinematic_model.a1
    }

    fn elbow_singularity(&self, joints: [f64; 6]) -> f64 {
        joints[2] + 90.0
    }

    fn wrist_singularity(&self, joints: [f64; 6]) -> f64 {
        joints[4]
    }

    pub fn cfx(&self, joints: [f64; 6]) -> i32 {
        let shoulder = (self.shoulder_singularity(joints) > 0.0) as i32;
        let elbow = (self.elbow_singularity(joints) < 0.0) as i32;
        let wrist = (self.wrist_singularity(joints) < 0.0) as i32;
        // cfx: 3-bit integer: (shoulder << 2) | (elbow << 1) | wrist
        (shoulder << 2) | (elbow << 1) | wrist
    }

    pub fn axis_configuration(&self, joints: [f64; 6]) -> [i32; 4] {
        let cfx = self.cfx(joints);
        let cf1 = self.quadrant(joints[0]);
        let cf4 = self.quadrant(joints[2]);
        let cf6 = self.quadrant(joints[5]);
        [cf1, cf4, cf6, cfx]
    }
}

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

    fn check_axis_configuration(
        axis_configuration: &AxisConfiguration,
        test_cases: &[([f64; 6], [i32; 4])],
    ) {
        for (joints, expected) in test_cases.iter() {
            let result = axis_configuration.axis_configuration(*joints);
            // Only compare cfx
            assert_eq!(result[3], expected[3], "joints: {:?}", joints);
        }
    }

    #[test]
    fn test_axis_configuration_5() {
        let axis_configuration = AxisConfiguration::new(&ABB_1660);
        let test_cases = [
            (
                [-103.1, -85.03, 19.06, -70.19, -35.87, 185.01],
                [0, 0, 0, 5],
            ),
            (
                [-116.97, -85.69, 16.82, -63.5, -39.63, 192.76],
                [0, 0, 0, 5],
            ),
            (
                [-128.14, -86.43, 13.04, -59.66, -40.66, 201.57],
                [0, 0, 0, 5],
            ),
            (
                [-124.68, -61.16, -20.4, 56.41, -38.79, -24.56],
                [0, 0, 0, 5],
            ),
            (
                [-127.36, -62.29, -16.83, 59.35, -35.14, -23.42],
                [0, 0, 0, 5],
            ),
        ];
        check_axis_configuration(&axis_configuration, &test_cases);
    }

    #[test]
    fn test_axis_configuration_4() {
        let axis_configuration = AxisConfiguration::new(&ABB_1660);
        let test_cases = [
            ([-103.1, -85.03, 19.06, 109.81, 35.87, 5.01], [0, 0, 0, 4]),
            ([-116.97, -85.69, 16.82, 116.5, 39.63, 12.76], [0, 0, 0, 4]),
            ([-128.14, -86.43, 13.04, 120.34, 40.66, 21.57], [0, 0, 0, 4]),
            (
                [-124.68, -61.16, -20.4, -123.59, 38.79, 155.44],
                [0, 0, 0, 4],
            ),
            (
                [-127.36, -62.29, -16.83, -120.65, 35.14, 156.58],
                [0, 0, 0, 4],
            ),
        ];
        check_axis_configuration(&axis_configuration, &test_cases);
    }
}
