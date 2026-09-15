//! Arm-local reach kernel: all eight OPW inverse-kinematics branches per pose,
//! with joint-limit margins, wrist-centre extension and Jacobian conditioning.

use nalgebra::{Isometry3, Matrix6, Vector3};
use rs_opw_kinematics::kinematic_traits::Kinematics;
use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;
use std::f64::consts::PI;

// Same tolerances rs-opw-kinematics uses to cross-check a candidate against FK.
const DISTANCE_TOLERANCE: f64 = 1e-6;
const ANGULAR_TOLERANCE: f64 = 1e-6;
// Below this |sin(theta5)| the J4/J6 split is taken from the full rotation instead.
const WRIST_SINGULARITY_SIN: f64 = 1e-8;

/// Per-pose output of the reach kernel. Joints are in radians in the user
/// (sign-corrected, offset) frame; a NaN row means the branch does not exist.
pub struct PoseReach {
    pub joints: [[f64; 6]; 8],
    pub extension: f64,
    pub wrist: [f64; 8],
    pub sigma_min: [f64; 8],
}

impl PoseReach {
    pub fn nan() -> Self {
        PoseReach {
            joints: [[f64::NAN; 6]; 8],
            extension: f64::NAN,
            wrist: [f64::NAN; 8],
            sigma_min: [f64::NAN; 8],
        }
    }
}

fn normalize_angle(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Enumerate the eight OPW branches for the flange pose (without ee offset).
///
/// Slot s = 4 * wrist_flip + 2 * shoulder + elbow, with the sign choices taken
/// on the raw OPW angles before offsets and sign corrections:
///   elbow 0/1:      theta3 = +acos(..) - psi3 / -acos(..) - psi3
///   shoulder 0/1:   theta1 = atan2(cy, cx) - atan2(b, ..) / the same + pi (arm reaching back)
///   wrist_flip 0/1: theta5 >= 0 / theta5 <= 0
/// This is the order rs-opw-kinematics builds its candidate list in.
pub fn all_branches(params: &Parameters, pose: &Isometry3<f64>) -> ([[f64; 6]; 8], f64, [f64; 8]) {
    let matrix = pose.rotation.to_rotation_matrix();
    let c = pose.translation.vector - params.c4 * matrix.transform_vector(&Vector3::z_axis());

    // Clamped so a wrist centre inside the |b| cylinder still yields a finite extension.
    let nx1 = ((c.x * c.x + c.y * c.y) - params.b * params.b)
        .max(0.0)
        .sqrt()
        - params.a1;

    let tmp1 = c.y.atan2(c.x);
    let tmp2 = params.b.atan2(nx1 + params.a1);
    let theta1_i = tmp1 - tmp2;
    let theta1_ii = tmp1 + tmp2 - PI;

    let tmp3 = c.z - params.c1;
    let s1_2 = nx1 * nx1 + tmp3 * tmp3;
    let tmp4 = nx1 + 2.0 * params.a1;
    let s2_2 = tmp4 * tmp4 + tmp3 * tmp3;
    let kappa_2 = params.a2 * params.a2 + params.c3 * params.c3;
    let kappa = kappa_2.sqrt();
    let c2_2 = params.c2 * params.c2;

    let s1 = s1_2.sqrt();
    let s2 = s2_2.sqrt();

    // Signed distance of the wrist centre to the annulus |c2 - L| <= d <= c2 + L,
    // taken over both shoulder configurations so it is negative only when no branch exists.
    let annulus = |d: f64| (params.c2 + kappa - d).min(d - (params.c2 - kappa).abs());
    let radius_xy = (c.x * c.x + c.y * c.y).sqrt();
    let extension = if radius_xy < params.b.abs() {
        // Inside the |b| cylinder no branch exists; report the distance to it.
        radius_xy - params.b.abs()
    } else {
        annulus(s1).max(annulus(s2))
    };

    let tmp13 = ((s1_2 + c2_2 - kappa_2) / (2.0 * s1 * params.c2)).acos();
    let tmp14 = nx1.atan2(tmp3);
    let theta2_i = -tmp13 + tmp14;
    let theta2_ii = tmp13 + tmp14;

    let tmp15 = ((s2_2 + c2_2 - kappa_2) / (2.0 * s2 * params.c2)).acos();
    let tmp16 = tmp4.atan2(tmp3);
    let theta2_iii = -tmp15 - tmp16;
    let theta2_iv = tmp15 - tmp16;

    let tmp9 = 2.0 * params.c2 * kappa;
    let tmp10 = params.a2.atan2(params.c3);
    let tmp11 = ((s1_2 - c2_2 - kappa_2) / tmp9).acos();
    let theta3_i = tmp11 - tmp10;
    let theta3_ii = -tmp11 - tmp10;
    let tmp12 = ((s2_2 - c2_2 - kappa_2) / tmp9).acos();
    let theta3_iii = tmp12 - tmp10;
    let theta3_iv = -tmp12 - tmp10;

    let theta1 = [theta1_i, theta1_i, theta1_ii, theta1_ii];
    let theta2 = [theta2_i, theta2_ii, theta2_iii, theta2_iv];
    let theta3 = [theta3_i, theta3_ii, theta3_iii, theta3_iv];

    let mut theta = [[f64::NAN; 6]; 8];
    let mut wrist = [f64::NAN; 8];

    for k in 0..4 {
        let (sin1, cos1) = theta1[k].sin_cos();
        let (s23, c23) = (theta2[k] + theta3[k]).sin_cos();

        let m = matrix[(0, 2)] * s23 * cos1 + matrix[(1, 2)] * s23 * sin1 + matrix[(2, 2)] * c23;
        // Clamp so an exact wrist singularity (|m| == 1 up to rounding) keeps its branches.
        let m = m.clamp(-1.0, 1.0);
        let sin5 = (1.0 - m * m).sqrt();
        let theta5 = sin5.atan2(m);

        let (theta4, theta6) = if sin5 < WRIST_SINGULARITY_SIN {
            // J4 and J6 are collinear: only theta4 + theta5_sign * theta6 is
            // determined. Put the whole rotation on J6 so the slot stays defined.
            let r00 =
                cos1 * c23 * matrix[(0, 0)] + sin1 * c23 * matrix[(1, 0)] - s23 * matrix[(2, 0)];
            let r10 = -sin1 * matrix[(0, 0)] + cos1 * matrix[(1, 0)];
            (0.0, r10.atan2(m * r00))
        } else {
            let theta4_y = matrix[(1, 2)] * cos1 - matrix[(0, 2)] * sin1;
            let theta4_x =
                matrix[(0, 2)] * c23 * cos1 + matrix[(1, 2)] * c23 * sin1 - matrix[(2, 2)] * s23;

            let theta6_y =
                matrix[(0, 1)] * s23 * cos1 + matrix[(1, 1)] * s23 * sin1 + matrix[(2, 1)] * c23;
            let theta6_x =
                -matrix[(0, 0)] * s23 * cos1 - matrix[(1, 0)] * s23 * sin1 - matrix[(2, 0)] * c23;
            (theta4_y.atan2(theta4_x), theta6_y.atan2(theta6_x))
        };

        theta[k] = [theta1[k], theta2[k], theta3[k], theta4, theta5, theta6];
        theta[k + 4] = [
            theta1[k],
            theta2[k],
            theta3[k],
            theta4 + PI,
            -theta5,
            theta6 - PI,
        ];
        wrist[k] = sin5;
        wrist[k + 4] = sin5;
    }

    let mut joints = [[f64::NAN; 6]; 8];
    for s in 0..8 {
        let mut valid = true;
        let mut q = [0.0; 6];
        for j in 0..6 {
            let angle = (theta[s][j] + params.offsets[j]) * params.sign_corrections[j] as f64;
            if !angle.is_finite() {
                valid = false;
                break;
            }
            q[j] = normalize_angle(angle);
        }
        if valid {
            joints[s] = q;
        } else {
            wrist[s] = f64::NAN;
        }
    }

    (joints, extension, wrist)
}

/// Geometric Jacobian of the point `tcp` (world frame) with respect to the
/// user-frame joints in radians. Rows 0..3 are translation (length unit per
/// radian), rows 3..6 rotation (radian per radian).
pub fn geometric_jacobian(
    params: &Parameters,
    joint_poses: &[Isometry3<f64>; 6],
    tcp: &Vector3<f64>,
) -> Matrix6<f64> {
    let mut jac = Matrix6::<f64>::zeros();
    for (i, pose) in joint_poses.iter().enumerate() {
        // rs-opw-kinematics joint frames rotate J1, J4, J6 about local z and J2, J3, J5 about local y.
        let local_axis = if i == 1 || i == 2 || i == 4 {
            Vector3::y_axis()
        } else {
            Vector3::z_axis()
        };
        let axis = pose.rotation * local_axis.into_inner() * params.sign_corrections[i] as f64;
        let lin = axis.cross(&(tcp - pose.translation.vector));
        jac.fixed_view_mut::<3, 1>(0, i).copy_from(&lin);
        jac.fixed_view_mut::<3, 1>(3, i).copy_from(&axis);
    }
    jac
}

/// Smallest singular value of the geometric Jacobian at `joints` (radians).
fn sigma_min(params: &Parameters, joint_poses: &[Isometry3<f64>; 6], tcp: &Vector3<f64>) -> f64 {
    let jac = geometric_jacobian(params, joint_poses, tcp);
    // Singular values of J are the square roots of the eigenvalues of J^T J;
    // the symmetric eigen solve is cheaper than a full SVD.
    let jtj = jac.transpose() * jac;
    let eig = jtj.symmetric_eigenvalues();
    eig.iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
        .sqrt()
}

/// Full reach evaluation for one target pose. `flange` is the target with the
/// end-effector offset removed; `ee_offset` is the TCP position expressed in
/// the flange frame, used to evaluate the Jacobian at the requested TCP.
pub fn reach_pose(
    robot: &OPWKinematics,
    params: &Parameters,
    flange: &Isometry3<f64>,
    ee_offset: &Vector3<f64>,
) -> PoseReach {
    let (mut joints, extension, mut wrist) = all_branches(params, flange);
    let mut sigma = [f64::NAN; 8];

    for s in 0..8 {
        if joints[s][0].is_nan() {
            continue;
        }
        let poses = robot.forward_with_joint_poses(&joints[s]);
        let flange_check = &poses[5];
        let dist = (flange_check.translation.vector - flange.translation.vector).norm();
        let ang = flange_check.rotation.angle_to(&flange.rotation);
        if dist > DISTANCE_TOLERANCE || ang > ANGULAR_TOLERANCE {
            joints[s] = [f64::NAN; 6];
            wrist[s] = f64::NAN;
            continue;
        }
        let tcp = flange_check.translation.vector + flange_check.rotation * ee_offset;
        sigma[s] = sigma_min(params, &poses, &tcp);
    }

    PoseReach {
        joints,
        extension,
        wrist,
        sigma_min: sigma,
    }
}
