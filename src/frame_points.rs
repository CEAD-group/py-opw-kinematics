//! World positions of fixed points on selected joint frames, evaluated along
//! the same chain as `forward_with_joint_poses` but without building poses.

use nalgebra::{Matrix3, Vector3};
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;

/// Frame selection resolved against one robot: each requested point is a
/// joint index (0..6 for J1..J6) and an offset in that joint's local frame.
pub struct FramePoints {
    a1: f64,
    a2: f64,
    b: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    sign: [f64; 6],
    offsets: [f64; 6],
    degrees: bool,
    /// Points per joint as (output slot, local offset).
    by_joint: [Vec<(usize, Vector3<f64>)>; 6],
    /// Joints whose rotation must be applied: a later joint or a nonzero offset needs it.
    needs_rotation: [bool; 6],
    /// Number of joints evaluated, one past the deepest selected joint.
    depth: usize,
    count: usize,
}

impl FramePoints {
    /// `offsets` are local to `joints[k]`; `params.offsets` must already be in radians.
    pub fn new(params: &Parameters, degrees: bool, points: &[(usize, Vector3<f64>)]) -> Self {
        let mut by_joint: [Vec<(usize, Vector3<f64>)>; 6] = Default::default();
        for (slot, &(joint, offset)) in points.iter().enumerate() {
            by_joint[joint].push((slot, offset));
        }
        let depth = points.iter().map(|&(j, _)| j + 1).max().unwrap_or(0);
        let mut needs_rotation = [false; 6];
        for (k, needs) in needs_rotation.iter_mut().enumerate().take(depth) {
            *needs = k + 1 < depth || by_joint[k].iter().any(|(_, p)| *p != Vector3::zeros());
        }
        FramePoints {
            a1: params.a1,
            a2: params.a2,
            b: params.b,
            c1: params.c1,
            c2: params.c2,
            c3: params.c3,
            c4: params.c4,
            sign: params.sign_corrections.map(|s| s as f64),
            offsets: params.offsets,
            degrees,
            by_joint,
            needs_rotation,
            depth,
            count: points.len(),
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Write `count() * 3` coordinates for one joint row; any NaN joint gives all NaN.
    pub fn eval(&self, joints: &[f64], out: &mut [f64]) {
        if joints.iter().any(|v| v.is_nan()) {
            out.fill(f64::NAN);
            return;
        }
        // Steps as in rs-opw-kinematics: translate in the parent frame, then rotate.
        let steps: [(Vector3<f64>, Axis); 6] = [
            (Vector3::new(0.0, 0.0, self.c1), Axis::Z),
            (Vector3::new(self.a1, self.b, 0.0), Axis::Y),
            (Vector3::new(0.0, 0.0, self.c2), Axis::Y),
            (Vector3::new(self.a2, 0.0, 0.0), Axis::Z),
            (Vector3::new(0.0, 0.0, self.c3), Axis::Y),
            (Vector3::new(0.0, 0.0, self.c4), Axis::Z),
        ];
        let mut rotation = Matrix3::identity();
        let mut translation = Vector3::zeros();
        for (k, (step, axis)) in steps.iter().enumerate().take(self.depth) {
            translation += rotation * step;
            if self.needs_rotation[k] {
                let raw = if self.degrees {
                    joints[k].to_radians()
                } else {
                    joints[k]
                };
                let q = raw * self.sign[k] - self.offsets[k];
                rotate(&mut rotation, *axis, q);
            }
            for &(slot, offset) in &self.by_joint[k] {
                let p = translation + rotation * offset;
                out[slot * 3..slot * 3 + 3].copy_from_slice(p.as_slice());
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Axis {
    Y,
    Z,
}

/// `rotation = rotation * R_axis(q)`, touching only the two columns that change.
fn rotate(rotation: &mut Matrix3<f64>, axis: Axis, q: f64) {
    let (s, c) = q.sin_cos();
    let (i, j, sign) = match axis {
        // Ry: x' = c x - s z, z' = s x + c z
        Axis::Y => (0, 2, -1.0),
        // Rz: x' = c x + s y, y' = -s x + c y
        Axis::Z => (0, 1, 1.0),
    };
    let ci = rotation.column(i).into_owned();
    let cj = rotation.column(j).into_owned();
    rotation.set_column(i, &(ci * c + cj * (sign * s)));
    rotation.set_column(j, &(cj * c - ci * (sign * s)));
}
