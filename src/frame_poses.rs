//! Selected joint frames for many joint rows, written frame-major.
//!
//! Each frame is built with the same operations as `forward_with_joint_poses`
//! and `joint_poses`, so the result is bit-identical to `batch_joint_poses`.

use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;

pub struct FramePoses {
    params: Parameters,
    degrees: bool,
    /// Frame indices as in `joint_poses`: 0..=5 for J1..J6, 6 for J6 with the ee transform.
    frames: Vec<usize>,
    ee: Option<(Matrix3<f64>, Vector3<f64>)>,
    /// Number of joints evaluated, one past the deepest joint any frame needs.
    depth: usize,
    /// 12 for a 3x4 matrix, 16 for 4x4 with the constant bottom row.
    stride: usize,
}

impl FramePoses {
    /// Callers check that every frame is in 0..=6 and that frame 6 comes with `ee`.
    pub fn new(
        params: &Parameters,
        degrees: bool,
        frames: Vec<usize>,
        ee: Option<(Matrix3<f64>, Vector3<f64>)>,
        homogeneous: bool,
    ) -> Self {
        let depth = frames.iter().map(|&f| f.min(5) + 1).max().unwrap_or(0);
        FramePoses {
            params: *params,
            degrees,
            frames,
            ee,
            depth,
            stride: if homogeneous { 16 } else { 12 },
        }
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Write row `i` of each frame's block; `blocks[k]` holds frame k for a run of rows.
    pub fn eval(&self, joints: &[f64], blocks: &mut [&mut [f64]], i: usize) {
        let s = self.stride;
        if joints.iter().any(|v| v.is_nan()) {
            for block in blocks.iter_mut() {
                block[i * s..(i + 1) * s].fill(f64::NAN);
            }
            return;
        }
        let p = &self.params;
        let q = |k: usize| {
            let raw = if self.degrees {
                joints[k].to_radians()
            } else {
                joints[k]
            };
            raw * p.sign_corrections[k] as f64 - p.offsets[k]
        };
        // Same steps and composition order as forward_with_joint_poses.
        let steps = [
            (Translation3::new(0.0, 0.0, p.c1), Vector3::z_axis()),
            (Translation3::new(p.a1, p.b, 0.0), Vector3::y_axis()),
            (Translation3::new(0.0, 0.0, p.c2), Vector3::y_axis()),
            (Translation3::new(p.a2, 0.0, 0.0), Vector3::z_axis()),
            (Translation3::new(0.0, 0.0, p.c3), Vector3::y_axis()),
            (Translation3::new(0.0, 0.0, p.c4), Vector3::z_axis()),
        ];
        let mut poses = [Isometry3::identity(); 6];
        for (k, (translation, axis)) in steps.iter().enumerate().take(self.depth) {
            let local =
                Isometry3::from_parts(*translation, UnitQuaternion::from_axis_angle(axis, q(k)));
            poses[k] = if k == 0 { local } else { poses[k - 1] * local };
        }
        for (block, &frame) in blocks.iter_mut().zip(&self.frames) {
            let pose = match (frame, &self.ee) {
                (6, Some((ee_rotation, ee_translation))) => {
                    // Same arithmetic as joint_poses, including its round trip through a quaternion.
                    let tcp = &poses[5];
                    let combined = tcp.rotation.to_rotation_matrix()
                        * Rotation3::from_matrix_unchecked(*ee_rotation);
                    let translation = tcp.translation.vector + combined * ee_translation;
                    Isometry3::from_parts(
                        Translation3::from(translation),
                        UnitQuaternion::from_rotation_matrix(&combined),
                    )
                }
                _ => poses[frame],
            };
            let m = pose.to_matrix();
            let out = &mut block[i * s..(i + 1) * s];
            for r in 0..s / 4 {
                for c in 0..4 {
                    out[r * 4 + c] = m[(r, c)];
                }
            }
        }
    }
}
