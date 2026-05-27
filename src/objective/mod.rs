pub mod binary;

pub use binary::BinaryObjective;

/// Maps raw scores + labels to per-sample gradients/hessians and provides an
/// initial constant score.
pub trait Objective: Send + Sync {
    fn init_score(&self, labels: &[f32]) -> f64;
    fn convert_output(&self, raw_scores: &[f64], out: &mut [f64]);
    /// Pack `[grad, hess]` pairs into a single buffer (one 8-byte load per row
    /// in the histogram hot loop).
    fn gradients_packed(&self, raw_scores: &[f64], labels: &[f32], out: &mut [[f32; 2]]);
}
