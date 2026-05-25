pub mod binary;

pub use binary::BinaryObjective;

/// Objective contract: maps raw scores + labels to per-sample gradients and hessians,
/// and provides an initial constant score for the boosting loop.
pub trait Objective: Send + Sync {
    fn init_score(&self, labels: &[f32]) -> f64;
    /// Convert raw additive scores to probabilities (or whatever the model emits).
    fn convert_output(&self, raw_scores: &[f64], out: &mut [f64]);
    /// Fill `grads` and `hesss` (length n) from raw scores and labels.
    fn gradients(&self, raw_scores: &[f64], labels: &[f32], grads: &mut [f32], hesss: &mut [f32]);

    /// Pack-aware variant: write `[grad, hess]` pairs into a single buffer.
    /// Default impl falls back to two separate writes via temp scratch; the
    /// performance-critical impl is on `BinaryObjective`.
    fn gradients_packed(&self, raw_scores: &[f64], labels: &[f32], out: &mut [[f32; 2]]) {
        let n = raw_scores.len();
        let mut g = vec![0f32; n];
        let mut h = vec![0f32; n];
        self.gradients(raw_scores, labels, &mut g, &mut h);
        for i in 0..n {
            out[i][0] = g[i];
            out[i][1] = h[i];
        }
    }
}
