use serde::{Deserialize, Serialize};

use super::Objective;

/// Binary logistic objective: probability = sigmoid(raw_score), with
/// gradient = p - y and hessian = p(1 - p).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinaryObjective;

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl Objective for BinaryObjective {
    fn init_score(&self, labels: &[f32]) -> f64 {
        let n = labels.len() as f64;
        let pos: f64 = labels.iter().map(|&y| y as f64).sum();
        let mean = (pos / n).clamp(1e-6, 1.0 - 1e-6);
        (mean / (1.0 - mean)).ln()
    }

    fn convert_output(&self, raw_scores: &[f64], out: &mut [f64]) {
        for (r, o) in raw_scores.iter().zip(out.iter_mut()) {
            *o = sigmoid(*r);
        }
    }

    fn gradients(&self, raw_scores: &[f64], labels: &[f32], grads: &mut [f32], hesss: &mut [f32]) {
        for i in 0..raw_scores.len() {
            let p = sigmoid(raw_scores[i]);
            grads[i] = (p - labels[i] as f64) as f32;
            hesss[i] = (p * (1.0 - p)).max(1e-6) as f32;
        }
    }

    /// Pack-aware variant: write `[grad, hess]` pairs into a single buffer so
    /// the histogram-build hot loop can fetch both values in one memory access
    /// (8-byte aligned), halving the per-row gather count vs separate arrays.
    fn gradients_packed(&self, raw_scores: &[f64], labels: &[f32], out: &mut [[f32; 2]]) {
        for i in 0..raw_scores.len() {
            let p = sigmoid(raw_scores[i]);
            out[i][0] = (p - labels[i] as f64) as f32;
            out[i][1] = (p * (1.0 - p)).max(1e-6) as f32;
        }
    }
}
