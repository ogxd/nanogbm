use super::Metric;
use crate::objective::binary::sigmoid;

const EPS: f64 = 1e-15;

pub struct BinaryLogloss;

impl Metric for BinaryLogloss {
    fn name(&self) -> &'static str {
        "binary_logloss"
    }
    fn lower_is_better(&self) -> bool {
        true
    }
    fn evaluate(&self, raw_scores: &[f64], labels: &[f32]) -> f64 {
        let n = raw_scores.len() as f64;
        let mut total = 0.0;
        for i in 0..raw_scores.len() {
            let p = sigmoid(raw_scores[i]).clamp(EPS, 1.0 - EPS);
            let y = labels[i] as f64;
            total += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        }
        total / n
    }
}
