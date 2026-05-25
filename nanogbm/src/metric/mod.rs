pub mod binary_logloss;

pub use binary_logloss::BinaryLogloss;

pub trait Metric: Send + Sync {
    fn name(&self) -> &'static str;
    /// Lower is better → `true`; higher is better → `false`.
    fn lower_is_better(&self) -> bool;
    /// Score from raw additive scores and labels.
    fn evaluate(&self, raw_scores: &[f64], labels: &[f32]) -> f64;
}
