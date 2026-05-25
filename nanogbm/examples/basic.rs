//! Train a binary classifier on a tiny synthetic dataset and print probabilities.
//!
//! Run with: `cargo run --release --example basic`

use nanogbm::{Config, DatasetBuilder, GbdtTrainer};
use nanogbm::predict::predict_proba;

fn main() {
    // 1000 rows, 5 features. Label is 1 when feature[0] + feature[1] > 0.
    let n_rows = 1000;
    let n_features = 5;
    let mut features = vec![0.0f64; n_rows * n_features];
    let mut labels = vec![0f32; n_rows];
    let mut state: u64 = 0xDEADBEEF;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    for i in 0..n_rows {
        for j in 0..n_features {
            features[i * n_features + j] = rand();
        }
        let z = features[i * n_features] + features[i * n_features + 1];
        labels[i] = if z > 0.0 { 1.0 } else { 0.0 };
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 100;
    cfg.learning_rate = 0.1;
    cfg.num_leaves = 15;
    cfg.seed = 0;

    let train = DatasetBuilder::from_rows(&features, n_rows, n_features, &labels, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&train, None).unwrap();

    let probs = predict_proba(&model, &features[..10 * n_features], 10, n_features);
    for (i, p) in probs.iter().enumerate() {
        println!("row {i}: label={} p={p:.4}", labels[i]);
    }
}
