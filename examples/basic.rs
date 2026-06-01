//! Train a binary classifier on a tiny synthetic dataset and print probabilities.
//!
//! Run with: `cargo run --release --example basic`

use nanogbm::{Config, FeatureBuilder, GbdtTrainer};

/// One training row. In a real service this is your domain object.
struct Row {
    f: [f64; 5],
}

fn main() {
    // 1000 rows, 5 features. Label is 1 when f[0] + f[1] > 0.
    let n_rows = 1000;
    let mut state: u64 = 0xDEADBEEF;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut rows = Vec::with_capacity(n_rows);
    let mut labels = vec![0f32; n_rows];
    for i in 0..n_rows {
        let f = [rand(), rand(), rand(), rand(), rand()];
        labels[i] = if f[0] + f[1] > 0.0 { 1.0 } else { 0.0 };
        rows.push(Row { f });
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 100;
    cfg.learning_rate = 0.1;
    cfg.num_leaves = 15;
    cfg.seed = 0;

    // Declare features: a name and a closure pulling the value from a Row.
    let fb = FeatureBuilder::<Row>::new()
        .add_continuous("f0", |r| r.f[0])
        .add_continuous("f1", |r| r.f[1])
        .add_continuous("f2", |r| r.f[2])
        .add_continuous("f3", |r| r.f[3])
        .add_continuous("f4", |r| r.f[4]);

    let model = GbdtTrainer::new(&cfg, &fb).fit(&rows, &labels, None).unwrap();

    let probs = model.predict_proba(&fb, &rows[..10]);
    for (i, p) in probs.iter().enumerate() {
        println!("row {i}: label={} p={p:.4}", labels[i]);
    }
}
