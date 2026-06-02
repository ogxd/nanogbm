//! Train with a validation set and early stopping. The trained model is automatically
//! truncated to the best iteration.
//!
//! Run with: `cargo run --release --example early_stopping`

use nanogbm::{Config, FeatureBuilder, GbdtTrainer};

const D: usize = 10;

struct Row {
    f: [f64; D],
}

fn make_data(n: usize, seed: u64) -> (Vec<Row>, Vec<f32>) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let weights: Vec<f64> = (0..D).map(|_| rand()).collect();
    let mut rows = Vec::with_capacity(n);
    let mut labels = vec![0f32; n];
    for i in 0..n {
        let mut f = [0.0f64; D];
        let mut z = 0.0;
        for (j, v) in f.iter_mut().enumerate() {
            *v = rand();
            z += weights[j] * *v;
        }
        z += 0.2 * rand();
        let p = 1.0 / (1.0 + (-z).exp());
        labels[i] = if rand() * 0.5 + 0.5 < p { 1.0 } else { 0.0 };
        rows.push(Row { f });
    }
    (rows, labels)
}

fn main() {
    let (tr, ty) = make_data(3000, 1);
    let (vr, vy) = make_data(1000, 2);

    let mut cfg = Config::default();
    cfg.num_iterations = 500;
    cfg.learning_rate = 0.05;
    cfg.num_leaves = 31;
    cfg.min_data_in_leaf = 20;
    cfg.lambda_l2 = 1.0;
    cfg.early_stopping_round = 25;
    cfg.verbose = true;
    cfg.seed = 0;

    let mut fb = FeatureBuilder::<Row>::new();
    for j in 0..D {
        fb = fb.add_continuous(format!("f{j}"), move |r| r.f[j]);
    }

    let model = GbdtTrainer::new(&cfg, &fb)
        .fit(&tr, &ty, Some((&vr, &vy)))
        .unwrap();

    let probs = model.predict_proba(&fb, &vr);
    let n = probs.len() as f64;
    let logloss = probs.iter().zip(&vy).map(|(&p, &y)| {
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        -((y as f64) * p.ln() + (1.0 - y as f64) * (1.0 - p).ln())
    }).sum::<f64>() / n;
    println!("trees={} valid_logloss={logloss:.5}", model.n_trees());
}
