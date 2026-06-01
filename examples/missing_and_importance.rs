//! Demonstrate two features at once:
//!   - NaN values in the input are handled natively (routed to whichever side maximizes gain).
//!   - Feature importance can be read off the trained model.
//!
//! Run with: `cargo run --release --example missing_and_importance`

use nanogbm::{Config, FeatureBuilder, GbdtTrainer};

struct Row {
    f: [f64; 6],
}

fn main() {
    let n = 2000;
    let d = 6;
    let mut s: u64 = 0xABCDEF;
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    // Only features 0 and 2 are predictive. The rest are noise.
    let mut rows = Vec::with_capacity(n);
    let mut labels = vec![0f32; n];
    for i in 0..n {
        let mut f = [0.0f64; 6];
        for v in f.iter_mut() {
            *v = rand();
        }
        let z = 2.0 * f[0] + f[2];
        labels[i] = if z > 0.0 { 1.0 } else { 0.0 };

        // Sprinkle ~10% NaNs into feature 3 (a noise column) and feature 0 (predictive).
        if rand() > 0.8 {
            f[3] = f64::NAN;
        }
        if rand() > 0.9 {
            f[0] = f64::NAN;
        }
        rows.push(Row { f });
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 100;
    cfg.num_leaves = 15;
    cfg.learning_rate = 0.1;
    cfg.seed = 0;

    let mut fb = FeatureBuilder::<Row>::new();
    for j in 0..d {
        // `move` so each closure captures its own `j`.
        fb = fb.add_continuous(format!("f{j}"), move |r| r.f[j]);
    }

    let model = GbdtTrainer::new(&cfg, &fb).fit(&rows, &labels, None).unwrap();

    let split = model.feature_importance_split();
    let gain = model.feature_importance_gain();
    println!("{}", fb.format_importance(&split, &gain));

    // Spot-check predictions on rows with missing values still work.
    let probs = model.predict_proba(&fb, &rows[..5]);
    for (i, p) in probs.iter().enumerate() {
        println!("row {i} {:?} -> p={p:.4}", rows[i].f);
    }
}
