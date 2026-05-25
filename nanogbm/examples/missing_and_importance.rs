//! Demonstrate two features at once:
//!   - NaN values in the input are handled natively (routed to whichever side maximizes gain).
//!   - Feature importance can be read off the trained model.
//!
//! Run with: `cargo run --release --example missing_and_importance`

use nanogbm::{Config, DatasetBuilder, GbdtTrainer};
use nanogbm::predict::predict_proba;

fn main() {
    let n = 2000;
    let d = 6;
    let mut features = vec![0.0f64; n * d];
    let mut labels = vec![0f32; n];
    let mut s: u64 = 0xABCDEF;
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    // Only features 0 and 2 are predictive. The rest are noise.
    for i in 0..n {
        for j in 0..d {
            features[i * d + j] = rand();
        }
        let z = 2.0 * features[i * d] + features[i * d + 2];
        labels[i] = if z > 0.0 { 1.0 } else { 0.0 };

        // Sprinkle ~10% NaNs into feature 3 (a noise column) and feature 0 (predictive).
        if rand() > 0.8 { features[i * d + 3] = f64::NAN; }
        if rand() > 0.9 { features[i * d] = f64::NAN; }
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 100;
    cfg.num_leaves = 15;
    cfg.learning_rate = 0.1;
    cfg.seed = 0;

    let ds = DatasetBuilder::from_rows(&features, n, d, &labels, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let split = model.feature_importance_split();
    let gain = model.feature_importance_gain();
    println!("feature | splits | gain");
    for j in 0..d {
        println!("  f{j}    |  {:5} | {:.3}", split[j], gain[j]);
    }

    // Spot-check predictions on rows with missing values still work.
    let probs = predict_proba(&model, &features[..5 * d], 5, d);
    for (i, p) in probs.iter().enumerate() {
        let row = &features[i * d..(i + 1) * d];
        println!("row {i} {row:?} -> p={p:.4}");
    }
}
