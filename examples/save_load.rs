//! Train a model, save it to disk, reload it, and verify predictions match.
//!
//! Run with: `cargo run --release --example save_load`

use nanogbm::{Config, DatasetBuilder, GbdtTrainer, Model};

fn main() {
    let n = 500;
    let d = 4;
    let mut features = vec![0.0f64; n * d];
    let mut labels = vec![0f32; n];
    let mut s: u64 = 12345;
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    for i in 0..n {
        for j in 0..d {
            features[i * d + j] = rand();
        }
        labels[i] = if features[i * d] - features[i * d + 1] > 0.0 {
            1.0
        } else {
            0.0
        };
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 50;
    cfg.num_leaves = 15;
    cfg.seed = 0;

    let ds = DatasetBuilder::from_rows(&features, n, d, &labels, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let path = std::env::temp_dir().join("nanogbm_example_model.bin");
    model.save(&path).unwrap();
    println!(
        "saved {} bytes to {}",
        std::fs::metadata(&path).unwrap().len(),
        path.display()
    );

    let loaded = Model::load(&path).unwrap();
    let p_orig = model.predict_proba(&features, n);
    let p_load = loaded.predict_proba(&features, n);
    let max_diff = p_orig
        .iter()
        .zip(&p_load)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("max prediction diff after round-trip: {max_diff:.2e}");
    assert!(max_diff < 1e-12);
}
