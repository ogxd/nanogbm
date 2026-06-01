//! Train a model, save it to disk, reload it, and verify predictions match.
//!
//! Run with: `cargo run --release --example save_load`

use nanogbm::{Config, FeatureBuilder, GbdtTrainer, Model};

struct Row {
    f: [f64; 4],
}

fn main() {
    let n = 500;
    let mut s: u64 = 12345;
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let mut rows = Vec::with_capacity(n);
    let mut labels = vec![0f32; n];
    for i in 0..n {
        let f = [rand(), rand(), rand(), rand()];
        labels[i] = if f[0] - f[1] > 0.0 { 1.0 } else { 0.0 };
        rows.push(Row { f });
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 50;
    cfg.num_leaves = 15;
    cfg.seed = 0;

    let fb = FeatureBuilder::<Row>::new()
        .add_continuous("f0", |r| r.f[0])
        .add_continuous("f1", |r| r.f[1])
        .add_continuous("f2", |r| r.f[2])
        .add_continuous("f3", |r| r.f[3]);

    let model = GbdtTrainer::new(&cfg, &fb).fit(&rows, &labels, None).unwrap();

    let path = std::env::temp_dir().join("nanogbm_example_model.bin");
    model.save(&path).unwrap();
    println!(
        "saved {} bytes to {}",
        std::fs::metadata(&path).unwrap().len(),
        path.display()
    );

    // The closures live in `fb`, not in the saved file — hand the same builder
    // to the reloaded model to predict.
    let loaded = Model::load(&path).unwrap();
    let p_orig = model.predict_proba(&fb, &rows);
    let p_load = loaded.predict_proba(&fb, &rows);
    let max_diff = p_orig
        .iter()
        .zip(&p_load)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("max prediction diff after round-trip: {max_diff:.2e}");
    assert!(max_diff < 1e-12);
}
