//! Train with a validation set and early stopping. The trained model is automatically
//! truncated to the best iteration.
//!
//! Run with: `cargo run --release --example early_stopping`

use nanogbm::metric::{BinaryLogloss, Metric};
use nanogbm::{Config, DatasetBuilder, GbdtTrainer};

fn make_data(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f32>) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let weights: Vec<f64> = (0..d).map(|_| rand()).collect();
    let mut features = vec![0.0; n * d];
    let mut labels = vec![0f32; n];
    for i in 0..n {
        let mut z = 0.0;
        for j in 0..d {
            let x = rand();
            features[i * d + j] = x;
            z += weights[j] * x;
        }
        z += 0.2 * rand();
        let p = 1.0 / (1.0 + (-z).exp());
        labels[i] = if rand() * 0.5 + 0.5 < p { 1.0 } else { 0.0 };
    }
    (features, labels)
}

fn main() {
    let d = 10;
    let (tx, ty) = make_data(3000, d, 1);
    let (vx, vy) = make_data(1000, d, 2);

    let mut cfg = Config::default();
    cfg.num_iterations = 500;
    cfg.learning_rate = 0.05;
    cfg.num_leaves = 31;
    cfg.min_data_in_leaf = 20;
    cfg.lambda_l2 = 1.0;
    cfg.early_stopping_round = 25;
    cfg.verbose = true;
    cfg.seed = 0;

    let schema = nanogbm::feature::Schema::all_numerical(d);
    let train = DatasetBuilder::from_rows(&tx, 3000, &schema, &ty, &cfg).unwrap();
    // Validation set MUST reuse the training bin mappers so val bins line up
    // with the trees' learned `threshold_bin`s.
    let valid =
        DatasetBuilder::from_rows_with_mappers(&vx, 1000, train.bin_mappers(), &vy).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&train, Some(&valid)).unwrap();

    let scores = model.predict_raw_scores(&vx, 1000);
    let logloss = BinaryLogloss.evaluate(&scores, &vy);
    println!("trees={} valid_logloss={logloss:.5}", model.n_trees());
}
