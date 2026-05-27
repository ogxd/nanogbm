use nanogbm::loss::binary_logloss;
use nanogbm::{Config, DatasetBuilder, GbdtTrainer, Model};
use rand::SeedableRng;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Generate a noisy linearly-separable binary classification problem. `train_n` rows
/// for training, `valid_n` for validation, with the same underlying weights/bias.
fn make_classification(
    train_n: usize,
    valid_n: usize,
    d: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f32>, Vec<f64>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let weights: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let bias: f64 = rng.gen_range(-0.5..0.5);
    let mut sample = |n: usize| -> (Vec<f64>, Vec<f32>) {
        let mut features = vec![0.0; n * d];
        let mut labels = vec![0f32; n];
        for i in 0..n {
            let mut z = bias;
            for j in 0..d {
                let x: f64 = rng.gen_range(-3.0..3.0);
                features[i * d + j] = x;
                z += weights[j] * x;
            }
            z += rng.gen_range(-0.4..0.4);
            let p = 1.0 / (1.0 + (-z).exp());
            labels[i] = if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 };
        }
        (features, labels)
    };
    let (tx, ty) = sample(train_n);
    let (vx, vy) = sample(valid_n);
    (tx, ty, vx, vy)
}

#[test]
fn trains_and_reduces_logloss_on_synthetic() {
    let n_train = 4000;
    let n_valid = 1000;
    let d = 8;
    let (train_x, train_y, valid_x, valid_y) = make_classification(n_train, n_valid, d, 42);

    let mut cfg = Config::default();
    cfg.num_iterations = 200;
    cfg.learning_rate = 0.1;
    cfg.num_leaves = 31;
    cfg.min_data_in_leaf = 20;
    cfg.max_bin = 64;
    cfg.feature_fraction = 1.0;
    cfg.bagging_fraction = 1.0;
    cfg.lambda_l2 = 1.0;
    cfg.early_stopping_round = 20;
    cfg.seed = 0;

    let train_ds = DatasetBuilder::from_rows(&train_x, n_train, d, &train_y, &cfg).unwrap();
    let valid_ds = DatasetBuilder::from_rows(&valid_x, n_valid, d, &valid_y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg)
        .fit(&train_ds, Some(&valid_ds))
        .unwrap();

    assert!(model.n_trees() > 0);

    // Compare logloss of init prediction vs. trained model on validation set.
    let init_only_scores = vec![model.init_score(); n_valid];
    let init_loss = binary_logloss(&init_only_scores, &valid_y);

    let final_scores = model.predict_raw_scores(&valid_x, n_valid);
    let final_loss = binary_logloss(&final_scores, &valid_y);

    println!("init_loss={init_loss:.5} final_loss={final_loss:.5}");
    assert!(
        final_loss < init_loss * 0.9,
        "model failed to improve appreciably: {init_loss} -> {final_loss}"
    );
    assert!(final_loss < 0.6, "final loss too high: {final_loss}");
}

#[test]
fn saved_model_round_trip_matches_predictions() {
    let n = 500;
    let d = 4;
    let (x, y, _, _) = make_classification(n, 1, d, 7);
    let mut cfg = Config::default();
    cfg.num_iterations = 10;
    cfg.num_leaves = 15;
    cfg.min_data_in_leaf = 10;
    cfg.max_bin = 32;
    let ds = DatasetBuilder::from_rows(&x, n, d, &y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let tmp = std::env::temp_dir().join("nanogbm_model.bin");
    model.save(&tmp).unwrap();
    let loaded = Model::load(&tmp).unwrap();

    let p1 = model.predict_proba(&x, n);
    let p2 = loaded.predict_proba(&x, n);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn predict_bin_and_raw_paths_agree() {
    let n = 400;
    let d = 3;
    let (x, y, _, _) = make_classification(n, 1, d, 9);

    let mut cfg = Config::default();
    cfg.num_iterations = 8;
    cfg.num_leaves = 11;
    cfg.min_data_in_leaf = 5;
    cfg.max_bin = 32;
    let ds = DatasetBuilder::from_rows(&x, n, d, &y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let raw = model.predict_raw_scores(&x, n);
    let bin_raw = model.predict_raw_scores_on_dataset(&ds);

    let max_abs = raw
        .iter()
        .zip(bin_raw.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs < 1e-9,
        "raw vs bin predictions diverged by {max_abs}"
    );
}

#[test]
fn predict_proba_binned_matches_raw_proba() {
    // Binned predict path must produce the same predictions as the raw f64
    // predict path: the trees carry both `threshold` and `threshold_bin`, and
    // the model now stores `bin_mappers` so the binned input gets the same
    // bin codes the trees were trained against.
    let n_train = 800;
    let n_eval = 1500;
    let d = 6;
    let (tx, ty, ex, _ey) = make_classification(n_train, n_eval, d, 17);

    let mut cfg = Config::default();
    cfg.num_iterations = 15;
    cfg.num_leaves = 21;
    cfg.min_data_in_leaf = 5;
    cfg.max_bin = 64;
    cfg.lambda_l2 = 0.5;

    let train_ds = DatasetBuilder::from_rows(&tx, n_train, d, &ty, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&train_ds, None).unwrap();

    let raw_proba = model.predict_proba(&ex, n_eval);
    let binned_proba = model.predict_proba_binned(&ex, n_eval);

    let max_abs = raw_proba
        .iter()
        .zip(binned_proba.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs < 1e-9,
        "raw vs binned proba diverged by {max_abs}"
    );
}

#[test]
fn binned_predict_survives_bincode_round_trip() {
    // Saved/restored Model must still produce identical binned predictions:
    // `bin_mappers` need to round-trip through bincode.
    let n = 600;
    let d = 4;
    let (x, y, ex, _) = make_classification(n, n / 2, d, 31);
    let mut cfg = Config::default();
    cfg.num_iterations = 12;
    cfg.num_leaves = 15;
    cfg.min_data_in_leaf = 10;
    cfg.max_bin = 48;
    let ds = DatasetBuilder::from_rows(&x, n, d, &y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let tmp = std::env::temp_dir().join("nanogbm_model_binned.bin");
    model.save(&tmp).unwrap();
    let loaded = Model::load(&tmp).unwrap();

    let before = model.predict_proba_binned(&ex, n / 2);
    let after = loaded.predict_proba_binned(&ex, n / 2);
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}
