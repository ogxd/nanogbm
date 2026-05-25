use nanogbm::feature::Schema;
use nanogbm::metric::{BinaryLogloss, Metric};
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

    let schema = Schema::all_numerical(d);
    let train_ds = DatasetBuilder::from_rows(&train_x, n_train, &schema, &train_y, &cfg).unwrap();
    let valid_ds = DatasetBuilder::from_rows(&valid_x, n_valid, &schema, &valid_y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg)
        .fit(&train_ds, Some(&valid_ds))
        .unwrap();

    assert!(!model.trees.is_empty());

    // Compare logloss of init prediction vs. trained model on validation set.
    let metric = BinaryLogloss;
    let init_only_scores = vec![model.init_score; n_valid];
    let init_loss = metric.evaluate(&init_only_scores, &valid_y);

    let final_scores = nanogbm::predict::predict_raw_scores(&model, &valid_x, n_valid, d);
    let final_loss = metric.evaluate(&final_scores, &valid_y);

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
    let schema = Schema::all_numerical(d);
    let ds = DatasetBuilder::from_rows(&x, n, &schema, &y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let tmp = std::env::temp_dir().join("nanogbm_model.bin");
    model.save(&tmp).unwrap();
    let loaded = Model::load(&tmp).unwrap();

    let p1 = nanogbm::predict::predict_proba(&model, &x, n, d);
    let p2 = nanogbm::predict::predict_proba(&loaded, &x, n, d);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn predict_bin_and_raw_paths_agree() {
    // Mixed dataset: 2 numerical, 1 categorical (column index 2). The categorical
    // values are integers; routing them via bin_mappers in the raw path must
    // give the same predictions as routing via the bin-encoded dataset.
    let n = 400;
    let d = 3;
    let (mut x, y, _, _) = make_classification(n, 1, 2, 9);
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut x_full = Vec::with_capacity(n * d);
    for row in 0..n {
        x_full.push(x[row * 2]);
        x_full.push(x[row * 2 + 1]);
        x_full.push(rng.gen_range(0..50) as f64);
    }
    x = x_full;

    let mut cfg = Config::default();
    cfg.num_iterations = 8;
    cfg.num_leaves = 11;
    cfg.min_data_in_leaf = 5;
    cfg.max_bin = 32;
    cfg.max_cat_bin = 64;
    let schema = Schema::with_categorical_at(d, &[2]);
    let ds = DatasetBuilder::from_rows(&x, n, &schema, &y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    let raw = nanogbm::predict::predict_raw_scores(&model, &x, n, d);

    let mut bin_raw = vec![model.init_score; n];
    for tree in &model.trees {
        for row in 0..n {
            bin_raw[row] += model.learning_rate * tree.predict_on_dataset(&ds, row);
        }
    }

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

/// Generate a synthetic problem where the label depends on a shuffled, high-
/// cardinality integer ID feature with no ordinal meaning. Returns (train_x,
/// train_y, valid_x, valid_y) with one column = id.
fn make_shuffled_id_problem(
    train_n: usize,
    valid_n: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f32>, Vec<f64>, Vec<f32>) {
    // Many IDs and few classes so the per-ID statistics still have plenty of
    // samples but the integer value carries no ordinal info. With >>max_bin
    // distinct values, ordered-scan numerical splits can't isolate individual
    // classes; categorical subset splits can.
    let n_ids = 1000u32;
    let n_classes = 3;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let class_probs = [0.05f64, 0.5, 0.95];
    let id_to_class: Vec<usize> = (0..n_ids).map(|_| rng.gen_range(0..n_classes)).collect();
    // Shuffle so the integer value carries no ordinal information w.r.t. label.
    let mut order: Vec<u32> = (0..n_ids).collect();
    order.shuffle(&mut rng);
    let mut id_perm: Vec<u32> = vec![0; n_ids as usize];
    for (i, &orig) in order.iter().enumerate() {
        id_perm[orig as usize] = i as u32;
    }
    let sample = |n: usize, rng: &mut ChaCha8Rng| -> (Vec<f64>, Vec<f32>) {
        let mut x = vec![0.0; n];
        let mut y = vec![0f32; n];
        for i in 0..n {
            let orig = rng.gen_range(0..n_ids);
            let shuf = id_perm[orig as usize];
            x[i] = shuf as f64;
            let p = class_probs[id_to_class[orig as usize]];
            y[i] = if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 };
        }
        (x, y)
    };
    let (tx, ty) = sample(train_n, &mut rng);
    let (vx, vy) = sample(valid_n, &mut rng);
    (tx, ty, vx, vy)
}

#[test]
fn categorical_treatment_beats_numerical_on_shuffled_ids() {
    let n_train = 30000;
    let n_valid = 5000;
    let (train_x, train_y, valid_x, valid_y) = make_shuffled_id_problem(n_train, n_valid, 11);

    let base = |cfg: &mut Config| {
        cfg.num_iterations = 80;
        cfg.learning_rate = 0.1;
        cfg.num_leaves = 31;
        cfg.min_data_in_leaf = 20;
        cfg.max_bin = 64;
        cfg.lambda_l2 = 1.0;
        cfg.seed = 0;
    };

    let metric = BinaryLogloss;

    // (1) numerical treatment of the ID column.
    let mut cfg_num = Config::default();
    base(&mut cfg_num);
    let schema_num = Schema::all_numerical(1);
    let train_num = DatasetBuilder::from_rows(&train_x, n_train, &schema_num, &train_y, &cfg_num).unwrap();
    let model_num = GbdtTrainer::new(&cfg_num).fit(&train_num, None).unwrap();
    let scores_num = nanogbm::predict::predict_raw_scores(&model_num, &valid_x, n_valid, 1);
    let loss_num = metric.evaluate(&scores_num, &valid_y);

    // (2) categorical treatment of the ID column.
    let mut cfg_cat = Config::default();
    base(&mut cfg_cat);
    cfg_cat.max_cat_bin = 1024;
    cfg_cat.max_cat_threshold = 32;
    let schema_cat = Schema::with_categorical_at(1, &[0]);
    let train_cat = DatasetBuilder::from_rows(&train_x, n_train, &schema_cat, &train_y, &cfg_cat).unwrap();
    let model_cat = GbdtTrainer::new(&cfg_cat).fit(&train_cat, None).unwrap();
    let scores_cat = nanogbm::predict::predict_raw_scores(&model_cat, &valid_x, n_valid, 1);
    let loss_cat = metric.evaluate(&scores_cat, &valid_y);

    println!(
        "loss_numerical={loss_num:.5} loss_categorical={loss_cat:.5} ratio={:.3}",
        loss_cat / loss_num
    );
    assert!(
        loss_cat < loss_num * 0.85,
        "categorical treatment should beat numerical by >=15%: num={loss_num} cat={loss_cat}"
    );
}

#[test]
fn categorical_model_round_trip_and_path_consistency() {
    let n = 1500;
    let d = 3;
    let (num_x, y, _, _) = make_classification(n, 1, 2, 21);
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut x = Vec::with_capacity(n * d);
    for row in 0..n {
        x.push(num_x[row * 2]);
        x.push(num_x[row * 2 + 1]);
        x.push(rng.gen_range(0..120) as f64);
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 30;
    cfg.num_leaves = 15;
    cfg.min_data_in_leaf = 10;
    cfg.max_bin = 64;
    cfg.max_cat_bin = 128;
    let schema = Schema::with_categorical_at(d, &[2]);

    let ds = DatasetBuilder::from_rows(&x, n, &schema, &y, &cfg).unwrap();
    let model = GbdtTrainer::new(&cfg).fit(&ds, None).unwrap();

    // Save and reload; predictions must match the in-memory model exactly.
    let tmp = std::env::temp_dir().join("nanogbm_cat_model.bin");
    model.save(&tmp).unwrap();
    let loaded = Model::load(&tmp).unwrap();
    let p1 = nanogbm::predict::predict_proba(&model, &x, n, d);
    let p2 = nanogbm::predict::predict_proba(&loaded, &x, n, d);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-12, "round-trip diverged: {a} vs {b}");
    }

    // Raw and bin paths must agree on the same rows.
    let raw = nanogbm::predict::predict_raw_scores(&model, &x, n, d);
    let mut bin_raw = vec![model.init_score; n];
    for tree in &model.trees {
        for row in 0..n {
            bin_raw[row] += model.learning_rate * tree.predict_on_dataset(&ds, row);
        }
    }
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
