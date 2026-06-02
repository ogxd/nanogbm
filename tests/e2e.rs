use nanogbm::loss::binary_logloss;
use nanogbm::{Config, FeatureBuilder, GbdtTrainer, Model};
use rand::SeedableRng;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// A training row is just a vector of `d` feature values.
type Row = Vec<f64>;

/// Build a `FeatureBuilder` with `d` numeric features indexing into a `Row`.
fn feature_builder(d: usize) -> FeatureBuilder<Row> {
    let mut fb = FeatureBuilder::<Row>::new();
    for j in 0..d {
        fb = fb.add_continuous(format!("f{j}"), move |r: &Row| r[j]);
    }
    fb
}

/// Logloss computed directly from predicted probabilities (the public predict
/// surface returns probabilities, not raw logits).
fn proba_logloss(probs: &[f64], labels: &[f32]) -> f64 {
    const EPS: f64 = 1e-15;
    let n = probs.len() as f64;
    let mut total = 0.0;
    for (&p, &y) in probs.iter().zip(labels.iter()) {
        let p = p.clamp(EPS, 1.0 - EPS);
        let y = y as f64;
        total += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    total / n
}

/// Generate a noisy linearly-separable binary classification problem. `train_n` rows
/// for training, `valid_n` for validation, with the same underlying weights/bias.
fn make_classification(
    train_n: usize,
    valid_n: usize,
    d: usize,
    seed: u64,
) -> (Vec<Row>, Vec<f32>, Vec<Row>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let weights: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let bias: f64 = rng.gen_range(-0.5..0.5);
    let mut sample = |n: usize| -> (Vec<Row>, Vec<f32>) {
        let mut rows = Vec::with_capacity(n);
        let mut labels = vec![0f32; n];
        for label in labels.iter_mut() {
            let mut z = bias;
            let mut row = vec![0.0; d];
            for j in 0..d {
                let x: f64 = rng.gen_range(-3.0..3.0);
                row[j] = x;
                z += weights[j] * x;
            }
            z += rng.gen_range(-0.4..0.4);
            let p = 1.0 / (1.0 + (-z).exp());
            *label = if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 };
            rows.push(row);
        }
        (rows, labels)
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

    let fb = feature_builder(d);
    let model = GbdtTrainer::new(&cfg, &fb)
        .fit(&train_x, &train_y, Some((&valid_x, &valid_y)))
        .unwrap();

    assert!(model.n_trees() > 0);

    // Compare logloss of init prediction vs. trained model on validation set.
    let init_only_scores = vec![model.init_score(); n_valid];
    let init_loss = binary_logloss(&init_only_scores, &valid_y);

    let final_probs = model.predict_proba(&fb, &valid_x);
    let final_loss = proba_logloss(&final_probs, &valid_y);

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
    let fb = feature_builder(d);
    let model = GbdtTrainer::new(&cfg, &fb).fit(&x, &y, None).unwrap();

    let tmp = std::env::temp_dir().join("nanogbm_model.bin");
    model.save(&tmp).unwrap();
    let loaded = Model::load(&tmp).unwrap();

    let p1 = model.predict_proba(&fb, &x);
    let p2 = loaded.predict_proba(&fb, &x);
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn predict_proba_is_batch_invariant() {
    // The single predict path walks tree-outer/row-inner over a row-major bin
    // buffer; a row's prediction must not depend on how many rows share the
    // batch (per-request batches are tiny, bulk eval batches are huge).
    let n = 400;
    let d = 3;
    let (x, y, _, _) = make_classification(n, 1, d, 9);

    let mut cfg = Config::default();
    cfg.num_iterations = 8;
    cfg.num_leaves = 11;
    cfg.min_data_in_leaf = 5;
    cfg.max_bin = 32;
    let fb = feature_builder(d);
    let model = GbdtTrainer::new(&cfg, &fb).fit(&x, &y, None).unwrap();

    let bulk = model.predict_proba(&fb, &x);
    // Same rows predicted one-by-one must match the bulk batch bit-for-bit.
    for (i, row) in x.iter().enumerate() {
        let single = model.predict_proba(&fb, std::slice::from_ref(row));
        assert_eq!(single[0], bulk[i], "row {i} differed between single and bulk predict");
    }
}

#[test]
fn categorical_split_learns_noncontiguous_subset() {
    // Feature 0 is noise; feature 1 is a categorical id in 0..12 whose label
    // depends on membership in a NON-contiguous "bidding" subset {1,4,7,9}.
    // A numeric threshold on the (arbitrary-order) id cannot separate this; a
    // per-node subset split can.
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let bidders = [1.0, 4.0, 7.0, 9.0];
    let mut make = |n: usize| -> (Vec<Row>, Vec<f32>) {
        let mut rows = Vec::with_capacity(n);
        let mut labels = vec![0f32; n];
        for label in labels.iter_mut() {
            let noise: f64 = rng.gen_range(-3.0..3.0);
            let cat = rng.gen_range(0..12) as f64;
            let p = if bidders.contains(&cat) { 0.85 } else { 0.1 };
            *label = if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 };
            rows.push(vec![noise, cat]);
        }
        (rows, labels)
    };
    let (tx, ty) = make(6000);
    let (ex, ey) = make(3000);

    let mut cfg = Config::default();
    cfg.num_iterations = 60;
    cfg.learning_rate = 0.1;
    cfg.num_leaves = 15;
    cfg.min_data_in_leaf = 20;
    cfg.max_bin = 32;
    cfg.feature_fraction = 1.0;
    cfg.bagging_fraction = 1.0;

    let fb = FeatureBuilder::<Row>::new()
        .add_continuous("noise", |r: &Row| r[0])
        .add_categorical("cat", |r: &Row| r[1] as i64);
    let model = GbdtTrainer::new(&cfg, &fb).fit(&tx, &ty, None).unwrap();

    // A categorical split must have been chosen.
    let cat_splits: usize = model.trees().iter().map(|t| t.category_sets.len()).sum();
    assert!(cat_splits > 0, "expected at least one categorical split");

    // The model must actually separate bidders from non-bidders through the
    // categorical (subset-membership) route.
    let probs = model.predict_proba(&fb, &ex);
    let ll = proba_logloss(&probs, &ey);
    assert!(ll < 0.45, "logloss {ll} too high; categorical subset not learned");
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
    let fb = feature_builder(d);
    let model = GbdtTrainer::new(&cfg, &fb).fit(&x, &y, None).unwrap();

    let tmp = std::env::temp_dir().join("nanogbm_model_binned.bin");
    model.save(&tmp).unwrap();
    let loaded = Model::load(&tmp).unwrap();

    let before = model.predict_proba(&fb, &ex);
    let after = loaded.predict_proba(&fb, &ex);
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}
