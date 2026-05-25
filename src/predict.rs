use crate::model::Model;
use crate::objective::Objective;
use crate::objective::binary::sigmoid;

/// Predict raw additive scores for a row-major feature matrix.
pub fn predict_raw_scores(model: &Model, features: &[f64], n_rows: usize, n_features: usize) -> Vec<f64> {
    let init = model.init_score;
    (0..n_rows)
        .map(|row| {
            let r = &features[row * n_features..(row + 1) * n_features];
            let mut s = init;
            for tree in &model.trees {
                s += model.learning_rate * tree.predict_raw(r);
            }
            s
        })
        .collect()
}

/// Predict probabilities (sigmoid of raw scores).
pub fn predict_proba(model: &Model, features: &[f64], n_rows: usize, n_features: usize) -> Vec<f64> {
    let raw = predict_raw_scores(model, features, n_rows, n_features);
    raw.into_iter().map(sigmoid).collect()
}

/// Apply an objective's output-transform to raw scores in place (convenience).
pub fn convert_with(obj: &dyn Objective, raw: &mut [f64]) {
    let raw_copy = raw.to_vec();
    obj.convert_output(&raw_copy, raw);
}
