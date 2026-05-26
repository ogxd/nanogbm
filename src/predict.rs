//! Free-function aliases for the inference methods on [`Model`].
//!
//! These exist for back-compatibility and one-line call sites. The canonical
//! API lives on [`Model`] itself ([`Model::predict_raw_scores`],
//! [`Model::predict_proba`], [`Model::predict_raw_scores_on_dataset`],
//! [`Model::predict_proba_on_dataset`]).

use crate::model::Model;
use crate::objective::Objective;

/// See [`Model::predict_raw_scores`]. `n_features` must equal `model.n_features()`.
pub fn predict_raw_scores(
    model: &Model,
    features: &[f64],
    n_rows: usize,
    n_features: usize,
) -> Vec<f64> {
    assert_eq!(
        n_features,
        model.n_features(),
        "n_features {} does not match model.n_features() {}",
        n_features,
        model.n_features()
    );
    model.predict_raw_scores(features, n_rows)
}

/// See [`Model::predict_proba`]. `n_features` must equal `model.n_features()`.
pub fn predict_proba(
    model: &Model,
    features: &[f64],
    n_rows: usize,
    n_features: usize,
) -> Vec<f64> {
    assert_eq!(
        n_features,
        model.n_features(),
        "n_features {} does not match model.n_features() {}",
        n_features,
        model.n_features()
    );
    model.predict_proba(features, n_rows)
}

/// Apply an objective's output-transform to raw scores in place.
pub fn convert_with(obj: &dyn Objective, raw: &mut [f64]) {
    let raw_copy = raw.to_vec();
    obj.convert_output(&raw_copy, raw);
}
