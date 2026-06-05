use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Hyperparameters for GBDT training. Names match LightGBM conventions.
///
/// Construct with `Config { num_iterations: 100, ..Config::default() }`.
/// Call [`Config::validate`] (the trainer does so automatically at the top of
/// `fit`) to surface out-of-range values as [`crate::Error::Config`] instead of
/// silently misbehaving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Maximum number of boosting rounds (trees) to fit. With
    /// `early_stopping_round > 0` and a validation set, training may stop
    /// sooner and the model is truncated to the best iteration.
    pub num_iterations: usize,

    /// Shrinkage applied to each tree's leaf values when added to the running
    /// score: `score += learning_rate * leaf_value`. Smaller values need more
    /// iterations but generalize better. Must be in `(0, 1]`.
    pub learning_rate: f64,

    /// Maximum number of leaves per tree. Together with `max_depth`, controls
    /// tree complexity. Must be `>= 2`.
    pub num_leaves: usize,

    /// Maximum tree depth. `-1` means unlimited (let `num_leaves` cap things).
    /// Use a positive cap to discourage very deep, narrow trees on noisy data.
    pub max_depth: i32,

    /// Minimum number of samples in a leaf. Splits that would produce a leaf
    /// with fewer rows are rejected. Higher = stronger regularization.
    pub min_data_in_leaf: usize,

    /// Minimum sum of hessians in a leaf. Cheaper proxy for "this leaf has
    /// enough signal to be worth splitting"; complements `min_data_in_leaf`.
    pub min_sum_hessian_in_leaf: f64,

    /// L1 regularization on leaf values (soft-threshold in the gain formula).
    pub lambda_l1: f64,

    /// L2 regularization on leaf values (denominator term in the gain formula).
    pub lambda_l2: f64,

    /// Minimum loss reduction required to accept a split. Defaults to `0.0`;
    /// raise it to prune trivially-improving splits on noisy datasets.
    pub min_gain_to_split: f64,

    /// Categorical split smoothing (LightGBM `cat_smooth`): added to each
    /// category's hessian when ranking categories by gradient ratio, so a
    /// low-count category can't jump to an extreme rank. Default `10.0`.
    pub cat_smooth: f64,

    /// Extra L2 regularization applied to categorical-split gains (LightGBM
    /// `cat_l2`), on top of `lambda_l2`. Default `10.0`.
    pub cat_l2: f64,

    /// Max categories on the smaller side of a categorical subset split
    /// (LightGBM `max_cat_threshold`): bounds partition granularity. Default `32`.
    pub max_cat_threshold: usize,

    /// Minimum rows for a category to participate in a categorical split
    /// (LightGBM `min_data_per_group`); rarer categories are forced to the
    /// default (right) side. Default `100`.
    pub min_data_per_group: usize,

    /// Maximum bins per numerical feature (includes one slot for the MISSING
    /// bin, so at most `max_bin - 1` real bins). Must be in `[2, 65535]`.
    pub max_bin: usize,

    /// Maximum bins per categorical feature (includes one slot for MISSING, so
    /// at most `max_cat_bins - 1` distinct categories keep their own bin; the
    /// rarer tail collapses into MISSING). Decoupled from `max_bin` because a
    /// modest numeric histogram resolution and a high categorical cardinality
    /// are independent needs: high-cardinality ids (placements, domains,
    /// networks) lose most of their signal if forced down to `max_bin`. When the
    /// widest column exceeds 256 bins, storage transparently widens to `u16`.
    /// Must be in `[2, 65535]`.
    pub max_cat_bins: usize,

    /// Minimum samples per histogram bin when fitting numerical bin mappers.
    /// Bins with fewer samples get merged into neighbors.
    pub min_data_in_bin: usize,

    /// Per-tree feature subsample fraction in `(0, 1]`. `1.0` disables.
    pub feature_fraction: f64,

    /// Row subsample fraction in `(0, 1]`. `1.0` disables. Only takes effect
    /// when `bagging_freq > 0`.
    pub bagging_fraction: f64,

    /// Re-sample bagged rows every `K` iterations. `0` disables bagging.
    pub bagging_freq: usize,

    /// Stop after this many iterations without validation-metric improvement.
    /// Only active when a validation set is passed to `fit`. `0` disables.
    /// On stop, the model is truncated to `best_iter + 1` trees.
    pub early_stopping_round: usize,

    /// Seed for the `ChaCha8Rng` used for bagging and feature subsampling.
    /// Same config + same data => byte-identical model.
    pub seed: u64,

    /// If `true`, log per-iteration validation/training metric and an
    /// end-of-fit timing breakdown to stderr.
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            min_data_per_group: 100,
            max_bin: 255,
            max_cat_bins: 255,
            min_data_in_bin: 3,
            feature_fraction: 1.0,
            bagging_fraction: 1.0,
            bagging_freq: 0,
            early_stopping_round: 0,
            seed: 0,
            verbose: false,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.num_leaves < 2 {
            return Err(Error::Config("num_leaves must be >= 2".into()));
        }
        if self.max_bin < 2 || self.max_bin > 65535 {
            return Err(Error::Config("max_bin must be in [2, 65535]".into()));
        }
        if self.max_cat_bins < 2 || self.max_cat_bins > 65535 {
            return Err(Error::Config("max_cat_bins must be in [2, 65535]".into()));
        }
        if !(0.0 < self.learning_rate && self.learning_rate <= 1.0) {
            return Err(Error::Config("learning_rate must be in (0, 1]".into()));
        }
        if !(0.0 < self.feature_fraction && self.feature_fraction <= 1.0) {
            return Err(Error::Config("feature_fraction must be in (0, 1]".into()));
        }
        if !(0.0 < self.bagging_fraction && self.bagging_fraction <= 1.0) {
            return Err(Error::Config("bagging_fraction must be in (0, 1]".into()));
        }
        Ok(())
    }
}
