use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Hyperparameters for GBDT training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub num_leaves: usize,
    pub max_depth: i32,
    pub min_data_in_leaf: usize,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub min_gain_to_split: f64,
    pub max_bin: usize,
    pub min_data_in_bin: usize,
    pub feature_fraction: f64,
    pub bagging_fraction: f64,
    pub bagging_freq: usize,
    pub early_stopping_round: usize,
    pub seed: u64,
    pub verbose: bool,
    /// Distinct values per categorical column; excess values collapse to MISSING.
    #[serde(default = "default_max_cat_bin")]
    pub max_cat_bin: usize,
    /// Smoothing in the sort key grad/(hess + cat_smooth) for categorical scanning.
    #[serde(default = "default_cat_smooth")]
    pub cat_smooth: f64,
    /// Additional L2 used during categorical split gain.
    #[serde(default = "default_cat_l2")]
    pub cat_l2: f64,
    /// Cap on the number of bins on the "left" side of a categorical subset split.
    #[serde(default = "default_max_cat_threshold")]
    pub max_cat_threshold: usize,
    /// Multiplier applied to gradient and hessian for positive (y >= 0.5) samples.
    #[serde(default = "default_pos_weight")]
    pub pos_weight: f64,
}

fn default_max_cat_bin() -> usize {
    1024
}
fn default_cat_smooth() -> f64 {
    10.0
}
fn default_cat_l2() -> f64 {
    10.0
}
fn default_max_cat_threshold() -> usize {
    32
}
fn default_pos_weight() -> f64 {
    1.0
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
            max_bin: 255,
            min_data_in_bin: 3,
            feature_fraction: 1.0,
            bagging_fraction: 1.0,
            bagging_freq: 0,
            early_stopping_round: 0,
            seed: 0,
            verbose: false,
            max_cat_bin: default_max_cat_bin(),
            cat_smooth: default_cat_smooth(),
            cat_l2: default_cat_l2(),
            max_cat_threshold: default_max_cat_threshold(),
            pos_weight: default_pos_weight(),
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
        if !(0.0 < self.learning_rate && self.learning_rate <= 1.0) {
            return Err(Error::Config("learning_rate must be in (0, 1]".into()));
        }
        if !(0.0 < self.feature_fraction && self.feature_fraction <= 1.0) {
            return Err(Error::Config("feature_fraction must be in (0, 1]".into()));
        }
        if !(0.0 < self.bagging_fraction && self.bagging_fraction <= 1.0) {
            return Err(Error::Config("bagging_fraction must be in (0, 1]".into()));
        }
        if self.max_cat_bin < 2 {
            return Err(Error::Config("max_cat_bin must be >= 2".into()));
        }
        if self.cat_smooth < 0.0 {
            return Err(Error::Config("cat_smooth must be >= 0".into()));
        }
        if self.cat_l2 < 0.0 {
            return Err(Error::Config("cat_l2 must be >= 0".into()));
        }
        if self.max_cat_threshold < 1 {
            return Err(Error::Config("max_cat_threshold must be >= 1".into()));
        }
        if self.pos_weight <= 0.0 || self.pos_weight.is_nan() {
            return Err(Error::Config("pos_weight must be > 0".into()));
        }
        Ok(())
    }
}
