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
        Ok(())
    }
}
