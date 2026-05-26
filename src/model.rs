use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::config::standard;
use serde::{Deserialize, Serialize};

use crate::dataset::{BinMapper, Dataset};
use crate::error::{Error, Result};
use crate::objective::binary::sigmoid;
use crate::tree::{SplitKind, Tree, bin_goes_left};

/// A trained GBDT model: ensemble of trees plus boosting metadata.
///
/// Fields are crate-private to keep model invariants (e.g. `bin_mappers.len() ==
/// n_features`) intact. Inspect a model through the accessor methods or the
/// `predict_*` methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub(crate) init_score: f64,
    pub(crate) learning_rate: f64,
    pub(crate) n_features: usize,
    pub(crate) trees: Vec<Tree>,
    /// Per-feature bin mapper, needed for routing raw-value categorical splits.
    /// `#[serde(default)]` keeps backward compatibility with pre-categorical models.
    #[serde(default)]
    pub(crate) bin_mappers: Vec<BinMapper>,
}

impl Model {
    /// Constant the boosting loop started from (the prior log-odds of the labels).
    pub fn init_score(&self) -> f64 {
        self.init_score
    }

    /// Shrinkage applied to every tree's leaf value at inference.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Number of input features this model expects per row.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of trees in the ensemble. After early stopping, this equals
    /// `best_iter + 1`, not the configured `num_iterations`.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// The trees themselves, in fit order.
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Per-feature bin mappers learned at training time. Reuse these when
    /// binning validation/inference data so val bins match train bins —
    /// see [`crate::dataset::DatasetBuilder::from_rows_with_mappers`].
    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }

    /// Bincode-serialize this model to `path`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        let bytes = bincode::serde::encode_to_vec(self, standard())
            .map_err(|e| Error::Serde(e.to_string()))?;
        w.write_all(&bytes)?;
        w.flush()?;
        Ok(())
    }

    /// Deserialize a model previously written by [`Model::save`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        let (model, _) = bincode::serde::decode_from_slice::<Model, _>(&buf, standard())
            .map_err(|e| Error::Serde(e.to_string()))?;
        Ok(model)
    }

    /// Number of times each feature was used as a split, indexed by feature.
    pub fn feature_importance_split(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.n_features];
        for tree in &self.trees {
            for node in &tree.nodes {
                counts[node.feature as usize] += 1;
            }
        }
        counts
    }

    /// Total split gain attributed to each feature, indexed by feature.
    pub fn feature_importance_gain(&self) -> Vec<f64> {
        let mut gains = vec![0.0f64; self.n_features];
        for tree in &self.trees {
            for node in &tree.nodes {
                gains[node.feature as usize] += node.gain;
            }
        }
        gains
    }

    /// Predict raw additive scores (pre-sigmoid logits) for a row-major feature
    /// matrix of shape `n_rows × self.n_features()`.
    ///
    /// # Panics
    /// Panics if `features.len() != n_rows * self.n_features()`.
    pub fn predict_raw_scores(&self, features: &[f64], n_rows: usize) -> Vec<f64> {
        let n_features = self.n_features;
        assert_eq!(
            features.len(),
            n_rows * n_features,
            "features.len() {} != n_rows {} * n_features {}",
            features.len(),
            n_rows,
            n_features
        );
        let init = self.init_score;
        let has_mappers = !self.bin_mappers.is_empty();
        (0..n_rows)
            .map(|row| {
                let r = &features[row * n_features..(row + 1) * n_features];
                let mut s = init;
                for tree in &self.trees {
                    let v = if has_mappers {
                        predict_tree_with_mappers(tree, r, &self.bin_mappers)
                    } else {
                        tree.predict_raw(r)
                    };
                    s += self.learning_rate * v;
                }
                s
            })
            .collect()
    }

    /// Predict probabilities (sigmoid of raw scores) for a row-major feature
    /// matrix of shape `n_rows × self.n_features()`.
    pub fn predict_proba(&self, features: &[f64], n_rows: usize) -> Vec<f64> {
        let raw = self.predict_raw_scores(features, n_rows);
        raw.into_iter().map(sigmoid).collect()
    }

    /// Predict raw additive scores against an already-binned dataset. Use this
    /// for fast inference paths where you can afford to bin once and predict
    /// many times; the dataset's bin mappers must match `self.bin_mappers()`.
    pub fn predict_raw_scores_on_dataset(&self, dataset: &Dataset) -> Vec<f64> {
        let n = dataset.n_rows();
        let mut scores = vec![self.init_score; n];
        for tree in &self.trees {
            for (row, s) in scores.iter_mut().enumerate() {
                *s += self.learning_rate * tree.predict_on_dataset(dataset, row);
            }
        }
        scores
    }

    /// Predict probabilities against an already-binned dataset.
    pub fn predict_proba_on_dataset(&self, dataset: &Dataset) -> Vec<f64> {
        let raw = self.predict_raw_scores_on_dataset(dataset);
        raw.into_iter().map(sigmoid).collect()
    }
}

/// Walk a single tree on a raw-value row, using `bin_mappers` to route through
/// the same code path as the bin-encoded predict (so categorical splits work).
#[inline]
fn predict_tree_with_mappers(tree: &Tree, row: &[f64], bin_mappers: &[BinMapper]) -> f64 {
    if tree.nodes.is_empty() {
        return tree.leaf_values[0];
    }
    let mut node_idx: i32 = 0;
    loop {
        let node = &tree.nodes[node_idx as usize];
        let feat = node.feature as usize;
        let v = row[feat];
        let go_left = match &node.kind {
            SplitKind::Numerical {
                threshold_value,
                missing_dir,
                ..
            } => {
                if !v.is_finite() {
                    matches!(missing_dir, crate::tree::MissingDir::Left)
                } else {
                    v <= *threshold_value
                }
            }
            SplitKind::Categorical { .. } => {
                let bin = bin_mappers[feat].value_to_bin(v);
                bin_goes_left(&node.kind, bin)
            }
        };
        let next = if go_left {
            node.left_child
        } else {
            node.right_child
        };
        if next < 0 {
            return tree.leaf_values[(!next) as usize];
        }
        node_idx = next;
    }
}
