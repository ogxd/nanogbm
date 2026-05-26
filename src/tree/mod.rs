pub mod histogram;
pub mod learner;
pub mod split;

use serde::{Deserialize, Serialize};

pub use learner::TreeLearner;

/// Direction the missing values go on a split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingDir {
    Left,
    Right,
}

/// A single internal split node in a tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitNode {
    pub feature: u32,
    /// Inclusive upper bound for the "left" branch in raw value space.
    pub threshold: f64,
    /// Inclusive upper bound for the "left" branch in bin-code space (for fast
    /// training-time prediction on a bin-encoded dataset).
    pub threshold_bin: u16,
    pub missing_dir: MissingDir,
    pub left_child: i32, // negative = leaf index (~leaf_idx), positive = internal node index
    pub right_child: i32,
    pub gain: f64,
}

/// A trained tree: collection of internal nodes + leaf values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<SplitNode>,
    pub leaf_values: Vec<f64>,
}

impl Tree {
    /// A trivial tree that returns `value` for any input.
    pub fn constant(value: f64) -> Self {
        Self {
            nodes: Vec::new(),
            leaf_values: vec![value],
        }
    }

    /// Predict for a single raw-value feature row.
    pub fn predict_raw(&self, row: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut node_idx: i32 = 0;
        loop {
            let node = &self.nodes[node_idx as usize];
            let v = row[node.feature as usize];
            let go_left = if !v.is_finite() {
                matches!(node.missing_dir, MissingDir::Left)
            } else {
                v <= node.threshold
            };
            let next = if go_left {
                node.left_child
            } else {
                node.right_child
            };
            if next < 0 {
                return self.leaf_values[(!next) as usize];
            }
            node_idx = next;
        }
    }

    /// Predict for a row given by its bin codes (fast path during training).
    pub fn predict_bin_row(&self, row_bins: &[u16]) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut node_idx: i32 = 0;
        loop {
            let node = &self.nodes[node_idx as usize];
            let bin = row_bins[node.feature as usize];
            let go_left = if bin == crate::dataset::MISSING_BIN {
                matches!(node.missing_dir, MissingDir::Left)
            } else {
                bin <= node.threshold_bin
            };
            let next = if go_left {
                node.left_child
            } else {
                node.right_child
            };
            if next < 0 {
                return self.leaf_values[(!next) as usize];
            }
            node_idx = next;
        }
    }

    /// Predict for one row of a column-major bin-encoded dataset, by walking
    /// the tree and indexing into per-feature columns.
    pub fn predict_on_dataset(&self, dataset: &crate::dataset::Dataset, row: usize) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut node_idx: i32 = 0;
        loop {
            let node = &self.nodes[node_idx as usize];
            let bin = dataset.feature_column(node.feature as usize)[row];
            let go_left = if bin == crate::dataset::MISSING_BIN {
                matches!(node.missing_dir, MissingDir::Left)
            } else {
                bin <= node.threshold_bin
            };
            let next = if go_left {
                node.left_child
            } else {
                node.right_child
            };
            if next < 0 {
                return self.leaf_values[(!next) as usize];
            }
            node_idx = next;
        }
    }
}
