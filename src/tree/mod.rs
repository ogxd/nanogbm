pub mod histogram;
pub mod learner;
pub mod split;

use serde::{Deserialize, Serialize};

pub use learner::TreeLearner;

/// Direction the missing values go on a numerical split. For categorical
/// splits missing is always routed right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingDir {
    Left,
    Right,
}

/// How a split partitions rows on its feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitKind {
    /// Ordered split: bin <= threshold_bin goes left. Missing follows `missing_dir`.
    Numerical {
        threshold_bin: u16,
        threshold_value: f64,
        missing_dir: MissingDir,
    },
    /// Subset split: any bin in `left_bins` goes left, everything else (including
    /// MISSING bin 0) goes right. `left_bins` is sorted ascending for fast lookup.
    Categorical { left_bins: Vec<u16> },
}

/// A single internal split node in a tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitNode {
    pub feature: u32,
    pub kind: SplitKind,
    pub left_child: i32,
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

    /// Predict for a single raw-value feature row. The numerical case looks at
    /// the raw `f64`; the categorical case bins the value via the model's
    /// `BinMapper` (passed alongside the model). Because that mapping isn't
    /// reachable from here, callers must go through `predict::predict_raw_scores`
    /// which threads the bin mappers in. For tree-only traversal with raw
    /// values, this method assumes numerical splits and would mis-route
    /// categoricals; the `Model::predict_raw_*` helpers route through bin codes
    /// instead. See `predict.rs`.
    pub fn predict_raw(&self, row: &[f64]) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut node_idx: i32 = 0;
        loop {
            let node = &self.nodes[node_idx as usize];
            let v = row[node.feature as usize];
            let go_left = match &node.kind {
                SplitKind::Numerical {
                    threshold_value,
                    missing_dir,
                    ..
                } => {
                    if !v.is_finite() {
                        matches!(missing_dir, MissingDir::Left)
                    } else {
                        v <= *threshold_value
                    }
                }
                // Categorical predict_raw is intentionally a no-op: there's no
                // bin mapper here, so we send MISSING (and any value) right.
                // Real categorical raw-path prediction goes through `predict.rs`.
                SplitKind::Categorical { .. } => false,
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
            let go_left = bin_goes_left(&node.kind, bin);
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
            let go_left = bin_goes_left(&node.kind, bin);
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

/// Per-node "does this bin code go left" decision. Categorical: bin must be in
/// `left_bins` (MISSING bin 0 is never included, so MISSING goes right).
#[inline]
pub(crate) fn bin_goes_left(kind: &SplitKind, bin: u16) -> bool {
    match kind {
        SplitKind::Numerical {
            threshold_bin,
            missing_dir,
            ..
        } => {
            if bin == crate::dataset::MISSING_BIN {
                matches!(missing_dir, MissingDir::Left)
            } else {
                bin <= *threshold_bin
            }
        }
        SplitKind::Categorical { left_bins } => left_bins.binary_search(&bin).is_ok(),
    }
}
