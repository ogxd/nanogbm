pub mod histogram;
pub mod learner;
pub mod split;

use serde::{Deserialize, Serialize};

pub use learner::TreeLearner;

/// `#[repr(u8)]` keeps the field at one byte inside [`SplitNode`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingDir {
    Left = 0,
    Right = 1,
}

/// Inference-tight split node. 16 bytes — 4 per cache line:
/// ```text
///   feature       u32  (4 bytes)
///   threshold_bin u16  (2 bytes)
///   missing_dir   u8   (1 byte)
///   _pad          1
///   left_child    i32  (4 bytes)
///   right_child   i32  (4 bytes)
/// ```
/// `threshold` (f64) and `gain` (f64) live on parallel arrays on [`Tree`]
/// because only the raw-f64 predict path and feature-importance reporter
/// read them; keeping them off the hot node fits 4× more nodes per line.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitNode {
    pub feature: u32,
    /// Numeric: inclusive left-branch bound in bin-code space. Categorical:
    /// index into [`Tree::category_sets`].
    pub threshold_bin: u16,
    pub missing_dir: MissingDir,
    /// `1` if this is a categorical (subset-membership) split, else `0`.
    pub is_categorical: u8,
    /// Negative encodes a leaf index as `!idx`; non-negative is an
    /// internal-node index.
    pub left_child: i32,
    pub right_child: i32,
}

/// True iff bit `bin` is set in the little-endian `u64` bitset `set`. Bins
/// beyond the bitset (e.g. unseen categories) count as not-in-the-left-set.
#[inline]
pub(crate) fn bitset_contains(set: &[u64], bin: usize) -> bool {
    let word = bin / 64;
    word < set.len() && (set[word] >> (bin % 64)) & 1 == 1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<SplitNode>,
    /// Parallel to `nodes`. Read only by [`Tree::predict_raw`].
    pub node_thresholds: Vec<f64>,
    /// Parallel to `nodes`. Read only by feature-importance reporting.
    pub node_gains: Vec<f64>,
    /// Left-branch bin bitsets for categorical nodes. A categorical
    /// [`SplitNode`] stores its index here in `threshold_bin`. Empty when the
    /// tree has no categorical splits.
    #[serde(default)]
    pub category_sets: Vec<Vec<u64>>,
    pub leaf_values: Vec<f64>,
}

impl Tree {
    /// A trivial tree that returns `value` for any input.
    pub fn constant(value: f64) -> Self {
        Self {
            nodes: Vec::new(),
            node_thresholds: Vec::new(),
            node_gains: Vec::new(),
            category_sets: Vec::new(),
            leaf_values: vec![value],
        }
    }

    /// Predict for one row of row-major `u16` bin codes (`row[feature]` is the
    /// bin code for that feature). Same routing as [`Tree::predict_on_columns`]
    /// but reads a contiguous per-row slice instead of column slices — used by
    /// the small-batch predict path where the row fits in L1 and the
    /// column-major materialization isn't worth its allocations.
    #[inline]
    pub fn predict_on_row_bins(&self, row: &[u16]) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut node_idx: i32 = 0;
        // SAFETY: child pointers address a valid node or an encoded leaf; `row`
        // has one entry per feature, and `node.feature` is bounded by the model
        // feature count (check_features upholds this on the public entry point).
        unsafe {
            loop {
                let node = self.nodes.get_unchecked(node_idx as usize);
                let bin = *row.get_unchecked(node.feature as usize);
                let go_left = if node.is_categorical == 1 {
                    bitset_contains(
                        self.category_sets.get_unchecked(node.threshold_bin as usize),
                        bin as usize,
                    )
                } else if bin == crate::dataset::MISSING_BIN {
                    matches!(node.missing_dir, MissingDir::Left)
                } else {
                    bin <= node.threshold_bin
                };
                let next = if go_left { node.left_child } else { node.right_child };
                if next < 0 {
                    return *self.leaf_values.get_unchecked((!next) as usize);
                }
                node_idx = next;
            }
        }
    }

    /// Predict for one row of a column-major bin-encoded dataset. The bin-width
    /// dispatch happens per call — fine for one-shot use, but batch paths
    /// (`Model::predict_*_on_dataset`) hoist it out of the row loop.
    pub fn predict_on_dataset(&self, dataset: &crate::dataset::Dataset, row: usize) -> f64 {
        use crate::dataset::with_columns;
        let feats: Vec<usize> = (0..dataset.n_features()).collect();
        with_columns!(dataset, feats, |cols| { self.predict_on_columns(&cols, row) })
    }

    /// Predict for one row from pre-collected column slices. Generic over
    /// `B: Bin` so the per-node comparison stays in the column's native
    /// width.
    #[inline]
    pub fn predict_on_columns<B: crate::dataset::Bin>(
        &self,
        columns: &[&[B]],
        row: usize,
    ) -> f64 {
        if self.nodes.is_empty() {
            return self.leaf_values[0];
        }
        let mut node_idx: i32 = 0;
        // SAFETY: child pointers come from the learner and address either a
        // valid node index or an encoded leaf. Column lengths cover `row` by
        // dataset builder invariant.
        unsafe {
            loop {
                let node = self.nodes.get_unchecked(node_idx as usize);
                let col = *columns.get_unchecked(node.feature as usize);
                let bin = *col.get_unchecked(row);
                let go_left = if node.is_categorical == 1 {
                    bitset_contains(
                        self.category_sets.get_unchecked(node.threshold_bin as usize),
                        bin.as_usize(),
                    )
                } else if bin == B::MISSING {
                    matches!(node.missing_dir, MissingDir::Left)
                } else {
                    bin.as_usize() <= node.threshold_bin as usize
                };
                let next = if go_left {
                    node.left_child
                } else {
                    node.right_child
                };
                if next < 0 {
                    return *self.leaf_values.get_unchecked((!next) as usize);
                }
                node_idx = next;
            }
        }
    }
}
