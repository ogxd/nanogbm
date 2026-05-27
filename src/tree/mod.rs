pub mod histogram;
pub mod learner;
pub mod split;

use serde::{Deserialize, Serialize};

pub use learner::TreeLearner;

/// Direction the missing values go on a split. `#[repr(u8)]` so it occupies
/// exactly one byte inside [`SplitNode`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingDir {
    Left = 0,
    Right = 1,
}

/// A single internal split node in a tree, **inference-tight**.
///
/// Layout (16 bytes, 4 nodes per cache line on x86-64 / aarch64):
/// ```text
///   feature       u32  (4 bytes)
///   threshold_bin u16  (2 bytes)
///   missing_dir   u8   (1 byte)
///   _pad          1
///   left_child    i32  (4 bytes)
///   right_child   i32  (4 bytes)
/// ```
///
/// The f64 raw `threshold` and the f64 `gain` used to live here but they're
/// only read by the raw-features predict path and the feature-importance
/// reporter respectively. Moving them to parallel arrays on [`Tree`] shrinks
/// this struct from ~40 to 16 bytes — 2-3× more nodes per cache line during
/// inference, which is the single biggest win for `predict_*_binned` and
/// `predict_*_on_dataset` on large eval batches.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitNode {
    pub feature: u32,
    /// Inclusive upper bound for the "left" branch in bin-code space.
    pub threshold_bin: u16,
    pub missing_dir: MissingDir,
    /// negative = leaf index (`!leaf_idx`), positive = internal node index.
    pub left_child: i32,
    pub right_child: i32,
}

/// A trained tree: collection of internal nodes + leaf values.
///
/// `node_thresholds` and `node_gains` are parallel arrays to `nodes` (same
/// length). They're kept off the inference-hot [`SplitNode`] because only
/// [`Tree::predict_raw`] reads `threshold`, and only
/// [`crate::Model::feature_importance_gain`] reads `gain`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<SplitNode>,
    /// Per-node raw-value threshold (parallel to `nodes`). Read only by
    /// [`Tree::predict_raw`]; the binned paths don't touch it.
    pub node_thresholds: Vec<f64>,
    /// Per-node split gain (parallel to `nodes`). Read only by
    /// feature-importance reporting.
    pub node_gains: Vec<f64>,
    pub leaf_values: Vec<f64>,
}

impl Tree {
    /// A trivial tree that returns `value` for any input.
    pub fn constant(value: f64) -> Self {
        Self {
            nodes: Vec::new(),
            node_thresholds: Vec::new(),
            node_gains: Vec::new(),
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
            let i = node_idx as usize;
            let node = &self.nodes[i];
            let threshold = self.node_thresholds[i];
            let v = row[node.feature as usize];
            let go_left = if !v.is_finite() {
                matches!(node.missing_dir, MissingDir::Left)
            } else {
                v <= threshold
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
    /// the tree and indexing into per-feature columns. Convenience wrapper
    /// around [`Tree::predict_on_columns`] — does an enum match on
    /// `dataset.bin_data` per call (acceptable for one-shot use; for batch
    /// predict the [`crate::Model::predict_*_on_dataset`] paths hoist this
    /// match out of the per-node loop).
    pub fn predict_on_dataset(&self, dataset: &crate::dataset::Dataset, row: usize) -> f64 {
        use crate::dataset::BinWidth;
        match dataset.bin_width() {
            BinWidth::U8 => {
                let cols: Vec<&[u8]> = (0..dataset.n_features())
                    .map(|f| dataset.feature_column_u8(f))
                    .collect();
                self.predict_on_columns(&cols, row)
            }
            BinWidth::U16 => {
                let cols: Vec<&[u16]> = (0..dataset.n_features())
                    .map(|f| dataset.feature_column_u16(f))
                    .collect();
                self.predict_on_columns(&cols, row)
            }
        }
    }

    /// Predict for one row given pre-collected column slices. Generic over the
    /// bin element type so the per-node comparison happens in the column's
    /// native width (u8 vs u16) — no per-node enum match, no per-element
    /// widening to u16.
    ///
    /// Hot path: callers (`predict_*_on_dataset`) collect `&[&[B]]` once per
    /// batch and reuse it across every row × tree, paying the dispatch cost
    /// once instead of per node visit.
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
        // SAFETY: node_idx is bootstrapped at 0 and only updated via
        // `left_child` / `right_child` values that the learner emits as valid
        // indices into `self.nodes` (or negative leaf encodings). `columns`
        // has one entry per dataset feature and each is long enough to cover
        // `row` (DatasetBuilder invariant).
        unsafe {
            loop {
                let node = self.nodes.get_unchecked(node_idx as usize);
                let col = *columns.get_unchecked(node.feature as usize);
                let bin = *col.get_unchecked(row);
                let go_left = if bin == B::MISSING {
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
