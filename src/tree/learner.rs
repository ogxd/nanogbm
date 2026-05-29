use std::cell::Cell;
use std::time::Duration;

use crate::config::Config;
use crate::dataset::{Bin, Dataset, with_column, with_columns};
use crate::tree::histogram::{
    FeatureHistogram, build_histograms_batched, build_histograms_batched_full,
};
use crate::tree::split::{SplitInfo, find_best_split_for_feature, threshold_leaf};
use crate::tree::{MissingDir, SplitNode, Tree};

/// Cumulative per-phase wall-clock counters. `Cell` because training is
/// single-threaded; this lets the buckets live behind a `&self` borrow.
#[derive(Default)]
pub struct TimingBuckets {
    pub hist_build: Cell<Duration>,
    pub hist_subtract: Cell<Duration>,
    pub split_search: Cell<Duration>,
    pub partition: Cell<Duration>,
}

impl TimingBuckets {
    fn add(cell: &Cell<Duration>, d: Duration) {
        cell.set(cell.get() + d);
    }
}

/// Negative child pointers encode a leaf index as `!idx`. Non-negative is an
/// internal-node index in `tree.nodes`.
#[inline]
fn encode_leaf(idx: usize) -> i32 {
    !(idx as i32)
}

struct LeafState {
    indices: Vec<u32>,
    sum_grad: f64,
    sum_hess: f64,
    count: u32,
    /// Histograms parallel to `feature_indices`.
    histograms: Vec<FeatureHistogram>,
    best_split: Option<SplitInfo>,
    /// Parent internal-node index in `Tree.nodes`, or `-1` at the root.
    parent_node_idx: i32,
    is_left_child: bool,
}

pub struct TreeLearner<'a> {
    config: &'a Config,
    dataset: &'a Dataset,
    pub timing: &'a TimingBuckets,
}

impl<'a> TreeLearner<'a> {
    pub fn new(config: &'a Config, dataset: &'a Dataset, timing: &'a TimingBuckets) -> Self {
        Self {
            config,
            dataset,
            timing,
        }
    }

    /// Grow a single tree on the provided sample of rows and features.
    ///
    /// `gradhess[row] = [grad, hess]` is the packed gradient/hessian pair
    /// (one 8-byte load = both values in the histogram hot loop).
    ///
    /// `is_full` must be `true` iff `row_indices == 0..dataset.n_rows()`; in
    /// that case the root histograms use the sequential `_full` path instead
    /// of the indices indirection.
    ///
    /// Returns the tree plus a `row_to_leaf` map of length `dataset.n_rows()`.
    /// Rows absent from `row_indices` remain mapped to leaf 0 (the root
    /// prediction).
    pub fn train_one_tree(
        &self,
        gradhess: &[[f32; 2]],
        row_indices: &[u32],
        feature_indices: &[usize],
        is_full: bool,
    ) -> (Tree, Vec<u32>) {
        let mut tree = Tree {
            nodes: Vec::new(),
            node_thresholds: Vec::new(),
            node_gains: Vec::new(),
            leaf_values: Vec::new(),
        };

        let mut row_to_leaf: Vec<u32> = vec![0u32; self.dataset.n_rows()];

        let t0 = std::time::Instant::now();
        let mut root_histograms: Vec<FeatureHistogram> = feature_indices
            .iter()
            .map(|&feat| FeatureHistogram::zeros(self.dataset.bin_mapper(feat).num_bins()))
            .collect();
        with_columns!(self.dataset, feature_indices, |cols| {
            if is_full {
                build_histograms_batched_full(&cols, gradhess, &mut root_histograms);
            } else {
                build_histograms_batched(&cols, row_indices, gradhess, &mut root_histograms);
            }
        });
        TimingBuckets::add(&self.timing.hist_build, t0.elapsed());

        let root_grad: f64 = row_indices
            .iter()
            .map(|&i| gradhess[i as usize][0] as f64)
            .sum();
        let root_hess: f64 = row_indices
            .iter()
            .map(|&i| gradhess[i as usize][1] as f64)
            .sum();
        let root_count = row_indices.len() as u32;

        tree.leaf_values
            .push(self.compute_leaf_value(root_grad, root_hess));

        let mut leaves: Vec<LeafState> = vec![LeafState {
            indices: row_indices.to_vec(),
            sum_grad: root_grad,
            sum_hess: root_hess,
            count: root_count,
            histograms: root_histograms,
            best_split: None,
            parent_node_idx: -1,
            is_left_child: false,
        }];

        self.update_best_split(&mut leaves[0], feature_indices);

        // Leaf-wise growth: at each step split the leaf with the highest gain,
        // until `num_leaves` is reached or no positive-gain split remains.
        while leaves.len() < self.config.num_leaves {
            let best_idx = leaves
                .iter()
                .enumerate()
                .filter_map(|(i, l)| l.best_split.as_ref().map(|s| (i, s.gain)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let Some(best_idx) = best_idx else { break };

            let parent = leaves.swap_remove(best_idx);
            let split = parent.best_split.clone().expect("checked above");

            // Exact left/right counts come from SplitInfo, so pre-size the Vecs
            // and use set_len to skip the per-push capacity check.
            let t_p = std::time::Instant::now();
            let mut left_indices: Vec<u32> = Vec::with_capacity(split.left_count as usize);
            let mut right_indices: Vec<u32> = Vec::with_capacity(split.right_count as usize);
            let missing_goes_left = matches!(split.missing_dir, MissingDir::Left);
            with_column!(self.dataset, split.feature, |col| {
                partition_indices(
                    col,
                    &parent.indices,
                    split.threshold_bin,
                    missing_goes_left,
                    &mut left_indices,
                    &mut right_indices,
                );
            });
            TimingBuckets::add(&self.timing.partition, t_p.elapsed());

            let left_leaf_idx = tree.leaf_values.len();
            tree.leaf_values
                .push(self.compute_leaf_value(split.left_sum_grad, split.left_sum_hess));
            let right_leaf_idx = tree.leaf_values.len();
            tree.leaf_values
                .push(self.compute_leaf_value(split.right_sum_grad, split.right_sum_hess));

            for &i in &left_indices {
                row_to_leaf[i as usize] = left_leaf_idx as u32;
            }
            for &i in &right_indices {
                row_to_leaf[i as usize] = right_leaf_idx as u32;
            }

            // Threshold and gain go into the parallel `node_thresholds` /
            // `node_gains` arrays — kept off the inference-hot SplitNode.
            let new_node_idx = tree.nodes.len() as i32;
            tree.nodes.push(SplitNode {
                feature: split.feature as u32,
                threshold_bin: split.threshold_bin,
                missing_dir: split.missing_dir,
                left_child: encode_leaf(left_leaf_idx),
                right_child: encode_leaf(right_leaf_idx),
            });
            tree.node_thresholds.push(split.threshold_value);
            tree.node_gains.push(split.gain);

            if parent.parent_node_idx >= 0 {
                let p = &mut tree.nodes[parent.parent_node_idx as usize];
                if parent.is_left_child {
                    p.left_child = new_node_idx;
                } else {
                    p.right_child = new_node_idx;
                }
            }

            // Sibling-by-subtraction: build histograms directly for the smaller
            // child and derive the larger via `parent - smaller`. Breaking this
            // invariant doubles tree-build time.
            let build_left_first = left_indices.len() <= right_indices.len();
            let small_indices: &Vec<u32> = if build_left_first {
                &left_indices
            } else {
                &right_indices
            };

            let t_h = std::time::Instant::now();
            let mut small_hists: Vec<FeatureHistogram> = feature_indices
                .iter()
                .enumerate()
                .map(|(slot, _)| FeatureHistogram::zeros(parent.histograms[slot].num_bins()))
                .collect();
            with_columns!(self.dataset, feature_indices, |cols| {
                build_histograms_batched(&cols, small_indices, gradhess, &mut small_hists);
            });
            TimingBuckets::add(&self.timing.hist_build, t_h.elapsed());

            let t_s = std::time::Instant::now();
            let mut large_hists: Vec<FeatureHistogram> = feature_indices
                .iter()
                .enumerate()
                .map(|(slot, _)| FeatureHistogram::zeros(parent.histograms[slot].num_bins()))
                .collect();
            for slot in 0..feature_indices.len() {
                FeatureHistogram::subtract_into(
                    &parent.histograms[slot],
                    &small_hists[slot],
                    &mut large_hists[slot],
                );
            }
            TimingBuckets::add(&self.timing.hist_subtract, t_s.elapsed());

            let (left_hists, right_hists) = if build_left_first {
                (small_hists, large_hists)
            } else {
                (large_hists, small_hists)
            };

            let mut left_leaf = LeafState {
                indices: left_indices,
                sum_grad: split.left_sum_grad,
                sum_hess: split.left_sum_hess,
                count: split.left_count,
                histograms: left_hists,
                best_split: None,
                parent_node_idx: new_node_idx,
                is_left_child: true,
            };
            let mut right_leaf = LeafState {
                indices: right_indices,
                sum_grad: split.right_sum_grad,
                sum_hess: split.right_sum_hess,
                count: split.right_count,
                histograms: right_hists,
                best_split: None,
                parent_node_idx: new_node_idx,
                is_left_child: false,
            };

            self.update_best_split(&mut left_leaf, feature_indices);
            self.update_best_split(&mut right_leaf, feature_indices);

            leaves.push(left_leaf);
            leaves.push(right_leaf);
        }

        (tree, row_to_leaf)
    }

    fn update_best_split(&self, leaf: &mut LeafState, feature_indices: &[usize]) {
        let t = std::time::Instant::now();
        if (leaf.count as usize) < 2 * self.config.min_data_in_leaf
            || leaf.sum_hess < 2.0 * self.config.min_sum_hessian_in_leaf
        {
            leaf.best_split = None;
            TimingBuckets::add(&self.timing.split_search, t.elapsed());
            return;
        }
        leaf.best_split = feature_indices
            .iter()
            .enumerate()
            .filter_map(|(slot, &feat)| {
                find_best_split_for_feature(
                    feat,
                    &leaf.histograms[slot],
                    self.dataset.bin_mapper(feat),
                    leaf.sum_grad,
                    leaf.sum_hess,
                    leaf.count,
                    self.config,
                )
            })
            .max_by(|a, b| a.gain.partial_cmp(&b.gain).unwrap_or(std::cmp::Ordering::Equal));
        TimingBuckets::add(&self.timing.split_search, t.elapsed());
    }

    #[inline]
    fn compute_leaf_value(&self, sum_grad: f64, sum_hess: f64) -> f64 {
        threshold_leaf(
            sum_grad,
            sum_hess,
            self.config.lambda_l1,
            self.config.lambda_l2,
        )
    }
}

/// Split `parent_indices` into left/right partitions using one feature column.
/// Generic over `B: Bin` so the comparison stays in the column's native width.
fn partition_indices<B: Bin>(
    feat_col: &[B],
    parent_indices: &[u32],
    threshold_bin: u16,
    missing_goes_left: bool,
    left_out: &mut Vec<u32>,
    right_out: &mut Vec<u32>,
) {
    let threshold = B::from_u16(threshold_bin);
    let n_parent = parent_indices.len();
    // SAFETY: caller pre-sized left_out/right_out to the exact split counts,
    // and feat_col covers every row index in parent_indices (dataset builder
    // invariant).
    unsafe {
        let lp = left_out.as_mut_ptr();
        let rp = right_out.as_mut_ptr();
        let mut li: usize = 0;
        let mut ri: usize = 0;
        let parent_ptr = parent_indices.as_ptr();
        let col_ptr = feat_col.as_ptr();
        for k in 0..n_parent {
            let i = *parent_ptr.add(k);
            let bin = *col_ptr.add(i as usize);
            let goes_left = if bin == B::MISSING {
                missing_goes_left
            } else {
                bin <= threshold
            };
            if goes_left {
                *lp.add(li) = i;
                li += 1;
            } else {
                *rp.add(ri) = i;
                ri += 1;
            }
        }
        left_out.set_len(li);
        right_out.set_len(ri);
    }
}
