use std::cell::Cell;
use std::time::Duration;

use crate::config::Config;
use crate::dataset::Dataset;
use crate::tree::histogram::{FeatureHistogram, build_histograms_batched, build_histograms_batched_full};
use crate::tree::split::{SplitInfo, find_best_split_for_feature, threshold_leaf};
use crate::tree::{MissingDir, SplitNode, Tree};

/// Cumulative per-phase wall-clock counters (one global instance, set from
/// gbdt.rs at end of fit). Cells because all training is single-threaded.
/// Per-phase wall-clock counters captured while training a single boosting
/// iteration. Visible via `Config::verbose`.
#[derive(Default)]
pub struct TimingBuckets {
    pub hist_build: Cell<Duration>,
    pub hist_subtract: Cell<Duration>,
    pub split_search: Cell<Duration>,
    pub partition: Cell<Duration>,
}

impl TimingBuckets {
    fn add(cell: &Cell<Duration>, d: Duration) { cell.set(cell.get() + d); }
}

/// Encode a leaf index `idx` (>= 0) as a negative child pointer.
#[inline]
fn encode_leaf(idx: usize) -> i32 {
    !(idx as i32)
}

struct LeafState {
    /// Training row indices that fall into this leaf.
    indices: Vec<u32>,
    sum_grad: f64,
    sum_hess: f64,
    count: u32,
    /// Histograms parallel to `feature_indices`.
    histograms: Vec<FeatureHistogram>,
    best_split: Option<SplitInfo>,
    /// Parent internal-node index in `Tree.nodes`. -1 if this is the root leaf.
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
        Self { config, dataset, timing }
    }

    /// Grow a single tree on the provided sample of rows and features.
    ///
    /// `gradhess[row] = [grad, hess]` is the packed per-row gradient/hessian
    /// pair (one 8-byte load = both values, halving gather pressure in the
    /// histogram-build hot loop vs separate f32 arrays).
    ///
    /// Returns the tree plus a `row_to_leaf` map of length `dataset.n_rows()`:
    /// for each row used in this tree, the index into `tree.leaf_values` of the
    /// leaf it landed in. Rows not in `row_indices` are mapped to leaf 0 (which
    /// is the root leaf; they receive the constant root prediction).
    pub fn train_one_tree(
        &self,
        gradhess: &[[f32; 2]],
        row_indices: &[u32],
        feature_indices: &[usize],
    ) -> (Tree, Vec<u32>) {
        let mut tree = Tree {
            nodes: Vec::new(),
            leaf_values: Vec::new(),
        };

        // Per-row current leaf index. Initially everyone is in leaf 0 (root).
        let mut row_to_leaf: Vec<u32> = vec![0u32; self.dataset.n_rows()];

        // Whether `row_indices` is exactly 0..n_rows. In that case the
        // histogram build over the full column is contiguous and we use the
        // gather-free `build_full` path.
        let full = row_indices.len() == self.dataset.n_rows() && {
            let mut ok = true;
            for (i, &r) in row_indices.iter().enumerate() {
                if r as usize != i {
                    ok = false;
                    break;
                }
            }
            ok
        };

        // Build root histograms over features (row-major batched: each row
        // touches gradhess once, then updates every feature's histogram in
        // lockstep).
        let t0 = std::time::Instant::now();
        let mut root_histograms: Vec<FeatureHistogram> = feature_indices
            .iter()
            .map(|&feat| FeatureHistogram::zeros(self.dataset.bin_mapper(feat).num_bins()))
            .collect();
        let root_columns: Vec<&[u16]> = feature_indices
            .iter()
            .map(|&feat| self.dataset.feature_column(feat))
            .collect();
        if full {
            build_histograms_batched_full(&root_columns, gradhess, &mut root_histograms);
        } else {
            build_histograms_batched(&root_columns, row_indices, gradhess, &mut root_histograms);
        }
        TimingBuckets::add(&self.timing.hist_build, t0.elapsed());

        let root_grad: f64 = row_indices.iter().map(|&i| gradhess[i as usize][0] as f64).sum();
        let root_hess: f64 = row_indices.iter().map(|&i| gradhess[i as usize][1] as f64).sum();
        let root_count = row_indices.len() as u32;

        // Allocate root leaf in tree.leaf_values; its value is set now and overwritten
        // if/when it gets split (leaving a dead entry, which is fine).
        tree.leaf_values.push(self.compute_leaf_value(root_grad, root_hess));

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

        // Leaf-wise growth: at each step, split the leaf with the highest gain,
        // until we reach `num_leaves` or no positive-gain split remains.
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

            // Partition parent.indices into left/right based on the split.
            // We know the exact left_count/right_count from the SplitInfo, so
            // pre-size the Vecs and use unchecked writes to avoid the per-push
            // capacity check + branch.
            let t_p = std::time::Instant::now();
            let feat_col = self.dataset.feature_column(split.feature);
            let n_parent = parent.indices.len();
            let mut left_indices: Vec<u32> = Vec::with_capacity(split.left_count as usize);
            let mut right_indices: Vec<u32> = Vec::with_capacity(split.right_count as usize);
            let missing_goes_left = matches!(split.missing_dir, MissingDir::Left);
            let threshold_bin = split.threshold_bin;
            unsafe {
                let lp = left_indices.as_mut_ptr();
                let rp = right_indices.as_mut_ptr();
                let mut li: usize = 0;
                let mut ri: usize = 0;
                let parent_ptr = parent.indices.as_ptr();
                let col_ptr = feat_col.as_ptr();
                for k in 0..n_parent {
                    let i = *parent_ptr.add(k);
                    let bin = *col_ptr.add(i as usize);
                    let goes_left = if bin == crate::dataset::MISSING_BIN {
                        missing_goes_left
                    } else {
                        bin <= threshold_bin
                    };
                    if goes_left {
                        *lp.add(li) = i;
                        li += 1;
                    } else {
                        *rp.add(ri) = i;
                        ri += 1;
                    }
                }
                left_indices.set_len(li);
                right_indices.set_len(ri);
            }
            TimingBuckets::add(&self.timing.partition, t_p.elapsed());

            // Allocate two new leaf slots.
            let left_leaf_idx = tree.leaf_values.len();
            tree.leaf_values.push(self.compute_leaf_value(split.left_sum_grad, split.left_sum_hess));
            let right_leaf_idx = tree.leaf_values.len();
            tree.leaf_values.push(self.compute_leaf_value(split.right_sum_grad, split.right_sum_hess));

            // Update per-row leaf assignment for the rows we just partitioned.
            for &i in &left_indices {
                row_to_leaf[i as usize] = left_leaf_idx as u32;
            }
            for &i in &right_indices {
                row_to_leaf[i as usize] = right_leaf_idx as u32;
            }

            // Append the new internal node.
            let new_node_idx = tree.nodes.len() as i32;
            tree.nodes.push(SplitNode {
                feature: split.feature as u32,
                threshold: split.threshold_value,
                threshold_bin: split.threshold_bin,
                missing_dir: split.missing_dir,
                left_child: encode_leaf(left_leaf_idx),
                right_child: encode_leaf(right_leaf_idx),
                gain: split.gain,
            });

            // Wire up the parent (if any) to point to this new internal node.
            if parent.parent_node_idx >= 0 {
                let p = &mut tree.nodes[parent.parent_node_idx as usize];
                if parent.is_left_child {
                    p.left_child = new_node_idx;
                } else {
                    p.right_child = new_node_idx;
                }
            }

            // Decide which child to build histograms for directly (the smaller one)
            // and derive the other by subtraction from the parent's histograms.
            let build_left_first = left_indices.len() <= right_indices.len();
            let small_indices: &Vec<u32> = if build_left_first { &left_indices } else { &right_indices };

            let t_h = std::time::Instant::now();
            let mut small_hists: Vec<FeatureHistogram> = feature_indices
                .iter()
                .enumerate()
                .map(|(slot, _)| FeatureHistogram::zeros(parent.histograms[slot].num_bins()))
                .collect();
            let small_columns: Vec<&[u16]> = feature_indices
                .iter()
                .map(|&feat| self.dataset.feature_column(feat))
                .collect();
            build_histograms_batched(&small_columns, small_indices, gradhess, &mut small_hists);
            TimingBuckets::add(&self.timing.hist_build, t_h.elapsed());

            let t_s = std::time::Instant::now();
            let mut large_hists: Vec<FeatureHistogram> = feature_indices
                .iter()
                .enumerate()
                .map(|(slot, _)| FeatureHistogram::zeros(parent.histograms[slot].num_bins()))
                .collect();
            for slot in 0..feature_indices.len() {
                FeatureHistogram::subtract_into(&parent.histograms[slot], &small_hists[slot], &mut large_hists[slot]);
            }
            TimingBuckets::add(&self.timing.hist_subtract, t_s.elapsed());

            let (left_hists, right_hists) = if build_left_first { (small_hists, large_hists) } else { (large_hists, small_hists) };

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

            // If max_depth is set, prune leaves that can't grow further by clearing
            // their best_split. (Simplified: we don't track depth precisely without
            // walking the tree, so we skip this for now if max_depth < 0.)
        }

        (tree, row_to_leaf)
    }

    /// Search all features for the best split of a leaf and store it.
    fn update_best_split(&self, leaf: &mut LeafState, feature_indices: &[usize]) {
        let t = std::time::Instant::now();
        if (leaf.count as usize) < 2 * self.config.min_data_in_leaf {
            leaf.best_split = None;
            TimingBuckets::add(&self.timing.split_search, t.elapsed());
            return;
        }
        if leaf.sum_hess < 2.0 * self.config.min_sum_hessian_in_leaf {
            leaf.best_split = None;
            TimingBuckets::add(&self.timing.split_search, t.elapsed());
            return;
        }
        let best = feature_indices
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
            .fold(SplitInfo::dummy_worst(), |a, b| if a.gain >= b.gain { a } else { b });

        leaf.best_split = if best.gain > f64::NEG_INFINITY { Some(best) } else { None };
        TimingBuckets::add(&self.timing.split_search, t.elapsed());
    }

    #[inline]
    fn compute_leaf_value(&self, sum_grad: f64, sum_hess: f64) -> f64 {
        threshold_leaf(sum_grad, sum_hess, self.config.lambda_l1, self.config.lambda_l2)
    }
}

impl SplitInfo {
    fn dummy_worst() -> Self {
        Self {
            feature: 0,
            threshold_bin: 0,
            threshold_value: 0.0,
            missing_dir: MissingDir::Left,
            gain: f64::NEG_INFINITY,
            left_sum_grad: 0.0,
            left_sum_hess: 0.0,
            left_count: 0,
            right_sum_grad: 0.0,
            right_sum_hess: 0.0,
            right_count: 0,
        }
    }
}
