use crate::config::Config;
use crate::dataset::BinMapper;
use crate::tree::histogram::FeatureHistogram;
use crate::tree::{MissingDir, SplitKind};

/// Best split found for a single feature.
#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub feature: usize,
    pub kind: SplitKind,
    pub gain: f64,
    pub left_sum_grad: f64,
    pub left_sum_hess: f64,
    pub left_count: u32,
    pub right_sum_grad: f64,
    pub right_sum_hess: f64,
    pub right_count: u32,
}

/// Apply L1/L2 to compute the optimal leaf weight given (G, H).
#[inline]
pub fn threshold_leaf(g: f64, h: f64, lambda_l1: f64, lambda_l2: f64) -> f64 {
    let g_thresh = if lambda_l1 > 0.0 {
        if g > lambda_l1 {
            g - lambda_l1
        } else if g < -lambda_l1 {
            g + lambda_l1
        } else {
            0.0
        }
    } else {
        g
    };
    -g_thresh / (h + lambda_l2)
}

/// Gain contribution of a node with sums (G, H), with L1/L2 regularization.
#[inline]
pub fn node_score(g: f64, h: f64, lambda_l1: f64, lambda_l2: f64) -> f64 {
    let g_thresh = if lambda_l1 > 0.0 {
        if g > lambda_l1 {
            g - lambda_l1
        } else if g < -lambda_l1 {
            g + lambda_l1
        } else {
            0.0
        }
    } else {
        g
    };
    (g_thresh * g_thresh) / (h + lambda_l2)
}

/// Dispatch: numerical vs categorical split search, picked from `bin_mapper`.
pub fn find_best_split_for_feature(
    feature: usize,
    hist: &FeatureHistogram,
    bin_mapper: &BinMapper,
    parent_grad: f64,
    parent_hess: f64,
    parent_count: u32,
    config: &Config,
) -> Option<SplitInfo> {
    if bin_mapper.is_categorical() {
        find_best_categorical_split(
            feature,
            hist,
            parent_grad,
            parent_hess,
            parent_count,
            config,
        )
    } else {
        find_best_numerical_split(
            feature,
            hist,
            bin_mapper,
            parent_grad,
            parent_hess,
            parent_count,
            config,
        )
    }
}

/// Ordered numerical split search: scan thresholds over bins, try sending
/// missing (bin 0) both left and right.
pub fn find_best_numerical_split(
    feature: usize,
    hist: &FeatureHistogram,
    bin_mapper: &BinMapper,
    parent_grad: f64,
    parent_hess: f64,
    parent_count: u32,
    config: &Config,
) -> Option<SplitInfo> {
    let num_bins = hist.num_bins();
    if num_bins <= 2 {
        return None;
    }
    let parent_score = node_score(parent_grad, parent_hess, config.lambda_l1, config.lambda_l2);

    let missing_grad = hist.bins[0].grad;
    let missing_hess = hist.bins[0].hess;
    let missing_count = hist.bins[0].count as i64;

    let mut best: Option<SplitInfo> = None;

    for &dir in &[MissingDir::Left, MissingDir::Right] {
        let mut left_grad = match dir {
            MissingDir::Left => missing_grad,
            MissingDir::Right => 0.0,
        };
        let mut left_hess = match dir {
            MissingDir::Left => missing_hess,
            MissingDir::Right => 0.0,
        };
        let mut left_count: i64 = match dir {
            MissingDir::Left => missing_count,
            MissingDir::Right => 0,
        };

        for t in 1..(num_bins - 1) {
            left_grad += hist.bins[t].grad;
            left_hess += hist.bins[t].hess;
            left_count += hist.bins[t].count as i64;

            let right_grad = parent_grad - left_grad;
            let right_hess = parent_hess - left_hess;
            let right_count = parent_count as i64 - left_count;

            if left_count < config.min_data_in_leaf as i64
                || right_count < config.min_data_in_leaf as i64
            {
                continue;
            }
            if left_hess < config.min_sum_hessian_in_leaf
                || right_hess < config.min_sum_hessian_in_leaf
            {
                continue;
            }

            let score = node_score(left_grad, left_hess, config.lambda_l1, config.lambda_l2)
                + node_score(right_grad, right_hess, config.lambda_l1, config.lambda_l2);
            let gain = (score - parent_score) * 0.5;
            if gain <= config.min_gain_to_split {
                continue;
            }

            let bin_idx = t - 1;
            let threshold_value = bin_mapper
                .upper_bounds()
                .get(bin_idx)
                .copied()
                .unwrap_or(f64::INFINITY);

            let candidate = SplitInfo {
                feature,
                kind: SplitKind::Numerical {
                    threshold_bin: t as u16,
                    threshold_value,
                    missing_dir: dir,
                },
                gain,
                left_sum_grad: left_grad,
                left_sum_hess: left_hess,
                left_count: left_count as u32,
                right_sum_grad: right_grad,
                right_sum_hess: right_hess,
                right_count: right_count as u32,
            };
            best = match best {
                Some(b) if b.gain >= candidate.gain => Some(b),
                _ => Some(candidate),
            };
        }
    }

    best
}

/// LightGBM-style categorical subset split. Real bins are sorted by
/// `grad/(hess+cat_smooth)` and we prefix-scan in both directions, keeping the
/// best (left, right) partition under the size/gain constraints. Missing
/// (bin 0) always goes right and is never in `left_bins`.
pub fn find_best_categorical_split(
    feature: usize,
    hist: &FeatureHistogram,
    parent_grad: f64,
    parent_hess: f64,
    parent_count: u32,
    config: &Config,
) -> Option<SplitInfo> {
    let num_bins = hist.num_bins();
    if num_bins <= 2 {
        return None;
    }

    let lambda_l1 = config.lambda_l1;
    let lambda_l2_eff = config.lambda_l2 + config.cat_l2;
    let parent_score = node_score(parent_grad, parent_hess, lambda_l1, lambda_l2_eff);

    let mut active: Vec<u16> = (1..num_bins as u16)
        .filter(|&b| hist.bins[b as usize].count > 0)
        .collect();
    if active.len() < 2 {
        return None;
    }

    let cat_smooth = config.cat_smooth;
    let sort_key = |b: u16| -> f64 {
        let bin = &hist.bins[b as usize];
        bin.grad / (bin.hess + cat_smooth)
    };
    // Deterministic ordering: ascending sort_key; tie-break by bin code so two
    // identical configs/datasets always produce the same split.
    active.sort_by(|&a, &b| {
        sort_key(a)
            .partial_cmp(&sort_key(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let n_active = active.len();
    let max_k = config.max_cat_threshold.min(n_active - 1);
    if max_k == 0 {
        return None;
    }

    let mut best: Option<SplitInfo> = None;

    // Both directions of the prefix scan (low-key-first and high-key-first).
    for reverse in [false, true] {
        let mut left_grad = 0.0_f64;
        let mut left_hess = 0.0_f64;
        let mut left_count: i64 = 0;
        let mut left_set: Vec<u16> = Vec::with_capacity(max_k);

        for step in 0..max_k {
            let pick_idx = if reverse { n_active - 1 - step } else { step };
            let b = active[pick_idx];
            let bin = &hist.bins[b as usize];
            left_grad += bin.grad;
            left_hess += bin.hess;
            left_count += bin.count as i64;
            left_set.push(b);

            let right_grad = parent_grad - left_grad;
            let right_hess = parent_hess - left_hess;
            let right_count = parent_count as i64 - left_count;

            if left_count < config.min_data_in_leaf as i64
                || right_count < config.min_data_in_leaf as i64
            {
                continue;
            }
            if left_hess < config.min_sum_hessian_in_leaf
                || right_hess < config.min_sum_hessian_in_leaf
            {
                continue;
            }

            let score = node_score(left_grad, left_hess, lambda_l1, lambda_l2_eff)
                + node_score(right_grad, right_hess, lambda_l1, lambda_l2_eff);
            let gain = (score - parent_score) * 0.5;
            if gain <= config.min_gain_to_split {
                continue;
            }

            let mut left_bins_sorted = left_set.clone();
            left_bins_sorted.sort_unstable();
            let candidate = SplitInfo {
                feature,
                kind: SplitKind::Categorical {
                    left_bins: left_bins_sorted,
                },
                gain,
                left_sum_grad: left_grad,
                left_sum_hess: left_hess,
                left_count: left_count as u32,
                right_sum_grad: right_grad,
                right_sum_hess: right_hess,
                right_count: right_count as u32,
            };
            best = match best {
                Some(b) if b.gain >= candidate.gain => Some(b),
                _ => Some(candidate),
            };
        }
    }

    best
}
