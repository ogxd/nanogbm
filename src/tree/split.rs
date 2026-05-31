use crate::config::Config;
use crate::dataset::BinMapper;
use crate::tree::MissingDir;
use crate::tree::histogram::FeatureHistogram;

/// Best split found for a single feature.
#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub feature: usize,
    /// Inclusive upper bound of the left branch in bin-code space (numeric
    /// splits only; unused when `cat_left_bins` is `Some`).
    pub threshold_bin: u16,
    pub threshold_value: f64,
    pub missing_dir: MissingDir,
    /// `Some(bins)` for a categorical subset split: the bin codes routed left.
    /// `None` for a numeric threshold split.
    pub cat_left_bins: Option<Vec<u16>>,
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

/// Find best split for a single feature using its histogram.
///
/// Returns `None` if no split satisfies the constraints. The "missing" bin is
/// bin 0; we try sending it both left and right.
pub fn find_best_split_for_feature(
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
        return None; // bin 0 (missing) + at most 1 real bin = nothing to split
    }
    let parent_score = node_score(parent_grad, parent_hess, config.lambda_l1, config.lambda_l2);

    let missing_grad = hist.bins[0].grad;
    let missing_hess = hist.bins[0].hess;
    let missing_count = hist.bins[0].count as i64;

    let mut best: Option<SplitInfo> = None;

    // Try both directions for missing values. For each direction, scan thresholds
    // over the real bins (1..num_bins-1, since splitting after the last bin gives
    // an empty right side).
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

        // bin t means: left = bins {missing-if-Left, 1..=t}, right = the rest
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

            let bin_idx = t - 1; // upper_bounds index for real bin t
            let threshold_value = bin_mapper
                .upper_bounds()
                .get(bin_idx)
                .copied()
                .unwrap_or(f64::INFINITY);

            let candidate = SplitInfo {
                feature,
                threshold_bin: t as u16,
                threshold_value,
                missing_dir: dir,
                cat_left_bins: None,
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

/// Find the best categorical (subset) split for a feature, following
/// LightGBM's `FindBestThresholdCategorical`.
///
/// Categories below `min_data_per_group` rows are dropped (forced to the
/// default/right side). With at most `max_cat_to_onehot` remaining categories
/// the split is one-vs-rest; otherwise categories are sorted by their
/// smoothed gradient ratio `grad / (hess + cat_smooth)` and the best subset is
/// found by scanning up to `max_cat_threshold` categories inward from *each*
/// end of that order (Fisher 1958: the optimal partition for the L2 objective
/// is contiguous in this order). Gains use `lambda_l2 + cat_l2`. The chosen
/// categories (returned in `cat_left_bins`) go left; everything else, including
/// dropped/rare categories and the missing bin, goes right. The ordering is
/// recomputed per node from current boosting residuals, so it does not leak the
/// way a fixed target-rate bin ordering would.
pub fn find_best_categorical_split(
    feature: usize,
    hist: &FeatureHistogram,
    parent_grad: f64,
    parent_hess: f64,
    parent_count: u32,
    config: &Config,
) -> Option<SplitInfo> {
    let l2 = config.lambda_l2 + config.cat_l2;
    let parent_score = node_score(parent_grad, parent_hess, config.lambda_l1, l2);

    // Only categories with enough data participate; the rest fall to the right.
    let num_bins = hist.num_bins();
    let mut used: Vec<usize> = (0..num_bins)
        .filter(|&b| hist.bins[b].count as usize >= config.min_data_per_group.max(1))
        .collect();
    if used.len() < 2 {
        return None;
    }

    // Gain of a candidate left group, or `None` if it violates a constraint.
    let eval = |lg: f64, lh: f64, lc: i64| -> Option<f64> {
        let rg = parent_grad - lg;
        let rh = parent_hess - lh;
        let rc = parent_count as i64 - lc;
        if lc < config.min_data_in_leaf as i64 || rc < config.min_data_in_leaf as i64 {
            return None;
        }
        if lh < config.min_sum_hessian_in_leaf || rh < config.min_sum_hessian_in_leaf {
            return None;
        }
        let score = node_score(lg, lh, config.lambda_l1, l2) + node_score(rg, rh, config.lambda_l1, l2);
        let gain = (score - parent_score) * 0.5;
        if gain <= config.min_gain_to_split { None } else { Some(gain) }
    };
    let make = |left_bins: Vec<u16>, lg: f64, lh: f64, lc: i64, gain: f64| SplitInfo {
        feature,
        threshold_bin: 0,
        threshold_value: 0.0,
        missing_dir: MissingDir::Left,
        cat_left_bins: Some(left_bins),
        gain,
        left_sum_grad: lg,
        left_sum_hess: lh,
        left_count: lc as u32,
        right_sum_grad: parent_grad - lg,
        right_sum_hess: parent_hess - lh,
        right_count: (parent_count as i64 - lc) as u32,
    };

    let mut best: Option<SplitInfo> = None;
    let consider = |cand: SplitInfo, best: &mut Option<SplitInfo>| {
        if best.as_ref().is_none_or(|b| cand.gain > b.gain) {
            *best = Some(cand);
        }
    };

    if used.len() <= config.max_cat_to_onehot {
        // Low cardinality: try each single category as the left group.
        for &b in &used {
            let (lg, lh, lc) = (hist.bins[b].grad, hist.bins[b].hess, hist.bins[b].count as i64);
            if let Some(gain) = eval(lg, lh, lc) {
                consider(make(vec![b as u16], lg, lh, lc, gain), &mut best);
            }
        }
    } else {
        // Sort by smoothed gradient ratio, then grow the left group inward from
        // each end up to `max_cat_threshold` categories.
        used.sort_by(|&a, &b| {
            let ra = hist.bins[a].grad / (hist.bins[a].hess + config.cat_smooth);
            let rb = hist.bins[b].grad / (hist.bins[b].hess + config.cat_smooth);
            ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let m = used.len();
        let limit = config.max_cat_threshold.min(m - 1);
        for forward in [true, false] {
            let (mut lg, mut lh, mut lc) = (0.0, 0.0, 0i64);
            let mut left_bins: Vec<u16> = Vec::with_capacity(limit);
            for step in 0..limit {
                let idx = if forward { used[step] } else { used[m - 1 - step] };
                lg += hist.bins[idx].grad;
                lh += hist.bins[idx].hess;
                lc += hist.bins[idx].count as i64;
                left_bins.push(idx as u16);
                if let Some(gain) = eval(lg, lh, lc) {
                    consider(make(left_bins.clone(), lg, lh, lc, gain), &mut best);
                }
            }
        }
    }

    best
}
