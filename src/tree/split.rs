use crate::config::Config;
use crate::dataset::BinMapper;
use crate::tree::MissingDir;
use crate::tree::histogram::FeatureHistogram;

/// Best split found for a single feature.
#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub feature: usize,
    /// Inclusive upper bound of the left branch in bin-code space.
    pub threshold_bin: u16,
    pub threshold_value: f64,
    pub missing_dir: MissingDir,
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

            if left_count < config.min_data_in_leaf as i64 || right_count < config.min_data_in_leaf as i64 {
                continue;
            }
            if left_hess < config.min_sum_hessian_in_leaf || right_hess < config.min_sum_hessian_in_leaf {
                continue;
            }

            let score = node_score(left_grad, left_hess, config.lambda_l1, config.lambda_l2)
                + node_score(right_grad, right_hess, config.lambda_l1, config.lambda_l2);
            let gain = (score - parent_score) * 0.5;
            if gain <= config.min_gain_to_split {
                continue;
            }

            let bin_idx = t - 1; // upper_bounds index for real bin t
            let threshold_value = bin_mapper.upper_bounds().get(bin_idx).copied().unwrap_or(f64::INFINITY);

            let candidate = SplitInfo {
                feature,
                threshold_bin: t as u16,
                threshold_value,
                missing_dir: dir,
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
