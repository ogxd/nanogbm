use serde::{Deserialize, Serialize};

use super::MISSING_BIN;

/// Per-feature mapping from raw f64 values to bin codes.
///
/// Bin 0 is reserved for missing (NaN). Bins 1..=N are real bins, where N is
/// determined at fit time and bounded by `max_bin`. `upper_bounds[i]` is the
/// inclusive upper boundary for bin `i + 1` (so bin 1 covers (-inf, upper_bounds[0]],
/// bin 2 covers (upper_bounds[0], upper_bounds[1]], etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinMapper {
    pub(crate) upper_bounds: Vec<f64>,
}

impl BinMapper {
    /// Build a bin mapper from a column of raw values using quantile-aware
    /// greedy binning. Skips NaNs when computing boundaries.
    pub fn fit(values: &[f64], max_bin: usize, min_data_in_bin: usize) -> Self {
        debug_assert!(max_bin >= 2);
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Self {
                upper_bounds: vec![],
            };
        }
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Distinct values with counts.
        let mut distinct_vals: Vec<f64> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        for &v in &finite {
            match distinct_vals.last() {
                Some(&last) if last == v => *counts.last_mut().unwrap() += 1,
                _ => {
                    distinct_vals.push(v);
                    counts.push(1);
                }
            }
        }
        let total = finite.len();
        let target_bins = max_bin.saturating_sub(1).max(1); // reserve bin 0 for missing
        let mut upper_bounds: Vec<f64> = Vec::new();

        if distinct_vals.len() <= target_bins {
            // Each distinct value gets its own bin; use midpoints as boundaries.
            for i in 0..distinct_vals.len().saturating_sub(1) {
                let mid = (distinct_vals[i] + distinct_vals[i + 1]) / 2.0;
                upper_bounds.push(mid);
            }
        } else {
            // Quantile bins: walk distinct values, accumulating count, and emit a
            // boundary whenever the current bin reaches the per-bin target size.
            let target_size = total.div_ceil(target_bins).max(min_data_in_bin).max(1);
            let mut cur_cnt: usize = 0;
            for i in 0..distinct_vals.len() - 1 {
                cur_cnt += counts[i];
                if cur_cnt >= target_size && upper_bounds.len() + 1 < target_bins {
                    let mid = (distinct_vals[i] + distinct_vals[i + 1]) / 2.0;
                    upper_bounds.push(mid);
                    cur_cnt = 0;
                }
            }
        }

        Self { upper_bounds }
    }

    /// Number of real (non-missing) bins.
    pub fn num_real_bins(&self) -> usize {
        self.upper_bounds.len() + 1
    }

    /// Total bin codes including missing.
    pub fn num_bins(&self) -> usize {
        self.num_real_bins() + 1
    }

    pub fn upper_bounds(&self) -> &[f64] {
        &self.upper_bounds
    }

    /// Map a raw f64 to a bin code. NaN → `MISSING_BIN`.
    #[inline]
    pub fn value_to_bin(&self, v: f64) -> u16 {
        if !v.is_finite() {
            return MISSING_BIN;
        }
        // Binary search: find first upper bound >= v, that index +1 is the bin.
        // partition_point returns first index where pred is false.
        let idx = self.upper_bounds.partition_point(|&ub| ub < v);
        (idx as u16) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn few_distinct_values_each_get_own_bin() {
        let vals = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let bm = BinMapper::fit(&vals, 16, 1);
        assert_eq!(bm.num_real_bins(), 3);
        assert_eq!(bm.value_to_bin(1.0), 1);
        assert_eq!(bm.value_to_bin(2.0), 2);
        assert_eq!(bm.value_to_bin(3.0), 3);
    }

    #[test]
    fn nan_maps_to_missing() {
        let vals = vec![1.0, 2.0, 3.0];
        let bm = BinMapper::fit(&vals, 16, 1);
        assert_eq!(bm.value_to_bin(f64::NAN), MISSING_BIN);
    }

    #[test]
    fn many_values_capped_at_max_bin() {
        let vals: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let bm = BinMapper::fit(&vals, 16, 1);
        assert!(bm.num_real_bins() <= 15);
    }
}
