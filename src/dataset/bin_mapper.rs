use std::collections::HashMap;

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
    /// Categorical mode: sorted `(value-bits, bin-code)` pairs for O(log n)
    /// lookup. Each distinct value gets its own bin code (no order implied);
    /// the learner finds an optimal subset partition per node. Empty for
    /// numeric features, which keep using `upper_bounds`.
    #[serde(default)]
    pub(crate) categories: Vec<(u64, u16)>,
    /// Number of real categorical bins (max assigned code); 0 for numeric.
    #[serde(default)]
    pub(crate) n_categorical_bins: u16,
}

impl BinMapper {
    /// Build a bin mapper from a column of raw values using quantile-aware
    /// greedy binning. Skips NaNs when computing boundaries.
    pub fn fit(values: &[f64], max_bin: usize, min_data_in_bin: usize) -> Self {
        debug_assert!(max_bin >= 2);
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Self::numeric(vec![]);
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

        Self::numeric(upper_bounds)
    }

    fn numeric(upper_bounds: Vec<f64>) -> Self {
        Self { upper_bounds, categories: Vec::new(), n_categorical_bins: 0 }
    }

    /// Build a categorical bin mapper: each distinct (finite) value becomes its
    /// own bin code. Bin codes carry no order, the learner finds an optimal
    /// subset partition per node (see `find_best_categorical_split`). When the
    /// distinct count exceeds `max_bin - 1`, only the most frequent categories
    /// get their own bin; the rare tail and missing/NaN/unseen all map to the
    /// missing bin (0), which the split treats as one more category.
    pub fn fit_categorical(values: &[f64], max_bin: usize) -> Self {
        debug_assert!(max_bin >= 2);
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for &v in values {
            if v.is_finite() {
                *counts.entry(v.to_bits()).or_insert(0) += 1;
            }
        }
        if counts.is_empty() {
            return Self::numeric(vec![]);
        }
        let target_bins = max_bin.saturating_sub(1).max(1);
        // Most frequent categories first; bits tie-break keeps it deterministic.
        let mut distinct: Vec<(u64, u64)> = counts.into_iter().collect();
        distinct.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        distinct.truncate(target_bins);

        let mut categories: Vec<(u64, u16)> =
            distinct.iter().enumerate().map(|(i, &(bits, _))| (bits, i as u16 + 1)).collect();
        let n_categorical_bins = categories.len() as u16;
        categories.sort_by_key(|&(b, _)| b);
        Self { upper_bounds: vec![], categories, n_categorical_bins }
    }

    /// Whether this mapper bins a categorical feature (subset splits) rather
    /// than a numeric one (threshold splits).
    #[inline]
    pub fn is_categorical(&self) -> bool {
        self.n_categorical_bins > 0
    }

    /// Number of real (non-missing) bins.
    pub fn num_real_bins(&self) -> usize {
        if self.n_categorical_bins > 0 {
            self.n_categorical_bins as usize
        } else {
            self.upper_bounds.len() + 1
        }
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
        if !self.categories.is_empty() {
            let bits = v.to_bits();
            return match self.categories.binary_search_by_key(&bits, |&(b, _)| b) {
                Ok(i) => self.categories[i].1,
                Err(_) => MISSING_BIN,
            };
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

    #[test]
    fn categorical_each_value_gets_its_own_bin() {
        let vals = vec![30.0, 10.0, 20.0, 30.0, 10.0, 20.0];
        let bm = BinMapper::fit_categorical(&vals, 16);
        assert!(bm.is_categorical());
        assert_eq!(bm.num_real_bins(), 3);
        let bins = [bm.value_to_bin(10.0), bm.value_to_bin(20.0), bm.value_to_bin(30.0)];
        assert!(bins.iter().all(|&b| b != MISSING_BIN));
        assert_eq!(bins.iter().copied().collect::<std::collections::HashSet<_>>().len(), 3);
        assert_eq!(bm.value_to_bin(99.0), MISSING_BIN);
        assert_eq!(bm.value_to_bin(f64::NAN), MISSING_BIN);
    }

    #[test]
    fn categorical_value_zero_does_not_collide_with_missing() {
        // A real category value of 0 (common for enums) must get its own bin,
        // distinct from MISSING_BIN, which is reserved for NaN / unseen values.
        let vals = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let bm = BinMapper::fit_categorical(&vals, 16);
        let zero_bin = bm.value_to_bin(0.0);
        assert_ne!(zero_bin, MISSING_BIN, "value 0.0 must not map to the missing bin");
        assert_ne!(zero_bin, bm.value_to_bin(1.0));
        assert_ne!(zero_bin, bm.value_to_bin(2.0));
        // Only genuinely unseen values and NaN land in the missing bin.
        assert_eq!(bm.value_to_bin(99.0), MISSING_BIN);
        assert_eq!(bm.value_to_bin(f64::NAN), MISSING_BIN);
    }

    #[test]
    fn categorical_keeps_most_frequent_when_capped() {
        let vals = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let bm = BinMapper::fit_categorical(&vals, 3);
        assert_eq!(bm.num_real_bins(), 2);
        assert_ne!(bm.value_to_bin(1.0), MISSING_BIN);
        assert_ne!(bm.value_to_bin(2.0), MISSING_BIN);
        assert_eq!(bm.value_to_bin(3.0), MISSING_BIN);
    }
}
