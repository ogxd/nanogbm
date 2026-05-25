use serde::{Deserialize, Serialize};

use super::MISSING_BIN;

/// Per-feature mapping from raw f64 values to bin codes.
///
/// Bin 0 is reserved for missing (NaN) and, for categorical columns, also for
/// values not observed during fit or pushed out by the frequency truncation in
/// `fit_categorical`. Real bins are `1..=N`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinMapper {
    /// Numerical column: quantile-binned with inclusive upper bounds.
    /// Bin 1 covers (-inf, upper_bounds[0]], bin 2 covers (upper_bounds[0], upper_bounds[1]], etc.
    Numerical { upper_bounds: Vec<f64> },
    /// Categorical column: exact-value lookup. `value_to_bin` is sorted by
    /// `i64` key for binary search. Bin codes are `1..=n_real_bins`.
    Categorical {
        value_to_bin: Vec<(i64, u16)>,
        n_real_bins: u16,
    },
}

impl BinMapper {
    /// Build a numerical bin mapper from a column of raw values using
    /// quantile-aware greedy binning. Skips NaNs when computing boundaries.
    pub fn fit_numerical(values: &[f64], max_bin: usize, min_data_in_bin: usize) -> Self {
        debug_assert!(max_bin >= 2);
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return Self::Numerical {
                upper_bounds: vec![],
            };
        }
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
        let target_bins = max_bin.saturating_sub(1).max(1);
        let mut upper_bounds: Vec<f64> = Vec::new();

        if distinct_vals.len() <= target_bins {
            for i in 0..distinct_vals.len().saturating_sub(1) {
                let mid = (distinct_vals[i] + distinct_vals[i + 1]) / 2.0;
                upper_bounds.push(mid);
            }
        } else {
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

        Self::Numerical { upper_bounds }
    }

    /// Back-compat alias for `fit_numerical`.
    pub fn fit(values: &[f64], max_bin: usize, min_data_in_bin: usize) -> Self {
        Self::fit_numerical(values, max_bin, min_data_in_bin)
    }

    /// Build a categorical bin mapper. Each finite value is cast to `i64`;
    /// distinct values are kept up to `max_cat_bin - 1` (bin 0 reserved for
    /// MISSING + unseen). When more distinct values exist than fit, keep the
    /// top by frequency; deterministic tiebreak prefers lower value first.
    pub fn fit_categorical(values: &[f64], max_cat_bin: usize) -> Self {
        debug_assert!(max_cat_bin >= 2);
        use std::collections::HashMap;
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &v in values {
            if v.is_finite() {
                *counts.entry(v as i64).or_insert(0) += 1;
            }
        }
        let mut by_freq: Vec<(i64, usize)> = counts.into_iter().collect();
        // Tiebreak: lower value first so the output is deterministic.
        by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let cap = max_cat_bin.saturating_sub(1);
        if by_freq.len() > cap {
            by_freq.truncate(cap);
        }
        // Re-sort the surviving values by their i64 key so we can binary-search.
        let mut kept: Vec<i64> = by_freq.into_iter().map(|(v, _)| v).collect();
        kept.sort_unstable();
        let n_real_bins = kept.len() as u16;
        let value_to_bin: Vec<(i64, u16)> = kept
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, (i as u16) + 1))
            .collect();
        Self::Categorical {
            value_to_bin,
            n_real_bins,
        }
    }

    /// Number of real (non-missing) bins.
    pub fn num_real_bins(&self) -> usize {
        match self {
            BinMapper::Numerical { upper_bounds } => upper_bounds.len() + 1,
            BinMapper::Categorical { n_real_bins, .. } => *n_real_bins as usize,
        }
    }

    /// Total bin codes including missing.
    pub fn num_bins(&self) -> usize {
        self.num_real_bins() + 1
    }

    /// Upper bounds for numerical mappers; empty slice for categorical.
    pub fn upper_bounds(&self) -> &[f64] {
        match self {
            BinMapper::Numerical { upper_bounds } => upper_bounds,
            BinMapper::Categorical { .. } => &[],
        }
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self, BinMapper::Categorical { .. })
    }

    /// Map a raw f64 to a bin code. NaN and unseen categorical values → `MISSING_BIN`.
    #[inline]
    pub fn value_to_bin(&self, v: f64) -> u16 {
        if !v.is_finite() {
            return MISSING_BIN;
        }
        match self {
            BinMapper::Numerical { upper_bounds } => {
                let idx = upper_bounds.partition_point(|&ub| ub < v);
                (idx as u16) + 1
            }
            BinMapper::Categorical { value_to_bin, .. } => {
                let key = v as i64;
                match value_to_bin.binary_search_by_key(&key, |(k, _)| *k) {
                    Ok(i) => value_to_bin[i].1,
                    Err(_) => MISSING_BIN,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn few_distinct_values_each_get_own_bin() {
        let vals = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let bm = BinMapper::fit_numerical(&vals, 16, 1);
        assert_eq!(bm.num_real_bins(), 3);
        assert_eq!(bm.value_to_bin(1.0), 1);
        assert_eq!(bm.value_to_bin(2.0), 2);
        assert_eq!(bm.value_to_bin(3.0), 3);
    }

    #[test]
    fn nan_maps_to_missing() {
        let vals = vec![1.0, 2.0, 3.0];
        let bm = BinMapper::fit_numerical(&vals, 16, 1);
        assert_eq!(bm.value_to_bin(f64::NAN), MISSING_BIN);
    }

    #[test]
    fn many_values_capped_at_max_bin() {
        let vals: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let bm = BinMapper::fit_numerical(&vals, 16, 1);
        assert!(bm.num_real_bins() <= 15);
    }

    #[test]
    fn fit_categorical_basic_and_invariants() {
        // 5 distinct values, lots of occurrences each.
        let vals: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let bm1 = BinMapper::fit_categorical(&vals, 16);
        assert_eq!(bm1.num_real_bins(), 5);
        for v in 0..5 {
            assert_ne!(bm1.value_to_bin(v as f64), MISSING_BIN);
        }
        // Insertion order should not change the mapping.
        let mut shuffled = vals.clone();
        shuffled.reverse();
        let bm2 = BinMapper::fit_categorical(&shuffled, 16);
        for v in 0..5 {
            assert_eq!(bm1.value_to_bin(v as f64), bm2.value_to_bin(v as f64));
        }
        // Unseen values go to MISSING.
        assert_eq!(bm1.value_to_bin(999.0), MISSING_BIN);
        assert_eq!(bm1.value_to_bin(f64::NAN), MISSING_BIN);
    }

    #[test]
    fn fit_categorical_truncates_by_frequency() {
        // 10 distinct values, but value v appears (v+1) times. With max_cat_bin = 5
        // we keep 4 real bins; the 4 most-frequent values are 9, 8, 7, 6.
        let mut vals: Vec<f64> = Vec::new();
        for v in 0..10i64 {
            for _ in 0..(v + 1) {
                vals.push(v as f64);
            }
        }
        let bm = BinMapper::fit_categorical(&vals, 5);
        assert_eq!(bm.num_real_bins(), 4);
        for v in [6i64, 7, 8, 9] {
            assert_ne!(
                bm.value_to_bin(v as f64),
                MISSING_BIN,
                "kept value {v} missing"
            );
        }
        for v in 0..6i64 {
            assert_eq!(
                bm.value_to_bin(v as f64),
                MISSING_BIN,
                "low-freq value {v} kept"
            );
        }
    }
}
