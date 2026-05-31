//! Declarative feature extraction.
//!
//! A [`FeatureBuilder<T>`] declares the columns of a training set as a list of
//! named extractor closures over your own row type `T`. Each feature names
//! itself, optionally overrides `max_bin`, and pulls one `f64` out of a `&T`
//! (return [`f64::NAN`] for a missing value). The *same* builder is handed to
//! both [`GbdtTrainer`](crate::GbdtTrainer) at train time and
//! [`Model`](crate::Model) at predict time — the closures can't be serialized,
//! so the model only stores bin mappers and feature names and re-extracts
//! through the builder you supply.

use std::hash::{BuildHasher, Hash};

use crate::config::Config;
use crate::dataset::{Dataset, builder};
use crate::error::Result;

/// One declared feature: a name, an optional per-feature `max_bin` override,
/// and a closure that extracts the raw value from a row. The closure should
/// return [`f64::NAN`] for a missing value — it maps to the reserved missing
/// bin like any other NaN.
struct Feature<T> {
    name: String,
    max_bin: Option<usize>,
    /// When true, the column is split as a categorical (per-node subset split)
    /// rather than a numeric threshold; see [`BinMapper::fit_categorical`].
    categorical: bool,
    extract: Box<dyn Fn(&T) -> f64>,
}

/// Declares the feature columns for a dataset of `T` rows.
///
/// Build it once and pass the same instance to training and prediction:
///
/// ```
/// use nanogbm::FeatureBuilder;
///
/// struct Row { age: f64, score: f64 }
///
/// let fb = FeatureBuilder::<Row>::new()
///     .add("age", None, |r| r.age)
///     .add("score", Some(128), |r| r.score);
/// assert_eq!(fb.len(), 2);
/// ```
pub struct FeatureBuilder<T> {
    features: Vec<Feature<T>>,
}

impl<T> Default for FeatureBuilder<T> {
    fn default() -> Self {
        Self {
            features: Vec::new(),
        }
    }
}

impl<T> FeatureBuilder<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a feature. `max_bin` of `None` falls back to the trainer's
    /// [`Config::max_bin`]; `extract` pulls the raw `f64` from a row (return
    /// `f64::NAN` for missing). Features are kept in declaration order, which
    /// is the feature index the model reports.
    pub fn add(
        mut self,
        name: impl Into<String>,
        max_bin: Option<usize>,
        extract: impl Fn(&T) -> f64 + 'static,
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
            max_bin,
            categorical: false,
            extract: Box::new(extract),
        });
        self
    }

    pub fn add_boolean(
        mut self,
        name: impl Into<String>,
        extract: impl Fn(&T) -> bool + 'static,
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
            max_bin: None,
            categorical: false,
            extract: Box::new(move |r| if extract(r) { 1.0 } else { 0.0 }),
        });
        self
    }

    pub fn add_float(self, name: impl Into<String>, extract: impl Fn(&T) -> f64 + 'static) -> Self {
        self.add(name, None, extract)
    }

    pub fn add_hash<H: Hash>(
        mut self,
        name: impl Into<String>,
        max_bin: Option<usize>,
        extract: impl Fn(&T) -> H + 'static,
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
            max_bin,
            categorical: false,
            extract: Box::new(move |r| {
                let h = extract(r);
                let h = gxhash::GxBuildHasher::with_seed(42).hash_one(&h);
                f64::from_bits(h)
            }),
        });
        self
    }

    /// Like [`add_hash`](Self::add_hash) but flags the column categorical: the
    /// learner splits it by optimal subset partition per node
    /// (see [`BinMapper::fit_categorical`](crate::dataset::BinMapper::fit_categorical))
    /// instead of a threshold on the (arbitrary) hashed order. Use for
    /// low-cardinality unordered ids.
    pub fn add_categorical<H: Hash>(
        mut self,
        name: impl Into<String>,
        max_bin: Option<usize>,
        extract: impl Fn(&T) -> H + 'static,
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
            max_bin,
            categorical: true,
            extract: Box::new(move |r| {
                let h = extract(r);
                let h = gxhash::GxBuildHasher::with_seed(42).hash_one(&h);
                f64::from_bits(h)
            }),
        });
        self
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Feature names in declaration (index) order.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.features.iter().map(|f| f.name.as_str())
    }

    /// Per-feature effective `max_bin`, substituting `default` wherever a
    /// feature left it unset.
    pub(crate) fn max_bins(&self, default: usize) -> Vec<usize> {
        self.features
            .iter()
            .map(|f| f.max_bin.unwrap_or(default))
            .collect()
    }

    /// Per-feature categorical flag, in declaration (index) order.
    pub(crate) fn categorical_flags(&self) -> Vec<bool> {
        self.features.iter().map(|f| f.categorical).collect()
    }

    /// Extract column-major `f64` data: `out[feat][row]`.
    pub(crate) fn extract_columns(&self, rows: &[T]) -> Vec<Vec<f64>> {
        self.features
            .iter()
            .map(|f| rows.iter().map(|r| (f.extract)(r)).collect())
            .collect()
    }

    /// Extract row-major `f64` data: `out[row * n_features + feat]`. Used by
    /// the raw-f64 predict path.
    pub(crate) fn extract_row_major(&self, rows: &[T]) -> Vec<f64> {
        let nf = self.features.len();
        let mut out = vec![0.0f64; rows.len() * nf];
        for (i, row) in rows.iter().enumerate() {
            for (j, f) in self.features.iter().enumerate() {
                out[i * nf + j] = (f.extract)(row);
            }
        }
        out
    }

    /// Extract features from `rows`, fit one [`BinMapper`](crate::dataset::BinMapper)
    /// per column (using each feature's `max_bin` or `config.max_bin`), and pack
    /// into a binned [`Dataset`]. This is the materialization step the trainer
    /// runs internally; it's public so you can bin once and reuse, or benchmark.
    pub fn build_dataset(&self, rows: &[T], labels: &[f32], config: &Config) -> Result<Dataset> {
        let columns = self.extract_columns(rows);
        let max_bins = self.max_bins(config.max_bin);
        let categorical = self.categorical_flags();
        builder::build_dataset(columns, &max_bins, &categorical, config.min_data_in_bin, labels)
    }

    /// Format a per-feature importance report as a sortable table, rows sorted
    /// by gain descending. `splits` and `gains` are usually
    /// [`Model::feature_importance_split`](crate::Model::feature_importance_split)
    /// and [`Model::feature_importance_gain`](crate::Model::feature_importance_gain).
    pub fn format_importance(&self, splits: &[u32], gains: &[f64]) -> String {
        let total_gain: f64 = gains.iter().sum();
        let total_splits: u32 = splits.iter().sum();
        let name_w = self
            .features
            .iter()
            .map(|f| f.name.len())
            .max()
            .unwrap_or(7)
            .max(7);
        let mut rows: Vec<(usize, &str, u32, f64)> = self
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                (
                    i,
                    f.name.as_str(),
                    *splits.get(i).unwrap_or(&0),
                    *gains.get(i).unwrap_or(&0.0),
                )
            })
            .collect();
        rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = String::new();
        out.push_str(&format!(
            "{:<idx$}  {:<name_w$}  {:>7}  {:>9}  {:>6}  {:>6}\n",
            "idx",
            "feature",
            "splits",
            "gain",
            "split%",
            "gain%",
            idx = 4,
            name_w = name_w
        ));
        for (i, name, sp, g) in &rows {
            let sp_pct = if total_splits > 0 {
                100.0 * *sp as f64 / total_splits as f64
            } else {
                0.0
            };
            let g_pct = if total_gain > 0.0 {
                100.0 * *g / total_gain
            } else {
                0.0
            };
            out.push_str(&format!(
                "{:<idx$}  {:<name_w$}  {:>7}  {:>9.3}  {:>5.1}%  {:>5.1}%\n",
                i,
                name,
                sp,
                g,
                sp_pct,
                g_pct,
                idx = 4,
                name_w = name_w
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Row {
        a: f64,
        b: f64,
    }

    #[test]
    fn declares_features_in_order() {
        let fb = FeatureBuilder::<Row>::new()
            .add("a", None, |r| r.a)
            .add("b", Some(32), |r| r.b);
        assert_eq!(fb.len(), 2);
        let names: Vec<_> = fb.names().collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn max_bins_fall_back_to_default() {
        let fb = FeatureBuilder::<Row>::new()
            .add("a", None, |r| r.a)
            .add("b", Some(32), |r| r.b);
        assert_eq!(fb.max_bins(255), vec![255, 32]);
    }

    #[test]
    fn extracts_columns_and_rows() {
        let fb = FeatureBuilder::<Row>::new()
            .add("a", None, |r| r.a)
            .add("b", None, |r| r.b);
        let rows = vec![Row { a: 1.0, b: 2.0 }, Row { a: 3.0, b: 4.0 }];
        assert_eq!(fb.extract_columns(&rows), vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
        assert_eq!(fb.extract_row_major(&rows), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
