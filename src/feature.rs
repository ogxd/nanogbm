//! Declarative feature extraction.
//!
//! A [`FeatureBuilder<T>`] declares the columns of a training set as a list of
//! named extractor closures over your own row type `T`. Each feature names
//! itself and pulls one `f64` out of a `&T`
//! (return [`f64::NAN`] for a missing value). The *same* builder is handed to
//! both [`GbdtTrainer`](crate::GbdtTrainer) at train time and
//! [`Model`](crate::Model) at predict time — the closures can't be serialized,
//! so the model only stores bin mappers and feature names and re-extracts
//! through the builder you supply.

use std::hash::{BuildHasher, Hash};

use crate::config::Config;
use crate::dataset::{Dataset, builder};
use crate::error::Result;

/// One declared feature: a name and a closure that extracts the raw value
/// from a row. The closure should return [`f64::NAN`] for a missing value —
/// it maps to the reserved missing bin like any other NaN.
struct Feature<T> {
    name: String,
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
///     .add_continuous("age", |r| r.age)
///     .add_continuous("score", |r| r.score);
/// assert_eq!(fb.len(), 2);
/// ```
pub struct FeatureBuilder<T> {
    features: Vec<Feature<T>>,
    /// Names of categorical features eligible for unknown-value augmentation
    /// (see [`crate::Config::unknown_aug_fraction`]). `None` means every
    /// categorical is eligible (the default); restrict it with
    /// [`with_augmentable_features`](Self::with_augmentable_features) to mask
    /// only the open-vocabulary columns that can see novel values in production.
    augmentable: Option<std::collections::HashSet<String>>,
}

impl<T> Default for FeatureBuilder<T> {
    fn default() -> Self {
        Self {
            features: Vec::new(),
            augmentable: None,
        }
    }
}

impl<T> FeatureBuilder<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_boolean(
        mut self,
        name: impl Into<String>,
        extract: impl Fn(&T) -> bool + 'static,
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
            categorical: true,
            extract: Box::new(move |r| if extract(r) { 1.0 } else { 0.0 }),
        });
        self
    }

    pub fn add_continuous<I: Into<f64>>(
        mut self, name: impl Into<String>,
        extract: impl Fn(&T) -> I + 'static
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
            categorical: false,
            extract: Box::new(move |r| extract(r).into()),
        });
        self
    }

    pub fn add_categorical<H: Hash>(
        mut self,
        name: impl Into<String>,
        extract: impl Fn(&T) -> H + 'static,
    ) -> Self {
        self.features.push(Feature {
            name: name.into(),
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

    /// Restrict unknown-value augmentation to the named categorical features.
    /// Names not present in the builder are ignored; non-categorical names have
    /// no effect (only categoricals are ever masked). Without this call, every
    /// categorical is eligible.
    pub fn with_augmentable_features(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.augmentable = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Per-feature categorical flag, in declaration (index) order.
    pub(crate) fn categorical_flags(&self) -> Vec<bool> {
        self.features.iter().map(|f| f.categorical).collect()
    }

    /// Per-feature mask eligibility for unknown-value augmentation, in
    /// declaration (index) order: categorical AND (no restriction set, or named
    /// in it).
    pub(crate) fn augmentable_flags(&self) -> Vec<bool> {
        self.features
            .iter()
            .map(|f| f.categorical && self.augmentable.as_ref().is_none_or(|s| s.contains(&f.name)))
            .collect()
    }

    /// Extract column-major `f64` data: `out[feat][row]`.
    pub(crate) fn extract_columns(&self, rows: &[T]) -> Vec<Vec<f64>> {
        self.features
            .iter()
            .map(|f| rows.iter().map(|r| (f.extract)(r)).collect())
            .collect()
    }

    /// Extract and bin features row-major into `out`: `out[row * n_features + feat]`
    /// is the `u16` bin code. Fuses extraction and binning in one pass with the
    /// supplied per-feature `mappers`, so no intermediate f64 column buffers
    /// are allocated. Used by the small-batch (per-request) predict path where
    /// allocation, not the tree walk, dominates. `mappers.len()` must equal the
    /// feature count.
    pub(crate) fn extract_bins_row_major(
        &self,
        rows: &[T],
        mappers: &[crate::dataset::BinMapper],
        out: &mut Vec<u16>,
    ) {
        let nf = self.features.len();
        debug_assert_eq!(mappers.len(), nf);
        out.clear();
        out.resize(rows.len() * nf, 0);
        for (i, row) in rows.iter().enumerate() {
            let base = i * nf;
            for (j, f) in self.features.iter().enumerate() {
                let v = (f.extract)(row);
                out[base + j] = mappers[j].value_to_bin(v);
            }
        }
    }

    /// Extract features from `rows`, fit one [`BinMapper`](crate::dataset::BinMapper)
    /// per column (each capped at `config.max_bin`), and pack into a binned
    /// [`Dataset`]. This is the materialization step the trainer runs
    /// internally; it's public so you can bin once and reuse, or benchmark.
    pub fn build_dataset(&self, rows: &[T], labels: &[f32], config: &Config) -> Result<Dataset> {
        let columns = self.extract_columns(rows);
        let categorical = self.categorical_flags();
        builder::build_dataset(
            columns,
            config.max_bin,
            config.max_cat_bins,
            &categorical,
            config.min_data_in_bin,
            labels,
        )
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
            .add_continuous("a", |r| r.a)
            .add_continuous("b", |r| r.b);
        assert_eq!(fb.len(), 2);
        let names: Vec<_> = fb.names().collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn augmentable_flags_default_all_categoricals_then_restricted() {
        let fb = FeatureBuilder::<Row>::new()
            .add_continuous("a", |r| r.a)
            .add_categorical("b", |r| r.b as i64)
            .add_categorical("c", |r| r.a as i64);
        // Default: every categorical eligible, numeric never.
        assert_eq!(fb.augmentable_flags(), vec![false, true, true]);
        // Restricted: only the named categorical (unknown name ignored).
        let fb = fb.with_augmentable_features(["b", "nonexistent"]);
        assert_eq!(fb.augmentable_flags(), vec![false, true, false]);
    }

    #[test]
    fn extracts_columns() {
        let fb = FeatureBuilder::<Row>::new()
            .add_continuous("a", |r| r.a)
            .add_continuous("b", |r| r.b);
        let rows = vec![Row { a: 1.0, b: 2.0 }, Row { a: 3.0, b: 4.0 }];
        assert_eq!(fb.extract_columns(&rows), vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    }
}
