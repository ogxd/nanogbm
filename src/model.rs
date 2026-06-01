use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::config::standard;
use serde::{Deserialize, Serialize};

use crate::dataset::{Bin, BinData, BinMapper, BinWidth, Dataset, with_columns};
use crate::error::{Error, Result};
use crate::feature::FeatureBuilder;
use crate::loss::sigmoid;
use crate::tree::Tree;

/// A trained GBDT model: ensemble of trees plus boosting metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub(crate) init_score: f64,
    pub(crate) learning_rate: f64,
    pub(crate) n_features: usize,
    /// Feature names in index order, copied from the training
    /// [`FeatureBuilder`]. Predict asserts the supplied builder matches these.
    pub(crate) feature_names: Vec<String>,
    /// Required by the binned inference path: per-node `threshold_bin` values
    /// in trees are calibrated to these specific mappers, so re-fitting would
    /// produce wrong predictions.
    pub(crate) bin_mappers: Vec<BinMapper>,
    pub(crate) trees: Vec<Tree>,
}

impl Model {
    /// Constant the boosting loop started from (the prior log-odds of the labels).
    pub fn init_score(&self) -> f64 {
        self.init_score
    }

    /// Shrinkage applied to every tree's leaf value at inference.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Number of input features this model expects per row.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Feature names in index order, as declared by the training builder.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Number of trees in the ensemble. After early stopping, this equals
    /// `best_iter + 1`, not the configured `num_iterations`.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// The trees themselves, in fit order.
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Bincode-serialize this model to `path`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        let bytes = bincode::serde::encode_to_vec(self, standard())
            .map_err(|e| Error::Serde(e.to_string()))?;
        w.write_all(&bytes)?;
        w.flush()?;
        Ok(())
    }

    /// Deserialize a model previously written by [`Model::save`].
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        let (model, _) = bincode::serde::decode_from_slice::<Model, _>(&buf, standard())
            .map_err(|e| Error::Serde(e.to_string()))?;
        Ok(model)
    }

    /// Number of times each feature was used as a split, indexed by feature.
    pub fn feature_importance_split(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.n_features];
        for tree in &self.trees {
            for node in &tree.nodes {
                counts[node.feature as usize] += 1;
            }
        }
        counts
    }

    /// Total split gain attributed to each feature, indexed by feature.
    pub fn feature_importance_gain(&self) -> Vec<f64> {
        let mut gains = vec![0.0f64; self.n_features];
        for tree in &self.trees {
            for (node, gain) in tree.nodes.iter().zip(tree.node_gains.iter()) {
                gains[node.feature as usize] += gain;
            }
        }
        gains
    }

    /// Assert the supplied builder declares exactly the features this model was
    /// trained on (same names, same order). Guards against predicting with a
    /// mismatched extractor set.
    fn check_features<T>(&self, features: &FeatureBuilder<T>) {
        assert_eq!(
            features.len(),
            self.n_features,
            "feature count mismatch: builder has {}, model expects {}",
            features.len(),
            self.n_features
        );
        for (i, (got, want)) in features.names().zip(self.feature_names.iter()).enumerate() {
            assert_eq!(
                got, want,
                "feature {i} name mismatch: builder has {got:?}, model expects {want:?}"
            );
        }
    }

    /// Predict raw additive scores (pre-sigmoid logits) for `rows`, extracting
    /// features through `features` (the same builder used to train). Walks the
    /// trees on raw `f64` values.
    ///
    /// # Panics
    /// Panics if `features` doesn't match the model's declared features.
    pub fn predict_raw_scores<T>(&self, features: &FeatureBuilder<T>, rows: &[T]) -> Vec<f64> {
        self.check_features(features);
        let n_features = self.n_features;
        let mut flat = features.extract_row_major(rows);
        // Categorical nodes route by bin-code set membership, so rewrite the
        // raw value of each categorical column to its bin code (matching the
        // binned path); numeric columns keep their raw f64.
        let cat_feats: Vec<usize> =
            self.bin_mappers.iter().enumerate().filter(|(_, m)| m.is_categorical()).map(|(i, _)| i).collect();
        if !cat_feats.is_empty() {
            for chunk in flat.chunks_mut(n_features.max(1)) {
                for &j in &cat_feats {
                    chunk[j] = self.bin_mappers[j].value_to_bin(chunk[j]) as f64;
                }
            }
        }
        let init = self.init_score;
        flat.chunks(n_features.max(1))
            .map(|r| {
                let mut s = init;
                for tree in &self.trees {
                    s += self.learning_rate * tree.predict_raw(r);
                }
                s
            })
            .collect()
    }

    /// Predict probabilities (sigmoid of raw scores) for `rows`.
    pub fn predict_proba<T>(&self, features: &FeatureBuilder<T>, rows: &[T]) -> Vec<f64> {
        let raw = self.predict_raw_scores(features, rows);
        raw.into_iter().map(sigmoid).collect()
    }

    /// Predict raw additive scores against an already-binned dataset. Faster
    /// than the raw f64 path when you can amortize binning across many
    /// predict calls.
    pub(crate) fn predict_raw_scores_on_dataset(&self, dataset: &Dataset) -> Vec<f64> {
        let n = dataset.n_rows();
        let mut scores = vec![self.init_score; n];
        let feats: Vec<usize> = (0..dataset.n_features()).collect();
        with_columns!(dataset, feats, |cols| {
            self.predict_into_with_columns(&cols, n, &mut scores);
        });
        scores
    }

    /// Tree-outer / row-inner accumulation: keeps the current tree's nodes hot
    /// in L1 across the full row sweep.
    fn predict_into_with_columns<B: Bin>(
        &self,
        columns: &[&[B]],
        n_rows: usize,
        scores: &mut [f64],
    ) {
        for tree in &self.trees {
            for (row, s) in scores.iter_mut().enumerate().take(n_rows) {
                *s += self.learning_rate * tree.predict_on_columns(columns, row);
            }
        }
    }

    /// Extract features through `features`, bin them with the training-time
    /// mappers, then walk trees on bin codes. Faster than
    /// [`Model::predict_raw_scores`] on batches (~10K+ rows): u8/u16
    /// comparisons instead of f64, ~8× smaller rows, no NaN check per node.
    /// Predictions match the raw path bit-for-bit.
    ///
    /// # Panics
    /// Panics if `features` doesn't match the model's declared features.
    pub fn predict_raw_scores_binned<T>(
        &self,
        features: &FeatureBuilder<T>,
        rows: &[T],
    ) -> Vec<f64> {
        self.check_features(features);
        let dataset = self.bin_for_predict(features.extract_columns(rows), rows.len());
        self.predict_raw_scores_on_dataset(&dataset)
    }

    /// Like [`Model::predict_proba`] but uses the binned inference path.
    pub fn predict_proba_binned<T>(&self, features: &FeatureBuilder<T>, rows: &[T]) -> Vec<f64> {
        let raw = self.predict_raw_scores_binned(features, rows);
        raw.into_iter().map(sigmoid).collect()
    }

    /// Bin column-major `columns` with `self.bin_mappers` and pack into a
    /// Dataset. Width choice mirrors the training-time dataset builder.
    fn bin_for_predict(&self, columns: Vec<Vec<f64>>, n_rows: usize) -> Dataset {
        let n_features = self.n_features;
        debug_assert_eq!(columns.len(), n_features);

        let max_num_bins = self
            .bin_mappers
            .iter()
            .map(|m| m.num_bins())
            .max()
            .unwrap_or(2);
        let width = if max_num_bins <= 256 {
            BinWidth::U8
        } else {
            BinWidth::U16
        };

        let bin_data = match width {
            BinWidth::U8 => BinData::U8(self.bin_columns::<u8>(&columns)),
            BinWidth::U16 => BinData::U16(self.bin_columns::<u16>(&columns)),
        };

        Dataset {
            n_rows,
            n_features,
            bin_data,
            bin_mappers: self.bin_mappers.clone(),
            labels: Vec::new(),
        }
    }

    fn bin_columns<B: Bin>(&self, columns: &[Vec<f64>]) -> Vec<Vec<B>> {
        columns
            .iter()
            .zip(self.bin_mappers.iter())
            .map(|(col, bm)| col.iter().map(|&v| B::from_u16(bm.value_to_bin(v))).collect())
            .collect()
    }
}
