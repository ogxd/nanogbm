use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::config::standard;
use serde::{Deserialize, Serialize};

use crate::dataset::{Bin, BinData, BinMapper, BinWidth, Dataset, with_columns};
use crate::error::{Error, Result};
use crate::loss::sigmoid;
use crate::tree::Tree;

/// A trained GBDT model: ensemble of trees plus boosting metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub(crate) init_score: f64,
    pub(crate) learning_rate: f64,
    pub(crate) n_features: usize,
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

    /// Predict raw additive scores (pre-sigmoid logits) for a row-major feature
    /// matrix of shape `n_rows × self.n_features()`.
    ///
    /// # Panics
    /// Panics if `features.len() != n_rows * self.n_features()`.
    pub fn predict_raw_scores(&self, features: &[f64], n_rows: usize) -> Vec<f64> {
        let n_features = self.n_features;
        assert_eq!(
            features.len(),
            n_rows * n_features,
            "features.len() {} != n_rows {} * n_features {}",
            features.len(),
            n_rows,
            n_features
        );
        let init = self.init_score;
        (0..n_rows)
            .map(|row| {
                let r = &features[row * n_features..(row + 1) * n_features];
                let mut s = init;
                for tree in &self.trees {
                    s += self.learning_rate * tree.predict_raw(r);
                }
                s
            })
            .collect()
    }

    /// Predict probabilities (sigmoid of raw scores) for a row-major feature
    /// matrix of shape `n_rows × self.n_features()`.
    pub fn predict_proba(&self, features: &[f64], n_rows: usize) -> Vec<f64> {
        let raw = self.predict_raw_scores(features, n_rows);
        raw.into_iter().map(sigmoid).collect()
    }

    /// Predict raw additive scores against an already-binned dataset. Faster
    /// than the raw f64 path when you can amortize binning across many
    /// predict calls.
    pub fn predict_raw_scores_on_dataset(&self, dataset: &Dataset) -> Vec<f64> {
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
            for row in 0..n_rows {
                scores[row] += self.learning_rate * tree.predict_on_columns(columns, row);
            }
        }
    }

    /// Predict probabilities against an already-binned dataset.
    pub fn predict_proba_on_dataset(&self, dataset: &Dataset) -> Vec<f64> {
        let raw = self.predict_raw_scores_on_dataset(dataset);
        raw.into_iter().map(sigmoid).collect()
    }

    /// Bin the inputs using the training-time mappers, then walk trees on bin
    /// codes. Faster than [`Model::predict_raw_scores`] on batches (~10K+
    /// rows): u8/u16 comparisons instead of f64, ~8× smaller rows, no NaN
    /// check per node. Predictions match the raw path bit-for-bit.
    ///
    /// # Panics
    /// Panics if `features.len() != n_rows * self.n_features()`.
    pub fn predict_raw_scores_binned(&self, features: &[f64], n_rows: usize) -> Vec<f64> {
        let dataset = self.bin_for_predict(features, n_rows);
        self.predict_raw_scores_on_dataset(&dataset)
    }

    /// Like [`Model::predict_proba`] but uses the binned inference path.
    pub fn predict_proba_binned(&self, features: &[f64], n_rows: usize) -> Vec<f64> {
        let raw = self.predict_raw_scores_binned(features, n_rows);
        raw.into_iter().map(sigmoid).collect()
    }

    /// Bin `features` with `self.bin_mappers` and pack into a Dataset. Width
    /// choice mirrors [`crate::dataset::DatasetBuilder`].
    fn bin_for_predict(&self, features: &[f64], n_rows: usize) -> Dataset {
        let n_features = self.n_features;
        assert_eq!(
            features.len(),
            n_rows * n_features,
            "features.len() {} != n_rows {} * n_features {}",
            features.len(),
            n_rows,
            n_features
        );
        assert_eq!(
            self.bin_mappers.len(),
            n_features,
            "model.bin_mappers.len() {} != n_features {}",
            self.bin_mappers.len(),
            n_features
        );

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
            BinWidth::U8 => BinData::U8(self.bin_columns::<u8>(features, n_rows, n_features)),
            BinWidth::U16 => BinData::U16(self.bin_columns::<u16>(features, n_rows, n_features)),
        };

        Dataset {
            n_rows,
            n_features,
            bin_data,
            bin_mappers: self.bin_mappers.clone(),
            labels: Vec::new(),
        }
    }

    fn bin_columns<B: Bin>(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Vec<Vec<B>> {
        (0..n_features)
            .map(|feat| {
                let bm = &self.bin_mappers[feat];
                let mut col: Vec<B> = Vec::with_capacity(n_rows);
                for row in 0..n_rows {
                    let v = features[row * n_features + feat];
                    col.push(B::from_u16(bm.value_to_bin(v)));
                }
                col
            })
            .collect()
    }
}
