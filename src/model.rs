use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::config::standard;
use serde::{Deserialize, Serialize};

use crate::dataset::BinMapper;
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

    /// Predict probabilities (sigmoid of the raw additive scores) for `rows`,
    /// extracting features through `features` (the same builder used to train).
    ///
    /// Bins each row through the training-time [`BinMapper`]s into a row-major
    /// scratch buffer, then walks the trees on bin codes (`u16` comparisons, no
    /// per-node NaN check). Walks tree-outer/row-inner so the current tree's
    /// nodes stay hot in L1 across the row sweep. This is the only inference
    /// path: it's the fastest at both per-request and bulk batch sizes.
    ///
    /// # Panics
    /// Panics if `features` doesn't match the model's declared features.
    pub fn predict_proba<T>(&self, features: &FeatureBuilder<T>, rows: &[T]) -> Vec<f64> {
        self.predict_raw_scores(features, rows).into_iter().map(sigmoid).collect()
    }

    /// Raw additive scores (pre-sigmoid logits). Private on purpose: the public
    /// serving surface is [`Model::predict_proba`]; raw logits aren't exposed.
    fn predict_raw_scores<T>(&self, features: &FeatureBuilder<T>, rows: &[T]) -> Vec<f64> {
        self.check_features(features);
        let nf = self.n_features.max(1);
        let mut bins: Vec<u16> = Vec::new();
        features.extract_bins_row_major(rows, &self.bin_mappers, &mut bins);

        let lr = self.learning_rate;
        let mut scores = vec![self.init_score; rows.len()];
        for tree in &self.trees {
            for (r, s) in scores.iter_mut().enumerate() {
                *s += lr * tree.predict_on_row_bins(&bins[r * nf..(r + 1) * nf]);
            }
        }
        scores
    }
}
