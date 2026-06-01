use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::config::Config;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::feature::FeatureBuilder;
use crate::loss;
use crate::model::Model;
use crate::tree::TreeLearner;
use crate::tree::learner::TimingBuckets;

/// Trains a GBDT model from a [`Config`] and a [`FeatureBuilder<T>`]. The
/// builder declares how each feature column is pulled from your row type `T`;
/// [`fit`](Self::fit) takes the rows and labels directly and bins internally.
pub struct GbdtTrainer<'a, T> {
    config: &'a Config,
    features: &'a FeatureBuilder<T>,
    weights: Option<&'a [f32]>,
}

impl<'a, T> GbdtTrainer<'a, T> {
    pub fn new(config: &'a Config, features: &'a FeatureBuilder<T>) -> Self {
        Self { config, features, weights: None }
    }

    /// Per-row sample weights (LightGBM `weight` column); the binary-logistic
    /// loss is scaled per row, carrying through to leaf values and split gains.
    /// Length must equal the training rows passed to [`fit`](Self::fit).
    pub fn with_weights(mut self, weights: &'a [f32]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Train a GBDT model on `rows`/`labels`. Features are extracted and binned
    /// via the [`FeatureBuilder`]. If `valid` is provided (its own rows +
    /// labels), evaluate `binary_logloss` after each iteration and apply early
    /// stopping (if configured).
    pub fn fit(
        &self,
        rows: &[T],
        labels: &[f32],
        valid: Option<(&[T], &[f32])>,
    ) -> Result<Model> {
        self.config.validate()?;
        if let Some(w) = self.weights {
            if w.len() != labels.len() {
                return Err(Error::Config(format!("weights len {} != labels len {}", w.len(), labels.len())));
            }
        }
        let train = self.features.build_dataset(rows, labels, self.config)?;
        let valid_ds = match valid {
            Some((vr, vl)) => Some(self.features.build_dataset(vr, vl, self.config)?),
            None => None,
        };
        self.fit_dataset(&train, valid_ds.as_ref())
    }

    /// Core boosting loop over already-binned datasets.
    fn fit_dataset(&self, train: &Dataset, valid: Option<&Dataset>) -> Result<Model> {

        let n = train.n_rows();
        let n_features = train.n_features();
        let init_score = loss::init_score(train.labels(), self.weights);

        let mut raw_scores = vec![init_score; n];
        // Packed [grad, hess] pairs — one 8-byte load per row in the histogram
        // hot loop, ~33% fewer memory ops vs separate f32 arrays.
        let mut gradhess: Vec<[f32; 2]> = vec![[0.0, 0.0]; n];

        let mut valid_raw_scores: Option<Vec<f64>> = valid.map(|v| vec![init_score; v.n_rows()]);

        let mut trees: Vec<crate::tree::Tree> = Vec::with_capacity(self.config.num_iterations);
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);

        let mut best_score = f64::INFINITY;
        let mut best_iter: usize = 0;
        let mut rounds_no_improve: usize = 0;

        // Allocated once and reused. `all_rows` and `all_features` are the
        // no-subsample slices; `bagged_rows` is rewritten in-place every
        // `bagging_freq` iterations.
        let all_rows: Vec<u32> = (0..n as u32).collect();
        let all_features: Vec<usize> = (0..n_features).collect();
        let mut bagged_rows: Vec<u32> = Vec::new();
        let bagging_on = self.config.bagging_fraction < 1.0 && self.config.bagging_freq > 0;
        let feature_subsample_on = self.config.feature_fraction < 1.0;

        let mut t_gradients = std::time::Duration::ZERO;
        let mut t_tree = std::time::Duration::ZERO;
        let mut t_update_scores = std::time::Duration::ZERO;
        let timing = TimingBuckets::default();

        for iter in 0..self.config.num_iterations {
            let t0 = std::time::Instant::now();
            loss::gradients_packed(&raw_scores, train.labels(), self.weights, &mut gradhess);
            t_gradients += t0.elapsed();

            let row_indices: &[u32] = if bagging_on {
                if iter % self.config.bagging_freq == 0 {
                    bagged_rows = sample_indices(n, self.config.bagging_fraction, &mut rng);
                }
                &bagged_rows
            } else {
                &all_rows
            };

            let feature_subsample: Vec<usize> = if feature_subsample_on {
                let mut all: Vec<usize> = (0..n_features).collect();
                all.shuffle(&mut rng);
                let k = ((n_features as f64 * self.config.feature_fraction).ceil() as usize).max(1);
                all.truncate(k);
                all.sort_unstable();
                all
            } else {
                Vec::new()
            };
            let feature_indices: &[usize] = if feature_subsample_on {
                &feature_subsample
            } else {
                &all_features
            };

            // `is_full` lets the learner skip the indices-indirection in the
            // root histogram build and use the sequential `_full` path.
            let is_full = !bagging_on;

            let t1 = std::time::Instant::now();
            let learner = TreeLearner::new(self.config, train, &timing);
            let (tree, row_to_leaf) =
                learner.train_one_tree(&gradhess, row_indices, feature_indices, is_full);
            t_tree += t1.elapsed();

            // Update raw_scores using the per-row leaf assignment recorded
            // during tree growth — avoids re-walking the tree per row. Bagged-
            // out rows stay mapped to leaf 0, which holds the root prediction.
            let t2 = std::time::Instant::now();
            let lr = self.config.learning_rate;
            for (row, s) in raw_scores.iter_mut().enumerate() {
                let leaf_idx = row_to_leaf[row] as usize;
                *s += lr * tree.leaf_values[leaf_idx];
            }
            t_update_scores += t2.elapsed();

            if let (Some(v), Some(vrs)) = (valid, valid_raw_scores.as_mut()) {
                vrs.iter_mut().enumerate().for_each(|(row, s)| {
                    *s += lr * tree.predict_on_dataset(v, row);
                });
                let score = loss::binary_logloss(vrs, v.labels());
                if self.config.verbose {
                    eprintln!("[{}] binary_logloss = {:.6}", iter + 1, score);
                }
                if score + 1e-12 < best_score {
                    best_score = score;
                    best_iter = iter;
                    rounds_no_improve = 0;
                } else {
                    rounds_no_improve += 1;
                }
                trees.push(tree);
                if self.config.early_stopping_round > 0
                    && rounds_no_improve >= self.config.early_stopping_round
                {
                    trees.truncate(best_iter + 1);
                    break;
                }
            } else {
                if self.config.verbose {
                    let train_score = loss::binary_logloss(&raw_scores, train.labels());
                    eprintln!("[{}] train binary_logloss = {:.6}", iter + 1, train_score);
                }
                trees.push(tree);
            }
        }

        if self.config.verbose {
            eprintln!(
                "[nanogbm] fit timing: gradients={:.2}s, tree_build={:.2}s (hist_build={:.2}s, hist_subtract={:.2}s, split_search={:.2}s, partition={:.2}s), update_scores={:.2}s",
                t_gradients.as_secs_f32(),
                t_tree.as_secs_f32(),
                timing.hist_build.get().as_secs_f32(),
                timing.hist_subtract.get().as_secs_f32(),
                timing.split_search.get().as_secs_f32(),
                timing.partition.get().as_secs_f32(),
                t_update_scores.as_secs_f32(),
            );
        }

        Ok(Model {
            init_score,
            learning_rate: self.config.learning_rate,
            n_features,
            feature_names: self.features.names().map(String::from).collect(),
            bin_mappers: train.bin_mappers().to_vec(),
            trees,
        })
    }
}

fn sample_indices(n: usize, fraction: f64, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let k = ((n as f64 * fraction).ceil() as usize).clamp(1, n);
    let mut all: Vec<u32> = (0..n as u32).collect();
    all.shuffle(rng);
    all.truncate(k);
    all.sort_unstable();
    all
}
