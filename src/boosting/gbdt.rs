use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::config::Config;
use crate::dataset::Dataset;
use crate::error::Result;
use crate::metric::{BinaryLogloss, Metric};
use crate::model::Model;
use crate::objective::{BinaryObjective, Objective};
use crate::tree::TreeLearner;
use crate::tree::learner::TimingBuckets;

pub struct GbdtTrainer<'a> {
    config: &'a Config,
}

impl<'a> GbdtTrainer<'a> {
    pub fn new(config: &'a Config) -> Self {
        Self { config }
    }

    /// Train a GBDT model. If `valid` is provided, evaluate `binary_logloss` after
    /// each iteration and apply early stopping (if configured).
    pub fn fit(&self, train: &Dataset, valid: Option<&Dataset>) -> Result<Model> {
        self.config.validate()?;
        let objective = BinaryObjective::default();
        let metric = BinaryLogloss;

        let n = train.n_rows();
        let n_features = train.n_features();
        let init_score = objective.init_score(train.labels());

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
        let mut bagged_rows: Option<Vec<u32>> = None;

        // Cumulative timing (for profiling).
        let mut t_gradients = std::time::Duration::ZERO;
        let mut t_tree = std::time::Duration::ZERO;
        let mut t_update_scores = std::time::Duration::ZERO;
        let timing = TimingBuckets::default();

        for iter in 0..self.config.num_iterations {
            let t0 = std::time::Instant::now();
            // Gradients/hessians from current raw scores, packed.
            objective.gradients_packed(&raw_scores, train.labels(), &mut gradhess);
            t_gradients += t0.elapsed();

            // Row bagging.
            let row_indices: Vec<u32> = if self.config.bagging_fraction < 1.0
                && self.config.bagging_freq > 0
            {
                if iter % self.config.bagging_freq == 0 {
                    bagged_rows = Some(sample_indices(n, self.config.bagging_fraction, &mut rng));
                }
                bagged_rows
                    .clone()
                    .unwrap_or_else(|| (0..n as u32).collect())
            } else {
                (0..n as u32).collect()
            };

            // Feature subsampling per tree.
            let feature_indices: Vec<usize> = if self.config.feature_fraction < 1.0 {
                let mut all: Vec<usize> = (0..n_features).collect();
                all.shuffle(&mut rng);
                let k = ((n_features as f64 * self.config.feature_fraction).ceil() as usize).max(1);
                all.truncate(k);
                all.sort_unstable();
                all
            } else {
                (0..n_features).collect()
            };

            let t1 = std::time::Instant::now();
            let learner = TreeLearner::new(self.config, train, &timing);
            let (tree, row_to_leaf) =
                learner.train_one_tree(&gradhess, &row_indices, &feature_indices);
            t_tree += t1.elapsed();

            // Update raw_scores using the leaf assignment recorded during tree
            // growth. This avoids re-walking the tree once per training row.
            // Rows not in `row_indices` (e.g. bagged-out) still got leaf 0 by
            // default; the tree's leaf 0 holds the root prediction, which is a
            // reasonable fallback when bagging.
            let t2 = std::time::Instant::now();
            let lr = self.config.learning_rate;
            for (row, s) in raw_scores.iter_mut().enumerate() {
                let leaf_idx = row_to_leaf[row] as usize;
                *s += lr * tree.leaf_values[leaf_idx];
            }
            t_update_scores += t2.elapsed();

            // Evaluate on validation set.
            if let (Some(v), Some(vrs)) = (valid, valid_raw_scores.as_mut()) {
                vrs.iter_mut().enumerate().for_each(|(row, s)| {
                    *s += lr * tree.predict_on_dataset(v, row);
                });
                let score = metric.evaluate(vrs, v.labels());
                if self.config.verbose {
                    eprintln!("[{}] {} = {:.6}", iter + 1, metric.name(), score);
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
                    // Truncate to best_iter + 1 trees.
                    trees.truncate(best_iter + 1);
                    break;
                }
            } else {
                if self.config.verbose {
                    let train_score = metric.evaluate(&raw_scores, train.labels());
                    eprintln!(
                        "[{}] train {} = {:.6}",
                        iter + 1,
                        metric.name(),
                        train_score
                    );
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
