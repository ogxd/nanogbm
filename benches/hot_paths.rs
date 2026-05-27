//! Microbenchmarks for the hot paths of `nanogbm`.
//!
//! Goal: per-core efficiency measurement. Each bench isolates one inner loop
//! so we can attribute speedups (or regressions) to a specific change.
//!
//! Run with: `cargo bench --bench hot_paths`
//! Filter:   `cargo bench --bench hot_paths -- hist_build`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use nanogbm::dataset::{BinMapper, Dataset, DatasetBuilder};
use nanogbm::loss;
use nanogbm::tree::histogram::{
    FeatureHistogram, build_histograms_batched, build_histograms_batched_full,
};
use nanogbm::tree::split::find_best_split_for_feature;
use nanogbm::{Config, GbdtTrainer};

/// Bench fixture sizes. Chosen to be representative of "real" workloads while
/// still amortizing the criterion timer (~50ns overhead per sample).
const N_ROWS: usize = 100_000;
const N_FEATURES: usize = 16;
const MAX_BIN: usize = 64; // 63 real + 1 missing — typical config
const SUBSAMPLE_FRACTION: f64 = 0.5;

/// One synthetic, fully-built fixture shared across benches.
struct Fixture {
    dataset: Dataset,
    gradhess: Vec<[f32; 2]>,
    /// 50% subsample of row indices (sorted), used by indexed paths.
    indices: Vec<u32>,
    /// Raw scores buffer (input to gradients).
    raw_scores: Vec<f64>,
    labels: Vec<f32>,
    cfg: Config,
}

fn build_fixture(seed: u64) -> Fixture {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Synthetic linearly-separable data, same shape as e2e but bigger.
    let weights: Vec<f64> = (0..N_FEATURES).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let bias: f64 = rng.gen_range(-0.5..0.5);
    let mut features = vec![0.0f64; N_ROWS * N_FEATURES];
    let mut labels = vec![0f32; N_ROWS];
    for i in 0..N_ROWS {
        let mut z = bias;
        for j in 0..N_FEATURES {
            let x: f64 = rng.gen_range(-3.0..3.0);
            features[i * N_FEATURES + j] = x;
            z += weights[j] * x;
        }
        z += rng.gen_range(-0.4..0.4);
        let p = 1.0 / (1.0 + (-z).exp());
        labels[i] = if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 };
    }

    let mut cfg = Config::default();
    cfg.num_iterations = 10;
    cfg.num_leaves = 31;
    cfg.min_data_in_leaf = 20;
    cfg.max_bin = MAX_BIN;
    cfg.learning_rate = 0.1;
    cfg.lambda_l2 = 1.0;
    cfg.seed = 0;

    let dataset =
        DatasetBuilder::from_rows(&features, N_ROWS, N_FEATURES, &labels, &cfg).unwrap();

    // Realistic raw_scores: not all zero (would skew sigmoid). Use a small
    // dispersion around the init log-odds.
    let init = loss::init_score(&labels);
    let raw_scores: Vec<f64> = (0..N_ROWS)
        .map(|_| init + rng.gen_range(-0.5..0.5))
        .collect();

    // Compute initial gradhess.
    let mut gradhess = vec![[0.0f32; 2]; N_ROWS];
    loss::gradients_packed(&raw_scores, &labels, &mut gradhess);

    // 50% subsample of indices, sorted.
    let mut indices: Vec<u32> = (0..N_ROWS as u32).collect();
    indices.shuffle(&mut rng);
    indices.truncate((N_ROWS as f64 * SUBSAMPLE_FRACTION) as usize);
    indices.sort_unstable();

    Fixture {
        dataset,
        gradhess,
        indices,
        raw_scores,
        labels,
        cfg,
    }
}

fn bench_gradients(c: &mut Criterion, fx: &Fixture) {
    let mut group = c.benchmark_group("gradients");
    group.throughput(Throughput::Elements(N_ROWS as u64));

    let mut out = vec![[0.0f32; 2]; N_ROWS];

    group.bench_function("gradients_packed", |b| {
        b.iter(|| {
            loss::gradients_packed(
                black_box(&fx.raw_scores),
                black_box(&fx.labels),
                black_box(&mut out),
            );
        })
    });
    group.finish();
}

fn bench_hist_build_one_feature(c: &mut Criterion, fx: &Fixture) {
    let mut group = c.benchmark_group("hist_build_one_feature");

    // Single-feature build_full (sequential, no indices indirection).
    let col = fx.dataset.feature_column_u8(0);
    let num_bins = fx.dataset.bin_mapper(0).num_bins();
    let mut hist = FeatureHistogram::zeros(num_bins);
    group.throughput(Throughput::Elements(N_ROWS as u64));
    group.bench_function("build_full", |b| {
        b.iter(|| {
            hist.build_full(black_box(col), black_box(&fx.gradhess));
            black_box(&hist);
        })
    });

    // Single-feature indexed build (50% subsample, sorted indices).
    let mut hist_idx = FeatureHistogram::zeros(num_bins);
    group.throughput(Throughput::Elements(fx.indices.len() as u64));
    group.bench_function("build_indexed_50pct", |b| {
        b.iter(|| {
            hist_idx.build(
                black_box(col),
                black_box(&fx.indices),
                black_box(&fx.gradhess),
            );
            black_box(&hist_idx);
        })
    });

    group.finish();
}

fn bench_hist_build_batched(c: &mut Criterion, fx: &Fixture) {
    let mut group = c.benchmark_group("hist_build_batched");

    let columns: Vec<&[u8]> = (0..N_FEATURES)
        .map(|f| fx.dataset.feature_column_u8(f))
        .collect();
    let mut hists: Vec<FeatureHistogram> = (0..N_FEATURES)
        .map(|f| FeatureHistogram::zeros(fx.dataset.bin_mapper(f).num_bins()))
        .collect();

    // Throughput in terms of (rows × features) — total scatter-adds done.
    group.throughput(Throughput::Elements((N_ROWS * N_FEATURES) as u64));
    group.bench_function("batched_full", |b| {
        b.iter(|| {
            build_histograms_batched_full(
                black_box(&columns),
                black_box(&fx.gradhess),
                black_box(&mut hists),
            );
            black_box(&hists);
        })
    });

    group.throughput(Throughput::Elements((fx.indices.len() * N_FEATURES) as u64));
    group.bench_function("batched_indexed_50pct", |b| {
        b.iter(|| {
            build_histograms_batched(
                black_box(&columns),
                black_box(&fx.indices),
                black_box(&fx.gradhess),
                black_box(&mut hists),
            );
            black_box(&hists);
        })
    });

    group.finish();
}

fn bench_hist_subtract(c: &mut Criterion, fx: &Fixture) {
    let mut group = c.benchmark_group("hist_subtract");

    // Build one realistic parent histogram + a smaller-child histogram, then
    // bench the subtraction inner loop.
    let col = fx.dataset.feature_column_u8(0);
    let num_bins = fx.dataset.bin_mapper(0).num_bins();
    let mut parent = FeatureHistogram::zeros(num_bins);
    parent.build_full(col, &fx.gradhess);
    let mut child = FeatureHistogram::zeros(num_bins);
    child.build(col, &fx.indices, &fx.gradhess);
    let mut out = FeatureHistogram::zeros(num_bins);

    group.throughput(Throughput::Elements(num_bins as u64));
    group.bench_function(BenchmarkId::new("subtract_into", num_bins), |b| {
        b.iter(|| {
            FeatureHistogram::subtract_into(
                black_box(&parent),
                black_box(&child),
                black_box(&mut out),
            );
            black_box(&out);
        })
    });
    group.finish();
}

fn bench_split_scan(c: &mut Criterion, fx: &Fixture) {
    let mut group = c.benchmark_group("split_scan");

    // One realistic histogram, then bench the full split-gain scan.
    let col = fx.dataset.feature_column_u8(0);
    let num_bins = fx.dataset.bin_mapper(0).num_bins();
    let mut hist = FeatureHistogram::zeros(num_bins);
    hist.build_full(col, &fx.gradhess);
    let bin_mapper: &BinMapper = fx.dataset.bin_mapper(0);

    // Parent sums = sum over all bins.
    let parent_grad: f64 = hist.bins.iter().map(|b| b.grad).sum();
    let parent_hess: f64 = hist.bins.iter().map(|b| b.hess).sum();
    let parent_count: u32 = hist.bins.iter().map(|b| b.count).sum();

    group.throughput(Throughput::Elements(num_bins as u64));
    group.bench_function(BenchmarkId::new("find_best_split", num_bins), |b| {
        b.iter(|| {
            let s = find_best_split_for_feature(
                0,
                black_box(&hist),
                black_box(bin_mapper),
                parent_grad,
                parent_hess,
                parent_count,
                black_box(&fx.cfg),
            );
            black_box(s);
        })
    });
    group.finish();
}

fn bench_score_update(c: &mut Criterion, _fx: &Fixture) {
    let mut group = c.benchmark_group("score_update");

    // The score update loop from gbdt.rs:
    //   for (row, s) in raw_scores.iter_mut().enumerate() {
    //       *s += lr * tree.leaf_values[row_to_leaf[row] as usize];
    //   }
    let n = N_ROWS;
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut scores = vec![0.0f64; n];
    let leaf_values: Vec<f64> = (0..31).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let row_to_leaf: Vec<u32> = (0..n).map(|_| rng.gen_range(0..31u32)).collect();
    let lr = 0.1f64;

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("score_update_loop", |b| {
        b.iter(|| {
            for (row, s) in scores.iter_mut().enumerate() {
                let leaf_idx = unsafe { *row_to_leaf.get_unchecked(row) } as usize;
                let v = unsafe { *leaf_values.get_unchecked(leaf_idx) };
                *s += lr * v;
            }
            black_box(&scores);
        })
    });
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion, fx: &Fixture) {
    let mut group = c.benchmark_group("end_to_end");
    // End-to-end is slower; keep sample count modest.
    group.sample_size(10);

    let mut cfg = fx.cfg.clone();
    cfg.num_iterations = 10;
    cfg.early_stopping_round = 0;

    group.bench_function("fit_10_iters", |b| {
        b.iter(|| {
            let model = GbdtTrainer::new(&cfg).fit(&fx.dataset, None).unwrap();
            black_box(model);
        })
    });
    group.finish();
}

fn all_benches(c: &mut Criterion) {
    let fx = build_fixture(42);
    bench_gradients(c, &fx);
    bench_hist_build_one_feature(c, &fx);
    bench_hist_build_batched(c, &fx);
    bench_hist_subtract(c, &fx);
    bench_split_scan(c, &fx);
    bench_score_update(c, &fx);
    bench_end_to_end(c, &fx);
}

criterion_group!(benches, all_benches);
criterion_main!(benches);
