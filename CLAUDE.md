# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`nanogbm` is a pure-Rust gradient boosting library, deliberately scoped to a narrow subset: **GBDT boosting only**, **binary logistic objective only**, **binary_logloss metric only**, CPU only, dense features (numerical with missing-value handling, plus native categorical subset splits), optional per-sample weights. No DART/GOSS/RF, no sparse input, no GPU, no FFI. Keep changes within this scope unless explicitly asked to expand it.

The library crate and the workspace are both called `nanogbm`.

Rust edition is **2024** (workspace-wide). Training is currently **single-threaded** — `TimingBuckets` uses `Cell` precisely because nothing runs in parallel. Treat any parallelism as a future change, not present state.

## Commands

```bash
cargo build --release
cargo test --release                              # unit tests + tests/e2e.rs
cargo test --release -p nanogbm <name>            # single test by name substring
cargo test --release --test e2e                   # only the e2e integration suite
cargo clippy --all-targets --release
cargo fmt
```

Always prefer `--release` — debug builds of the training loop are orders of magnitude slower and skew any perf observation. Set `Config::verbose = true` to get per-iteration validation scores and a per-phase timing dump at end of fit (`hist_build`, `hist_subtract`, `split_search`, `partition`, gradients, score updates).

## Workspace layout

Single crate:
- `nanogbm/` — the library. Public re-exports from `lib.rs`: `Config`, `Dataset`, `FeatureBuilder`, `GbdtTrainer`, `Model`, `Error`, `Result`.

The public entry point is `FeatureBuilder<T>`: you declare each feature as `(name, optional max_bin, closure: Fn(&T) -> f64)` over your own row type. `GbdtTrainer::new(&cfg, &fb).fit(&rows, &labels, valid)` extracts + bins via the builder and trains; `Model::predict_*(&fb, &rows)` re-extracts through the *same* builder (closures aren't serializable — the model only stores bin mappers and feature names). `Dataset` is the internal binned matrix, built by `FeatureBuilder::build_dataset`; there is no longer a `DatasetBuilder`.

Workspace deps: `serde`, `bincode` v2 (with the `serde` feature), `rand` + `rand_chacha`, `thiserror`. No `rayon`, no math crates — everything is hand-rolled.

## How it works internally (big picture)

The model is just **a list of small decision trees**. To predict, you walk every tree, sum up the numbers in the leaves you land in, and squash the total through a sigmoid to get a probability. Training builds those trees one at a time, where each new tree is fit to correct the errors of the trees built so far. That's the whole algorithm.

The rest is performance tricks. End-to-end flow of one `GbdtTrainer::fit` call:

1. **Bucket all feature values once, up front** (`dataset::bin_mapper`, `dataset::builder`). Real-valued columns are slow to split on — you'd have to consider every distinct value as a possible split point. So before training, each feature column is bucketed into at most ~255 buckets using quantiles, and the raw `f64` is replaced by a small `u16` bucket id. From here on the trees only ever look at bucket ids. Bucket **0 is reserved for "missing" (NaN)**. The mapping (`BinMapper`) is fit on training data and **reused as-is** for validation and inference — never refit it. The `Dataset` is stored **column-major** because every hot loop iterates one feature across many rows.

2. **Outer loop: build trees one at a time** (`boosting::gbdt`). Keep a running per-row score (a logit) initialized to the prior log-odds of the labels. At each iteration:
   - For every row, compute two numbers describing "how wrong is the current score for this row, and how confidently": gradient and hessian of the logistic loss. They're packed as `[f32; 2]` so the inner loop loads both in one 8-byte read.
   - Optionally sample a fraction of rows and/or features (cheap regularization). RNG is `ChaCha8Rng` seeded from `Config::seed` — deterministic.
   - Build one new tree that, on the sampled rows, predicts those gradients (`train_one_tree`, step 3).
   - Add `learning_rate * leaf_value` to each row's running score. The learner returns a `row → leaf` map so we don't re-walk the tree per row.
   - If a validation set is provided, score it, track best iteration, and stop early after `early_stopping_round` iterations without improvement. **On stop, trees are truncated to `best_iter + 1`** so the saved model is the best one, not the last.

3. **Inner loop: grow one tree** (`tree::learner`, `tree::histogram`, `tree::split`). Start with one leaf containing all (sampled) rows. Repeatedly pick the leaf whose best split improves the loss the most, and split it in two. Stop when `num_leaves` is hit, or when no remaining leaf has a split that passes the `min_data_in_leaf` / `min_gain_to_split` thresholds. Finding the best split is the hot path:
   - For the leaf's rows, build a small table per feature: for each bucket id, the sum of gradients, sum of hessians, and row count. The best split is then a single scan over those tables — the regularized gain formula tells you the best place to cut, with L1/L2 penalties from `lambda_l1` / `lambda_l2`. Missing values (bucket 0) are routed to whichever side maximizes gain; that direction is recorded on the node so prediction reproduces it.
   - **Sibling-by-subtraction** (load-bearing perf trick): after a split, only build those tables from scratch for the **smaller** child. The larger child's tables are computed as `parent − smaller`. Breaking this invariant doubles tree-build time.
   - Tree storage: internal nodes are `SplitNode`s in `tree.nodes`; leaf values are in `tree.leaf_values`. Child pointers use a sentinel — **negative means leaf, encoded as `!(idx as i32)`**; non-negative is an internal-node index.

4. **Inference** (`predict`). A `Model` is `{ init_score, learning_rate, n_features, trees }`. Walk each tree, sum `init_score + Σ learning_rate * leaf_value`, sigmoid for probabilities. Two code paths exist — predicting from raw `f64` inputs (which bucketize on the fly using the model's `BinMapper`s) and predicting from an already-bucketed `Dataset` (used internally during validation). **They must produce identical predictions** — the `e2e` test enforces it.

5. **Serialization** (`model.rs`). `bincode` v2 with `serde` derives. Any layout change to `Tree`, `SplitNode`, `BinMapper`, or `Model` breaks previously saved models.

## Tests

`nanogbm/tests/e2e.rs` is the integration suite and protects three properties:
- **convergence** on a synthetic problem (training logloss actually decreases),
- **bincode round-trip** of a trained model,
- **bin-path vs raw-path prediction consistency**.

If you touch binning, splits, missing-direction logic, or serialization, run the full e2e suite — unit tests alone won't catch path-consistency or round-trip regressions.

## Determinism

Same `Config` + same data must produce identical models. All randomness (bagging, feature subsampling) flows through a single `ChaCha8Rng` seeded from `Config::seed`. If a change introduces non-determinism, that's a bug.

## Config

`Config` exposes the usual GBDT knobs: `num_iterations`, `num_leaves`, `learning_rate`, `min_data_in_leaf`, `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `max_bin`, `min_data_in_bin`, `bagging_fraction`, `bagging_freq`, `feature_fraction`, `early_stopping_round`, `seed`, `verbose`. The names are stable; downstream code depends on them.
