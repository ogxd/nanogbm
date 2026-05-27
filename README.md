# nanogbm

A small gradient boosting library, in pure Rust, with a deliberately narrow
scope: **GBDT only, binary classification only, CPU only, dense numerical
features**. No DART/GOSS/RF, no multiclass, no ranking, no regression, no
sparse inputs, no GPU, no FFI bindings.

What you get in return is a few thousand lines of code you can read end to
end and actually follow — useful both as a learning artifact and as a
no-FFI dependency in a Rust service.

```toml
[dependencies]
nanogbm = "0.3"
```

```rust
use nanogbm::{Config, DatasetBuilder, GbdtTrainer};

let cfg = Config { num_iterations: 100, learning_rate: 0.1, num_leaves: 31, ..Config::default() };
let train = DatasetBuilder::from_rows(&features, n_rows, n_features, &labels, &cfg)?;
let model = GbdtTrainer::new(&cfg).fit(&train, None)?;
let probs = model.predict_proba(&features, n_rows);
```

## Why does this exist?

LightGBM and XGBoost are excellent and you should reach for them whenever
you can. They're also large C++ codebases with non-trivial build systems,
and to actually *understand* what they do, you eventually have to sit down
with a histogram-based learner small enough to fit in your head. That's the
primary purpose of this code.

The secondary purpose is practical: when you want to train a model from
inside a Rust service, a pure-Rust crate is a much smaller commitment than
linking C++ through an FFI shim. `cargo build` and that's it.  

Performance-wise, nanogbm is supposed to be as fast, if not faster, than LightGBM, thanks to optimizations tailored to the GBDT + histogram learner + binary logistic loss subset.

## What's actually in the box

- **GBDT.** Trees built one at a time, each one fitting the gradient of the
  loss so far.
- **Binary logistic loss.** Only. The objective is hardcoded on purpose.
- **Histogram learner with sibling-by-subtraction.** After a split, only the
  smaller child's histograms are built from scratch; the larger sibling is
  recovered by subtracting from the parent. This is the load-bearing perf
  trick — `CLAUDE.md` has the details.
- **Missing values handled at the split.** Bucket 0 is reserved for NaN, and
  the learner picks per-node which side missing values go to, by gain.
- **Early stopping** that actually truncates the model to the best
  iteration, so the model you save is the one that won — not whatever the
  loop happened to land on when it gave up.
- **Determinism.** Same `Config` + same data → byte-identical model. All
  randomness flows through a single `ChaCha8Rng` seeded from `Config::seed`.
- **Bincode v2 serialization** with serde derives. Stable across runs;
  re-check after layout changes to `Tree`, `SplitNode`, `BinMapper`, or
  `Model`.
- **A feature-encoding helper layer** (`nanogbm::feature`). You write one
  `encode_into` function that pushes `num`, `bool`, `cat`, `cat_hashed`, or
  `multi_hot` values into a sink, and run it twice — once with
  `DiscoverySink` to derive a `Schema`, then with `SliceSink` per row on the
  hot path. Worth being precise here: the schema *knows* which columns are
  categorical (the feature-importance printer uses it), but the learner does
  **not** do native categorical splits. `cat(v)` writes `v as f64`,
  `cat_hashed` writes a hash bucket index as `f64`, and the trees then split
  those columns numerically like any other feature. If you need true subset
  splits, expand to one-hot via `multi_hot` and let the learner work on
  that.

## What's not in the box

| Thing                            | Status                                              |
|----------------------------------|-----------------------------------------------------|
| Multiclass / regression / rank   | No                                                  |
| Native categorical splits        | No — categoricals encode to numeric, see `feature`  |
| Sparse input                     | No                                                  |
| DART / GOSS / RF mode            | No                                                  |
| GPU                              | No                                                  |
| Multithreading                   | No (single-threaded today, not a principle)         |
| Python / C / WASM bindings       | No                                                  |

The single-thread limitation is a current fact, not a design principle:
`TimingBuckets` uses `Cell` specifically because nothing runs in parallel
yet. Parallelism may come later, but it would be a deliberate change.

## Examples

```
cargo run --release --example basic
cargo run --release --example early_stopping
cargo run --release --example missing_and_importance
cargo run --release --example save_load
```

Always run in `--release`; debug builds of the training loop are orders of
magnitude slower and will skew any timing observation. Set
`Config::verbose = true` to get per-iteration validation scores and an
end-of-fit timing dump (`hist_build`, `hist_subtract`, `split_search`,
`partition`, gradients, score updates) — useful when you want to see where
the time actually went.

## Tests

```
cargo test --release
cargo test --release --test e2e
```

The integration suite (`tests/e2e.rs`) protects three things and you
should care about all of them:

1. **Convergence** on a synthetic problem — if it can't fit easy data, it
   can't fit hard data.
2. **Bincode round-trip** — save, load, predict, identical results.
3. **Bin-path vs raw-path prediction consistency** — predicting from raw
   `f64` and predicting from a pre-bucketed `Dataset` must produce
   bit-identical outputs. If you touch binning, splits, missing-direction
   logic, or serialization, run this.

## A reading order, if you're here to learn

1. `boosting/gbdt.rs` — the outer loop. Build N trees, each one fitting the
   gradient of the loss the previous trees haven't explained.
2. `tree/learner.rs` — the inner loop. Grow one tree leaf-wise until you hit
   `num_leaves` or no leaf has a profitable split left.
3. `tree/histogram.rs` + `tree/split.rs` — the part that's actually fast.
   Per-feature gradient/hessian histograms, regularized gain formula,
   missing-direction selection.
4. `dataset/bin_mapper.rs` — how a column of `f64` becomes a column of
   `u16` bucket ids, and why bucket 0 is special.
5. `predict.rs` — walk the trees, sum, sigmoid. The whole inference path.

## License

MIT.
