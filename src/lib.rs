//! A small, pure-Rust gradient boosting library with a deliberately narrow
//! scope: **GBDT only, binary classification only, CPU only, dense numerical
//! features**. No DART/GOSS/RF, no multiclass, no ranking, no regression, no
//! sparse inputs, no GPU, no FFI bindings.
//!
//! # Quickstart
//!
//! ```no_run
//! use nanogbm::{Config, DatasetBuilder, GbdtTrainer};
//!
//! # let n_rows = 1000usize;
//! # let n_features = 5usize;
//! # let features: Vec<f64> = vec![0.0; n_rows * n_features];
//! # let labels: Vec<f32> = vec![0.0; n_rows];
//! let cfg = Config {
//!     num_iterations: 100,
//!     learning_rate: 0.1,
//!     num_leaves: 31,
//!     ..Config::default()
//! };
//!
//! let train = DatasetBuilder::from_rows(&features, n_rows, n_features, &labels, &cfg)?;
//! let model = GbdtTrainer::new(&cfg).fit(&train, None)?;
//! let probs = model.predict_proba(&features, n_rows);
//! # Ok::<(), nanogbm::Error>(())
//! ```
//!
//! # Highlights
//!
//! - Histogram learner with sibling-by-subtraction.
//! - Missing values handled at the split (NaN bucket, per-node direction by gain).
//! - Early stopping truncates the model to the best iteration.
//! - Deterministic: same [`Config`] + same data → byte-identical model. All
//!   randomness flows through a single `ChaCha8Rng` seeded from
//!   [`Config::seed`].
//! - Bincode v2 + serde serialization of [`Model`].
//!
//! See the `examples/` directory for runnable end-to-end programs.

pub mod boosting;
pub mod config;
pub mod dataset;
pub mod error;
pub mod metric;
pub mod model;
pub mod objective;
pub mod tree;

pub use boosting::GbdtTrainer;
pub use config::Config;
pub use dataset::{Dataset, DatasetBuilder};
pub use error::{Error, Result};
pub use model::Model;
