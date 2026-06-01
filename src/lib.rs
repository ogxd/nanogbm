//! A small, pure-Rust gradient boosting library with a deliberately narrow
//! scope: **GBDT only, binary classification only, CPU only, dense numerical
//! features**. No DART/GOSS/RF, no multiclass, no ranking, no regression, no
//! sparse inputs, no GPU, no FFI bindings.
//!
//! # Quickstart
//!
//! Declare your features as named closures over your own row type, then train
//! and predict by handing the same [`FeatureBuilder`] to both sides.
//!
//! ```no_run
//! use nanogbm::{Config, FeatureBuilder, GbdtTrainer};
//!
//! struct Row { age: f64, income: f64, active: bool }
//! # let rows: Vec<Row> = Vec::new();
//! # let labels: Vec<f32> = Vec::new();
//!
//! let cfg = Config {
//!     num_iterations: 100,
//!     learning_rate: 0.1,
//!     num_leaves: 31,
//!     ..Config::default()
//! };
//!
//! let fb = FeatureBuilder::<Row>::new()
//!     .add_continuous("age", |r| r.age)
//!     .add_continuous("income", |r| r.income)
//!     .add_boolean("active", |r| r.active);
//!
//! let model = GbdtTrainer::new(&cfg, &fb).fit(&rows, &labels, None)?;
//! let probs = model.predict_proba(&fb, &rows);
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
pub mod feature;
pub mod loss;
pub mod model;
pub mod tree;

pub use boosting::GbdtTrainer;
pub use config::Config;
pub use dataset::Dataset;
pub use error::{Error, Result};
pub use feature::FeatureBuilder;
pub use model::Model;
