//! nanogbm: a pure-Rust gradient boosting library (GBDT, binary classification, CPU only).

pub mod boosting;
pub mod config;
pub mod dataset;
pub mod error;
pub mod feature;
pub mod metric;
pub mod model;
pub mod objective;
pub mod predict;
pub mod tree;

pub use boosting::GbdtTrainer;
pub use config::Config;
pub use dataset::{Dataset, DatasetBuilder};
pub use error::{Error, Result};
pub use model::Model;
