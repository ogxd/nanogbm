pub mod bin_mapper;
pub mod builder;

pub use bin_mapper::BinMapper;
pub use builder::DatasetBuilder;

use serde::{Deserialize, Serialize};

/// Bin code reserved for missing (NaN) values.
pub const MISSING_BIN: u16 = 0;

/// Bin-encoded training dataset, column-major.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub(crate) n_rows: usize,
    pub(crate) n_features: usize,
    /// Column-major bin codes: `bin_data[feat][row]`.
    pub(crate) bin_data: Vec<Vec<u16>>,
    pub(crate) bin_mappers: Vec<BinMapper>,
    pub(crate) labels: Vec<f32>,
}

impl Dataset {
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn labels(&self) -> &[f32] {
        &self.labels
    }

    pub fn feature_column(&self, feat: usize) -> &[u16] {
        &self.bin_data[feat]
    }

    pub fn bin_mapper(&self, feat: usize) -> &BinMapper {
        &self.bin_mappers[feat]
    }

    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }
}
