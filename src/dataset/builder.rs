use super::{BinMapper, Dataset};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::feature::Schema;

/// Builds a `Dataset` from raw row-major or column-major f64 features.
///
/// The categorical/numerical nature of each column is taken from `schema` (the
/// same schema your `FeatureSink`-based encoder produced). Use
/// [`Schema::all_numerical`] when calling with raw test data that has no
/// FeatureSink schema.
pub struct DatasetBuilder;

impl DatasetBuilder {
    /// Build from row-major data: `features` has length `n_rows * schema.len()`,
    /// `labels` has length `n_rows`.
    pub fn from_rows(features: &[f64], n_rows: usize, schema: &Schema, labels: &[f32], config: &Config) -> Result<Dataset> {
        let n_features = schema.len();
        if features.len() != n_rows * n_features {
            return Err(Error::Shape(format!("features len {} != n_rows {} * n_features {}", features.len(), n_rows, n_features)));
        }
        if labels.len() != n_rows {
            return Err(Error::Shape(format!("labels len {} != n_rows {}", labels.len(), n_rows)));
        }

        let columns: Vec<Vec<f64>> = (0..n_features)
            .map(|feat| {
                let mut col = Vec::with_capacity(n_rows);
                for row in 0..n_rows {
                    col.push(features[row * n_features + feat]);
                }
                col
            })
            .collect();

        Self::from_columns(columns, schema, labels, config)
    }

    /// Build from already column-major data.
    pub fn from_columns(columns: Vec<Vec<f64>>, schema: &Schema, labels: &[f32], config: &Config) -> Result<Dataset> {
        let n_features = columns.len();
        if n_features == 0 {
            return Err(Error::Shape("no features".into()));
        }
        if schema.len() != n_features {
            return Err(Error::Shape(format!("schema has {} columns but data has {}", schema.len(), n_features)));
        }
        let n_rows = columns[0].len();
        if labels.len() != n_rows {
            return Err(Error::Shape(format!("labels len {} != n_rows {}", labels.len(), n_rows)));
        }
        for (i, c) in columns.iter().enumerate() {
            if c.len() != n_rows {
                return Err(Error::Shape(format!("column {} has len {}, expected {}", i, c.len(), n_rows)));
            }
        }

        let (bin_mappers, bin_data): (Vec<BinMapper>, Vec<Vec<u16>>) = columns
            .iter()
            .enumerate()
            .map(|(feat, col)| {
                let bm = if schema.is_categorical(feat) {
                    BinMapper::fit_categorical(col, config.max_cat_bin)
                } else {
                    BinMapper::fit_numerical(col, config.max_bin, config.min_data_in_bin)
                };
                let bins: Vec<u16> = col.iter().map(|&v| bm.value_to_bin(v)).collect();
                (bm, bins)
            })
            .unzip();

        Ok(Dataset {
            n_rows,
            n_features,
            bin_data,
            bin_mappers,
            labels: labels.to_vec(),
        })
    }
}
