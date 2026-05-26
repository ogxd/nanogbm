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
///
/// **Validation / test sets must reuse the training bin mappers.** Refitting
/// quantile boundaries on a different sample produces bins that don't line up
/// with the trained tree's `threshold_bin`, which biases validation scores and
/// breaks early stopping. Use [`DatasetBuilder::from_rows_with_mappers`] or
/// [`DatasetBuilder::from_columns_with_mappers`], passing
/// `train.bin_mappers()` (or `model.bin_mappers()`), for every dataset that
/// isn't your initial training set.
pub struct DatasetBuilder;

impl DatasetBuilder {
    /// Build a training dataset from row-major data, fitting new bin mappers.
    /// `features` has length `n_rows * schema.len()`, `labels` has length `n_rows`.
    pub fn from_rows(features: &[f64], n_rows: usize, schema: &Schema, labels: &[f32], config: &Config) -> Result<Dataset> {
        let n_features = schema.len();
        let columns = rows_to_columns(features, n_rows, n_features)?;
        Self::from_columns(columns, schema, labels, config)
    }

    /// Build a training dataset from column-major data, fitting new bin mappers.
    pub fn from_columns(columns: Vec<Vec<f64>>, schema: &Schema, labels: &[f32], config: &Config) -> Result<Dataset> {
        let n_features = columns.len();
        if schema.len() != n_features {
            return Err(Error::Shape(format!("schema has {} columns but data has {}", schema.len(), n_features)));
        }
        let n_rows = check_columns(&columns, labels)?;

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

    /// Build a validation/test dataset from row-major data, **reusing
    /// pre-fitted bin mappers** (typically `train.bin_mappers()` or
    /// `model.bin_mappers()`). This is the correct way to bin any dataset
    /// after the first training set — without it, val bins don't line up with
    /// the trees' learned `threshold_bin`s.
    pub fn from_rows_with_mappers(features: &[f64], n_rows: usize, mappers: &[BinMapper], labels: &[f32]) -> Result<Dataset> {
        let n_features = mappers.len();
        let columns = rows_to_columns(features, n_rows, n_features)?;
        Self::from_columns_with_mappers(columns, mappers, labels)
    }

    /// Column-major counterpart of [`DatasetBuilder::from_rows_with_mappers`].
    pub fn from_columns_with_mappers(columns: Vec<Vec<f64>>, mappers: &[BinMapper], labels: &[f32]) -> Result<Dataset> {
        let n_features = columns.len();
        if mappers.len() != n_features {
            return Err(Error::Shape(format!("mappers has {} entries but data has {} columns", mappers.len(), n_features)));
        }
        let n_rows = check_columns(&columns, labels)?;

        let bin_data: Vec<Vec<u16>> = columns
            .iter()
            .zip(mappers.iter())
            .map(|(col, bm)| col.iter().map(|&v| bm.value_to_bin(v)).collect())
            .collect();

        Ok(Dataset {
            n_rows,
            n_features,
            bin_data,
            bin_mappers: mappers.to_vec(),
            labels: labels.to_vec(),
        })
    }
}

fn rows_to_columns(features: &[f64], n_rows: usize, n_features: usize) -> Result<Vec<Vec<f64>>> {
    if n_features == 0 {
        return Err(Error::Shape("no features".into()));
    }
    if features.len() != n_rows * n_features {
        return Err(Error::Shape(format!("features len {} != n_rows {} * n_features {}", features.len(), n_rows, n_features)));
    }
    Ok((0..n_features)
        .map(|feat| {
            let mut col = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                col.push(features[row * n_features + feat]);
            }
            col
        })
        .collect())
}

fn check_columns(columns: &[Vec<f64>], labels: &[f32]) -> Result<usize> {
    if columns.is_empty() {
        return Err(Error::Shape("no features".into()));
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
    Ok(n_rows)
}
