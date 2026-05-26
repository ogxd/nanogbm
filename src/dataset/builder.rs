use super::{Bin, BinData, BinMapper, BinWidth, Dataset};
use crate::config::Config;
use crate::error::{Error, Result};

/// Builds a `Dataset` from raw row-major or column-major f64 features.
pub struct DatasetBuilder;

impl DatasetBuilder {
    /// Build from row-major data: `features` has length `n_rows * n_features`,
    /// `labels` has length `n_rows`.
    pub fn from_rows(
        features: &[f64],
        n_rows: usize,
        n_features: usize,
        labels: &[f32],
        config: &Config,
    ) -> Result<Dataset> {
        let columns = rows_to_columns(features, n_rows, n_features)?;
        Self::from_columns(columns, labels, config)
    }

    /// Build from already column-major data.
    pub fn from_columns(
        columns: Vec<Vec<f64>>,
        labels: &[f32],
        config: &Config,
    ) -> Result<Dataset> {
        let n_features = columns.len();
        let n_rows = check_columns(&columns, labels)?;

        // Storage width: u8 if every column fits in a byte. `config.max_bin`
        // is the upper bound on `num_bins` per column (BinMapper enforces it),
        // so we can decide globally without inspecting fitted mappers.
        let width = if config.max_bin <= 256 {
            BinWidth::U8
        } else {
            BinWidth::U16
        };

        let bin_mappers: Vec<BinMapper> = columns
            .iter()
            .map(|col| BinMapper::fit(col, config.max_bin, config.min_data_in_bin))
            .collect();

        let bin_data = match width {
            BinWidth::U8 => BinData::U8(encode_columns::<u8>(&columns, &bin_mappers)),
            BinWidth::U16 => BinData::U16(encode_columns::<u16>(&columns, &bin_mappers)),
        };

        Ok(Dataset {
            n_rows,
            n_features,
            bin_data,
            bin_mappers,
            labels: labels.to_vec(),
        })
    }
}

/// Encode each f64 column into bins of type `B` using its fitted [`BinMapper`].
/// `BinMapper::value_to_bin` always returns u16 — we narrow via [`Bin::from_u16`].
fn encode_columns<B: Bin>(columns: &[Vec<f64>], mappers: &[BinMapper]) -> Vec<Vec<B>> {
    columns
        .iter()
        .zip(mappers.iter())
        .map(|(col, bm)| col.iter().map(|&v| B::from_u16(bm.value_to_bin(v))).collect())
        .collect()
}

fn rows_to_columns(features: &[f64], n_rows: usize, n_features: usize) -> Result<Vec<Vec<f64>>> {
    if n_features == 0 {
        return Err(Error::Shape("no features".into()));
    }
    if features.len() != n_rows * n_features {
        return Err(Error::Shape(format!(
            "features len {} != n_rows {} * n_features {}",
            features.len(),
            n_rows,
            n_features
        )));
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
        return Err(Error::Shape(format!(
            "labels len {} != n_rows {}",
            labels.len(),
            n_rows
        )));
    }
    for (i, c) in columns.iter().enumerate() {
        if c.len() != n_rows {
            return Err(Error::Shape(format!(
                "column {} has len {}, expected {}",
                i,
                c.len(),
                n_rows
            )));
        }
    }
    Ok(n_rows)
}
