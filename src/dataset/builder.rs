use super::{Bin, BinData, BinMapper, BinWidth, Dataset};
use crate::error::{Error, Result};

/// Build a binned [`Dataset`] from column-major `f64` data.
///
/// `max_bins[i]` is the per-feature bin cap (already resolved against the
/// config default by the caller); each column gets its own [`BinMapper`] fit
/// to that cap. Storage width is `u8` when every column fits in a byte
/// (`max_bin <= 256`), else `u16`.
pub(crate) fn build_dataset(
    columns: Vec<Vec<f64>>,
    max_bins: &[usize],
    min_data_in_bin: usize,
    labels: &[f32],
) -> Result<Dataset> {
    let n_features = columns.len();
    let n_rows = check_columns(&columns, labels)?;
    debug_assert_eq!(max_bins.len(), n_features);

    // u8 storage iff every column's bin domain fits in a byte.
    let width = if max_bins.iter().all(|&b| b <= 256) {
        BinWidth::U8
    } else {
        BinWidth::U16
    };

    let bin_mappers: Vec<BinMapper> = columns
        .iter()
        .zip(max_bins)
        .map(|(col, &mb)| BinMapper::fit(col, mb, min_data_in_bin))
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

/// Encode each f64 column into bins of type `B` using its fitted [`BinMapper`].
/// `BinMapper::value_to_bin` always returns u16 — we narrow via [`Bin::from_u16`].
fn encode_columns<B: Bin>(columns: &[Vec<f64>], mappers: &[BinMapper]) -> Vec<Vec<B>> {
    columns
        .iter()
        .zip(mappers.iter())
        .map(|(col, bm)| col.iter().map(|&v| B::from_u16(bm.value_to_bin(v))).collect())
        .collect()
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
