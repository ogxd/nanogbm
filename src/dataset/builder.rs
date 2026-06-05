use super::{Bin, BinData, BinMapper, BinWidth, Dataset};
use crate::error::{Error, Result};

/// Build a binned [`Dataset`] from column-major `f64` data.
///
/// Numeric columns are capped at `max_bin`, categorical columns at
/// `max_cat_bins` (each column gets its own [`BinMapper`]). Storage width is
/// `u8` when the widest fitted column fits in a byte (`num_bins <= 256`), else
/// `u16`. Because categoricals can hold many more bins than `max_bin`, the
/// width is decided from the *actual* fitted bin domains, not from `max_bin`.
pub(crate) fn build_dataset(
    columns: Vec<Vec<f64>>,
    max_bin: usize,
    max_cat_bins: usize,
    categorical: &[bool],
    min_data_in_bin: usize,
    labels: &[f32],
) -> Result<Dataset> {
    let n_features = columns.len();
    let n_rows = check_columns(&columns, labels)?;
    debug_assert_eq!(categorical.len(), n_features);

    // Fit a bin mapper per column first (read-only over the f64 staging), so the
    // storage width can be chosen from the widest realized bin domain.
    let bin_mappers: Vec<BinMapper> = columns
        .iter()
        .zip(categorical)
        .map(|(col, &cat)| if cat { BinMapper::fit_categorical(col, max_cat_bins) } else { BinMapper::fit(col, max_bin, min_data_in_bin) })
        .collect();

    // u8 storage iff every column's bin domain fits in a byte.
    let max_bins = bin_mappers.iter().map(|m| m.num_bins()).max().unwrap_or(0);
    let width = if max_bins <= 256 { BinWidth::U8 } else { BinWidth::U16 };

    // Encode each column, freeing its f64 storage the moment it has been binned,
    // so the full f64 matrix and the full binned matrix never coexist (the f64
    // staging is the memory peak).
    let bin_data = match width {
        BinWidth::U8 => BinData::U8(encode_and_consume::<u8>(columns, &bin_mappers)),
        BinWidth::U16 => BinData::U16(encode_and_consume::<u16>(columns, &bin_mappers)),
    };

    Ok(Dataset {
        n_rows,
        n_features,
        bin_data,
        bin_mappers,
        labels: labels.to_vec(),
    })
}

/// Bin-encode each column with its fitted [`BinMapper`], dropping each f64
/// column as soon as it is encoded. `BinMapper::value_to_bin` always returns
/// u16, which we narrow via [`Bin::from_u16`].
fn encode_and_consume<B: Bin>(columns: Vec<Vec<f64>>, mappers: &[BinMapper]) -> Vec<Vec<B>> {
    let mut out = Vec::with_capacity(columns.len());
    for (col, bm) in columns.into_iter().zip(mappers) {
        out.push(col.iter().map(|&v| B::from_u16(bm.value_to_bin(v))).collect());
        // `col` is dropped here, freeing its f64 storage before the next column.
    }
    out
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
