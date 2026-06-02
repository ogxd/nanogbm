use super::{Bin, BinData, BinMapper, BinWidth, Dataset};
use crate::error::{Error, Result};

/// Build a binned [`Dataset`] from column-major `f64` data.
///
/// `max_bin` is the bin cap applied to every column; each column gets its own
/// [`BinMapper`] fit to that cap. Storage width is `u8` when the cap fits in a
/// byte (`max_bin <= 256`), else `u16`.
pub(crate) fn build_dataset(
    columns: Vec<Vec<f64>>,
    max_bin: usize,
    categorical: &[bool],
    min_data_in_bin: usize,
    labels: &[f32],
) -> Result<Dataset> {
    let n_features = columns.len();
    let n_rows = check_columns(&columns, labels)?;
    debug_assert_eq!(categorical.len(), n_features);

    // u8 storage iff the bin domain fits in a byte.
    let width = if max_bin <= 256 {
        BinWidth::U8
    } else {
        BinWidth::U16
    };

    // Fit and bin-encode each column, freeing its f64 storage the moment it has
    // been binned, so the full f64 matrix and the full binned matrix never
    // coexist (the f64 staging is the memory peak).
    let mut bin_mappers: Vec<BinMapper> = Vec::with_capacity(n_features);
    let bin_data = match width {
        BinWidth::U8 => BinData::U8(bin_and_consume::<u8>(columns, categorical, max_bin, min_data_in_bin, &mut bin_mappers)),
        BinWidth::U16 => BinData::U16(bin_and_consume::<u16>(columns, categorical, max_bin, min_data_in_bin, &mut bin_mappers)),
    };

    Ok(Dataset {
        n_rows,
        n_features,
        bin_data,
        bin_mappers,
        labels: labels.to_vec(),
    })
}

/// Fit a [`BinMapper`] per column and bin-encode it, dropping each f64 column as
/// soon as it is encoded. `mappers` is filled in column order alongside the
/// returned binned columns. `BinMapper::value_to_bin` always returns u16, which
/// we narrow via [`Bin::from_u16`].
fn bin_and_consume<B: Bin>(columns: Vec<Vec<f64>>, categorical: &[bool], max_bin: usize, min_data_in_bin: usize, mappers: &mut Vec<BinMapper>) -> Vec<Vec<B>> {
    let mut out = Vec::with_capacity(columns.len());
    for (col, &cat) in columns.into_iter().zip(categorical) {
        let bm = if cat { BinMapper::fit_categorical(&col, max_bin) } else { BinMapper::fit(&col, max_bin, min_data_in_bin) };
        out.push(col.iter().map(|&v| B::from_u16(bm.value_to_bin(v))).collect());
        mappers.push(bm);
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
