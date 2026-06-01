pub mod bin_mapper;
pub(crate) mod builder;

pub use bin_mapper::BinMapper;

use serde::{Deserialize, Serialize};

/// Bin code reserved for missing (NaN) values. `0u8` or `0u16` both represent
/// missing.
pub const MISSING_BIN: u16 = 0;

/// Per-element type for bin-encoded column data. Hot loops are generic over
/// this so a single inner loop serves u8 and u16 columns. Prefer `B::MISSING`
/// over [`MISSING_BIN`] so the compiler can compare in the column's native
/// width (one cycle) instead of widening every element to u16.
pub trait Bin: Copy + PartialEq + PartialOrd + Send + Sync + 'static {
    const MISSING: Self;
    fn as_usize(self) -> usize;
    /// Narrow a u16 bin code to this type. For `u8`, values above 255 are
    /// truncated — only call on columns whose `num_bins <= 256`.
    fn from_u16(v: u16) -> Self;
}

impl Bin for u8 {
    const MISSING: Self = 0;
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v as u8
    }
}

impl Bin for u16 {
    const MISSING: Self = 0;
    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_u16(v: u16) -> Self {
        v
    }
}

/// Storage width for binned column data. `U8` (chosen when `max_bin <= 256`)
/// halves column-read bandwidth in the histogram hot loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinWidth {
    U8,
    U16,
}

/// Column-major bin-encoded data. All columns share one width so the
/// dispatch happens once per call, not per (row, feature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinData {
    U8(Vec<Vec<u8>>),
    U16(Vec<Vec<u16>>),
}

impl BinData {
    pub fn width(&self) -> BinWidth {
        match self {
            BinData::U8(_) => BinWidth::U8,
            BinData::U16(_) => BinWidth::U16,
        }
    }
    pub fn n_features(&self) -> usize {
        match self {
            BinData::U8(v) => v.len(),
            BinData::U16(v) => v.len(),
        }
    }
}

/// Bin-encoded training dataset, column-major.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub(crate) n_rows: usize,
    pub(crate) n_features: usize,
    /// Column-major bin codes. See [`BinData`] for the width choice.
    pub(crate) bin_data: BinData,
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

    pub fn bin_width(&self) -> BinWidth {
        self.bin_data.width()
    }

    /// Get a u8 column. Panics if `bin_width() != U8`.
    pub fn feature_column_u8(&self, feat: usize) -> &[u8] {
        match &self.bin_data {
            BinData::U8(v) => &v[feat],
            BinData::U16(_) => panic!("feature_column_u8 called on a U16 dataset"),
        }
    }

    /// Get a u16 column. Panics if `bin_width() != U16`.
    pub fn feature_column_u16(&self, feat: usize) -> &[u16] {
        match &self.bin_data {
            BinData::U16(v) => &v[feat],
            BinData::U8(_) => panic!("feature_column_u16 called on a U8 dataset"),
        }
    }

    /// Read a single bin code, widening u8 → u16. Hot loops should dispatch
    /// on `bin_width()` and call the type-stable accessor instead.
    #[inline]
    pub fn feature_bin(&self, feat: usize, row: usize) -> u16 {
        match &self.bin_data {
            BinData::U8(v) => v[feat][row] as u16,
            BinData::U16(v) => v[feat][row],
        }
    }

    pub fn bin_mapper(&self, feat: usize) -> &BinMapper {
        &self.bin_mappers[feat]
    }

    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }
}

/// Run `$body` once per dataset bin width with `$cols` bound to a
/// `Vec<&[u8]>` or `Vec<&[u16]>` containing the requested feature columns.
///
/// Centralizes the u8/u16 match so callers don't duplicate the dispatch
/// arms. The body is monomorphized per width by the compiler.
macro_rules! with_columns {
    ($ds:expr, $feats:expr, |$cols:ident| $body:block) => {
        match $ds.bin_width() {
            $crate::dataset::BinWidth::U8 => {
                let $cols: Vec<&[u8]> =
                    $feats.iter().map(|&f| $ds.feature_column_u8(f)).collect();
                $body
            }
            $crate::dataset::BinWidth::U16 => {
                let $cols: Vec<&[u16]> =
                    $feats.iter().map(|&f| $ds.feature_column_u16(f)).collect();
                $body
            }
        }
    };
}

/// Single-column counterpart to [`with_columns!`].
macro_rules! with_column {
    ($ds:expr, $feat:expr, |$col:ident| $body:block) => {
        match $ds.bin_width() {
            $crate::dataset::BinWidth::U8 => {
                let $col: &[u8] = $ds.feature_column_u8($feat);
                $body
            }
            $crate::dataset::BinWidth::U16 => {
                let $col: &[u16] = $ds.feature_column_u16($feat);
                $body
            }
        }
    };
}

pub(crate) use with_column;
pub(crate) use with_columns;
