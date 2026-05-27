pub mod bin_mapper;
pub mod builder;

pub use bin_mapper::BinMapper;
pub use builder::DatasetBuilder;

use serde::{Deserialize, Serialize};

/// Bin code reserved for missing (NaN) values. Same numeric value regardless
/// of the storage width — `0u8` or `0u16` both represent missing.
pub const MISSING_BIN: u16 = 0;

/// Per-element type for bin-encoded column data. Hot loops are generic over
/// this so a single inner loop can serve both u8 and u16 columns.
///
/// Use `B::MISSING` instead of comparing against [`MISSING_BIN`] directly; that
/// way the compiler can compare u8-to-u8 (one cycle) instead of widening
/// every element to u16 to match a u16 constant.
pub trait Bin: Copy + PartialEq + PartialOrd + Send + Sync + 'static {
    const MISSING: Self;
    fn as_usize(self) -> usize;
    /// Narrow a u16 bin code (as produced by [`BinMapper::value_to_bin`]) into
    /// this type. For [`u8`], values above 255 are truncated — only call on
    /// columns whose `num_bins <= 256`.
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

/// Storage width for binned column data, chosen once per [`Dataset`] at build
/// time based on `config.max_bin`. `U8` is used when every column's bin count
/// fits in a byte (`max_bin <= 256`) — that halves column-read bandwidth in
/// the histogram-build hot loop, which is the dominant cost on large
/// workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinWidth {
    U8,
    U16,
}

/// Column-major bin-encoded data, in either u8 or u16 storage.
///
/// All columns of a given dataset use the same width: the type is chosen
/// globally so that the hot-loop dispatch happens once per call instead of
/// per (row, feature). For workloads where most columns would fit in u8 but
/// one is u16, that one column forces all to u16 — accept that as the cost
/// of a single inner-loop body.
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

    /// Read a single bin code out of a column, widening to u16 if stored as
    /// u8. Convenient for one-off lookups (the [`crate::tree::Tree`] predict
    /// path). Hot loops should dispatch on `bin_width()` and call the
    /// type-stable accessor instead.
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
