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

    /// Append "unknown"-value synthetic rows in place (see
    /// [`crate::Config::unknown_aug_fraction`]). For each existing row, with
    /// probability `fraction`, append a copy with one randomly chosen
    /// *maskable* feature forced to [`MISSING_BIN`], carrying the same label.
    /// This trains the bin-0 fallback path that unseen category values map to.
    /// `maskable[f]` selects which columns are eligible (typically categorical
    /// open-vocabulary features; see `FeatureBuilder::augmentable_flags`).
    ///
    /// Returns the base-row index for each appended row (parallel to the new
    /// rows), so the caller can extend per-row weights consistently. No-op
    /// (returns empty) when `fraction <= 0` or no feature is maskable.
    pub(crate) fn augment_unknown(&mut self, fraction: f64, maskable: &[bool], rng: &mut impl rand::Rng) -> Vec<usize> {
        let cat_feats: Vec<usize> = (0..self.n_features).filter(|&f| maskable.get(f).copied().unwrap_or(false)).collect();
        if fraction <= 0.0 || cat_feats.is_empty() {
            return Vec::new();
        }
        let base_rows = self.n_rows;
        // Plan: (base_row, feature_to_mask) for each accepted row.
        let mut plan: Vec<(usize, usize)> = Vec::new();
        for row in 0..base_rows {
            if rng.r#gen::<f64>() < fraction {
                let m = cat_feats[rng.gen_range(0..cat_feats.len())];
                plan.push((row, m));
            }
        }
        if plan.is_empty() {
            return Vec::new();
        }
        match &mut self.bin_data {
            BinData::U8(cols) => append_masked(cols, &plan, 0u8),
            BinData::U16(cols) => append_masked(cols, &plan, 0u16),
        }
        for &(base, _) in &plan {
            self.labels.push(self.labels[base]);
        }
        self.n_rows += plan.len();
        plan.into_iter().map(|(base, _)| base).collect()
    }
}

/// Copy each planned base row across every column, then overwrite the masked
/// feature's value with `missing` (bin 0).
fn append_masked<B: Bin>(cols: &mut [Vec<B>], plan: &[(usize, usize)], missing: B) {
    for (feat, col) in cols.iter_mut().enumerate() {
        col.reserve(plan.len());
        for &(base, mask) in plan {
            let v = if feat == mask { missing } else { col[base] };
            col.push(v);
        }
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

#[cfg(test)]
mod augment_tests {
    use super::MISSING_BIN;
    use super::builder::build_dataset;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn augment_unknown_masks_one_categorical_per_synthetic_row() {
        // Column 0: categorical (3 values), column 1: numeric.
        let cols = vec![vec![10.0, 20.0, 30.0, 10.0], vec![1.0, 2.0, 3.0, 4.0]];
        let labels = vec![1.0f32, 0.0, 1.0, 0.0];
        let mut ds = build_dataset(cols, 16, 16, &[true, false], 1, &labels).unwrap();
        let base_rows = ds.n_rows();

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let base_idx = ds.augment_unknown(1.0, &[true, false], &mut rng);

        // fraction 1.0 → every base row spawns one synthetic row.
        assert_eq!(base_idx.len(), base_rows);
        assert_eq!(ds.n_rows(), base_rows * 2);

        for (i, &base) in base_idx.iter().enumerate() {
            let row = base_rows + i;
            // Label carried over from the base row.
            assert_eq!(ds.labels()[row], ds.labels()[base]);
            // Categorical column 0 is the only maskable feature, so it must be
            // MISSING (0) on every synthetic row; numeric column 1 is untouched.
            assert_eq!(ds.feature_bin(0, row), MISSING_BIN);
            assert_eq!(ds.feature_bin(1, row), ds.feature_bin(1, base));
        }
    }

    #[test]
    fn augment_unknown_noop_without_categoricals() {
        let cols = vec![vec![1.0, 2.0, 3.0]];
        let labels = vec![1.0f32, 0.0, 1.0];
        let mut ds = build_dataset(cols, 16, 16, &[false], 1, &labels).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert!(ds.augment_unknown(1.0, &[false], &mut rng).is_empty());
        assert_eq!(ds.n_rows(), 3);
    }
}
