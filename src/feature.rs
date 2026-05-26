//! Feature encoding helpers.
//!
//! Users write a single `encode_into` function that pushes columns into a
//! [`FeatureSink`]. Two sinks are provided: [`DiscoverySink`] runs once to
//! derive the [`Schema`] (column names and kinds); [`SliceSink`] writes values
//! into a `&mut [f64]` on the hot path.

use gxhash::GxHasher;
use std::hash::{Hash, Hasher};

const HASH_SEED: i64 = 0x6e616e6f67626d; // "nanogbm"

#[inline]
fn hash_to_bucket<H: Hash>(value: &H, buckets: u32) -> u32 {
    let mut h = GxHasher::with_seed(HASH_SEED);
    value.hash(&mut h);
    let b = buckets.max(1);
    (h.finish() as u32) % b
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Numeric,
    Categorical,
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: &'static str,
    pub kind: ColumnKind,
}

#[derive(Debug, Clone, Default)]
pub struct Schema {
    columns: Vec<Column>,
}

impl Schema {
    pub fn len(&self) -> usize {
        self.columns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.columns.iter().map(|c| c.name)
    }
    pub fn categorical_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.columns
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (c.kind == ColumnKind::Categorical).then_some(i))
    }

    /// Format a per-feature importance report as a sortable table.
    /// `splits` and `gains` are usually `model.feature_importance_split()` and
    /// `model.feature_importance_gain()`. Rows are sorted by gain descending.
    pub fn format_importance(&self, splits: &[u32], gains: &[f64]) -> String {
        let total_gain: f64 = gains.iter().sum();
        let total_splits: u32 = splits.iter().sum();
        let name_w = self
            .columns
            .iter()
            .map(|c| c.name.len())
            .max()
            .unwrap_or(4)
            .max(7);
        let mut rows: Vec<(usize, &Column, u32, f64)> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    i,
                    c,
                    *splits.get(i).unwrap_or(&0),
                    *gains.get(i).unwrap_or(&0.0),
                )
            })
            .collect();
        rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = String::new();
        out.push_str(&format!(
            "{:<idx$}  {:<name_w$}  {:<5}  {:>7}  {:>9}  {:>6}  {:>6}\n",
            "idx",
            "feature",
            "kind",
            "splits",
            "gain",
            "split%",
            "gain%",
            idx = 4,
            name_w = name_w
        ));
        for (i, c, sp, g) in &rows {
            let kind = match c.kind {
                ColumnKind::Numeric => "num",
                ColumnKind::Categorical => "cat",
            };
            let sp_pct = if total_splits > 0 {
                100.0 * *sp as f64 / total_splits as f64
            } else {
                0.0
            };
            let g_pct = if total_gain > 0.0 {
                100.0 * *g / total_gain
            } else {
                0.0
            };
            out.push_str(&format!(
                "{:<idx$}  {:<name_w$}  {:<5}  {:>7}  {:>9.3}  {:>5.1}%  {:>5.1}%\n",
                i,
                c.name,
                kind,
                sp,
                g,
                sp_pct,
                g_pct,
                idx = 4,
                name_w = name_w
            ));
        }
        out
    }
}

/// Sink that the user's `encode_into` writes into. The hot-path impl is
/// [`SliceSink`]; [`DiscoverySink`] runs once at startup to build the schema.
pub trait FeatureSink {
    fn num(&mut self, name: &'static str, v: f64);
    fn bool(&mut self, name: &'static str, v: bool);
    fn cat(&mut self, name: &'static str, v: i32);
    fn cat_hashed<H: Hash>(&mut self, name: &'static str, buckets: u32, v: &H);
    fn multi_hot<I: IntoIterator<Item = i32>>(
        &mut self,
        name: &'static str,
        min: i32,
        max: i32,
        values: I,
    );
}

#[derive(Debug, Default)]
pub struct DiscoverySink {
    schema: Schema,
}

impl DiscoverySink {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn into_schema(self) -> Schema {
        self.schema
    }
    fn push(&mut self, name: &'static str, kind: ColumnKind) {
        self.schema.columns.push(Column { name, kind });
    }
}

impl FeatureSink for DiscoverySink {
    fn num(&mut self, name: &'static str, _v: f64) {
        self.push(name, ColumnKind::Numeric);
    }
    fn bool(&mut self, name: &'static str, _v: bool) {
        self.push(name, ColumnKind::Numeric);
    }
    fn cat(&mut self, name: &'static str, _v: i32) {
        self.push(name, ColumnKind::Categorical);
    }
    fn cat_hashed<H: Hash>(&mut self, name: &'static str, _buckets: u32, _v: &H) {
        self.push(name, ColumnKind::Categorical);
    }
    fn multi_hot<I: IntoIterator<Item = i32>>(
        &mut self,
        name: &'static str,
        min: i32,
        max: i32,
        _values: I,
    ) {
        let n = (max - min + 1).max(1) as usize;
        for _ in 0..n {
            self.push(name, ColumnKind::Numeric);
        }
    }
}

pub struct SliceSink<'a> {
    out: &'a mut [f64],
    i: usize,
}

impl<'a> SliceSink<'a> {
    pub fn new(out: &'a mut [f64]) -> Self {
        Self { out, i: 0 }
    }
    pub fn position(&self) -> usize {
        self.i
    }

    #[inline]
    fn write(&mut self, v: f64) {
        self.out[self.i] = v;
        self.i += 1;
    }
}

impl<'a> FeatureSink for SliceSink<'a> {
    #[inline]
    fn num(&mut self, _name: &'static str, v: f64) {
        self.write(v);
    }
    #[inline]
    fn bool(&mut self, _name: &'static str, v: bool) {
        self.write(if v { 1.0 } else { 0.0 });
    }
    #[inline]
    fn cat(&mut self, _name: &'static str, v: i32) {
        self.write(v as f64);
    }
    #[inline]
    fn cat_hashed<H: Hash>(&mut self, _name: &'static str, buckets: u32, v: &H) {
        self.write(hash_to_bucket(v, buckets) as f64);
    }
    #[inline]
    fn multi_hot<I: IntoIterator<Item = i32>>(
        &mut self,
        _name: &'static str,
        min: i32,
        max: i32,
        values: I,
    ) {
        let n = (max - min + 1).max(1) as usize;
        let start = self.i;
        for k in 0..n {
            self.out[start + k] = 0.0;
        }
        for v in values {
            if v < min || v > max {
                continue;
            }
            self.out[start + (v - min) as usize] = 1.0;
        }
        self.i += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc<S: FeatureSink>(s: &mut S) {
        s.num("a", 1.5);
        s.bool("b", true);
        s.cat("c", 7);
        s.cat_hashed("d", 1024, &"hello");
        s.multi_hot("e", 0, 2, [0, 2]);
    }

    #[test]
    fn discover_records_schema() {
        let mut d = DiscoverySink::new();
        enc(&mut d);
        let schema = d.into_schema();
        assert_eq!(schema.len(), 7); // 1+1+1+1+3
        let kinds: Vec<_> = schema.columns().iter().map(|c| c.kind).collect();
        assert_eq!(
            kinds,
            vec![
                ColumnKind::Numeric,
                ColumnKind::Numeric,
                ColumnKind::Categorical,
                ColumnKind::Categorical,
                ColumnKind::Numeric,
                ColumnKind::Numeric,
                ColumnKind::Numeric,
            ]
        );
        let cats: Vec<_> = schema.categorical_indices().collect();
        assert_eq!(cats, vec![2, 3]);
    }

    #[test]
    fn slice_writes_expected_values() {
        let mut d = DiscoverySink::new();
        enc(&mut d);
        let n = d.into_schema().len();
        let mut out = vec![0.0; n];
        let mut s = SliceSink::new(&mut out);
        enc(&mut s);
        assert_eq!(s.position(), n);
        assert_eq!(out[0], 1.5);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 7.0);
        assert!(out[3] >= 0.0 && out[3] < 1024.0);
        // multi-hot slots: [0]=1, [1]=0, [2]=1
        assert_eq!(out[4], 1.0);
        assert_eq!(out[5], 0.0);
        assert_eq!(out[6], 1.0);
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash_to_bucket(&"x", 1024), hash_to_bucket(&"x", 1024));
        assert_eq!(
            hash_to_bucket(&(7u32, 12i32), 1024),
            hash_to_bucket(&(7u32, 12i32), 1024)
        );
    }
}
