use crate::dataset::Bin;

/// One histogram bin: gradient sum, hessian sum, row count.
/// 24 bytes (8 + 8 + 4 + 4 padding) — sized to one machine word triple so
/// a single cache line covers 2-3 bins on x86-64 / Apple Silicon.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct HistBin {
    pub grad: f64,
    pub hess: f64,
    pub count: u32,
    _pad: u32,
}

/// Per-feature gradient/hessian histogram bins (AoS layout).
///
/// AoS (one `HistBin` per bin) was chosen over SoA (three separate Vecs)
/// because the histogram-build hot loop touches all three fields of the same
/// bin together; AoS makes that one cache-line access instead of three.
#[derive(Debug, Clone)]
pub struct FeatureHistogram {
    /// `num_bins` entries (including bin 0 = missing).
    pub bins: Vec<HistBin>,
}

impl FeatureHistogram {
    pub fn zeros(num_bins: usize) -> Self {
        Self {
            bins: vec![HistBin::default(); num_bins],
        }
    }

    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }

    pub fn clear(&mut self) {
        for b in self.bins.iter_mut() {
            *b = HistBin::default();
        }
    }

    /// Accumulate gradient/hessian for `indices` rows on a single feature column.
    /// `gradhess[row] = [grad, hess]` — packed so one gather pulls both values.
    ///
    /// Generic over the column element type `B: Bin` so u8 and u16 columns
    /// share one inner loop. Inner loop uses `get_unchecked` to elide bounds
    /// checks: bins are bounded by `num_bins` (BinMapper invariant), rows are
    /// bounded by column.len() (DatasetBuilder invariant); both are upheld by
    /// nanogbm's own pipeline.
    pub fn build<B: Bin>(&mut self, column: &[B], indices: &[u32], gradhess: &[[f32; 2]]) {
        self.clear();
        let bins = self.bins.as_mut_ptr();
        unsafe {
            for &i in indices {
                let row = i as usize;
                let bin = (*column.get_unchecked(row)).as_usize();
                let gh = *gradhess.get_unchecked(row);
                let b = bins.add(bin);
                (*b).grad += gh[0] as f64;
                (*b).hess += gh[1] as f64;
                (*b).count += 1;
            }
        }
    }

    /// Root-level fast path: accumulate over ALL rows in column order (no
    /// `indices` indirection). Sequential reads on both column and gradhess.
    pub fn build_full<B: Bin>(&mut self, column: &[B], gradhess: &[[f32; 2]]) {
        self.clear();
        debug_assert_eq!(column.len(), gradhess.len());
        let n = column.len();
        let bins = self.bins.as_mut_ptr();
        unsafe {
            for row in 0..n {
                let bin = (*column.get_unchecked(row)).as_usize();
                let gh = *gradhess.get_unchecked(row);
                let b = bins.add(bin);
                (*b).grad += gh[0] as f64;
                (*b).hess += gh[1] as f64;
                (*b).count += 1;
            }
        }
    }

    /// Compute `self = parent - sibling` (parent-minus-sibling histogram trick).
    ///
    /// The loop body operates on independent bins with no carried dependency
    /// across iterations, so LLVM autovectorizes the f64 grad/hess subtraction
    /// (NEON f64x2 / AVX f64x4). The `count` field is a separate u32 sub.
    pub fn subtract_into(
        parent: &FeatureHistogram,
        sibling: &FeatureHistogram,
        out: &mut FeatureHistogram,
    ) {
        let n = parent.num_bins();
        debug_assert_eq!(n, sibling.num_bins());
        debug_assert_eq!(n, out.num_bins());
        let p = parent.bins.as_ptr();
        let s = sibling.bins.as_ptr();
        let o = out.bins.as_mut_ptr();
        // SAFETY: all three slices have length `n` (checked above).
        unsafe {
            for i in 0..n {
                let pi = &*p.add(i);
                let si = &*s.add(i);
                *o.add(i) = HistBin {
                    grad: pi.grad - si.grad,
                    hess: pi.hess - si.hess,
                    count: pi.count - si.count,
                    _pad: 0,
                };
            }
        }
    }
}

/// Row-major batched histogram build: walks `indices` once and updates all
/// `histograms` in lockstep. This reads `gradhess[row]` ONE TIME per row
/// (instead of `n_features` times in the naive per-feature loop), cutting
/// gather traffic on the largest array dramatically — `gradhess` is typically
/// far bigger than L2/L3, while the histograms themselves (~12 KB per feature)
/// fit in L1.
///
/// `columns.len()` must equal `histograms.len()`; bins are addressed by
/// `column[row] as usize` and must be in range of each histogram's `num_bins`.
pub fn build_histograms_batched<B: Bin>(
    columns: &[&[B]],
    indices: &[u32],
    gradhess: &[[f32; 2]],
    histograms: &mut [FeatureHistogram],
) {
    debug_assert_eq!(columns.len(), histograms.len());
    for h in histograms.iter_mut() {
        h.clear();
    }
    let n_feat = columns.len();
    // Pre-extract raw pointers so the inner loop doesn't reborrow on every iter.
    let bin_ptrs: Vec<*mut HistBin> = histograms.iter_mut().map(|h| h.bins.as_mut_ptr()).collect();
    let col_ptrs: Vec<*const B> = columns.iter().map(|c| c.as_ptr()).collect();
    // SAFETY: every column has length >= max(indices), every histogram has
    // capacity matching its column's bin domain (caller upholds via DatasetBuilder).
    unsafe {
        for &i in indices {
            let row = i as usize;
            let gh = *gradhess.get_unchecked(row);
            let g = gh[0] as f64;
            let h = gh[1] as f64;
            for fi in 0..n_feat {
                let bin = (*col_ptrs.get_unchecked(fi).add(row)).as_usize();
                let b = (*bin_ptrs.get_unchecked(fi)).add(bin);
                (*b).grad += g;
                (*b).hess += h;
                (*b).count += 1;
            }
        }
    }
}

/// Row-major batched build for the root level (sequential row iteration, no
/// `indices` indirection).
pub fn build_histograms_batched_full<B: Bin>(
    columns: &[&[B]],
    gradhess: &[[f32; 2]],
    histograms: &mut [FeatureHistogram],
) {
    debug_assert_eq!(columns.len(), histograms.len());
    for h in histograms.iter_mut() {
        h.clear();
    }
    let n = gradhess.len();
    let n_feat = columns.len();
    let bin_ptrs: Vec<*mut HistBin> = histograms.iter_mut().map(|h| h.bins.as_mut_ptr()).collect();
    let col_ptrs: Vec<*const B> = columns.iter().map(|c| c.as_ptr()).collect();
    unsafe {
        for row in 0..n {
            let gh = *gradhess.get_unchecked(row);
            let g = gh[0] as f64;
            let h = gh[1] as f64;
            for fi in 0..n_feat {
                let bin = (*col_ptrs.get_unchecked(fi).add(row)).as_usize();
                let b = (*bin_ptrs.get_unchecked(fi)).add(bin);
                (*b).grad += g;
                (*b).hess += h;
                (*b).count += 1;
            }
        }
    }
}
