//! Binary logistic loss: sigmoid, init score, packed gradients/hessians, and
//! the binary logloss evaluation metric. One file because the project is
//! scope-locked to binary classification.

const LOGLOSS_EPS: f64 = 1e-15;

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Branchless f32 sigmoid using a polynomial `exp2` approximation. The
/// straight-line math lets LLVM autovectorize the caller's loop (NEON f32x4 /
/// AVX f32x8). < 1e-4 abs error over `x ∈ [-30, 30]`; saturates beyond.
#[inline]
pub fn sigmoid_fast_f32(x: f32) -> f32 {
    let x = x.clamp(-30.0, 30.0);
    let ax = x.abs();
    let y = -ax * std::f32::consts::LOG2_E;
    let yi = y.floor();
    let yf = y - yi;
    // Minimax degree-5 polynomial for 2^yf over [0, 1).
    let p = 1.0
        + yf
            * (std::f32::consts::LN_2
                + yf * (0.2402265 + yf * (0.0555041 + yf * (0.0096181 + yf * 0.0013427))));
    // 2^yi via direct injection into the f32 exponent.
    let yi_i = (yi as i32).clamp(-126, 127);
    let pow2_yi = f32::from_bits(((yi_i + 127) as u32) << 23);
    let e = pow2_yi * p;
    let s_neg = e / (1.0 + e);
    let s_pos = 1.0 - s_neg;
    if x >= 0.0 { s_pos } else { s_neg }
}

/// Prior log-odds of the labels (weighted by `weights` if given). The boosting
/// loop starts from this constant.
pub fn init_score(labels: &[f32], weights: Option<&[f32]>) -> f64 {
    let (pos, total) = match weights {
        None => (labels.iter().map(|&y| y as f64).sum::<f64>(), labels.len() as f64),
        Some(w) => {
            let mut pos = 0.0;
            let mut total = 0.0;
            for (&y, &wi) in labels.iter().zip(w) {
                let wi = wi as f64;
                pos += wi * y as f64;
                total += wi;
            }
            (pos, total)
        }
    };
    let mean = (pos / total).clamp(1e-6, 1.0 - 1e-6);
    (mean / (1.0 - mean)).ln()
}

/// Write packed `[grad, hess]` pairs from current raw scores and labels. With
/// `weights`, each row's grad and hess are scaled by its weight (LightGBM
/// `weight` column), which carries through the histogram sums into leaf values
/// and split gains. Packed so the histogram hot loop fetches both with one
/// 8-byte load.
pub fn gradients_packed(raw_scores: &[f64], labels: &[f32], weights: Option<&[f32]>, out: &mut [[f32; 2]]) {
    match weights {
        None => gradients_unweighted(raw_scores, labels, out),
        Some(w) => gradients_weighted(raw_scores, labels, w, out),
    }
}

#[inline]
fn gradients_unweighted(raw_scores: &[f64], labels: &[f32], out: &mut [[f32; 2]]) {
    let n = raw_scores.len();
    for i in 0..n {
        let s = raw_scores[i] as f32;
        let p = sigmoid_fast_f32(s);
        let y = labels[i];
        out[i][0] = p - y;
        let h = p * (1.0 - p);
        out[i][1] = if h > 1e-6 { h } else { 1e-6 };
    }
}

#[inline]
fn gradients_weighted(raw_scores: &[f64], labels: &[f32], w: &[f32], out: &mut [[f32; 2]]) {
    let n = raw_scores.len();
    for i in 0..n {
        let s = raw_scores[i] as f32;
        let p = sigmoid_fast_f32(s);
        let y = labels[i];
        let wi = w[i];
        out[i][0] = wi * (p - y);
        let h = p * (1.0 - p);
        let h = if h > 1e-6 { h } else { 1e-6 };
        out[i][1] = wi * h;
    }
}

/// Binary cross-entropy on raw additive scores. Lower is better.
pub fn binary_logloss(raw_scores: &[f64], labels: &[f32]) -> f64 {
    let n = raw_scores.len() as f64;
    let mut total = 0.0;
    for i in 0..raw_scores.len() {
        let p = sigmoid(raw_scores[i]).clamp(LOGLOSS_EPS, 1.0 - LOGLOSS_EPS);
        let y = labels[i] as f64;
        total += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }
    total / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sigmoid_matches_libm_within_tolerance() {
        for i in -300i32..=300 {
            let x = i as f32 * 0.1;
            let fast = sigmoid_fast_f32(x) as f64;
            let exact = sigmoid(x as f64);
            assert!((fast - exact).abs() < 1e-4);
        }
    }

    #[test]
    fn fast_sigmoid_saturates_at_extremes() {
        assert!((sigmoid_fast_f32(50.0) - 1.0).abs() < 1e-6);
        assert!(sigmoid_fast_f32(-50.0).abs() < 1e-6);
    }
}
