//! Numerical primitives, written to mirror `underdog_woof/runtime.py` operation by operation.
//!
//! The constants matter as much as the arithmetic: the reference uses numpy's *float32 eps* for
//! the speech model's RMS scaling and `1e-5` for the layer norms, and it approximates `erf` with
//! Abramowitz & Stegun 7.1.26 because numpy has none. All three are reproduced here, since
//! anything else shifts the output by a few dB of SNR.

use ndarray::{Array1, Array2};

/// numpy's `np.finfo(np.float32).eps`, used by `Speech._rinv`.
pub const F32_EPS: f32 = 1.192_092_9e-7;
/// Epsilon used by `_layer_norm` and the decoder's `_ln`.
pub const LN_EPS: f32 = 1e-5;

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7 — the reference's `_erf`.
pub fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-a * a).exp();
    y.copysign(x)
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * (1.0 / 2f32.sqrt())))
}

#[inline]
pub fn elu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        x.exp() - 1.0
    }
}

/// `Speech._rinv`: `1 / sqrt(mean(x^2) + eps)`, applied to a matvec *output*.
pub fn inverse_rms(x: &Array1<f32>) -> f32 {
    let d = x.len() as f32;
    1.0 / ((x.dot(x) / d) + F32_EPS).sqrt()
}

/// Per-row RMS scaling (`xn = x / sqrt(mean(x^2) + eps)`) used by the prefix pass.
pub fn rms_rows(x: &Array2<f32>) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let d = row.len() as f32;
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let gain = 1.0 / (sum_sq / d + F32_EPS).sqrt();
        row.mapv_inplace(|v| v * gain);
    }
    out
}

/// `_layer_norm` / decoder `_ln`, with the optional learned gain and bias the decoder applies.
pub fn layer_norm_rows(
    x: &Array2<f32>,
    eps: f32,
    gain: Option<&Array1<f32>>,
    bias: Option<&Array1<f32>>,
) -> Array2<f32> {
    let mut out = x.clone();
    for mut row in out.rows_mut() {
        let d = row.len() as f32;
        let mean: f32 = row.iter().sum::<f32>() / d;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d;
        let gain_rms = 1.0 / (var + eps).sqrt();
        for (j, v) in row.iter_mut().enumerate() {
            let mut t = (*v - mean) * gain_rms;
            if let Some(g) = gain {
                t *= g[j];
            }
            if let Some(b) = bias {
                t += b[j];
            }
            *v = t;
        }
    }
    out
}

/// Layer norm over a single vector (the flow head's blocks).
pub fn layer_norm_row(h: &Array1<f32>, eps: f32) -> Array1<f32> {
    let d = h.len() as f32;
    let mean: f32 = h.iter().sum::<f32>() / d;
    let var: f32 = h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d;
    let gain = 1.0 / (var + eps).sqrt();
    h.mapv(|v| (v - mean) * gain)
}

/// Row-wise softmax; `-inf` entries are excluded (masks are applied before calling this).
pub fn softmax_inplace(s: &mut Array2<f32>) {
    for mut row in s.rows_mut() {
        let mut max = f32::NEG_INFINITY;
        for v in row.iter() {
            if *v > max {
                max = *v;
            }
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// `rotate_half`: `[v[half..], v[..half]]`. Used by the speech model's RoPE.
pub fn rotate_half(v: &Array1<f32>) -> Array1<f32> {
    let half = v.len() / 2;
    let mut out = Array1::zeros(v.len());
    for i in 0..half {
        out[i] = v[half + i];
        out[half + i] = v[i];
    }
    out
}

/// 8-way unrolled dot product; LLVM turns this into FMA (AVX2/NEON) without intrinsics.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        for k in 0..8 {
            acc[k] += a[o + k] * b[o + k];
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    for i in chunks * 8..n {
        sum += a[i] * b[i];
    }
    sum
}

/// `out[i] = q[i] * scale`, one int8 row expanded to f32.
#[inline]
pub fn dequantise_row(q: &[i8], scale: f32, out: &mut [f32]) {
    for (o, v) in out.iter_mut().zip(q.iter()) {
        *o = f32::from(*v) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut s = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]).unwrap();
        softmax_inplace(&mut s);
        for row in s.rows() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row sums to {sum}");
        }
    }

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.05).collect();
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-4);
    }
}
