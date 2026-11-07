//! Numerical primitives, written to mirror `underdog_woof/runtime.py` operation by operation.
//!
//! The constants matter as much as the arithmetic: the reference uses numpy's *float32 eps* for
//! the speech model's RMS scaling and `1e-5` for the layer norms, and it approximates `erf` with
//! Abramowitz & Stegun 7.1.26 because numpy has none. All three are reproduced here, since
//! anything else shifts the output by a few dB of SNR.

use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array1, Array2};

/// Whether matrix-vector products may quantise their activations to int8 (W8A8).
///
/// Off by default: the reference multiplies dequantised f32 weights by f32 activations, and matching
/// it is the point of the port. On, the weights stay int8 *and* the activation is quantised, so the
/// product accumulates in i32 and no f32 copy of the weight row is ever materialised -- 4x less
/// memory traffic on the side that is actually bound. The cost is accuracy, measured with
/// `tools/parity.py`, not assumed.
///
/// A process-global rather than a parameter: `matvec` is called from a dozen places inside the
/// speech model, and this is set once before any synthesis starts.
static W8A8: AtomicBool = AtomicBool::new(false);

pub fn set_w8a8(enabled: bool) {
    W8A8.store(enabled, Ordering::Relaxed);
}

#[inline]
pub fn w8a8() -> bool {
    W8A8.load(Ordering::Relaxed)
}

/// Symmetric per-vector int8 quantisation: the scale is `max|x| / 127`, so the largest magnitude maps
/// to -127 or 127 and everything else keeps the mantissa the range allows.
pub fn quantise_vector(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return (vec![0i8; x.len()], 0.0);
    }
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    let q = x
        .iter()
        .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

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

/// Whether the fused int8×f32 dot can run on AVX2/FMA (x86_64 only).
pub fn simd_dot_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `scale · Σ q_i · x_i`, dequantising in registers.
///
/// Two int8 weights per instruction-pair: load 8 bytes, widen to 8 int32s, convert to f32, FMA
/// against the activation. That removes the dequantised scratch entirely -- the copy was the cost
/// (profiling: 156 µs for a 448x1344 block against ~36 µs for the integer kernel), and trying to do
/// the same thing in portable Rust made it *slower* (213 µs) because the per-element widen defeated
/// the vectoriser. This is the same idea with the vector instructions written down.
pub fn dot_i8_scaled(q: &[i8], x: &[f32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_dot_available() {
            // SAFETY: `simd_dot_available` checked AVX2 and FMA on this CPU; the loads are bounded
            // by the loop conditions against the shorter slice.
            return unsafe { dot_i8_scaled_avx2(q, x, scale) };
        }
    }
    let n = q.len().min(x.len());
    let mut sum = 0f32;
    for i in 0..n {
        sum += f32::from(q[i]) * x[i];
    }
    sum * scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_scaled_avx2(q: &[i8], x: &[f32], scale: f32) -> f32 {
    use std::arch::x86_64::*;

    let n = q.len().min(x.len());
    let (mut acc0, mut acc1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (qp, xp) = (q.as_ptr(), x.as_ptr());
    let mut i = 0;
    while i + 16 <= n {
        let q0 = _mm_loadl_epi64(qp.add(i) as *const __m128i);
        let q1 = _mm_loadl_epi64(qp.add(i + 8) as *const __m128i);
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)),
            _mm256_loadu_ps(xp.add(i)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)),
            _mm256_loadu_ps(xp.add(i + 8)),
            acc1,
        );
        i += 16;
    }
    while i + 8 <= n {
        let q0 = _mm_loadl_epi64(qp.add(i) as *const __m128i);
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)),
            _mm256_loadu_ps(xp.add(i)),
            acc0,
        );
        i += 8;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);
    while i < n {
        sum += f32::from(*qp.add(i)) * *xp.add(i);
        i += 1;
    }
    sum * scale
}

/// `Σ q_i · x_i` for int8 weights against f32 activations.
///
/// The row scale is applied by the caller *after* the sum, which is the same product as dequantising
/// first (`Σ (q_i·s)·x_i = s·Σ q_i·x_i`) up to float rounding — and it removes the scratch buffer the
/// dequantise-first version writes and reads back, which profiling showed to be the cost: 195 µs for a
/// 448x1344 row-block against 36 µs for the integer path, for the same arithmetic.
#[inline]
pub fn dot_i8_f32(q: &[i8], x: &[f32]) -> f32 {
    let n = q.len().min(x.len());
    let mut acc = [0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        for k in 0..8 {
            acc[k] += f32::from(q[o + k]) * x[o + k];
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    for i in chunks * 8..n {
        sum += f32::from(q[i]) * x[i];
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
    fn quantise_vector_bounds_the_error_by_half_a_step() {
        let x: Vec<f32> = (0..512)
            .map(|i| ((i as f32 * 0.37).sin() * 1.7) + 0.05)
            .collect();
        let (q, scale) = quantise_vector(&x);
        assert_eq!(q.len(), x.len());
        assert!(scale > 0.0);
        // every element is within half a quantisation step of where it came from
        for (orig, qi) in x.iter().zip(q.iter()) {
            let back = f32::from(*qi) * scale;
            assert!(
                (orig - back).abs() <= scale * 0.5 + 1e-6,
                "{orig} -> {back} (step {scale})"
            );
        }
        // and the largest magnitude actually reaches the top of the range
        assert_eq!(q.iter().map(|v| v.unsigned_abs()).max(), Some(127));
    }

    #[test]
    fn quantise_vector_handles_an_all_zero_activation() {
        let (q, scale) = quantise_vector(&[0.0; 8]);
        assert!(q.iter().all(|v| *v == 0));
        assert_eq!(scale, 0.0);
    }

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
