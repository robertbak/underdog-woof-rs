//! The decoder's latent transformer: two RoPE layers with causal, windowed attention, plus the
//! key/value cache that lets a streaming push transform only its own frames.

use ndarray::{s, Array1, Array2, Array3};

use super::DecoderConfig;
use crate::nn::{gelu, layer_norm_rows, softmax_inplace};
use crate::weights::Weights;
use crate::Result;

pub(super) struct TransformerLayer {
    ln1: (Array1<f32>, Array1<f32>),
    ln2: (Array1<f32>, Array1<f32>),
    qkv: String,
    out: String,
    fc1: String,
    fc2: String,
}

impl TransformerLayer {
    pub(super) fn load(weights: &Weights, index: usize) -> Result<Self> {
        let name = |suffix: &str| format!("dec.tf.{index}.{suffix}");
        Ok(TransformerLayer {
            ln1: (
                weights.vector(&name("ln1.0"))?,
                weights.vector(&name("ln1.1"))?,
            ),
            ln2: (
                weights.vector(&name("ln2.0"))?,
                weights.vector(&name("ln2.1"))?,
            ),
            qkv: name("qkv"),
            out: name("o"),
            fc1: name("fc1"),
            fc2: name("fc2"),
        })
    }

    /// Materialise and transpose a grouped-int8 matrix: `(in, out)`.
    ///
    /// The copy is deliberate. Handing sgemm a transposed *view* changes the result here by 2.6 dB
    /// against the reference, so the dense copy is the behaviour that matches numpy.
    fn transposed_weights(&self, weights: &Weights, name: &str) -> Result<Array2<f32>> {
        Ok(weights.grouped(name)?.dequantise_2d().t().to_owned())
    }

    /// Norm + `qkv` projection + RoPE for `rows` positions starting at absolute `start_pos`.
    fn project(
        &self,
        h: &Array2<f32>,
        weights: &Weights,
        cfg: &DecoderConfig,
        inv_freq: &Array1<f32>,
        start_pos: usize,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let rows = h.nrows();
        let (heads, head_dim, dim) = (cfg.heads, cfg.head_dim, cfg.dim);
        let (cos, sin) = rope(start_pos, rows, head_dim, inv_freq);

        let y = layer_norm_rows(h, cfg.eps, Some(&self.ln1.0), Some(&self.ln1.1));
        let qkv = y.dot(&self.transposed_weights(weights, &self.qkv)?); // (rows, 3*dim)

        let mut q = Array3::<f32>::zeros((heads, rows, head_dim));
        let mut k = Array3::<f32>::zeros((heads, rows, head_dim));
        let mut v = Array3::<f32>::zeros((heads, rows, head_dim));
        for t in 0..rows {
            for head in 0..heads {
                for j in 0..head_dim {
                    q[[head, t, j]] = qkv[[t, head * head_dim + j]];
                    k[[head, t, j]] = qkv[[t, dim + head * head_dim + j]];
                    v[[head, t, j]] = qkv[[t, 2 * dim + head * head_dim + j]];
                }
            }
        }
        // decoder RoPE negates the first half of the rotated part; rotate out of place
        let (q_in, k_in) = (q.clone(), k.clone());
        for head in 0..heads {
            for t in 0..rows {
                for j in 0..head_dim {
                    let (c, s_) = (cos[[t, j]], sin[[t, j]]);
                    let qr = if j < head_dim / 2 {
                        -q_in[[head, t, j + head_dim / 2]]
                    } else {
                        q_in[[head, t, j - head_dim / 2]]
                    };
                    let kr = if j < head_dim / 2 {
                        -k_in[[head, t, j + head_dim / 2]]
                    } else {
                        k_in[[head, t, j - head_dim / 2]]
                    };
                    q[[head, t, j]] = q_in[[head, t, j]] * c + qr * s_;
                    k[[head, t, j]] = k_in[[head, t, j]] * c + kr * s_;
                }
            }
        }
        Ok((q, k, v))
    }

    /// Attention of `q` (absolute positions `q_start..`) against `k`/`v` (`k_start..`), causal and
    /// windowed.
    fn attend(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
        cfg: &DecoderConfig,
        q_start: usize,
        k_start: usize,
    ) -> Array2<f32> {
        let (heads, head_dim, dim) = (cfg.heads, cfg.head_dim, cfg.dim);
        let (rows, keys) = (q.shape()[1], k.shape()[1]);
        let mut attended = Array2::<f32>::zeros((rows, dim));
        for head in 0..heads {
            let qh = q.slice(s![head, .., ..]).to_owned();
            let kh = k.slice(s![head, .., ..]).to_owned();
            let vh = v.slice(s![head, .., ..]).to_owned();
            let mut scores = qh.dot(&kh.t()) * (1.0 / (head_dim as f32).sqrt());
            for query in 0..rows {
                let q_abs = q_start + query;
                for key in 0..keys {
                    // causal, and no older than the sliding window
                    let k_abs = k_start + key;
                    if k_abs > q_abs || k_abs + cfg.window <= q_abs {
                        scores[[query, key]] = f32::NEG_INFINITY;
                    }
                }
            }
            softmax_inplace(&mut scores);
            let out = scores.dot(&vh);
            for query in 0..rows {
                for j in 0..head_dim {
                    attended[[query, head * head_dim + j]] = out[[query, j]];
                }
            }
        }
        attended
    }

    /// Everything after attention: output projection, residual, then the gated FFN.
    fn tail(
        &self,
        h: &Array2<f32>,
        attended: &Array2<f32>,
        weights: &Weights,
        cfg: &DecoderConfig,
    ) -> Result<Array2<f32>> {
        let h = h + attended.dot(&self.transposed_weights(weights, &self.out)?);
        let y = layer_norm_rows(&h, cfg.eps, Some(&self.ln2.0), Some(&self.ln2.1));
        let mut gated = y.dot(&self.transposed_weights(weights, &self.fc1)?);
        gated.mapv_inplace(gelu);
        Ok(h + gated.dot(&self.transposed_weights(weights, &self.fc2)?))
    }

    /// One transformer layer over `h`, whose first row sits at absolute position `start_pos` (RoPE
    /// needs absolute positions: a streaming caller passes a non-zero start when it has dropped a
    /// prefix it no longer needs).
    pub(super) fn run(
        &self,
        h: &Array2<f32>,
        weights: &Weights,
        cfg: &DecoderConfig,
        inv_freq: &Array1<f32>,
        start_pos: usize,
    ) -> Result<Array2<f32>> {
        let (q, k, v) = self.project(h, weights, cfg, inv_freq, start_pos)?;
        let attended = self.attend(&q, &k, &v, cfg, start_pos, start_pos);
        self.tail(h, &attended, weights, cfg)
    }

    /// As [`Self::run`], but keys and values from previous pushes are reused instead of recomputed.
    pub(super) fn run_cached(
        &self,
        h: &Array2<f32>,
        weights: &Weights,
        cfg: &DecoderConfig,
        inv_freq: &Array1<f32>,
        start_pos: usize,
        cache: &mut LayerCache,
    ) -> Result<Array2<f32>> {
        let (q, k, v) = self.project(h, weights, cfg, inv_freq, start_pos)?;
        cache.push(&k, &v, start_pos, cfg.window);
        let attended = self.attend(&q, cache.k(), cache.v(), cfg, start_pos, cache.start());
        self.tail(h, &attended, weights, cfg)
    }
}

/// Keys and values for one decoder layer, kept between streaming pushes.
#[derive(Default)]
pub struct LayerCache {
    /// absolute position of `k`'s first row
    start: usize,
    k: Array3<f32>,
    v: Array3<f32>,
}

impl LayerCache {
    fn start(&self) -> usize {
        self.start
    }

    fn k(&self) -> &Array3<f32> {
        &self.k
    }

    fn v(&self) -> &Array3<f32> {
        &self.v
    }

    /// Drop keys/values this push cannot see, then append its own.
    ///
    /// The bound is the *first* query of this push (`start_pos`) minus the window: trimming to what
    /// the last query needs would drop keys the first one still sees.
    fn push(&mut self, k: &Array3<f32>, v: &Array3<f32>, start_pos: usize, window: usize) {
        let earliest = start_pos.saturating_sub(window.saturating_sub(1));
        let drop = earliest.saturating_sub(self.start).min(self.k.shape()[1]);
        if drop > 0 {
            self.k = self.k.slice(s![.., drop.., ..]).to_owned();
            self.v = self.v.slice(s![.., drop.., ..]).to_owned();
            self.start += drop;
        }
        let (heads, new, head_dim) = (k.shape()[0], k.shape()[1], k.shape()[2]);
        let old = self.k.shape()[1];
        if old == 0 && self.k.shape()[0] != heads {
            self.k = Array3::<f32>::zeros((heads, 0, head_dim));
            self.v = Array3::<f32>::zeros((heads, 0, head_dim));
            self.start = start_pos;
        }
        let mut nk = Array3::<f32>::zeros((heads, old + new, head_dim));
        let mut nv = Array3::<f32>::zeros((heads, old + new, head_dim));
        if old > 0 {
            nk.slice_mut(s![.., ..old, ..]).assign(&self.k);
            nv.slice_mut(s![.., ..old, ..]).assign(&self.v);
        }
        nk.slice_mut(s![.., old.., ..]).assign(k);
        nv.slice_mut(s![.., old.., ..]).assign(v);
        self.k = nk;
        self.v = nv;
    }
}

/// Decoder hyper-parameters, read from the manifest.
fn rope(
    start: usize,
    rows: usize,
    head_dim: usize,
    inv_freq: &Array1<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let mut cos = Array2::<f32>::zeros((rows, head_dim));
    let mut sin = Array2::<f32>::zeros((rows, head_dim));
    for t in 0..rows {
        for j in 0..head_dim / 2 {
            let angle = (start + t) as f32 * inv_freq[j];
            cos[[t, j]] = angle.cos();
            cos[[t, j + head_dim / 2]] = angle.cos();
            sin[[t, j]] = angle.sin();
            sin[[t, j + head_dim / 2]] = angle.sin();
        }
    }
    (cos, sin)
}
