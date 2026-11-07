//! The maths inside a frame: the prefix pass over voice+text tokens, one transformer step, the
//! conditioning vector and the flow head.
//!
//! Split out of `speech.rs` because these four functions are where an optimisation lands or does not:
//! `step_block` runs eight times per frame — 25 frames a second — so its allocations and its matvecs
//! are the hot path.

use ndarray::{s, Array1, Array2, Array3};

use crate::nn::{self, inverse_rms, layer_norm_row, rms_rows, rotate_half, silu, LN_EPS};

use super::Speech;

impl Speech {
    pub(super) fn prefix_block(
        &mut self,
        layer: usize,
        x: &Array2<f32>,
        rows: usize,
    ) -> Array2<f32> {
        let (heads, head_dim, hidden, ffn) = (self.heads, self.head_dim, self.hidden, self.ffn);
        let scale = self.attention_scale;
        let [wqkv, wo, w1, w2] = &self.blocks[layer];

        let qkv = wqkv.matmul_t(&rms_rows(x)); // (rows, 3*heads*head_dim)
        let mut q = Array3::<f32>::zeros((heads, rows, head_dim));
        let mut k = Array3::<f32>::zeros((heads, rows, head_dim));
        let mut v = Array3::<f32>::zeros((heads, rows, head_dim));
        for t in 0..rows {
            let cos = self.cos.slice(s![t, ..]).to_owned();
            let sin = self.sin.slice(s![t, ..]).to_owned();
            for h in 0..heads {
                for (part, dst) in [(0usize, &mut q), (1, &mut k), (2, &mut v)] {
                    let base = (part * heads + h) * head_dim;
                    let row = qkv.slice(s![t, base..base + head_dim]).to_owned();
                    // no rotation on the value stream
                    let rotated = if part < 2 {
                        &row * &cos + &rotate_half(&row) * &sin
                    } else {
                        row
                    };
                    dst.slice_mut(s![h, t, ..]).assign(&rotated);
                }
            }
        }

        let mut attended = Array3::<f32>::zeros((heads, rows, head_dim));
        for h in 0..heads {
            let qh = q.slice(s![h, .., ..]).to_owned();
            let kh = k.slice(s![h, .., ..]).to_owned();
            let vh = v.slice(s![h, .., ..]).to_owned();
            self.keys[layer].slice_mut(s![h, ..rows, ..]).assign(&kh);
            self.values[layer].slice_mut(s![h, ..rows, ..]).assign(&vh);

            let mut scores = qh.dot(&kh.t()) * scale;
            for r in 0..rows {
                for c in (r + 1)..rows {
                    scores[[r, c]] = f32::NEG_INFINITY;
                }
            }
            nn::softmax_inplace(&mut scores);
            attended.slice_mut(s![h, .., ..]).assign(&scores.dot(&vh));
        }

        let mut merged = Array2::<f32>::zeros((rows, hidden));
        for t in 0..rows {
            for h in 0..heads {
                for j in 0..head_dim {
                    merged[[t, h * head_dim + j]] = attended[[h, t, j]];
                }
            }
        }
        let x = x + wo.matmul_t(&merged);

        let gated = w1.matmul_t(&rms_rows(&x)); // (rows, 2*ffn)
        let mut gate = Array2::<f32>::zeros((rows, ffn));
        for t in 0..rows {
            for j in 0..ffn {
                gate[[t, j]] = silu(gated[[t, j]]) * gated[[t, ffn + j]];
            }
        }
        x + w2.matmul_t(&gate)
    }

    /// `_block1`: one incremental step at position `pos` (KV cache append + FFN).
    pub(super) fn step_block(&mut self, layer: usize, x: &Array1<f32>, pos: usize) -> Array1<f32> {
        let (heads, head_dim, ffn) = (self.heads, self.head_dim, self.ffn);
        let scale = self.attention_scale;
        let [wqkv, wo, w1, w2] = &self.blocks[layer];

        let inverse = {
            let _s = crate::profile::span(crate::profile::Part::Elementwise);
            inverse_rms(x)
        };
        let qkv = {
            let _s = crate::profile::span(crate::profile::Part::Matvec);
            wqkv.matvec(x).mapv(|v| v * inverse)
        };
        let cos = self.cos.slice(s![pos, ..]).to_owned();
        let sin = self.sin.slice(s![pos, ..]).to_owned();

        let mut queries = Array2::<f32>::zeros((heads, head_dim));
        let attention = crate::profile::span(crate::profile::Part::Attention);
        for h in 0..heads {
            let q = qkv.slice(s![h * head_dim..(h + 1) * head_dim]).to_owned();
            let k = qkv
                .slice(s![(heads + h) * head_dim..(heads + h + 1) * head_dim])
                .to_owned();
            let v = qkv
                .slice(s![
                    (2 * heads + h) * head_dim..(2 * heads + h + 1) * head_dim
                ])
                .to_owned();
            queries
                .slice_mut(s![h, ..])
                .assign(&(&q * &cos + &rotate_half(&q) * &sin));
            let rotated_k = &k * &cos + &rotate_half(&k) * &sin;
            self.keys[layer]
                .slice_mut(s![h, pos, ..])
                .assign(&rotated_k);
            self.values[layer].slice_mut(s![h, pos, ..]).assign(&v);
        }

        let mut attended = Array1::<f32>::zeros(heads * head_dim);
        for h in 0..heads {
            let q = queries.slice(s![h, ..]).to_owned();
            let kh = self.keys[layer].slice(s![h, ..pos + 1, ..]).to_owned();
            let vh = self.values[layer].slice(s![h, ..pos + 1, ..]).to_owned();
            let scores = kh.dot(&q) * scale;
            let mut weights = Array2::from_shape_fn((1, pos + 1), |(_, j)| scores[j]);
            nn::softmax_inplace(&mut weights);
            let value = weights.row(0).dot(&vh);
            attended
                .slice_mut(s![h * head_dim..(h + 1) * head_dim])
                .assign(&value);
        }
        drop(attention);
        let x = {
            let _s = crate::profile::span(crate::profile::Part::Matvec);
            x + wo.matvec(&attended)
        };

        let gated = {
            let _s = crate::profile::span(crate::profile::Part::Matvec);
            w1.matvec(&x)
        };
        let inverse = {
            let _s = crate::profile::span(crate::profile::Part::Elementwise);
            inverse_rms(&x)
        };
        let mut gate = Array1::<f32>::zeros(ffn);
        {
            let _s = crate::profile::span(crate::profile::Part::Elementwise);
            for j in 0..ffn {
                gate[j] = silu(gated[j] * inverse) * (gated[ffn + j] * inverse);
            }
        }
        let _s = crate::profile::span(crate::profile::Part::Matvec);
        x + w2.matvec(&gate)
    }

    /// `_cond`: the conditioning vector for the flow head plus the EOS logit.
    pub(super) fn condition(&self, x: &Array1<f32>) -> (Array1<f32>, f32) {
        let inverse = inverse_rms(x);
        let cond = self.cond_embed.0.matvec(x).mapv(|v| v * inverse) + &self.cond_embed.1;
        let s = cond.slice(s![..self.head_width]).mapv(silu);
        (s, cond[self.head_width])
    }

    /// `_head`: one flow step, `z = noise + head(cond, noise)`.
    pub(super) fn flow(&self, cond: &Array1<f32>, noise: &Array1<f32>) -> Array1<f32> {
        let w = self.head_width;
        let modulation = self.modulations.0.matvec(cond) + &self.modulations.1;
        let mut h = self.hidden_in.0.matvec(noise) + &self.hidden_in.1;

        for block in 0..self.depth {
            let o = 3 * w * block;
            let normalised = layer_norm_row(&h, LN_EPS);
            let y = modulation.slice(s![o..o + w]).to_owned()
                + &(&normalised * &modulation.slice(s![o + w..o + 2 * w]).to_owned());
            let (w1, b1) = &self.head_blocks[block][0];
            let (w2, b2) = &self.head_blocks[block][1];
            let mut hidden = w1.matvec(&y) + b1;
            hidden.mapv_inplace(silu);
            let update = w2.matvec(&hidden) + b2;
            h = &h + &(update * &modulation.slice(s![o + 2 * w..o + 3 * w]).to_owned());
        }

        let o = 3 * w * self.depth;
        let normalised = layer_norm_row(&h, LN_EPS);
        let inner = modulation.slice(s![o..o + w]).to_owned()
            + &(&normalised * &modulation.slice(s![o + w..o + 2 * w]).to_owned());
        self.hidden_out.0.matvec(&inner) + &self.hidden_out.1
    }
}
