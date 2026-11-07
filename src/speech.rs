//! The speech model: text/voice prefix, one incremental transformer step per latent frame, and the
//! one-step flow head that produces each frame.
//!
//! Ported from `Speech` in `underdog_woof/runtime.py`, keeping the same order of operations so the
//! output stays within ~30 dB of the reference's own float32 noise.

use ndarray::{s, Array1, Array2, Array3};

use crate::audio;
use crate::nn::{self, inverse_rms, layer_norm_row, rms_rows, rotate_half, silu, LN_EPS};
use crate::rng::Mt19937;
use crate::weights::{Linear, Weights};
use crate::Result;

/// Latent frames allocated before the cache grows: 128 frames = 5.1 s, which covers a typical
/// 120-character utterance without a reallocation.
const INITIAL_CAPACITY: usize = 128;

/// Sampling knobs, as the reference exposes them.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub eos_threshold: f32,
    pub max_frames: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        // the reference's `Woof()` defaults
        Sampling {
            temperature: 0.5,
            eos_threshold: 0.9,
            max_frames: 750,
        }
    }
}

pub struct Speech {
    layers: usize,
    hidden: usize,
    heads: usize,
    ffn: usize,
    head_width: usize,
    depth: usize,
    latent: usize,
    head_dim: usize,
    max_frames: usize,

    blocks: Vec<[Linear; 4]>, // wqkv, wo, w1, w2
    cond_embed: (Linear, Array1<f32>),
    modulations: (Linear, Array1<f32>),
    latent_in: (Linear, Array1<f32>),
    hidden_in: (Linear, Array1<f32>),
    hidden_out: (Linear, Array1<f32>),
    head_blocks: Vec<[(Linear, Array1<f32>); 2]>,

    text_embedding: Array2<f32>,
    bos: Array1<f32>,
    voice: Array2<f32>,
    /// RoPE tables, `(capacity, head_dim)`; rebuilt only when the cache grows.
    cos: Array2<f32>,
    sin: Array2<f32>,

    /// per-layer key/value caches, `(heads, capacity, head_dim)`.
    ///
    /// `capacity` starts small and doubles on demand. The reference (and the first version here)
    /// allocates the full 1100-frame table up front: 8 layers x 2 x 7 heads x 1100 x 64 x f32 =
    /// 31.5 MB, for utterances that are a ~125-frame sentence in practice.
    keys: Vec<Array3<f32>>,
    values: Vec<Array3<f32>>,
    /// Current KV capacity in frames; grows on demand up to `max_frames`.
    capacity: usize,
    attention_scale: f32,
}

impl Speech {
    pub fn new(w: &Weights, cfg: &serde_json::Value, voice: &str) -> Result<Self> {
        let layers = field(cfg, "L")?;
        let hidden = field(cfg, "d")?;
        let heads = field(cfg, "H")?;
        let ffn = field(cfg, "ff")?;
        let head_width = field(cfg, "hw")?;
        let depth = field(cfg, "depth")?;
        let latent = field(cfg, "latent")?;
        let max_frames = field(cfg, "max_len")?;
        let head_dim = hidden / heads;

        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            blocks.push([
                w.linear(&format!("sp.layers.{i}.0"))?,
                w.linear(&format!("sp.layers.{i}.1"))?,
                w.linear(&format!("sp.layers.{i}.2"))?,
                w.linear(&format!("sp.layers.{i}.3"))?,
            ]);
        }
        let mut head_blocks = Vec::with_capacity(depth);
        for j in 0..depth {
            head_blocks.push([
                (
                    w.linear(&format!("sp.hblocks.{j}.0"))?,
                    w.vector(&format!("sp.hblocks.{j}.0.b"))?,
                ),
                (
                    w.linear(&format!("sp.hblocks.{j}.1"))?,
                    w.vector(&format!("sp.hblocks.{j}.1.b"))?,
                ),
            ]);
        }
        let pair = |name: &str| -> Result<(Linear, Array1<f32>)> {
            Ok((
                w.linear(&format!("sp.{name}"))?,
                w.vector(&format!("sp.{name}.b"))?,
            ))
        };

        // start at a few seconds and grow; see `ensure_capacity`
        let capacity = INITIAL_CAPACITY.min(max_frames);
        let (cos, sin) = rope_tables(capacity, head_dim, 10_000.0);
        Ok(Speech {
            layers,
            hidden,
            heads,
            ffn,
            head_width,
            depth,
            latent,
            head_dim,
            max_frames,
            blocks,
            cond_embed: pair("ce")?,
            modulations: pair("ada")?,
            latent_in: pair("lat_in")?,
            hidden_in: pair("h_inp")?,
            hidden_out: pair("h_out")?,
            head_blocks,
            text_embedding: w.matrix("sp.text_emb")?,
            bos: w.vector("sp.bos")?,
            voice: w.matrix(&format!("sp.voice.{voice}"))?,
            cos,
            sin,
            keys: (0..layers)
                .map(|_| Array3::<f32>::zeros((heads, capacity, head_dim)))
                .collect(),
            values: (0..layers)
                .map(|_| Array3::<f32>::zeros((heads, capacity, head_dim)))
                .collect(),
            capacity,
            attention_scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// Make room for `needed` positions, doubling the cache (and rebuilding the rope tables) when it
    /// does not fit. Values already cached are preserved, so positions never shift.
    fn ensure_capacity(&mut self, needed: usize) -> Result<()> {
        if needed <= self.capacity {
            return Ok(());
        }
        if needed > self.max_frames {
            return Err(format!(
                "utterance needs {needed} latent frames, the model's tables allow {}",
                self.max_frames
            )
            .into());
        }
        let mut capacity = self.capacity.max(1);
        while capacity < needed {
            capacity = (capacity * 2).min(self.max_frames);
        }
        let (heads, _, head_dim) = self.keys[0].dim();
        for (keys, values) in self.keys.iter_mut().zip(self.values.iter_mut()) {
            let mut grown_keys = Array3::<f32>::zeros((heads, capacity, head_dim));
            let mut grown_values = Array3::<f32>::zeros((heads, capacity, head_dim));
            grown_keys
                .slice_mut(s![.., ..self.capacity, ..])
                .assign(&*keys);
            grown_values
                .slice_mut(s![.., ..self.capacity, ..])
                .assign(&*values);
            *keys = grown_keys;
            *values = grown_values;
        }
        let (cos, sin) = rope_tables(capacity, head_dim, 10_000.0);
        self.cos = cos;
        self.sin = sin;
        self.capacity = capacity;
        Ok(())
    }

    /// Replace the voice prefix. Everything else in the model is voice-independent, so this is all
    /// it takes to serve a different voice: 28 KiB of state, not a second copy of the model.
    pub fn set_voice(&mut self, prefix: Array2<f32>) -> Result<()> {
        if prefix.ncols() != self.hidden {
            return Err(format!(
                "voice prefix has {} columns, the model expects {}",
                prefix.ncols(),
                self.hidden
            )
            .into());
        }
        self.voice = prefix;
        Ok(())
    }

    /// Number of latent frames per second of audio (unused by the decoder, kept for reporting).
    pub fn latent_rate(&self) -> f32 {
        25.0
    }

    /// Generate `(frames, latent)` for a tokenised text, drawing noise exactly like the reference.
    ///
    /// `on_chunk` is called with each batch of frames as it is produced; `chunk_frames == 0` means
    /// "once, at the end". The autoregressive loop is unchanged — the callback only observes, so the
    /// audio is identical either way.
    pub fn latents_streaming(
        &mut self,
        token_ids: &[usize],
        sampling: Sampling,
        rng: &mut Mt19937,
        chunk_frames: usize,
        mut on_chunk: impl FnMut(&Array2<f32>) -> Result<()>,
    ) -> Result<Array2<f32>> {
        let prefix_rows = self.voice.nrows() + token_ids.len() + 1;
        let mut x = Array2::<f32>::zeros((prefix_rows, self.hidden));
        x.slice_mut(s![..self.voice.nrows(), ..])
            .assign(&self.voice);
        for (t, &id) in token_ids.iter().enumerate() {
            if id >= self.text_embedding.nrows() {
                return Err(format!("token id {id} outside the text embedding").into());
            }
            x.slice_mut(s![self.voice.nrows() + t, ..])
                .assign(&self.text_embedding.row(id));
        }
        x.slice_mut(s![prefix_rows - 1, ..]).assign(&self.bos);

        self.ensure_capacity(prefix_rows + 1)?;
        for layer in 0..self.layers {
            let _s = crate::profile::span(crate::profile::Part::Prefix);
            x = self.prefix_block(layer, &x, prefix_rows);
        }

        let mut last = x.row(prefix_rows - 1).to_owned();
        let (mut cond, mut eos) = self.condition(&last);
        let eos_logit = (sampling.eos_threshold / (1.0 - sampling.eos_threshold)).ln();
        let max_frames = sampling.max_frames.min(self.max_frames - prefix_rows);

        let mut frames = Array2::<f32>::zeros((max_frames, self.latent));
        let mut produced = 0usize;
        let mut emitted = 0usize;
        for f in 0..max_frames {
            // the reference emits at least four frames before it may stop
            if f > 3 && eos > eos_logit {
                break;
            }
            let noise = rng.randn(self.latent) * sampling.temperature;
            let z = {
                let _s = crate::profile::span(crate::profile::Part::Flow);
                &noise + &self.flow(&cond, &noise)
            };
            frames.slice_mut(s![f, ..]).assign(&z);
            produced = f + 1;

            last = {
                let _s = crate::profile::span(crate::profile::Part::Matvec);
                self.latent_in.0.matvec(&z) + &self.latent_in.1
            };
            self.ensure_capacity(prefix_rows + f + 1)?;
            for layer in 0..self.layers {
                last = self.step_block(layer, &last, prefix_rows + f);
            }
            let (c, e) = {
                let _s = crate::profile::span(crate::profile::Part::Flow);
                self.condition(&last)
            };
            cond = c;
            eos = e;
            crate::profile::frame();

            if chunk_frames > 0 && produced - emitted >= chunk_frames {
                on_chunk(&frames.slice(s![emitted..produced, ..]).to_owned())?;
                emitted = produced;
            }
        }
        if chunk_frames > 0 && produced > emitted {
            on_chunk(&frames.slice(s![emitted..produced, ..]).to_owned())?;
        }
        Ok(frames.slice(s![..produced, ..]).to_owned())
    }

    /// `_prefix_block`: full causal self-attention over the voice/text prefix.
    fn prefix_block(&mut self, layer: usize, x: &Array2<f32>, rows: usize) -> Array2<f32> {
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
    fn step_block(&mut self, layer: usize, x: &Array1<f32>, pos: usize) -> Array1<f32> {
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
    fn condition(&self, x: &Array1<f32>) -> (Array1<f32>, f32) {
        let inverse = inverse_rms(x);
        let cond = self.cond_embed.0.matvec(x).mapv(|v| v * inverse) + &self.cond_embed.1;
        let s = cond.slice(s![..self.head_width]).mapv(silu);
        (s, cond[self.head_width])
    }

    /// `_head`: one flow step, `z = noise + head(cond, noise)`.
    fn flow(&self, cond: &Array1<f32>, noise: &Array1<f32>) -> Array1<f32> {
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

    /// Frame cap a draw for `text` may reach before it counts as truncated, i.e. the reference's
    /// `frames_max` for this utterance.
    pub fn frame_cap(&self, text: &str) -> usize {
        let prefix_rows = self.voice.nrows() + audio::encode_text(text).len() + 1;
        self.max_frames.saturating_sub(prefix_rows)
    }

    /// Generate all latents for a tokenised text (no incremental callback).
    pub fn latents(
        &mut self,
        token_ids: &[usize],
        sampling: Sampling,
        rng: &mut Mt19937,
    ) -> Result<Array2<f32>> {
        self.latents_streaming(token_ids, sampling, rng, 0, |_| Ok(()))
    }

    /// Convenience: tokenise + sample, i.e. `Woof.__call__` up to the latents.
    pub fn latents_for_text(
        &mut self,
        text: &str,
        sampling: Sampling,
        rng: &mut Mt19937,
    ) -> Result<Array2<f32>> {
        self.latents(&audio::encode_text(text), sampling, rng)
    }
}

fn field(cfg: &serde_json::Value, key: &str) -> Result<usize> {
    cfg[key]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("speech config missing `{key}`").into())
}

/// RoPE tables, built exactly like `Speech.__init__`:
/// `cos = concat(cos(ang), cos(ang))`, `sin = concat(-sin(ang), sin(ang))`.
fn rope_tables(max_len: usize, head_dim: usize, base: f32) -> (Array2<f32>, Array2<f32>) {
    let inv: Vec<f32> = (0..head_dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let mut cos = Array2::<f32>::zeros((max_len, head_dim));
    let mut sin = Array2::<f32>::zeros((max_len, head_dim));
    for pos in 0..max_len {
        for j in 0..head_dim / 2 {
            let angle = pos as f32 * inv[j];
            cos[[pos, j]] = angle.cos();
            cos[[pos, j + head_dim / 2]] = angle.cos();
            sin[[pos, j]] = -angle.sin();
            sin[[pos, j + head_dim / 2]] = angle.sin();
        }
    }
    (cos, sin)
}
