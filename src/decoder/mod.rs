//! The audio decoder: a small transformer over latent frames followed by a SEANet stack that
//! upsamples them to 24 kHz.
//!
//! Two halves, mirroring `Decoder` in the reference:
//!
//! * stages `0..split` run once over the whole latent sequence,
//! * the rest run in row chunks, because the high-rate convolutions are memory-hungry (the
//!   reference calls this the "chunked fused" path; it also keeps streaming state across chunks).

pub mod conv;

use ndarray::{s, Array1, Array2};

use crate::weights::Weights;
use crate::Result;
mod stages;
mod transformer;

use conv::{build_stage, Stage};
use stages::{run_stages, run_stages_dumped};
use transformer::{LayerCache, TransformerLayer};

#[derive(Clone, Copy, Debug)]
pub struct DecoderConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub window: usize,
    pub eps: f32,
    pub chunk: usize,
    pub split: usize,
    pub dim: usize,
}

impl DecoderConfig {
    pub fn from_manifest(cfg: &serde_json::Value) -> Result<Self> {
        let dec = &cfg["dec"];
        let get = |k: &str| dec[k].as_u64().map(|v| v as usize);
        let heads = get("nh").ok_or("decoder config missing `nh`")?;
        let head_dim = get("hd").ok_or("decoder config missing `hd`")?;
        Ok(DecoderConfig {
            heads,
            head_dim,
            window: get("window").unwrap_or(250),
            eps: dec["eps"].as_f64().unwrap_or(1e-5) as f32,
            chunk: get("chunk").unwrap_or(8),
            split: get("split").unwrap_or(6),
            dim: heads * head_dim,
        })
    }
}

pub struct Decoder {
    cfg: DecoderConfig,
    /// per-latent de-normalisation applied before the input projection (`mean`, `std` in the file)
    latent_mean: Array1<f32>,
    latent_std: Array1<f32>,
    input: (Array2<f32>, Array1<f32>),
    inv_freq: Array1<f32>,
    layers: Vec<TransformerLayer>,
    stages: Vec<Stage>,
}

/// Timing breakdown, so callers can report where the decoder spends its time.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecoderTimings {
    pub transformer: f32,
    pub head_stages: f32,
    pub tail_stages: f32,
}

impl Decoder {
    /// `cfg` is the manifest's `cfg` object (`{"sp": …, "dec": …, "seanet": […]}`).
    pub fn new(weights: &Weights, cfg: &serde_json::Value) -> Result<Self> {
        let config = DecoderConfig::from_manifest(cfg)?;
        let layers = (0..)
            .take_while(|i| weights.has(&format!("dec.tf.{i}.qkv")))
            .map(|i| TransformerLayer::load(weights, i))
            .collect::<Result<Vec<_>>>()?;
        let stages = cfg["seanet"]
            .as_array()
            .ok_or("manifest missing `cfg.seanet`")?
            .iter()
            .enumerate()
            .map(|(i, stage)| build_stage(weights, stage, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Decoder {
            latent_mean: weights.vector("mean")?,
            latent_std: weights.vector("std")?,
            input: (weights.matrix("dec.inp_w")?, weights.vector("dec.inp_b")?),
            inv_freq: weights.vector("dec.inv_freq")?,
            layers,
            stages,
            cfg: config,
        })
    }

    pub fn config(&self) -> DecoderConfig {
        self.cfg
    }

    /// Latent dimension the decoder expects (for the de-normalisation).
    pub fn latent_dim(&self) -> usize {
        self.latent_mean.len()
    }

    /// De-normalise + run the latent transformer: the CPU half of the pipeline that the resident
    /// GPU back-end still runs on the host.
    #[cfg_attr(not(feature = "gpu-resident"), allow(dead_code))]
    pub(crate) fn prepare(&self, latents: &Array2<f32>, weights: &Weights) -> Result<Array2<f32>> {
        let latents = self.denormalise(latents)?;
        self.transformer(&latents, weights, 0)
    }

    #[cfg_attr(not(feature = "gpu-resident"), allow(dead_code))]
    pub(crate) fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Run one stage on the CPU. Gives back-ends a way to compare a device result against the
    /// reference without owning the stage list.
    #[cfg_attr(not(feature = "gpu-resident"), allow(dead_code))]
    pub(crate) fn run_stage(
        &mut self,
        index: usize,
        x: Array2<f32>,
        weights: &Weights,
    ) -> Result<Array2<f32>> {
        let stage = self
            .stages
            .get_mut(index)
            .ok_or_else(|| format!("stage {index} out of range"))?;
        run_stages(x, std::slice::from_mut(stage), weights)
    }

    #[cfg_attr(not(feature = "gpu-resident"), allow(dead_code))]
    pub(crate) fn reset_states(&mut self) {
        for stage in self.stages.iter_mut() {
            stage.reset();
        }
    }

    /// Diagnostic hook: with `WOOF_DUMP_STAGES=<dir>`, write the transformer output and each
    /// stage's output as raw little-endian f32 (rows x cols) so they can be diffed against the
    /// reference implementation. Kept in the library because it is the fastest way to localise a
    /// parity regression.
    fn dump(name: &str, x: &Array2<f32>) {
        let Ok(dir) = std::env::var("WOOF_DUMP_STAGES") else {
            return;
        };
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in x.iter() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let path =
            std::path::Path::new(&dir).join(format!("{name}_{}x{}.f32", x.nrows(), x.ncols()));
        let _ = std::fs::write(path, bytes);
    }

    /// Raw latents `(frames, latent)` from [`crate::Speech`] -> 24 kHz waveform.
    ///
    /// The de-normalisation (`latents * std + mean`) is applied here on purpose: the reference
    /// does it at the `WoofNP.__call__` boundary, and a caller that forgets costs ~70 dB of parity.
    pub fn decode(
        &mut self,
        latents: &Array2<f32>,
        weights: &Weights,
        timings: &mut DecoderTimings,
    ) -> Result<Array1<f32>> {
        let started = std::time::Instant::now();
        let latents = self.denormalise(latents)?;
        // tile by the attention window: the full n x n pass is O(n^2) for no benefit. Measured on a
        // 573-frame utterance it is worth ~11% of the whole pipeline (11.5 s vs 12.9 s) -- the
        // speech model dominates, not the decoder
        let block = self.cfg.window.max(1) * 4;
        let h = self.transformer_tiled(&latents, weights, block)?;
        Self::dump("00_transformer", &h);
        timings.transformer = started.elapsed().as_secs_f32();

        for stage in self.stages.iter_mut() {
            stage.reset();
        }

        let head_started = std::time::Instant::now();
        let x = run_stages_dumped(h, &mut self.stages[..self.cfg.split], weights, 0)?;
        timings.head_stages = head_started.elapsed().as_secs_f32();

        let tail_started = std::time::Instant::now();
        let latent_rows = latents.nrows();
        let chunk = self.cfg.chunk * x.nrows() / latent_rows.max(1);
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < x.nrows() {
            let end = (start + chunk).min(x.nrows());
            let mut piece = x.slice(s![start..end, ..]).to_owned();
            piece = run_stages_dumped(
                piece,
                &mut self.stages[self.cfg.split..],
                weights,
                self.cfg.split,
            )?;
            pieces.push(piece);
            start = end;
        }
        timings.tail_stages = tail_started.elapsed().as_secs_f32();

        let total: usize = pieces.iter().map(|p| p.nrows()).sum();
        let mut out = Array1::<f32>::zeros(total);
        let mut at = 0;
        for piece in pieces {
            for row in 0..piece.nrows() {
                out[at] = piece[[row, 0]];
                at += 1;
            }
        }
        Ok(out)
    }

    /// `latents * std + mean`, per latent dimension.
    fn denormalise(&self, latents: &Array2<f32>) -> Result<Array2<f32>> {
        if latents.ncols() != self.latent_mean.len() {
            return Err(format!(
                "latents have {} columns, the decoder expects {}",
                latents.ncols(),
                self.latent_mean.len()
            )
            .into());
        }
        let mut out = latents.clone();
        for mut row in out.rows_mut() {
            for j in 0..row.len() {
                row[j] = row[j] * self.latent_std[j] + self.latent_mean[j];
            }
        }
        Ok(out)
    }

    /// Incremental decoding: feed latent frames as they are generated and get audio for the new
    /// frames back.
    ///
    /// Valid because every stage of the decoder is causal — the transformer's mask is causal and
    /// windowed, and each conv/convt carries its own state. The transformer is re-run over the
    /// accumulated latents and only the new rows are pushed through the SEANet stack, so the result
    /// matches whole-sequence decoding; the transformer's repeat cost is a few percent (it is ~8 %
    /// of decoder time and grows with the number of chunks).
    pub fn start_streaming(&mut self) -> StreamingDecoder<'_> {
        for stage in self.stages.iter_mut() {
            stage.reset();
        }
        let layers = self.layers.len();
        StreamingDecoder {
            decoder: self,
            caches: (0..layers).map(|_| LayerCache::default()).collect(),
            total: 0,
            emitted: 0,
            transformer_s: 0.0,
            stages_s: 0.0,
        }
    }

    /// Latent transformer over *new* rows only, reusing cached keys and values.
    ///
    /// This is what makes a small `--chunk-frames` cheap: each push runs the transformer over the
    /// frames it was given, not over everything decoded so far.
    fn transformer_cached(
        &self,
        latents: &Array2<f32>,
        weights: &Weights,
        start_pos: usize,
        caches: &mut [LayerCache],
    ) -> Result<Array2<f32>> {
        let mut h = latents.dot(&self.input.0) + &self.input.1;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            h = layer.run_cached(&h, weights, &self.cfg, &self.inv_freq, start_pos, cache)?;
        }
        Ok(h)
    }

    /// Latent transformer over a whole sequence, tiled so the cost stays linear.
    ///
    /// The attention mask is causal *and* windowed, so row `i` cannot see anything older than
    /// `i - window + 1`; running the transformer over the full sequence therefore computes an
    /// n x n score matrix whose older half is discarded. Tiling the sequence into window-sized
    /// blocks, each with its own overlap of `window - 1` history rows, gives the same rows at a
    /// fraction of the work (exactly the same rows up to float noise: the shapes handed to the
    /// matrix multiply differ, so the last bit of the result can too).
    fn transformer_tiled(
        &self,
        latents: &Array2<f32>,
        weights: &Weights,
        block: usize,
    ) -> Result<Array2<f32>> {
        let rows = latents.nrows();
        let window = self.cfg.window.max(1);
        let block = block.max(1).min(rows.max(1));
        let mut out = Array2::<f32>::zeros((rows, self.cfg.dim));
        let mut start = 0;
        while start < rows {
            let end = (start + block).min(rows);
            let keep = (window.saturating_sub(1)).min(start);
            let from = start - keep;
            let piece = latents.slice(s![from..end, ..]).to_owned();
            let hidden = self.transformer(&piece, weights, from)?;
            out.slice_mut(s![start..end, ..])
                .assign(&hidden.slice(s![keep.., ..]));
            start = end;
        }
        Ok(out)
    }

    /// Latent transformer over `latents`, whose first row is absolute position `start_pos`.
    fn transformer(
        &self,
        latents: &Array2<f32>,
        weights: &Weights,
        start_pos: usize,
    ) -> Result<Array2<f32>> {
        let mut h = latents.dot(&self.input.0) + &self.input.1;
        for layer in &self.layers {
            h = layer.run(&h, weights, &self.cfg, &self.inv_freq, start_pos)?;
        }
        Ok(h)
    }
}

pub struct StreamingDecoder<'a> {
    decoder: &'a mut Decoder,
    /// Keys and values per transformer layer, so a push transforms only its own frames.
    caches: Vec<LayerCache>,
    /// Absolute frames pushed so far: where in the sequence the next push starts (RoPE).
    total: usize,
    emitted: usize,
    /// wall time in the transformer and in the SEANet stages, so `--timings` can report streamed
    /// utterances as well as whole ones
    transformer_s: f32,
    stages_s: f32,
}

impl StreamingDecoder<'_> {
    /// Append `frames` latents and return the audio for them.
    ///
    /// Only the new rows go through the transformer, with keys and values from earlier pushes
    /// reused, and through the SEANet stages, which carry their own conv state. Measured over a
    /// 4 s sentence at 13 frames per chunk: the transformer is 15 ms of a 160 ms push, so the cache
    /// is worth ~10% here (2.2x -> 2.4x realtime); the other 145 ms is [`Conv::taps`] dequantising
    /// all 31M decoder weights on every push. That is the cost that makes small chunks expensive,
    /// and caching the f32 taps would cost 118 MiB of RSS — see the chunk-size section of the
    /// README for the trade.
    pub fn push(&mut self, frames: &Array2<f32>, weights: &Weights) -> Result<Array1<f32>> {
        if frames.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }
        let started = std::time::Instant::now();
        let prepared = self.decoder.denormalise(frames)?;
        let hidden =
            self.decoder
                .transformer_cached(&prepared, weights, self.total, &mut self.caches)?;
        self.transformer_s += started.elapsed().as_secs_f32();
        self.total += frames.nrows();

        let stages_started = std::time::Instant::now();
        let x = run_stages(hidden, &mut self.decoder.stages, weights)?;
        self.stages_s += stages_started.elapsed().as_secs_f32();
        self.emitted += frames.nrows();
        Ok(x.slice(s![.., 0]).to_owned())
    }

    /// Wall time spent in the transformer and in the stages so far.
    pub fn timings(&self) -> (f32, f32) {
        (self.transformer_s, self.stages_s)
    }

    /// Frames pushed so far (absolute, not just the retained tail).
    pub fn frames(&self) -> usize {
        self.total
    }
}
