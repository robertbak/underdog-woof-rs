//! Resident-chain decoder: the SEANet stack runs on the device with weights cached and activations
//! never returning to the host between stages.
//!
//! # Why
//!
//! The per-op path ([`Gpu`] as a [`crate::backend::MatMul`]) uploads operands, dispatches, copies
//! the result back and blocks on `map_async` for *every* matmul. Measured on a Tab S9
//! (Adreno 740 / Vulkan): decoder **2.12 s** against **0.58 s** on the CPU — the decoder is ~10
//! sequential stages, and each round trip costs ~0.5 ms to do ~20 µs of work.
//!
//! # Division of labour
//!
//! The latent transformer stays on the host (105 × 512 ≈ 30 ms); batching it per frame would be
//! thousands of tiny dispatches, the same trap the speech model avoids. The device gets the
//! SEANet stages, which are 91 % of decoder time and are properly batched.
//!
//! # Verification
//!
//! [`ResidentDecoder::verify`] runs each stage twice from the same input — device, and
//! [`crate::decoder::Decoder::run_stage`] on the CPU — and reports the SNR per stage. It continues
//! from the CPU result so one bad stage cannot smear into the next, which makes the first
//! divergence unambiguous.

use ndarray::{Array1, Array2};

use crate::decoder::conv::{Conv, ConvT, Stage};
use crate::decoder::Decoder;
use crate::gpu::{ConvT as ConvTDispatch, Gemm, Gpu};
use crate::weights::Weights;
use crate::Result;

/// A convolution's device resources: weights expanded once, state kept across chunks.
struct ConvDevice {
    /// one `(Cin, Cout)` matrix per tap. Measured on a Tab S9: accumulating taps as separate
    /// GEMMs (0.80 s) beats gathering them into one via im2col (0.91 s) — the gather traffic
    /// costs more than the dispatches it saves on a bandwidth-bound mobile GPU.
    taps: Vec<wgpu::Buffer>,
    cin: usize,
    cout: usize,
    bias: wgpu::Buffer,
    state: wgpu::Buffer,
}

impl ConvDevice {
    fn new(gpu: &Gpu, key: &str, conv: &Conv, weights: &Weights) -> Result<Self> {
        let per_tap = conv.taps(weights)?;
        let (cin, cout, taps) = (conv.cin, conv.cout, conv.taps);
        Ok(ConvDevice {
            taps: per_tap
                .iter()
                .enumerate()
                .map(|(j, tap)| gpu.upload_cached(&format!("{key}.tap{j}"), tap))
                .collect(),
            cin,
            cout,
            bias: gpu.upload_cached_slice(
                &format!("{key}.b"),
                conv.bias.as_slice().expect("dense bias"),
            ),
            state: gpu.alloc((taps - 1).max(1) * cin),
        })
    }

    fn reset(&self, gpu: &Gpu) {
        gpu.fill(&self.state, 0.0);
    }

    /// `out[t] = bias + (residual) + Σ_j x[t + j - (k-1)] · tap[j]`
    fn run(
        &self,
        gpu: &Gpu,
        x: &wgpu::Buffer,
        rows: usize,
        residual: Option<&wgpu::Buffer>,
    ) -> (wgpu::Buffer, usize) {
        let history = self.taps.len() - 1;
        let extended = gpu.alloc((history + rows) * self.cin);
        if history > 0 {
            gpu.copy(&self.state, 0, &extended, 0, history * self.cin);
        }
        gpu.copy(x, 0, &extended, history * self.cin, rows * self.cin);
        if history > 0 {
            // this chunk's tail becomes the next chunk's history
            gpu.copy(
                &extended,
                rows * self.cin,
                &self.state,
                0,
                history * self.cin,
            );
        }

        let out = gpu.alloc(rows * self.cout);
        gpu.seed(&out, &self.bias, residual, rows, self.cout);
        for (j, tap) in self.taps.iter().enumerate() {
            gpu.gemm(
                &extended,
                tap,
                &out,
                Gemm::new(rows, self.cin, self.cout).offset(j).accumulate(),
            );
        }
        (out, self.cout)
    }
}

/// A transposed convolution's device resources.
struct ConvTDevice {
    matrix: wgpu::Buffer,
    bias: wgpu::Buffer,
    carry: wgpu::Buffer,
    stride: usize,
    cout: usize,
    width: usize, // k*Cout with k = 2*stride
}

impl ConvTDevice {
    fn new(gpu: &Gpu, key: &str, convt: &ConvT, weights: &Weights) -> Result<Self> {
        let matrix = convt.matrix(weights)?;
        Ok(ConvTDevice {
            matrix: gpu.upload_cached(&format!("{key}.w"), &matrix),
            bias: gpu.upload_cached_slice(
                &format!("{key}.b"),
                convt.bias.as_slice().expect("dense bias"),
            ),
            carry: gpu.alloc(convt.bias.len()),
            stride: convt.stride,
            cout: convt.cout,
            width: matrix.ncols(),
        })
    }

    fn reset(&self, gpu: &Gpu) {
        gpu.fill(&self.carry, 0.0);
    }

    fn run(&self, gpu: &Gpu, x: &wgpu::Buffer, rows: usize, cols: usize) -> (wgpu::Buffer, usize) {
        let z = gpu.alloc(rows * self.width);
        gpu.gemm(x, &self.matrix, &z, Gemm::new(rows, cols, self.width));
        let out = gpu.alloc(rows * self.stride * self.cout);
        gpu.convt(
            &z,
            &self.bias,
            &self.carry,
            &out,
            ConvTDispatch::new(rows, self.cout, self.stride),
        );
        (out, self.cout)
    }
}

enum DeviceStage {
    Conv(ConvDevice),
    ConvT(ConvTDevice),
    Res(ConvDevice, ConvDevice),
    Elu,
}

impl DeviceStage {
    fn new(gpu: &Gpu, index: usize, stage: &Stage, weights: &Weights) -> Result<Self> {
        Ok(match stage {
            Stage::Elu => DeviceStage::Elu,
            Stage::Conv(conv) => {
                DeviceStage::Conv(ConvDevice::new(gpu, &format!("s{index}"), conv, weights)?)
            }
            Stage::ConvT(convt) => {
                DeviceStage::ConvT(ConvTDevice::new(gpu, &format!("s{index}"), convt, weights)?)
            }
            Stage::Res(a, b) => DeviceStage::Res(
                ConvDevice::new(gpu, &format!("s{index}a"), a, weights)?,
                ConvDevice::new(gpu, &format!("s{index}b"), b, weights)?,
            ),
        })
    }

    fn reset(&self, gpu: &Gpu) {
        match self {
            DeviceStage::Conv(c) => c.reset(gpu),
            DeviceStage::ConvT(c) => c.reset(gpu),
            DeviceStage::Res(a, b) => {
                a.reset(gpu);
                b.reset(gpu);
            }
            DeviceStage::Elu => {}
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            DeviceStage::Conv(_) => "conv",
            DeviceStage::ConvT(_) => "convt",
            DeviceStage::Res(..) => "res",
            DeviceStage::Elu => "elu",
        }
    }

    /// Mirrors one iteration of [`crate::decoder::run_stages`].
    fn run(
        &self,
        gpu: &Gpu,
        x: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) -> (wgpu::Buffer, usize, usize) {
        match self {
            DeviceStage::Elu => {
                let out = gpu.alloc(rows * cols);
                gpu.elu(x, &out, rows * cols);
                (out, rows, cols)
            }
            DeviceStage::Conv(conv) => {
                let (out, cout) = conv.run(gpu, x, rows, None);
                (out, rows, cout)
            }
            DeviceStage::ConvT(convt) => {
                let (out, cout) = convt.run(gpu, x, rows, cols);
                (out, rows * convt.stride, cout)
            }
            DeviceStage::Res(a, b) => {
                let elu = gpu.alloc(rows * cols);
                gpu.elu(x, &elu, rows * cols);
                let (first, c1) = a.run(gpu, &elu, rows, None);
                let elu2 = gpu.alloc(rows * c1);
                gpu.elu(&first, &elu2, rows * c1);
                let (second, c2) = b.run(gpu, &elu2, rows, Some(x));
                (second, rows, c2)
            }
        }
    }
}

/// Runs a decoder's SEANet stages on the device.
pub struct ResidentDecoder<'a> {
    gpu: &'a Gpu,
    stages: Vec<DeviceStage>,
}

impl<'a> ResidentDecoder<'a> {
    pub fn new(gpu: &'a Gpu, decoder: &Decoder, weights: &Weights) -> Result<Self> {
        let stages = decoder
            .stages()
            .iter()
            .enumerate()
            .map(|(i, stage)| DeviceStage::new(gpu, i, stage, weights))
            .collect::<Result<Vec<_>>>()?;
        Ok(ResidentDecoder { gpu, stages })
    }

    /// Latents in, waveform out: two host transfers for the whole chain.
    pub fn decode(
        &mut self,
        decoder: &mut Decoder,
        latents: &Array2<f32>,
        weights: &Weights,
    ) -> Result<Array1<f32>> {
        let cfg = decoder.config();
        if latents.ncols() != decoder.latent_dim() {
            return Err(format!(
                "latents have {} columns, the decoder expects {}",
                latents.ncols(),
                decoder.latent_dim()
            )
            .into());
        }
        decoder.reset_states();
        for stage in &self.stages {
            stage.reset(self.gpu);
        }

        // host half: de-normalisation + latent transformer (~30 ms)
        let h = decoder.prepare(latents, weights)?;
        // one submit for the whole chain; per-op submits are several ms each on Android
        let batch = std::env::var("WOOF_NO_BATCH").is_err();
        if batch {
            self.gpu.begin_batch();
        }
        let latent_rows = latents.nrows();
        let mut x = self.gpu.upload(&h);
        let (mut rows, mut cols) = (h.nrows(), h.ncols());
        for stage in &self.stages[..cfg.split] {
            let (out, r, c) = stage.run(self.gpu, &x, rows, cols);
            x = out;
            rows = r;
            cols = c;
        }

        // device half: high-rate stages in row chunks, state carried across chunk boundaries
        let chunk = (cfg.chunk * rows / latent_rows.max(1)).max(1);
        let mut pieces: Vec<(wgpu::Buffer, usize, usize)> = Vec::new();
        let mut start = 0;
        while start < rows {
            let end = (start + chunk).min(rows);
            let count = end - start;
            let mut current = self.gpu.alloc(count * cols);
            self.gpu.copy(&x, start * cols, &current, 0, count * cols);
            let (mut this_rows, mut this_cols) = (count, cols);
            for stage in &self.stages[cfg.split..] {
                let (out, r, c) = stage.run(self.gpu, &current, this_rows, this_cols);
                current = out;
                this_rows = r;
                this_cols = c;
            }
            pieces.push((current, this_rows, this_cols));
            start = end;
        }
        if batch {
            self.gpu.end_batch();
        }
        let pieces: Vec<Array2<f32>> = pieces
            .iter()
            .map(|(buffer, rows, cols)| self.gpu.download(buffer, *rows, *cols))
            .collect();

        let total: usize = pieces.iter().map(|p| p.nrows()).sum();
        let mut out = vec![0f32; total];
        let mut at = 0;
        for piece in pieces {
            for row in 0..piece.nrows() {
                out[at] = piece[[row, 0]];
                at += 1;
            }
        }
        Ok(Array1::from(out))
    }

    /// Per-stage parity check against the CPU runner, reporting `(stage, kind, snr_db)`.
    pub fn verify(
        &mut self,
        decoder: &mut Decoder,
        latents: &Array2<f32>,
        weights: &Weights,
    ) -> Result<Vec<(usize, &'static str, f32)>> {
        let mut report = Vec::new();
        let mut reference = decoder.prepare(latents, weights)?;

        for (index, stage) in self.stages.iter().enumerate() {
            stage.reset(self.gpu);
            let input = self.gpu.upload(&reference);
            let (device_buf, rows, cols) =
                stage.run(self.gpu, &input, reference.nrows(), reference.ncols());
            let device = self.gpu.download(&device_buf, rows, cols);

            decoder.reset_states();
            let cpu = decoder.run_stage(index, reference.clone(), weights)?;
            report.push((index, stage.kind(), snr(&cpu, &device)));

            // continue from the CPU result: a bad stage cannot contaminate the next comparison
            reference = cpu;
        }
        Ok(report)
    }
}

/// SNR of `got` against `want` in dB (both must share a shape).
fn snr(want: &Array2<f32>, got: &Array2<f32>) -> f32 {
    if want.dim() != got.dim() {
        return f32::NAN;
    }
    let noise: f32 = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    let signal: f32 = want.iter().map(|v| v * v).sum::<f32>().sqrt();
    20.0 * (signal / noise.max(1e-20)).log10()
}

#[allow(dead_code)]
fn _keep_array1_import(_: &Array1<f32>) {}
