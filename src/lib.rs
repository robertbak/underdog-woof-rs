//! Underdog Woof, ported to Rust.
//!
//! A line-by-line port of the numpy runtime (`underdog_woof/runtime.py`) that keeps the model's
//! int8 weights quantised in memory: the reference dequantises everything to f32 at load, which is
//! where its ~205 MB of resident weights come from, while the quantised form is ~39 MB.
//!
//! Layout:
//!
//! * [`weights`] — manifest/blob loading and the two quantisation layouts (`Linear`, `GroupedInt8`)
//! * [`nn`] — the numerical primitives, including the reference's exact constants
//! * [`speech`] — text/voice prefix, incremental transformer step, flow head, sampling loop
//! * [`decoder`] — latent transformer plus the SEANet stack, in the reference's two-phase shape
//! * [`backend`] — a `MatMul` trait so the batched stages can run on CPU or GPU
//! * [`gpu`] — wgpu back-end (feature `gpu`), verified `A · B` kernel and epilogues
//! * [`cpu`] — topology detection and the pinned thread pool (big.LITTLE matters a great deal here)
//! * [`rng`] — the torch-compatible MT19937 + Box-Muller sampler
//! * [`audio`] — vocabulary, tokenisation, WAV output
//! * [`normalise`] — raw text -> spoken form (staged port of the reference's rules)
//! * [`textsplit`] — sentence/clause splitting so audio can start before the text ends
//! * [`bench`] — repeatable latency/throughput/guard benchmark corpus
//! * [`quality`] — the reference's bad-take guard (`stall`/`spc`/cap) and the retry policy
//! * [`voice`] — `Voice`: weights + models + streaming synthesis
//! * [`serve`] — `--serve`: the OpenAI-shaped HTTP server, in the binary

pub mod audio;
pub mod backend;
pub mod bench;
pub mod cpu;
pub mod decoder;
pub mod nn;
pub mod normalise;
pub mod profile;
pub mod quality;
pub mod rng;
pub mod serve;
pub mod speech;
pub mod textsplit;
pub mod voice;
pub mod weights;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use decoder::Decoder;
/// Re-exported so the binary can build arrays without depending on `ndarray` directly.
pub mod ndarray_compat {
    pub use ndarray::{Array1, Array2};
}

pub use rng::Mt19937;
pub use speech::{Sampling, Speech};
pub use voice::{Utterance, Voice};
pub use weights::Weights;

/// Default limit on one HTTP request: ~10 minutes of speech. Nothing in the synthesiser needs it --
/// text is split into sentences and streamed one at a time -- so it is an endpoint guard, and
/// `--max-chars` overrides it (0 = no limit on the command line; the server falls back to this).
pub const MAX_TEXT_CHARS: usize = 4000;

/// Crate-wide result type: boxed errors keep the call sites readable.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
