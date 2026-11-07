//! One-stop synthesis: weights + speech model + decoder, streamed utterance by utterance.
//!
//! The reference synthesises the whole text before returning any audio (`Woof.__call__` splits,
//! then concatenates). For playback that is the wrong shape: a five-sentence paragraph means
//! seconds of silence. [`Voice::stream`] yields the waveform of each utterance as soon as it is
//! ready, which is what a player or an HTTP endpoint wants.

use ndarray::{Array1, Array2};

use crate::decoder::{Decoder, DecoderTimings};
use crate::quality::{self, Policy as RetryPolicy, Take};
use crate::textsplit::{self, Options as SplitOptions};
use crate::{audio, Mt19937, Result, Sampling, Speech, Weights};

/// Everything needed to turn text into audio.
pub struct Voice {
    speech: Speech,
    decoder: Decoder,
    sampling: Sampling,
    voice: String,
    retries: RetryPolicy,
    /// emit audio every N latent frames instead of per utterance (0 = per utterance)
    chunk_frames: usize,
}

/// One streamed utterance.
pub struct Utterance {
    pub text: String,
    /// raw latents the decoder consumed (frames x latent)
    pub latents: ndarray::Array2<f32>,
    pub samples: Array1<f32>,
    pub timings: DecoderTimings,
    /// seconds of audio in this utterance
    pub duration: f32,
    /// takes drawn for this utterance (1 = accepted first time)
    pub draws: usize,
    /// whether the kept take failed the guard
    pub bad: bool,
    /// wall time spent generating this utterance, all draws included
    pub generation_s: f32,
}

impl Voice {
    /// `cfg` is the manifest's `cfg` object.
    pub fn new(
        weights: &Weights,
        cfg: &serde_json::Value,
        voice: &str,
        sampling: Sampling,
    ) -> Result<Self> {
        Ok(Voice {
            speech: Speech::new(weights, &cfg["sp"], voice)?,
            decoder: Decoder::new(weights, cfg)?,
            sampling,
            voice: voice.to_string(),
            retries: RetryPolicy::default(),
            chunk_frames: 0,
        })
    }

    /// Emit audio every `frames` latent frames rather than once per utterance.
    ///
    /// Chunked emission forces a single take for the affected utterances: a take that has reached
    /// the listener cannot be redrawn, so the guard cannot apply. Use it for the first utterance
    /// (where latency is felt) and leave the rest to the guard.
    pub fn with_chunk_frames(mut self, frames: usize) -> Self {
        self.chunk_frames = frames;
        self
    }

    /// Set the streaming chunk size on a reused instance (the builder form consumes `self`).
    pub fn set_chunk_frames(&mut self, frames: usize) {
        self.chunk_frames = frames;
    }

    /// Override the take-selection policy (see [`crate::quality`]).
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retries = policy;
        self
    }

    /// Synthesise `text` as one utterance, drawing takes until one passes the guard or the policy's
    /// attempts are exhausted (the best take is kept).
    pub fn say(&mut self, text: &str, weights: &Weights, rng: &mut Mt19937) -> Result<Utterance> {
        self.say_at(text, weights, rng, 0)
    }

    /// Synthesise `text` and hand each batch of samples to `on_samples` as it is decoded.
    ///
    /// With `chunk_frames > 0` this is the low-latency path: audio leaves every N frames (~40 ms
    /// each) instead of after the whole utterance. Retries are skipped here by construction.
    pub fn say_progressive(
        &mut self,
        text: &str,
        weights: &Weights,
        rng: &mut Mt19937,
        index: usize,
        on_samples: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<Utterance> {
        let chunk = self.chunk_frames;
        if chunk == 0 {
            let utterance = self.say_at(text, weights, rng, index)?;
            on_samples(utterance.samples.as_slice().unwrap_or(&[]))?;
            return Ok(utterance);
        }

        let started = std::time::Instant::now();
        let ids = crate::audio::encode_text(text);

        // The decoder runs on its own thread, fed through a bounded queue, so the sampler generates
        // chunk k+1 while chunk k is being vocoded. Only valid where the take cannot be redrawn (the
        // first utterance, or attempts = 0): emitting audio while generating forfeits the guard, which
        // is exactly why the chunked path is single-take in the first place.
        //
        // Both channels are bounded: four chunks of latents in, four chunks of samples back (26 latent
        // frames is 1.04 s of audio, so ~4 s of pending work and ~200 KB of samples at most). Decoded
        // samples are drained on this thread between chunks, which keeps the sink on its own thread and
        // keeps emission in order.
        let (jobs_tx, jobs_rx) = std::sync::mpsc::sync_channel::<Option<Array2<f32>>>(4);
        let (out_tx, out_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(4);
        let mut collected: Vec<f32> = Vec::new();
        let speech = &mut self.speech;
        let decoder = &mut self.decoder;
        let sampling = self.sampling;

        let (latents, timings) =
            std::thread::scope(|scope| -> Result<(Array2<f32>, (f32, f32))> {
                let worker = scope.spawn(move || -> Result<(f32, f32)> {
                    let mut stream = decoder.start_streaming();
                    while let Ok(job) = jobs_rx.recv() {
                        let Some(frames) = job else { break };
                        let samples = stream.push(&frames, weights)?;
                        out_tx
                            .send(samples.as_slice().expect("dense samples").to_vec())
                            .map_err(|_| "the sampler stopped")?;
                    }
                    Ok(stream.timings())
                });

                // For the utterance the listener is actually waiting for, the first chunk is waited for, so
                // time-to-first-audio does not pay for the overlap. Every other utterance overlaps from its
                // first chunk: nobody is listening to its start yet, and waiting cost ~50 ms per utterance
                // of throughput (15 utterances: 7.1 s instead of 5.9 s on the bench corpus).
                let mut first = index == 0;
                let produced = speech.latents_streaming(&ids, sampling, rng, chunk, |frames| {
                    jobs_tx
                        .send(Some(frames.to_owned()))
                        .map_err(|_| "the decoder thread stopped")?;
                    if first {
                        first = false;
                        let samples = out_rx.recv().map_err(|_| "the decoder thread stopped")?;
                        collected.extend_from_slice(&samples);
                        on_samples(&samples)?;
                        return Ok(());
                    }
                    // otherwise take whatever finished while this chunk was being generated
                    while let Ok(samples) = out_rx.try_recv() {
                        collected.extend_from_slice(&samples);
                        on_samples(&samples)?;
                    }
                    Ok(())
                });
                // Always end the worker before propagating a sampling error: `thread::scope` joins at the
                // end of the closure, so returning early through `?` would leave the worker blocked in
                // recv() and hang. A text too long for the model's tables reaches exactly this path.
                jobs_tx.send(None).ok();
                let timings = worker.join().expect("decoder thread panicked")?;
                let latents = produced?;
                while let Ok(samples) = out_rx.try_recv() {
                    collected.extend_from_slice(&samples);
                    on_samples(&samples)?;
                }
                Ok((latents, timings))
            })?;
        let samples = Array1::from(collected);
        let (transformer, stages) = timings;
        Ok(Utterance {
            text: text.to_string(),
            duration: samples.len() as f32 / crate::audio::SAMPLE_RATE as f32,
            latents,
            samples,
            timings: DecoderTimings {
                transformer,
                head_stages: stages,
                tail_stages: 0.0,
            },
            draws: 1,
            bad: false,
            generation_s: started.elapsed().as_secs_f32(),
        })
    }

    /// As [`Voice::say`], with the utterance's index deciding whether the first-utterance exemption
    /// applies.
    pub fn say_at(
        &mut self,
        text: &str,
        weights: &Weights,
        rng: &mut Mt19937,
        index: usize,
    ) -> Result<Utterance> {
        let frame_cap = self.speech.frame_cap(text);
        let attempts = self.retries.attempts_for(index);
        let started = std::time::Instant::now();
        let (take, used): (Take, usize) = quality::best_take(
            || {
                let latents = self.speech.latents_for_text(text, self.sampling, rng)?;
                let mut timings = DecoderTimings::default();
                let samples = self.decoder.decode(&latents, weights, &mut timings)?;
                let mut take = Take::new(latents, samples, text, frame_cap);
                take.timings = timings;
                Ok(take)
            },
            attempts,
        )?;
        let generation_s = started.elapsed().as_secs_f32();
        if std::env::var("WOOF_TRACE").is_ok() {
            eprintln!(
                "utterance {index}: {} chars, attempts allowed {attempts}, draws {}, bad {}, {:.2}s audio",
                text.chars().count(),
                used + 1,
                take.is_bad(),
                take.samples.len() as f32 / audio::SAMPLE_RATE as f32
            );
        }
        Ok(Utterance {
            text: text.to_string(),
            latents: take.latents.clone(),
            duration: take.samples.len() as f32 / audio::SAMPLE_RATE as f32,
            samples: take.samples.clone(),
            timings: take.timings,
            draws: used + 1,
            bad: take.is_bad(),
            generation_s,
        })
    }

    /// Synthesise `text` utterance by utterance, yielding each as it completes.
    ///
    /// The returned closure-iterator borrows `self`, so a caller can stream into a socket or a
    /// player while the next sentence is still being computed.
    pub fn stream<'a>(
        &'a mut self,
        text: &str,
        weights: &'a Weights,
        rng: &'a mut Mt19937,
        options: SplitOptions,
    ) -> impl Iterator<Item = Result<Utterance>> + 'a {
        let pieces = textsplit::split_text(text, options);
        // the index matters: the retry policy may exempt the first utterance
        pieces
            .into_iter()
            .enumerate()
            .map(move |(index, piece)| self.say_at(&piece, weights, rng, index))
    }

    /// Switch voice in place (the per-voice state is a 28 KiB prefix).
    pub fn set_voice(&mut self, weights: &Weights, name: &str) -> Result<()> {
        let prefix = weights.voice_prefix(name)?;
        self.speech.set_voice(prefix)?;
        self.voice = name.to_string();
        Ok(())
    }

    /// The voice currently loaded.
    pub fn voice(&self) -> &str {
        &self.voice
    }

    /// Frames per emission (0 = one emission per utterance).
    pub fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }

    /// Synthesise `text` utterance by utterance, handing samples to `sink` as they become
    /// available, and returning what was synthesised.
    ///
    /// With `chunk_frames > 0` the **first** utterance is decoded incrementally, so audio starts
    /// after a few frames rather than after the whole sentence (and, necessarily, without the
    /// guard). Every later utterance keeps normal take selection. This is the single loop the CLI
    /// and the benchmark share, so they cannot drift apart.
    pub fn render(
        &mut self,
        text: &str,
        weights: &Weights,
        rng: &mut Mt19937,
        options: SplitOptions,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<Vec<Utterance>> {
        let pieces = textsplit::split_text(text, options);
        let mut out = Vec::with_capacity(pieces.len());
        for (index, piece) in pieces.iter().enumerate() {
            let progressive = index == 0 && self.chunk_frames > 0;
            let utterance = if progressive {
                self.say_progressive(piece, weights, rng, index, sink)?
            } else {
                let utterance = self.say_at(piece, weights, rng, index)?;
                sink(utterance.samples.as_slice().unwrap_or(&[]))?;
                utterance
            };
            out.push(utterance);
        }
        Ok(out)
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_yields_one_utterance_per_sentence() {
        // only the splitting half is unit-testable without weights
        let pieces = textsplit::split_text("One. Two. Three.", SplitOptions::default());
        assert_eq!(pieces.len(), 3);
    }
}
