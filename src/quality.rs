//! Take selection: the reference's guard against bad draws, and the knob that trades it for latency.
//!
//! `Woof._piece` draws a take, looks at the audio and the text, and redraws up to `retries` times if
//! the take is bad — keeping the best of them. "Bad" means any of:
//!
//! * a run of quiet 40 ms frames at least `stall_s` long between the first and last voiced frame,
//! * more than `max_spc` seconds of audio per input character (a runaway draw),
//! * hitting the frame cap (the draw was truncated).
//!
//! Takes are scored `(bad, run, spc)` and compared lexicographically, so a clean take always wins.
//!
//! This guard needs the whole utterance before it can decide, which is why audio is emitted per
//! sentence rather than per frame: you cannot redraw audio that has already been played. Callers
//! that care more about latency than about the occasional bad take can set [`Policy::attempts`] to
//! 0 for the first utterance — see [`Policy::first_utterance_single_take`].

use ndarray::{Array1, Array2};

use crate::audio::SAMPLE_RATE;

/// Frame hop for the silence scan: 960 samples = 40 ms at 24 kHz.
const HOP: usize = 960;
/// Frames quieter than this count as silence.
const QUIET_DBFS: f32 = -50.0;
/// A silence run at least this long makes a take bad.
pub const STALL_S: f32 = 1.0;
/// More than this many seconds of audio per input character makes a take bad.
pub const MAX_SPC: f32 = 0.095;

/// How many takes to draw, and whether the first utterance is exempt.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    /// Extra draws beyond the first. `0` = single take, which bounds latency at the cost of
    /// shipping the occasional bad draw.
    pub attempts: usize,
    /// Stream the first utterance as a single take regardless of [`Policy::attempts`]: it is the one
    /// the listener is waiting for, so its worst case becomes 1× rather than (`attempts`+1)×.
    pub first_utterance_single_take: bool,
}

impl Default for Policy {
    fn default() -> Self {
        // the reference's `Woof(retries=3)`, with no exemption
        Policy {
            attempts: 3,
            first_utterance_single_take: false,
        }
    }
}

impl Policy {
    /// Attempts to use for the utterance at `index`.
    pub fn attempts_for(&self, index: usize) -> usize {
        if index == 0 && self.first_utterance_single_take {
            0
        } else {
            self.attempts
        }
    }
}

/// Longest run of quiet frames, in seconds, strictly between the first and last voiced frame.
pub fn stall_seconds(samples: &[f32]) -> f32 {
    let frames = samples.len() / HOP;
    if frames < 3 {
        return 0.0;
    }
    let quiet: Vec<bool> = (0..frames)
        .map(|i| {
            let frame = &samples[i * HOP..(i + 1) * HOP];
            let rms = (frame.iter().map(|v| v * v).sum::<f32>() / HOP as f32).sqrt();
            20.0 * (rms + 1e-9).log10() < QUIET_DBFS
        })
        .collect();
    let voiced: Vec<usize> = quiet
        .iter()
        .enumerate()
        .filter(|(_, q)| !**q)
        .map(|(i, _)| i)
        .collect();
    if voiced.len() < 2 {
        return 0.0;
    }
    let (first, last) = (voiced[0], voiced[voiced.len() - 1]);
    let mut best = 0usize;
    let mut run = 0usize;
    for &is_quiet in &quiet[first..=last] {
        run = if is_quiet { run + 1 } else { 0 };
        best = best.max(run);
    }
    best as f32 * HOP as f32 / SAMPLE_RATE as f32
}

/// One synthsis draw, with the measurements the guard uses.
pub struct Take {
    pub latents: Array2<f32>,
    pub samples: Array1<f32>,
    pub stall_s: f32,
    pub seconds_per_char: f32,
    pub truncated: bool,
    /// where the decoder spent this take's time, carried through so `--timings` can report it
    pub timings: crate::decoder::DecoderTimings,
}

impl Take {
    pub fn new(latents: Array2<f32>, samples: Array1<f32>, text: &str, frame_cap: usize) -> Self {
        let stall = stall_seconds(samples.as_slice().unwrap_or(&[]));
        let seconds_per_char =
            samples.len() as f32 / SAMPLE_RATE as f32 / text.chars().count().max(1) as f32;
        Take {
            truncated: latents.nrows() >= frame_cap,
            latents,
            samples,
            stall_s: stall,
            seconds_per_char,
            timings: crate::decoder::DecoderTimings::default(),
        }
    }

    pub fn is_bad(&self) -> bool {
        self.stall_s >= STALL_S || self.seconds_per_char > MAX_SPC || self.truncated
    }

    /// Lexicographic `(bad, run, spc)`, exactly as the reference compares them.
    fn score(&self) -> (bool, f32, f32) {
        (self.is_bad(), self.stall_s, self.seconds_per_char)
    }
}

/// Draw up to `attempts + 1` takes and keep the best, stopping early on a clean one.
///
/// `draw` is called afresh for each attempt and is expected to advance the sampler, so successive
/// takes differ — the reference relies on the same behaviour.
pub fn best_take(
    mut draw: impl FnMut() -> crate::Result<Take>,
    attempts: usize,
) -> crate::Result<(Take, usize)> {
    let mut best: Option<Take> = None;
    for attempt in 0..=attempts {
        let take = draw()?;
        let clean = !take.is_bad();
        let better = match &best {
            None => true,
            Some(current) => take.score() < current.score(),
        };
        if better {
            best = Some(take);
        }
        if clean {
            return Ok((best.expect("set above"), attempt));
        }
    }
    Ok((best.expect("at least one attempt"), attempts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(seconds: f32) -> Vec<f32> {
        vec![0.0; (seconds * SAMPLE_RATE as f32) as usize]
    }

    fn tone(seconds: f32) -> Vec<f32> {
        (0..(seconds * SAMPLE_RATE as f32) as usize)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect()
    }

    #[test]
    fn stall_measures_the_longest_internal_silence() {
        let mut wav = tone(0.5);
        wav.extend(silence(1.2));
        wav.extend(tone(0.5));
        let run = stall_seconds(&wav);
        assert!((run - 1.2).abs() < 0.08, "measured {run}");
    }

    #[test]
    fn stall_ignores_leading_and_trailing_silence() {
        let mut wav = silence(2.0);
        wav.extend(tone(0.5));
        wav.extend(silence(2.0));
        assert_eq!(stall_seconds(&wav), 0.0);
    }

    #[test]
    fn policy_exempts_only_the_first_utterance() {
        let policy = Policy {
            attempts: 3,
            first_utterance_single_take: true,
        };
        assert_eq!(policy.attempts_for(0), 0);
        assert_eq!(policy.attempts_for(1), 3);
        assert_eq!(Policy::default().attempts_for(0), 3);
    }

    #[test]
    fn best_take_stops_at_the_first_clean_draw() {
        let mut calls = 0;
        let (_, used) = best_take(
            || {
                calls += 1;
                Ok(Take {
                    latents: Array2::<f32>::zeros((4, 32)),
                    samples: Array1::from(tone(1.0)),
                    stall_s: 0.0,
                    seconds_per_char: 0.01,
                    truncated: false,
                    timings: crate::decoder::DecoderTimings::default(),
                })
            },
            3,
        )
        .unwrap();
        assert_eq!(used, 0);
        assert_eq!(calls, 1);
    }
}
