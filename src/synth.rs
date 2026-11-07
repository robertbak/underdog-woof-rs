//! Synthesis plumbing for the command line: where the audio goes, on what thread, and what happens
//! when a reader hangs up.
//!
//! The model lives in the library; this is the part that turns utterances into a file, a pipe or a
//! player's stdin, and keeps the producer ahead of whatever is consuming it.

use woof_rs::{audio, Mt19937, Result, Weights};

use crate::cli::Args;

/// Let a closed reader end the process the way it ends `yes | head`: silently, with 141.
///
/// Rust ignores SIGPIPE, so `woof-rs ... --pcm-stdout | head -c 100` would otherwise print
/// "error: Broken pipe" after the reader has *succeeded* in what it asked for. Only the command
/// line does this -- `--serve` keeps the default disposition so a client hanging up mid-response
/// cannot kill the server.
/// Let a player downstream run without starving: the default pipe holds 64 KiB, which is 1.4 s of
/// 24 kHz mono, so a sentence that takes longer than that to synthesise arrives too late and the
/// player falls silent between lines. 1 MiB (~22 s) lets the model stay several sentences ahead,
/// while the player's own buffer still decides playback latency.
///
/// Best effort: the limit is `/proc/sys/fs/pipe-max-size` (1 MiB by default) and raising it needs
/// privileges, so failure is expected on constrained kernels and changes nothing.
pub fn widen_stdout_pipe() {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::fcntl(1, libc::F_SETPIPE_SZ, 1 << 20);
    }
}

pub fn die_on_sigpipe() {
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// Fill `--text` from stdin when it was omitted (or given as `-`) and stdin is a pipe.
///
/// Only when something is actually piped: a bare `woof-rs` on a terminal should say `--text is
/// required`, not hang waiting for EOF.
pub fn read_piped_text(args: &mut Args) -> Result<()> {
    let wants = args.text.trim().is_empty() || args.text.trim() == "-";
    let wants_normalise = args.normalise_text.as_deref() == Some("-");
    if !wants && !wants_normalise {
        return Ok(());
    }
    #[cfg(unix)]
    let piped = unsafe { libc::isatty(0) } == 0;
    #[cfg(not(unix))]
    let piped = false;
    if !piped {
        return Ok(());
    }
    let mut text = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
    // a trailing newline from echo/cat is not part of the sentence
    let text = text.trim_end_matches(['\n', '\r']).to_string();
    if args.max_chars > 0 && text.chars().count() > args.max_chars {
        return Err(format!(
            "stdin holds {} characters; --max-chars allows {}. Text is split into sentences and \
             streamed one at a time, so raise it (or --max-chars 0) if you meant to synthesise \
             all of it",
            text.chars().count(),
            args.max_chars
        )
        .into());
    }
    if wants_normalise {
        args.normalise_text = Some(text);
    } else {
        args.text = text;
    }
    Ok(())
}

/// "1 utterance" / "3 utterances", because "(s)" in user-facing output is a smell.
fn plural(n: usize, singular: &str) -> String {
    if n == 1 {
        format!("{n} {singular}")
    } else {
        format!("{n} {singular}s")
    }
}

/// How audio leaves the process.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// buffer everything, write one WAV at the end
    Whole,
    /// write each utterance as it lands
    Stream,
    /// raw PCM on stdout, for a player or a pipe
    StreamPcm,
}

/// The one synthesis path. The reference splits its input into utterances and concatenates the
/// results (`Woof.__call__`), so this does too; `--stream` writes each utterance (or, with
/// `--chunk-frames`, each batch of frames) as it arrives instead of buffering.
pub fn synthesise(
    args: &Args,
    weights: &Weights,
    manifest: &serde_json::Value,
    output: Output,
) -> Result<()> {
    use woof_rs::textsplit;

    let started = std::time::Instant::now();
    let mut voice = woof_rs::Voice::new(weights, &manifest["cfg"], &args.voice, args.sampling)?
        .with_retry_policy(woof_rs::quality::Policy {
            attempts: args.retries,
            first_utterance_single_take: args.no_retry_first,
        })
        .with_chunk_frames(args.chunk_frames);
    let mut rng = Mt19937::new(args.seed);

    let options = textsplit::Options {
        clause_max: textsplit::CLAUSE_MAX,
        first_max: args.first_max,
    };
    // `Woof(normalize=True)` normalises first, then splits: numbers, dates and codes become words
    // the model can read. `--no-normalise` reproduces `normalize=False`.
    if args.max_chars > 0 && args.text.chars().count() > args.max_chars {
        return Err(format!(
            "--text holds {} characters; --max-chars allows {}",
            args.text.chars().count(),
            args.max_chars
        )
        .into());
    }
    let text = if args.no_normalise {
        args.text.clone()
    } else {
        woof_rs::normalise::normalise(&args.text)
    };
    let pieces = textsplit::split_text(&text, options).len();
    // A buffered --out holds every sample in memory: 96 KiB per second of audio. Long text is
    // fine streamed and wasteful buffered, so mention it before spending minutes synthesising.
    if output == Output::Whole && text.chars().count() > 20_000 {
        let seconds = text.chars().count() as f32 / 15.0;
        eprintln!(
            "woof: note: --out buffers all {:.0} min of audio (~{:.0} MiB); --stream or \
             --pcm-stdout writes it as it is made",
            seconds / 60.0,
            seconds * 96.0 / 1024.0
        );
    }

    let mut wav = if output == Output::Stream {
        Some(audio::StreamingWav::create(&args.out)?)
    } else {
        None
    };
    let mut stdout = std::io::stdout();
    let mut buffered: Vec<f32> = Vec::new();
    let mut latents: Vec<f32> = Vec::new();
    let (mut audio_seconds, mut first_audio) = (0.0f32, None::<f32>);

    // Streaming synthesises on one thread and writes on another. Without it the next utterance is
    // started only after the previous one has been played out, so every boundary costs whatever the
    // model needs to produce its first frame (0.3-0.7 s at one thread) -- heard as a pause between
    // lines. With it the model runs ahead into a bounded queue and the player never starves, as long
    // as average synthesis beats playback (1.7x realtime at one thread, 2.9x at fourteen).
    const MESSAGE_SAMPLES: usize = 24000 / 4; // 0.25 s per handoff: fine-grained enough to start fast
    const AHEAD_MESSAGES: usize = 64; // the model may run 16 s ahead; ~1.5 MiB

    // only the buffered mode uses this; streaming has its own producer sink below
    let mut sink = |samples: &[f32]| -> Result<()> {
        buffered.extend_from_slice(samples);
        if first_audio.is_none() && !samples.is_empty() {
            first_audio = Some(started.elapsed().as_secs_f32());
            if args.verbose {
                eprintln!(
                    "first audio after {:.2}s ({} to synthesise)",
                    first_audio.unwrap_or_default(),
                    plural(pieces, "utterance")
                );
            }
        }
        Ok(())
    };

    // one loop for both modes, shared with the benchmark
    let rendered = if output == Output::Whole {
        voice.render(&text, weights, &mut rng, options, &mut sink)?
    } else {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(AHEAD_MESSAGES);
        std::thread::scope(|scope| -> Result<Vec<woof_rs::voice::Utterance>> {
            let producer = scope.spawn(move || -> Result<Vec<woof_rs::voice::Utterance>> {
                let mut pending: Vec<f32> = Vec::with_capacity(2 * MESSAGE_SAMPLES);
                let mut push = |samples: &[f32]| -> Result<()> {
                    pending.extend_from_slice(samples);
                    while pending.len() >= MESSAGE_SAMPLES {
                        let rest = pending.split_off(MESSAGE_SAMPLES);
                        tx.send(std::mem::replace(&mut pending, rest))
                            .map_err(|_| "the writer stopped")?;
                    }
                    Ok(())
                };
                let rendered = voice.render(&text, weights, &mut rng, options, &mut push)?;
                if !pending.is_empty() {
                    tx.send(pending).ok(); // the writer may already have gone away
                }
                Ok(rendered)
            });
            let mut emitted = 0.0f32;
            for samples in rx {
                if first_audio.is_none() {
                    first_audio = Some(started.elapsed().as_secs_f32());
                    if args.verbose {
                        eprintln!(
                            "first audio after {:.2}s ({} to synthesise)",
                            first_audio.unwrap_or_default(),
                            plural(pieces, "utterance")
                        );
                    }
                }
                if args.verbose {
                    emitted += samples.len() as f32 / 24_000.0;
                    eprintln!(
                        "emit {:6.2}s audio at {:6.2}s wall (+{:.2}s)",
                        emitted,
                        started.elapsed().as_secs_f32(),
                        samples.len() as f32 / 24_000.0
                    );
                }
                match output {
                    Output::Stream => wav.as_mut().expect("wav writer").write(&samples)?,
                    _ => {
                        use std::io::Write as _;
                        stdout.write_all(&audio::pcm16(&samples))?;
                        stdout.flush()?;
                    }
                }
            }
            producer.join().expect("synthesis thread panicked")
        })?
    };
    let utterances = rendered.len();
    for (index, utterance) in rendered.iter().enumerate() {
        audio_seconds += utterance.duration;
        latents.extend(utterance.latents.iter());
        if args.timings {
            let decoder = utterance.timings.transformer
                + utterance.timings.head_stages
                + utterance.timings.tail_stages;
            eprintln!(
                "utterance {}: {:>4} chars {:>4} frames {:>6.2}s audio | speech {:>6.2}s decoder {:>6.2}s (transformer {:.2} head {:.2} tail {:.2}) | {} take(s){}",
                index + 1,
                utterance.text.chars().count(),
                utterance.latents.nrows(),
                utterance.duration,
                (utterance.generation_s - decoder).max(0.0),
                decoder,
                utterance.timings.transformer,
                utterance.timings.head_stages,
                utterance.timings.tail_stages,
                utterance.draws,
                if utterance.bad { " (guard: bad take)" } else { "" },
            );
        }
    }

    if output == Output::Whole {
        let samples = woof_rs::ndarray_compat::Array1::from(buffered);
        audio::write_wav(&args.out, samples.as_slice().expect("contiguous samples"))?;
    }
    if let Some(wav) = wav {
        wav.finish()?;
    }
    if let Some(path) = &args.dump_latents {
        let mut bytes = Vec::with_capacity(latents.len() * 4);
        for value in &latents {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
    }

    if args.verbose {
        eprintln!(
            "{}, {:.2}s of audio, {:.2}s wall ({:.1}x realtime)",
            plural(utterances, "utterance"),
            audio_seconds,
            started.elapsed().as_secs_f32(),
            audio_seconds / started.elapsed().as_secs_f32().max(1e-6)
        );
    }
    if output != Output::StreamPcm && args.verbose {
        eprintln!("wrote {}", args.out.display());
    }
    Ok(())
}
