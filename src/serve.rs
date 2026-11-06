//! `--serve`: an OpenAI-shaped HTTP server, in the binary.
//!
//! The Python wrapper existed only for HTTP framing — normalisation, synthesis and streaming all
//! live here now — so this replaces it, and is the only shape that ships on Android, where there is
//! no Python runtime.
//!
//! HTTP is `tiny_http`, not a hand-rolled parser and not an async stack: synthesis is CPU-bound and
//! serialised behind one lock, so nothing is waiting on IO and `tokio` would buy only binary size.
//! `tiny_http` also gives correct chunked transfer encoding, which is exactly what streaming needs.
//!
//! Endpoints, matching what llama-swap and existing clients expect:
//!
//! ```text
//! GET  /                    identity JSON (llama-swap preloads by probing this)
//! GET  /health              200 "ok"          (llama-swap's checkEndpoint)
//! GET  /v1/models           {"data":[{"id":"juno"},…]}
//! GET  /props               voices, sample rate, backend
//! POST /v1/audio/speech     audio/wav, or chunked audio/pcm when "stream": true
//! ```
//!
//! `POST /v1/audio/speech` takes the OpenAI speech fields plus two of ours:
//! `{"model":"juno","input":"…","voice":"juno"|"mira","response_format":"wav"|"pcm",
//!   "stream":false,"seed":0,"chunk_frames":13,"speed":1.0}`.

use std::io::Read;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

use tiny_http::{Header, Response, Server, StatusCode};

use crate::{audio, textsplit, Mt19937, Result, Sampling, Voice, Weights};

/// Backpressure for the streaming relay: two 8 KiB PCM blocks in flight.
const STREAM_BACKLOG: usize = 2;

struct State {
    weights: Weights,
    max_chars: usize,
    default_voice: String,
    chunk_frames: usize,
    /// A single `Voice`. Each voice costs only a 28 KiB prefix (the rest of the model is shared), so
    /// switching is a swap rather than a second copy — measured 131 MB with a Voice per name versus
    /// 90 MB with one.
    voice: Voice,
}

impl State {
    /// Point `voice` at `name` and return it.
    ///
    /// Switching is a 28 KiB prefix swap, not a second copy of the model; an unknown name falls back
    /// to the default so a typo degrades instead of failing the request. Takes the fields rather than
    /// `&mut self` so callers can hold `weights` and `voice` at the same time.
    fn swap_voice<'a>(
        weights: &Weights,
        voice: &'a mut Voice,
        default_voice: &str,
        name: &str,
    ) -> Result<&'a mut Voice> {
        let wanted = if weights.voice_prefix(name).is_ok() {
            name
        } else {
            default_voice
        };
        if voice.voice() != wanted {
            voice.set_voice(weights, wanted)?;
        }
        Ok(voice)
    }
}

/// Options for [`serve`].
pub struct Options {
    /// Refuse requests longer than this many characters (the text is split into sentences either
    /// way, so this guards the endpoint rather than the synthesiser).
    pub max_chars: usize,
    pub port: u16,
    pub host: String,
    pub voice: String,
    pub chunk_frames: usize,
    pub retries: usize,
}

/// Run the server until the process is killed.
pub fn serve(
    weights: Weights,
    cfg: serde_json::Value,
    sampling: Sampling,
    options: Options,
) -> Result<()> {
    // build the voice up front: a broken weight set should fail at startup, and the first request
    // should not pay for the model build
    let voice = Voice::new(&weights, &cfg, &options.voice, sampling)?
        .with_retry_policy(crate::quality::Policy {
            attempts: options.retries,
            first_utterance_single_take: false,
        })
        .with_chunk_frames(options.chunk_frames);
    let state = Arc::new(Mutex::new(State {
        weights,
        max_chars: options.max_chars,
        default_voice: options.voice.clone(),
        chunk_frames: options.chunk_frames,
        voice,
    }));

    let address = format!("{}:{}", options.host, options.port);
    let server = Server::http(&address).map_err(|e| format!("cannot bind {address}: {e}"))?;
    println!(
        "woof-rs serving on http://{address} (default voice {}, chunk_frames {})",
        options.voice, options.chunk_frames
    );

    for request in server.incoming_requests() {
        let state = Arc::clone(&state);
        // one thread per request; the synthesis lock serialises the actual work
        std::thread::spawn(move || {
            if let Err(e) = route(request, state) {
                println!("request failed: {e}");
            }
        });
    }
    Ok(())
}

fn route(request: tiny_http::Request, state: Arc<Mutex<State>>) -> Result<()> {
    let method = request.method().as_str().to_string();
    let path = request.url().split('?').next().unwrap_or("/").to_string();

    match (method.as_str(), path.as_str()) {
        ("GET", "/") => {
            let default_voice = state.lock().expect("state poisoned").default_voice.clone();
            json(
                request,
                200,
                serde_json::json!({
                    "name": "underdog-woof",
                    "kind": "tts",
                    "voices": audio::voices(),
                    "sample_rate": audio::SAMPLE_RATE,
                    "default_voice": default_voice,
                    "backend": "rust",
                    "endpoint": "/v1/audio/speech",
                }),
            )
        }
        ("GET", "/health") | ("GET", "/health/") => text(request, 200, "ok"),
        ("GET", "/v1/models") | ("GET", "/models") => {
            let data: Vec<serde_json::Value> = audio::voices()
                .iter()
                .map(|v| serde_json::json!({"id": v, "object": "model", "owned_by": "underdog"}))
                .collect();
            json(
                request,
                200,
                serde_json::json!({"object": "list", "data": data}),
            )
        }
        ("GET", "/props") => {
            let default_voice = state.lock().expect("state poisoned").default_voice.clone();
            json(
                request,
                200,
                serde_json::json!({
                    "model_path": "underdog-woof",
                    "voices": audio::voices(),
                    "sample_rate": audio::SAMPLE_RATE,
                    "backend": "rust",
                    "default_voice": default_voice,
                    "default_generation_settings": {},
                }),
            )
        }
        ("POST", "/v1/audio/speech") | ("POST", "/audio/speech") => speech(request, state),
        ("GET", _) | ("POST", _) => json(
            request,
            404,
            serde_json::json!({"error": {"message": "no route"}}),
        ),
        _ => json(
            request,
            405,
            serde_json::json!({"error": {"message": "method not allowed"}}),
        ),
    }
}

fn speech(mut request: tiny_http::Request, state: Arc<Mutex<State>>) -> Result<()> {
    let mut body = String::new();
    let _ = request.as_reader().read_to_string(&mut body);
    let body: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return json(
                request,
                400,
                serde_json::json!({"error": {"message": format!("bad json: {e}")}}),
            )
        }
    };

    let text = body["input"]
        .as_str()
        .or_else(|| body["text"].as_str())
        .unwrap_or("");
    if text.trim().is_empty() {
        return json(
            request,
            400,
            serde_json::json!({"error": {"message": "`input` must be non-empty text"}}),
        );
    }
    let max_chars = state.lock().expect("state poisoned").max_chars;
    if max_chars > 0 && text.chars().count() > max_chars {
        return json(
            request,
            413,
            serde_json::json!({"error": {"message": format!(
                "`input` is {} characters; this server accepts {max_chars}",
                text.chars().count())}}),
        );
    }
    let speed = body["speed"].as_f64().unwrap_or(1.0);
    if (speed - 1.0).abs() > 1e-6 {
        return json(
            request,
            400,
            serde_json::json!({"error": {"message": "Woof has no speed control; only speed=1.0 is supported"}}),
        );
    }
    let format = body["response_format"]
        .as_str()
        .unwrap_or("wav")
        .to_lowercase();
    if format != "wav" && format != "pcm" {
        return json(
            request,
            400,
            serde_json::json!({"error": {"message": format!("response_format {format:?} unsupported; use wav or pcm")}}),
        );
    }
    let stream_requested = body["stream"].as_bool().unwrap_or(false);
    let seed = body["seed"].as_u64().unwrap_or(0) as u32;
    let voice_name = body["voice"]
        .as_str()
        .or_else(|| body["model"].as_str())
        .unwrap_or("juno")
        .to_lowercase();

    // normalisation is ours now (the reference's normalize=True path)
    let text = crate::normalise::normalise(text);
    let options = textsplit::Options {
        clause_max: textsplit::CLAUSE_MAX,
        first_max: 0,
    };

    let mut guard = state.lock().expect("state poisoned");
    let chunk_frames = body["chunk_frames"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(guard.chunk_frames);
    // validate the voice now, so a bad name is answered rather than streamed as silence
    {
        let State {
            weights,
            voice,
            default_voice,
            ..
        } = &mut *guard;
        State::swap_voice(weights, voice, default_voice, &voice_name)?;
    }
    drop(guard);

    if stream_requested || format == "pcm" {
        // Streaming runs the synthesis on a producer thread that owns the lock for the duration;
        // this handler only returns the response body. The channel's bound is the backpressure, so a
        // slow client cannot make the synthesiser buffer a whole utterance.
        let reader = ChannelReader::new(
            Arc::clone(&state),
            text,
            voice_name.clone(),
            seed,
            options,
            chunk_frames,
        );
        let headers = vec![
            header("Content-Type", "audio/pcm"),
            header("X-Sample-Rate", &audio::SAMPLE_RATE.to_string()),
            header("X-Channels", "1"),
            header("X-Sample-Format", "s16le"),
            header("Cache-Control", "no-store"),
        ];
        let response = Response::new(StatusCode(200), headers, reader, None, None);
        return Ok(request.respond(response)?);
    }

    let mut collected: Vec<f32> = Vec::new();
    {
        let mut guard = state.lock().expect("state poisoned");
        let State {
            weights,
            voice,
            default_voice,
            ..
        } = &mut *guard;
        let voice = State::swap_voice(weights, voice, default_voice, &voice_name)?;
        voice.set_chunk_frames(chunk_frames);
        let mut rng = Mt19937::new(seed);
        let mut sink = |chunk: &[f32]| -> Result<()> {
            collected.extend_from_slice(chunk);
            Ok(())
        };
        voice.render(&text, weights, &mut rng, options, &mut sink)?;
    }
    let payload = audio::wav_bytes(&collected);
    let response = Response::from_data(payload).with_header(header("Content-Type", "audio/wav"));
    Ok(request.respond(response)?)
}

/// A `Read` that lazily starts the synthesiser on first use and yields its PCM blocks.
///
/// The producer runs on its own thread and takes the state lock itself, so the request handler can
/// return a chunked response immediately and `tiny_http` can relay blocks as they appear.
struct ChannelReader {
    state: Arc<Mutex<State>>,
    text: String,
    voice: String,
    seed: u32,
    options: textsplit::Options,
    chunk_frames: usize,
    blocks: Option<Receiver<Vec<u8>>>,
    pending: Vec<u8>,
    offset: usize,
}

impl ChannelReader {
    fn new(
        state: Arc<Mutex<State>>,
        text: String,
        voice: String,
        seed: u32,
        options: textsplit::Options,
        chunk_frames: usize,
    ) -> Self {
        ChannelReader {
            state,
            text,
            voice,
            seed,
            options,
            chunk_frames,
            blocks: None,
            pending: Vec::new(),
            offset: 0,
        }
    }

    fn start(&mut self) {
        let (sender, receiver) = sync_channel::<Vec<u8>>(STREAM_BACKLOG);
        self.blocks = Some(receiver);
        let state = Arc::clone(&self.state);
        let text = std::mem::take(&mut self.text);
        let voice_name = self.voice.clone();
        let (seed, options, chunk_frames) = (self.seed, self.options, self.chunk_frames);
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let mut first: Option<f32> = None;
            let mut total = 0usize;
            let mut rng = Mt19937::new(seed);
            let mut guard = state.lock().expect("state poisoned");
            let State {
                weights,
                voice,
                default_voice,
                ..
            } = &mut *guard;
            let Ok(voice) = State::swap_voice(weights, voice, default_voice, &voice_name) else {
                return;
            };
            voice.set_chunk_frames(chunk_frames);
            let mut sink = |samples: &[f32]| -> Result<()> {
                if samples.is_empty() {
                    return Ok(());
                }
                if first.is_none() {
                    first = Some(started.elapsed().as_secs_f32());
                    println!(
                        "stream: first audio after {:.2}s (voice={voice_name}, {chunk_frames} frames/chunk)",
                        first.unwrap_or(0.0)
                    );
                }
                let pcm = audio::pcm16(samples);
                total += pcm.len();
                let _ = sender.send(pcm);
                Ok(())
            };
            if let Err(e) = voice.render(&text, weights, &mut rng, options, &mut sink) {
                println!("stream: synthesis failed: {e}");
            }
            println!(
                "stream: {:.2}s of audio relayed",
                total as f32 / 2.0 / audio::SAMPLE_RATE as f32
            );
        });
    }
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.blocks.is_none() {
            self.start();
        }
        // serve the remainder of the block in hand before pulling another
        if self.offset < self.pending.len() {
            let n = (self.pending.len() - self.offset).min(buf.len());
            buf[..n].copy_from_slice(&self.pending[self.offset..self.offset + n]);
            self.offset += n;
            return Ok(n);
        }
        let Some(receiver) = &self.blocks else {
            return Ok(0);
        };
        match receiver.recv() {
            Ok(block) => {
                let n = block.len().min(buf.len());
                buf[..n].copy_from_slice(&block[..n]);
                self.pending = block;
                self.offset = n;
                Ok(n)
            }
            Err(_) => Ok(0), // producer finished
        }
    }
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

fn json(request: tiny_http::Request, code: u16, body: serde_json::Value) -> Result<()> {
    let payload = serde_json::to_vec(&body)?;
    let response = Response::from_data(payload)
        .with_status_code(StatusCode(code))
        .with_header(header("Content-Type", "application/json"));
    Ok(request.respond(response)?)
}

fn text(request: tiny_http::Request, code: u16, body: &str) -> Result<()> {
    let response = Response::from_string(body)
        .with_status_code(StatusCode(code))
        .with_header(header("Content-Type", "text/plain"));
    Ok(request.respond(response)?)
}
