//! Command line surface: the flag list, the parsed form of it, and the parser.
//!
//! Kept apart from `main.rs` so the run paths read as what they do (load, synthesise, serve) and so
//! the usage text sits next to the fields it documents.

use std::path::PathBuf;

use woof_rs::{Result, Sampling};

const USAGE: &str = "\
woof-rs — Underdog Woof in Rust

USAGE:
    woof-rs --text <TEXT> [OPTIONS]

OPTIONS:
    --text <TEXT>          text to speak. Omit it (or pass -) to read it from stdin, so text can be
                           piped: echo hello | woof-rs --out hi.wav
    --out <PATH>           output WAV (default: out.wav)
    --model <PATH>         weights: the reference's woof_int8.npz directly (no Python needed), or a
                           prefix written by tools/export.py -> .bin + .json.
                           Omitted: the cached download is used, and fetched if absent
    --offline              never download; fail if the weights are not already cached
    --voice <NAME>         juno | mira (default: juno)
    --seed <N>             sampler seed (default: 0); same text + seed -> identical audio
    --temp <F>             sampling temperature (default: 0.5)
    --eos <F>              EOS threshold (default: 0.9)
    --max-frames <N>       hard cap on latent frames (default: 750)
    --threads <N>          worker threads (default 4; 0 = auto, one per fast core). Fewer leaves no
                           cushion for streaming: at parity with playback every chunk boundary becomes
                           an audible gap (measured: 2.05 s of silence in 6 gaps at one thread, 0.77 s
                           in 3 at two, 0.67 s in 3 at four). 2 -> 4 costs ~3% more CPU for +25% speed.
    --malloc-arenas <N>    cap glibc's per-thread malloc arenas (0 = glibc default). Each arena
                           reserves 64 MiB of address space; 2 keeps VmSize near 190 MiB instead
                           of ~1.7 GiB with 20 threads, with no effect on resident memory
    --cpu-mask <LIST>      explicit core list, e.g. 3,4,5,6,7 (overrides detection)
    --gpu                  use the wgpu back-end for the decoder's batched stages
    --serve                run an OpenAI-shaped HTTP server instead of synthesising once
    --port <N>             with --serve, the port to listen on (required)
    --host <ADDR>          with --serve, the address to bind (default 127.0.0.1)
    --stream               synthesise sentence by sentence and write as audio arrives
    --pcm-stdout           with --stream, write raw 16-bit PCM to stdout instead of a WAV
    --first-max <N>        with --stream, shorten the first utterance to N characters
    --retries <N>          extra draws per utterance when a take fails the guard (default 3)
    --verbose, -v          say what it is doing: thread count, weight load time, time to first audio,
                           a line per emission, and the summary. Off by default, so a run is silent
                           unless it fails (errors are always printed)
    --max-chars <N>        refuse input longer than N characters (0 = no limit). The text is split
                           into sentences and streamed one at a time, so this is a guard, not a
                           memory limit -- the HTTP server keeps its own 4000 unless this is set
    --chunk-frames <N>     emit audio every N latent frames (~40 ms each) for the first utterance,
                           instead of after the whole sentence (default 26 = ~1 s; 0 disables).
                           Streaming restarts the decoder transformer on the whole prefix per chunk,
                           so small chunks are expensive: measured 0.6x realtime on an N150 NAS at 13
                           frames vs 0.9x at 26 and 1.1x at 52 (unchunked 1.1x).
                           Chunked emission implies a single take: audio already played cannot be
                           redrawn. Only affects streaming/--pcm-stdout; a buffered --out file keeps
                           the guard
    --no-retry-first       never redraw the first utterance: bounds time-to-first-sound at 1x
    --gpu-resident         run the whole SEANet stack on the device (needs `gpu-resident`)
    --verify-gpu           report per-stage device-vs-CPU parity and exit (needs `gpu-resident`)
    --timings              print per-stage timings to stderr
    --dump-latents <PATH>  write the raw f32 latents (frames x latent)
    --dump-tensor <NAME> <PATH>
                           write one tensor as raw f32 (grouped-int8 tensors are expanded)
    --bench                run the built-in benchmark corpus and exit
    --bench-file <PATH>    benchmark the lines of a file instead (one text per line)
    --bench-json <PATH>    also write the benchmark results as JSON
    --no-normalise         pass --text through verbatim (the reference's normalize=False)
    --normalise <TEXT>     print the spoken form of TEXT and exit (normaliser port check)
    --normalise-file <PATH>
                           normalise each line of a file, one per line (batch, for the fuzz corpus)
    --bench-dot            measure the dequantise+dot kernel and exit
    --bench-gpu            verify the GPU matmul against the CPU and exit
    -h, --help             this message
";

#[derive(Debug)]
pub struct Args {
    pub text: String,
    pub out: PathBuf,
    /// `None` means "use the cached download, fetching it if needed"
    pub model: Option<PathBuf>,
    pub voice: String,
    pub seed: u32,
    pub sampling: Sampling,
    pub threads: usize,
    pub offline: bool,
    pub verbose: bool,
    pub malloc_arenas: usize,
    pub max_chars: usize,
    pub cpu_mask: Option<Vec<usize>>,
    pub gpu: bool,
    pub serve: bool,
    pub port: u16,
    pub host: String,
    pub stream: bool,
    pub pcm_stdout: bool,
    pub first_max: usize,
    pub retries: usize,
    pub no_retry_first: bool,
    pub chunk_frames: usize,
    pub gpu_resident: bool,
    pub verify_gpu: bool,
    pub timings: bool,
    pub dump_latents: Option<PathBuf>,
    pub dump_tensor: Option<(String, PathBuf)>,
    pub bench: bool,
    pub bench_file: Option<PathBuf>,
    pub bench_json: Option<PathBuf>,
    pub normalise_text: Option<String>,
    pub no_normalise: bool,
    pub normalise_file: Option<PathBuf>,
    pub bench_dot: bool,
    pub bench_gpu: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            text: String::new(),
            out: PathBuf::from("out.wav"),
            model: None,
            voice: "juno".to_string(),
            seed: 0,
            sampling: Sampling::default(),
            // four: synthesis must beat playback with room to spare (see --threads)
            threads: 4,
            offline: false,
            verbose: false,
            malloc_arenas: 0,
            max_chars: 0,
            cpu_mask: None,
            gpu: false,
            serve: false,
            port: 0,
            host: "127.0.0.1".to_string(),
            stream: false,
            pcm_stdout: false,
            first_max: 0,
            retries: 3,
            no_retry_first: false,
            chunk_frames: 26,
            gpu_resident: false,
            verify_gpu: false,
            timings: false,
            dump_latents: None,
            dump_tensor: None,
            bench: false,
            bench_file: None,
            bench_json: None,
            normalise_text: None,
            no_normalise: false,
            normalise_file: None,
            bench_dot: false,
            bench_gpu: false,
        }
    }
}

pub fn parse_args() -> Result<Option<Args>> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-V" | "--version" => {
                println!(
                    "{} {} ({} build, Underdog Woof normaliser{})",
                    env!("CARGO_PKG_NAME"),
                    env!("CARGO_PKG_VERSION"),
                    if cfg!(debug_assertions) {
                        "debug"
                    } else {
                        "release"
                    },
                    if cfg!(feature = "gpu-resident") {
                        ", gpu-resident"
                    } else {
                        ""
                    }
                );
                return Ok(None);
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--text" => args.text = value()?,
            "--out" => args.out = PathBuf::from(value()?),
            "--model" => args.model = Some(PathBuf::from(value()?)),
            "--offline" => args.offline = true,
            "--voice" => args.voice = value()?,
            "--seed" => args.seed = value()?.parse().map_err(|_| "--seed expects a number")?,
            "--temp" => {
                args.sampling.temperature =
                    value()?.parse().map_err(|_| "--temp expects a number")?
            }
            "--eos" => {
                args.sampling.eos_threshold =
                    value()?.parse().map_err(|_| "--eos expects a number")?
            }
            "--max-frames" => {
                args.sampling.max_frames = value()?
                    .parse()
                    .map_err(|_| "--max-frames expects a number")?
            }
            "--threads" => {
                args.threads = value()?.parse().map_err(|_| "--threads expects a number")?
            }
            "--verbose" | "-v" => args.verbose = true,
            "--max-chars" => {
                args.max_chars = value()?
                    .parse()
                    .map_err(|_| "--max-chars expects a number")?
            }
            "--malloc-arenas" => {
                args.malloc_arenas = value()?
                    .parse()
                    .map_err(|_| "--malloc-arenas expects a number")?
            }
            "--cpu-mask" => {
                let list = value()?;
                args.cpu_mask = Some(
                    list.split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect::<Vec<usize>>(),
                );
            }
            "--gpu" => args.gpu = true,
            "--serve" => args.serve = true,
            "--port" => args.port = value()?.parse().map_err(|_| "--port expects a number")?,
            "--host" => args.host = value()?,
            "--stream" => args.stream = true,
            "--pcm-stdout" => args.pcm_stdout = true,
            "--first-max" => {
                args.first_max = value()?
                    .parse()
                    .map_err(|_| "--first-max expects a number")?
            }
            "--retries" => {
                args.retries = value()?.parse().map_err(|_| "--retries expects a number")?
            }
            "--no-retry-first" => args.no_retry_first = true,
            "--chunk-frames" => {
                args.chunk_frames = value()?
                    .parse()
                    .map_err(|_| "--chunk-frames expects a number")?
            }
            "--gpu-resident" => args.gpu_resident = true,
            "--verify-gpu" => args.verify_gpu = true,
            "--timings" => args.timings = true,
            "--dump-latents" => args.dump_latents = Some(PathBuf::from(value()?)),
            "--dump-tensor" => {
                let name = value()?;
                args.dump_tensor = Some((name, PathBuf::from(value()?)));
            }
            "--bench" => args.bench = true,
            "--bench-file" => args.bench_file = Some(PathBuf::from(value()?)),
            "--bench-json" => args.bench_json = Some(PathBuf::from(value()?)),
            "--no-normalise" => args.no_normalise = true,
            "--normalise" => args.normalise_text = Some(value()?),
            "--normalise-file" => args.normalise_file = Some(PathBuf::from(value()?)),
            "--bench-dot" => args.bench_dot = true,
            "--bench-gpu" => args.bench_gpu = true,
            // a bare argument is the text, so `woof-rs "hello" --voice mira` works as well as
            // `woof-rs --text hello`; more than one is a mistake worth reporting
            other if !other.starts_with('-') => {
                if args.text.is_empty() {
                    args.text = other.to_string();
                } else {
                    return Err(format!(
                        "text given twice ({:?} and {:?}); pass it once",
                        args.text, other
                    )
                    .into());
                }
            }
            other => return Err(format!("unknown option {other}\n\n{USAGE}").into()),
        }
    }
    Ok(Some(args))
}
