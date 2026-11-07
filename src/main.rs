//! `woof-rs` command line: text in, WAV out.
//!
//! Everything model-shaped lives in the library; this file only dispatches: parse arguments, load
//! weights, choose the back-end, and print one summary line.

use woof_rs::backend::{CpuMatMul, MatMul};
// the resident-decoder path writes a WAV of its own, so it needs the writer; the CPU path gets all
// its audio through synth.rs
#[cfg(feature = "gpu-resident")]
use woof_rs::audio;
#[cfg(feature = "gpu-resident")]
use woof_rs::decoder::Decoder;
#[cfg(feature = "gpu-resident")]
use woof_rs::Speech;
use woof_rs::{cpu, Mt19937, Result, Weights};

mod cli;
mod synth;

use cli::{parse_args, Args};
use synth::{die_on_sigpipe, read_piped_text, synthesise, widen_stdout_pipe, Output};

/// Counts allocations while `WOOF_PROFILE` is set, so the profile can say how much of a frame is
/// spent handing memory back and forth. One relaxed load per allocation when profiling is off.
struct CountingAllocator;

unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        woof_rs::profile::count_alloc(layout.size());
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        woof_rs::profile::count_alloc(layout.size());
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        woof_rs::profile::count_alloc(new_size);
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn main() {
    woof_rs::profile::init_from_env();
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    woof_rs::profile::report();
}

fn run() -> Result<()> {
    if std::env::args().any(|a| a == "--w8a8") {
        woof_rs::nn::set_w8a8(true);
    }
    let Some(mut args) = parse_args()? else {
        return Ok(());
    };
    read_piped_text(&mut args)?;

    // Pin before the pool exists: worker threads inherit the mask, which is the only thing that
    // works on Android (see cpu.rs).
    // cap the malloc arenas before any worker thread exists: each one reserves 64 MiB of address
    // space, and the pool is what creates them
    match cpu::limit_malloc_arenas(args.malloc_arenas) {
        Some(n) => eprintln!("malloc arenas capped at {n}"),
        None if args.malloc_arenas > 0 => {
            eprintln!("warning: mallopt(M_ARENA_MAX) not supported here")
        }
        None => {}
    }

    let cores = args.cpu_mask.clone().unwrap_or_else(cpu::fast_cores);
    let threads = cpu::init_pool(args.threads, &cores);
    eprintln!(
        "threads: {threads} on cores {}",
        if cores.is_empty() {
            "all".to_string()
        } else {
            format!("{cores:?}")
        }
    );

    if args.bench_dot {
        return bench_dot();
    }
    #[cfg(feature = "gpu")]
    if args.bench_gpu {
        return bench_gpu();
    }
    #[cfg(not(feature = "gpu"))]
    if args.bench_gpu {
        return Err("--bench-gpu needs the `gpu` feature".into());
    }

    if let Some(path) = &args.normalise_file {
        let text = std::fs::read_to_string(path)?;
        let mut out = String::with_capacity(text.len() * 2);
        for line in text.lines() {
            out.push_str(&woof_rs::normalise::normalise(line));
            out.push('\n');
        }
        use std::io::Write as _;
        std::io::stdout().write_all(out.as_bytes())?;
        return Ok(());
    }
    if let Some(text) = &args.normalise_text {
        die_on_sigpipe();
        println!("{}", woof_rs::normalise::normalise(text));
        return Ok(());
    }
    // `--serve` synthesises per request, so it needs no --text
    if args.text.trim().is_empty() && !args.serve {
        return Err("--text is required (try --help)".into());
    }

    let backend: Box<dyn MatMul> = if args.gpu {
        #[cfg(feature = "gpu")]
        {
            let gpu = woof_rs::gpu::Gpu::new()?;
            eprintln!("gpu: {}", gpu.name());
            Box::new(gpu)
        }
        #[cfg(not(feature = "gpu"))]
        {
            return Err("--gpu needs the `gpu` feature".into());
        }
    } else {
        Box::new(CpuMatMul)
    };
    let _ = backend.name();

    let started = std::time::Instant::now();
    let model_path = match &args.model {
        Some(path) => path.clone(),
        None => woof_rs::weights::fetch::ensure_npz(args.offline)?,
    };
    let (weights, cfg) = Weights::load_any(&model_path)?;
    eprintln!("weights: {:.2}s", started.elapsed().as_secs_f32());

    let manifest = serde_json::json!({ "cfg": cfg });
    if args.serve {
        if args.port == 0 {
            return Err("--serve needs --port".into());
        }
        return woof_rs::serve::serve(
            weights,
            manifest["cfg"].clone(),
            args.sampling,
            woof_rs::serve::Options {
                port: args.port,
                host: args.host.clone(),
                voice: args.voice.clone(),
                chunk_frames: args.chunk_frames,
                retries: args.retries,
                max_chars: args.max_chars,
            },
        );
    }

    if let Some((name, path)) = &args.dump_tensor {
        return dump_tensor(&weights, name, path);
    }

    #[cfg(feature = "gpu-resident")]
    if args.verify_gpu || args.gpu_resident {
        // the device path works on one utterance (the parity reference), so keep it single-shot
        let mut speech = Speech::new(&weights, &manifest["cfg"]["sp"], &args.voice)?;
        let mut decoder = Decoder::new(&weights, &manifest["cfg"])?;
        let mut rng = Mt19937::new(args.seed);
        let latents = speech.latents_for_text(&args.text, args.sampling, &mut rng)?;

        let gpu = woof_rs::gpu::Gpu::new()?;
        eprintln!("gpu: {}", gpu.name());
        let mut resident = woof_rs::gpu::ResidentDecoder::new(&gpu, &decoder, &weights)?;
        if args.verify_gpu {
            for (index, kind, snr) in resident.verify(&mut decoder, &latents, &weights)? {
                eprintln!("stage {index:02} {kind:5} vs cpu: {snr:6.1} dB");
            }
            return Ok(());
        }
        let started = std::time::Instant::now();
        let waveform = resident.decode(&mut decoder, &latents, &weights)?;
        eprintln!(
            "resident decode: {} samples in {:.2}s",
            waveform.len(),
            started.elapsed().as_secs_f32()
        );
        audio::write_wav(&args.out, waveform.as_slice().expect("contiguous waveform"))?;
        eprintln!("wrote {}", args.out.display());
        return Ok(());
    }
    #[cfg(not(feature = "gpu-resident"))]
    if args.verify_gpu || args.gpu_resident {
        return Err("--gpu-resident/--verify-gpu need the `gpu-resident` feature".into());
    }

    if args.bench || args.bench_file.is_some() {
        return run_benchmark(&args, &weights, &manifest);
    }

    let output = if !args.stream {
        Output::Whole
    } else if args.pcm_stdout {
        Output::StreamPcm
    } else {
        Output::Stream
    };
    die_on_sigpipe();
    if output == Output::StreamPcm {
        widen_stdout_pipe();
    }
    synthesise(&args, &weights, &manifest, output)
}

/// Run the corpus (or `--bench-file`) once, printing a table and a summary.
fn run_benchmark(args: &Args, weights: &Weights, manifest: &serde_json::Value) -> Result<()> {
    use woof_rs::bench;

    let texts: Vec<String> = match &args.bench_file {
        Some(path) => std::fs::read_to_string(path)?
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect(),
        None => bench::CORPUS.iter().map(|s| (*s).to_string()).collect(),
    };
    let mut voice = woof_rs::Voice::new(weights, &manifest["cfg"], &args.voice, args.sampling)?
        .with_retry_policy(woof_rs::quality::Policy {
            attempts: args.retries,
            first_utterance_single_take: args.no_retry_first,
        })
        .with_chunk_frames(args.chunk_frames);
    let mut rng = Mt19937::new(args.seed);
    let split = woof_rs::textsplit::Options {
        clause_max: woof_rs::textsplit::CLAUSE_MAX,
        first_max: args.first_max,
    };

    let summary = bench::run(&mut voice, weights, &mut rng, &texts, split)?;
    print!("{}", bench::report(&summary));
    if let Some(path) = &args.bench_json {
        std::fs::write(path, serde_json::to_vec_pretty(&bench::to_json(&summary))?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// Run the corpus (or `--bench-file`) once, printing a table and a summary.
fn dump_tensor(weights: &Weights, name: &str, path: &std::path::Path) -> Result<()> {
    let values: Vec<f32> = if let Ok(grouped) = weights.grouped(name) {
        grouped.dequantise().iter().copied().collect()
    } else if let Ok(vector) = weights.vector(name) {
        vector.iter().copied().collect()
    } else {
        let matrix = weights.matrix(name)?;
        matrix.iter().copied().collect()
    };
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    eprintln!("wrote {} values to {}", values.len(), path.display());
    Ok(())
}

/// Throughput of the hot kernel: dequantise one int8 row, then a vectorised f32 dot.
fn bench_dot() -> Result<()> {
    let n = 2432usize;
    let q: Vec<i8> = (0..n).map(|i| ((i % 251) as i32 - 125) as i8).collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut scratch = vec![0f32; n];
    let reps = 200_000usize;

    let started = std::time::Instant::now();
    let mut acc = 0f32;
    for _ in 0..reps {
        woof_rs::nn::dequantise_row(&q, 0.5, &mut scratch);
        acc += woof_rs::nn::dot(&scratch, &x);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let macs = reps as f64 * n as f64;
    eprintln!(
        "dequantise+dot: {:.2} GMAC/s [acc {acc:.3}]",
        macs / elapsed / 1e9
    );
    Ok(())
}

#[cfg(feature = "gpu")]
fn bench_gpu() -> Result<()> {
    use ndarray::Array2;
    let gpu = woof_rs::gpu::Gpu::new()?;
    eprintln!("gpu: {}", gpu.name());
    for (m, k, n) in [
        (105usize, 512usize, 384usize),
        (5040, 192, 96),
        (630, 48, 48),
        (17, 33, 5),
    ] {
        let a = Array2::<f32>::from_shape_fn((m, k), |(i, j)| {
            ((i * 7 + j * 3) % 11) as f32 * 0.01 - 0.05
        });
        let b =
            Array2::<f32>::from_shape_fn((k, n), |(i, j)| ((i * 5 + j) % 13) as f32 * 0.02 - 0.12);
        let cpu = a.dot(&b);
        let gpu_out = gpu.matmul(&a, &b);
        let max_diff = (&cpu - &gpu_out)
            .mapv(f32::abs)
            .iter()
            .fold(0f32, |x, y| x.max(*y));
        let scale = cpu.iter().fold(0f32, |x, y| x.max(y.abs())).max(1e-9);
        eprintln!(
            "gemm {m}x{k}x{n}: max|d| {max_diff:.3e} (rel {:.2e})",
            max_diff / scale
        );
    }
    Ok(())
}
