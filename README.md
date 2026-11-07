# woof-rs — Underdog Woof in Rust

[![ci](https://github.com/robertbak/underdog-woof-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/robertbak/underdog-woof-rs/actions/workflows/ci.yml)

A port of [`ConwayResearch/Underdog-Woof`](https://huggingface.co/ConwayResearch/Underdog-Woof)'s
numpy runtime (`underdog_woof/runtime.py`), keeping the model's **int8 weights int8**: the
reference dequantises everything to f32 at load (~205 MB resident), while the quantised form is
**38.7 MB**.

## Licence and credit

This is a **Derivative** of Underdog Woof under the
[Underdog Non-Commercial License 1.0](LICENSE), inherited unchanged — see [NOTICE](NOTICE) and
[LICENSE](LICENSE), both copied verbatim from upstream. Free for non-commercial use; **any commercial
use, including a hosted service, needs a commercial licence** from Underdog (license@conway.tech).

If you use it, the licence requires credit:

* **Published audio** — put "Underdog Woof" or "underdog.ai" in the title or description.
* **Apps and products** — show **"Voice by Underdog Woof"**, linking to
  [underdog.ai](https://underdog.ai), where people can see it. Audio-only products carry it in the
  about or settings screen and in the listing.
* **This kind of file** — say the work is derived from Underdog Woof and link the licence.
* **Redistribution** — ship `LICENSE` and `NOTICE` unchanged, and say what you changed (below).

Two restrictions worth stating plainly: no training, fine-tuning or distillation of another speech
model on Woof or its audio, and no cloning or imitating a real person's voice without consent.

This is an independent port. It is not affiliated with, sponsored by, or endorsed by Underdog, and
"Underdog Woof" appears here only as the credit the licence requires.

The **weights are not in this repository** — they are generated from the upstream model by
`tools/export.py` (see *Quick start*), so nothing here redistributes them.

## What this port changes

Recorded because the licence asks for it, and because a port that hides its changes is useless to
whoever reads it next:

* **Written in Rust**, no numpy and no Python at run time. The reference's numerics are reproduced
  from `underdog_woof/runtime.py`, including its RNG (MT19937 + Box-Muller) and its exact constants.
* **int8 stays int8.** The reference dequantises every weight to f32 at load (~205 MB resident); the
  exported blob is 38.7 MB and stays quantised, dequantised per row inside the matvec.
* **Streaming**: sentence-at-a-time synthesis, frame-level chunked emission, a prefetch queue, and an
  OpenAI-shaped HTTP server (`--serve`), none of which the reference has.
* **Text normaliser ported**: the reference's `normalize.py` and its respell lexicon, reimplemented
  as regex tables generated from the original, and checked byte-for-byte against it.
* **Bad-take guard**: a stalled or mispronounced take is redrawn, with the policy measured rather
  than guessed.
* **CPU thread pool** with topology-aware pinning, and an optional wgpu back-end for the decoder.
* **Not changed**: the model, its voices and its numerics. Parity against the reference is measured
  (`tools/parity.py`), not asserted.

## Quick start

```bash
git clone https://github.com/robertbak/underdog-woof-rs && cd underdog-woof-rs
cargo build --release                       # CPU build, 4.1 MiB
cargo test --release                        # 50 unit tests
tools/install.sh                            # build -> ~/bin/woof-rs (+ say.sh) -> cargo clean
woof-rs --version                           # name, version and build features

# no weights given: it fetches them from upstream once, verifies the SHA-256, caches them
woof-rs "Your order ships on March 3rd." --out hi.wav

# or point it at a file you already have (the reference's own .npz needs no conversion)
woof-rs --model /path/to/woof_int8.npz "Your order ships on March 3rd." --out hi.wav
woof-rs --model /tmp/woof-i8 "Piped text." --pcm-stdout --stream | pw-cat --playback --raw \
    --format=s16 --rate=24000 --channels=1 -
woof-rs --model /tmp/woof-i8 --serve --port 5816 --voice juno
```

**No Python is needed to run the model.** With no `--model`, the first run downloads the reference's
`woof_int8.npz` from Underdog's Hugging Face repository into `$XDG_CACHE_HOME/woof-rs` (or
`~/.cache/woof-rs`), verifies the pinned SHA-256 and caches it — 38.7 MB, once. It is fetched from
upstream rather than shipped here because the weights are theirs, not ours to redistribute; `--offline`
refuses to touch the network, and `--model` skips all of it. The download uses `curl` or `wget` rather
than a TLS stack, because pulling in rustls to fetch one file would cost more binary than the whole
synthesiser.

`--model` also takes the reference's `woof_int8.npz` directly:
its entries are stored (uncompressed) `.npy` files and `__tree__` is a JSON description of the nesting,
so the loader reads it in Rust — same weights, bit-identical audio, 0.03 s to load. `tools/export.py`
still exists for the alternate layout (one flat blob plus a small manifest, 0.02 s to load, one file
per tensor instead of a 40 MB archive) and for the `--f32` A/B of dequantised weights, but it is
optional now. Python is needed only for the parity and normaliser harnesses.

## Layout

```
src/
  lib.rs            crate root, module map, Result alias, text limits
  main.rs           binary entry: dispatch only (load, synthesise, serve, bench)
  cli.rs            flags, the parsed Args, and the usage text
  synth.rs          where audio goes: buffered WAV, streaming WAV, PCM pipe; prefetch queue,
                    SIGPIPE and pipe-size handling
  audio.rs          vocabulary + tokeniser + 16-bit PCM WAV writer
  rng.rs            torch-compatible MT19937 + Box-Muller (the sampler's RNG)
  nn.rs             numeric primitives, including the reference's exact constants
  weights/          manifest blob loading; Linear (int8 + per-row scales) and GroupedInt8
  speech.rs         text/voice prefix, incremental transformer step, flow head, sampling loop
  voice.rs          one utterance: take, guard, retries, chunked streaming
  quality.rs        the bad-take guard and its measured policy
  decoder/          latent transformer + SEANet stack, in the reference's two-phase shape
    conv.rs         causal conv / transposed conv / residual, with streaming state
  backend.rs        MatMul trait so the batched stages can run on CPU or GPU
  cpu.rs            core-topology detection and the pinned thread pool
  gpu/              wgpu back-end (feature `gpu`), verified A·B kernel + epilogues
  serve.rs          OpenAI-shaped HTTP server (tiny_http), wav or chunked PCM
  bench.rs          corpus harness: time-to-first-samples, RTF, guard draws by length
  textsplit.rs      the reference's sentence/clause splitter
  normalise/        the reference's text normaliser
    mod.rs          orchestration, fallback verbaliser
    pretokens/      num_tokens, Roman numerals, money/units
      roman.rs      IV -> four, Henry VIII -> Henry the eighth
      money.rs      $4.99, 10 kg, x^2, 12 million
    prepass/        the 23 pre-passes, one function per stage, in reference order
      urls.rs       hosts, schemes, e-mail, file names
      dates.rs      slashed, ISO, day/month, ranges, years
      addresses.rs  house numbers, street types, the abbreviation table
      numbers.rs    times, phones, units, identifiers, codes, durations, symbols
      support.rs    table lookups, alternations, pluralisation
    respell.rs      the 24,883-entry pronunciation lexicon
    tables.rs       generated lookup tables (tools/gen_normalise_tables.py)
    re.rs           cached fancy-regex helpers
tools/
  export.py         numpy runtime -> flat f32/int8 blob + JSON manifest
  parity.py         Rust vs numpy on the same text + seed, reports SNR
  normalise_diff.py 317 reference cases, byte-for-byte
  normalise_fuzz.py randomised cases against the reference normaliser
  gen_normalise_tables.py  regenerates src/normalise/tables.rs from the reference
  install.sh        build -> ~/bin/woof-rs (+ say.sh) -> cargo clean
  say.sh            speak text through the default sink
```

`tables.rs`, `data/respell.tsv` and `tests/data/normalise_cases.tsv` are generated from the reference,
not written: 713 lines of tables with no decisions in them, kept in the tree so the build needs no
Python. They are Derivative content like everything else here, and their headers say so.

Two rules the code follows, both learned the hard way:

* **The reference is the specification.** Operation order, the `float32` epsilon the speech model
  uses, `1e-5` in the layer norms, and the Abramowitz-Stegun `erf` are all reproduced exactly —
  anything else shows up as tens of dB of SNR.
* **Boundaries own their invariants.** The decoder applies its own latent de-normalisation
  (`lat * std + mean`) because a caller that forgets it costs ~70 dB; the CPU layer owns thread
  affinity because `sched_setaffinity` from a worker thread is a no-op on Android.

## Bad-take guard and its latency knob

The reference redraws a take when it is bad — a silence run of ≥ 1 s inside the audio, more than
0.095 s of audio per input character, or hitting the frame cap — up to `retries` times, keeping the
best by `(bad, stall, spc)`. `src/quality.rs` reproduces that, and `--retries N` sets it
(default 3, the reference's value).

`--no-retry-first` exempts the **first** utterance, and `--chunk-frames N` decodes it
incrementally (see below). That matters because the guard's worst case is
4× generation time, and the first utterance is the one the listener is waiting for. Measured on a
4-utterance sample (77/89/44/3 chars), sentence-level decoding:

| policy | draws for the short last utterance | wall |
|---|---|---|
| `--retries 3` (reference) | 4 (all fail `spc`: three characters cannot produce ≥ 0.095 s/char of audio) | 2.71 s |
| `--no-retry-first` | 4 (unchanged — it is not the first utterance) | 2.65 s |
| `--retries 0` | 1 | 2.12 s |

Note what that table says: **short utterances can never pass the `spc` test**, so for a text whose
first piece is short, `--no-retry-first` (or `--retries 0`) removes up to 3 wasted draws from the
latency-critical path while producing byte-identical audio (the guard keeps the first take anyway).
`--timings` prints the per-utterance draw counts (and where the decoder spent its time);
`WOOF_TRACE=1` prints the same for the reference-style redraw path.

## Text in: `--text`, a positional, or stdin

```
woof-rs --text "Your order ships on March 3rd." --out hi.wav
woof-rs "Your order ships on March 3rd." --out hi.wav          # same thing
echo "Piped text." | woof-rs --out hi.wav                      # or `say`, which is a wrapper for this
cat notes.txt | woof-rs --out notes.wav                        # one utterance per sentence
echo "1,250 on March 3rd" | woof-rs --normalise -              # inspect the spoken form
```

A missing `--text` (or `-`) reads stdin, but only when stdin is actually a pipe — a bare `woof-rs` on a
terminal still says `--text is required` rather than hanging on EOF. A trailing newline from
`echo`/`cat` is trimmed. `say` is `tools/say.sh` installed to `~/bin`, which pipes
`--pcm-stdout --stream` into the default sink, so all three forms work there too.

**Length is a queue, not a limit.** Text is split into ≤120-character utterances and each is
synthesised as the previous one plays, so a 6,000-character file is just 80 sentences in a row —
`cat notes.txt | woof-rs --out notes.wav` needs no special handling. The one thing that scales with
length is a buffered `--out`, which holds every sample (96 KiB per second of audio, so ~137 MiB for a
24-minute file); above 20,000 characters the CLI says so and suggests `--stream`. `--max-chars N`
imposes a hard limit if you want one (default: none). The HTTP server is the exception: it refuses
requests over 4,000 characters (`--max-chars 0` there means "use the 4,000 default", since an endpoint
should not accept a novel from any client).

Because the command line restores the default `SIGPIPE` disposition, `woof-rs … | head -c 9600` exits
141 and silent, like any other pipe tool, instead of reporting a broken pipe after the reader got what
it asked for.

## Serving (`--serve`)

The binary is the server: normalisation, synthesis and chunked streaming all live in it, so a
deployment needs no Python, no numpy and no repo checkout.

```
woof-rs --model …/models/woof-i8 --serve --port 8080 --voice juno --chunk-frames 13
```

| endpoint | purpose |
|---|---|
| `GET /` | identity JSON — llama-swap preloads a proxy seat by probing this |
| `GET /health` | `200 ok` (llama-swap's `checkEndpoint`) |
| `GET /v1/models`, `GET /props` | voices, sample rate, backend |
| `POST /v1/audio/speech` | OpenAI speech API; `audio/wav`, or chunked `audio/pcm` when `"stream": true` |

Request: `{"model":"juno","input":"…","voice":"juno"|"mira","response_format":"wav"|"pcm",
"stream":false,"seed":0,"chunk_frames":13,"speed":1.0}`. `speed` other than 1.0 is a 400 (Woof has
no speed control and resampling would change pitch).

Measured through llama-swap's proxy, numbers-and-dates text: **wav 8.44 s in 2.40 s**, **streaming
first byte at 0.29 s**, voice override honoured.

**Server mode needs no prefetch queue and is the smoothest path there is.** Reading a chunked response
at exactly 24 kHz — the same pacing test that exposed the CLI's gaps — gave **first byte 0.26 s, 16.60 s
of audio, 0 stalls**: the TCP send buffer (megabytes) is a far better cushion than a 64 KiB pipe, so
the model stays far ahead of a client that plays as it receives. Idle, the server holds 4 pool threads,
95 MiB RSS and ~1% CPU.

Requests are **serialised by one lock** (measured: solo 1.00 s; two in parallel, 1.02 s and 2.05 s, so
the second waits for the first to finish streaming). That is the right trade for a single seat — no CPU
contention, no latency surprises — and the reason the lock is not released between chunks: interleaving
two streams would halve each one's speed to gain nothing. A client that hangs up mid-stream stops
synthesis at the next write (`llama-swap` logs "recovered from upstream disconnection"); the server
stays healthy.

**Why `tiny_http` and not `tokio`/`hyper` or a hand-rolled server.** Synthesis is CPU-bound and
serialised behind one lock, so there is no IO concurrency for an async runtime to win — `tokio` would
add megabytes and an executor for nothing. A hand-rolled `std::net` server was the first attempt and
was the wrong call: HTTP request parsing and chunked transfer encoding are exactly the code you do
not want to get subtly wrong. `tiny_http` is blocking (matching the model), brings correct framing
plus chunked encoding, and costs four small crates. Streaming is a `Read` implementation that starts
the synthesiser on first read and drains a bounded channel, so a slow client cannot make the server
buffer a whole utterance.

## Speed work (v0.1.3): +65% throughput, first audio 0.15 -> 0.11 s

Three changes, each measured interleaved against the binary it replaced (the load on this box ranges
1.4-5.3 depending on what else is running, so absolute numbers are only comparable within a session;
the deltas are the claim):

| change | what it was | effect |
|---|---|---|
| tiled dequantise | the matvec wrote a dequantised row into a heap scratch and read it back; now 256-float tiles in a stack buffer, so the block stays in L1 | +13% |
| decoder overlap | the decoder (34% of wall time) ran serially behind the sampler; it now runs on a worker thread, fed by a bounded queue, so chunk k+1 is generated while k is vocoded | +13% |
| AVX2 fused dot | load 8 int8 weights -> widen -> convert -> FMA, no scratch at all | +30% |

```
bench corpus, 4 threads      baseline 6.3x  ->  8.2-8.6x realtime
time to first audio (median) 0.15 s        ->  0.11 s
per-frame cost, 1 thread     8.5 ms        ->  2.8 ms
peak RSS                     136 MB        ->  141 MB
```

Measured, not asserted, and three things were rejected on the measurements rather than shipped:

* **per-frame allocation removal** — cut 6,880 allocations/frame to 6,551 and changed wall time by
  nothing. The allocator is not the bottleneck here.
* **dequantising inside the dot in portable Rust** — 195 -> 213 µs/call: the per-element `i8 -> f32`
  widening defeats the vectoriser. The AVX2 version of the same idea is 42 µs/call.
* **attention via `ndarray` views** (drop the per-head copies) — 95 -> 128 µs/call end to end slower,
  because `dot` on a view leaves the BLAS path.

Quality held: the proxy compares duration, 20 ms energy envelope and level against the pre-work takes
(all pass), and correctness improved — decoder parity went 85.3 -> **90.5 dB** over 72 frames because
the new summation orders land closer to numpy. `tools/quality_proxy.py` and `benchmarks/profile.md`
carry the detail; `WOOF_PROFILE=1` reproduces the profile.

What is left, in the profile's terms: attention (25% of a frame, needs a hand-written kernel) and the
voice/text prefix (5.1 ms per utterance, worth under 4% of first audio). Both are recorded as future
work rather than claimed.

## Threads: default 4

The default is **four workers**; `--threads 0` asks for auto (one per fast core), and `--threads N`
pins the count. Thresholds below are from before the speed work in the section above — current numbers
are 7.4x at one thread and 8.2-8.6x at four. Measured on the 10-text `--bench` corpus (51 s of audio,
desktop i5-14600KF):

| threads | throughput | first audio (median) |
|---|---|---|
| 1 | 3.6x realtime | 0.31 s |
| 2 | 5.3x | 0.21 s |
| **4 (default)** | **6.2x** | **0.19 s** |
| auto (14) | 6.4x | 0.17 s |

Scaling is poor — 1 → 4 threads is 1.7x, 4 → 14 is 1.03x — because the per-frame matvec is bound by
memory and cache, not FLOPs (the same reason the GPU path lost on the tablet). One thread is no longer
a sensible default: synthesis has to beat playback, not merely match it, and one thread leaves no
cushion for streaming. Pacing a reader at exactly 24 kHz and counting the moments it has nothing to
play (16.6 s of audio, 5 sentences, machine at load ~10):

| threads | silence | gaps |
|---|---|---|
| 1 | 2.05 s | 6 |
| 2 | 0.77 s | 3 |
| **4 (default)** | **0.62 s** | **2** |
| auto (14) | 0.59 s | 2 |

Those gap figures predate the dequantisation fix below; with it, four threads are comfortably clear of
the boundary. Two remains the light option when power matters more than a tenth of a second.

Pool ramp-up is the remaining cost of anything above one thread: topology detection has to be right,
and the workers compete with whatever else owns the machine.

Core detection still decides *which* cores the pool may use: `cpu_capacity` when it is non-uniform
(ARM big.LITTLE), otherwise one logical CPU per physical core via topology, so SMT siblings are never
counted as cores. Intel hybrid x86 reports `cpu_capacity = 1024` for every core unless the kernel
enables ITMT, which is why the topology fallback exists (and clock-rate filtering does not work:
E-cores at 4.0 GHz are 0.755 of the P-cores' 5.3).

## `--w8a8`: int8 activations, and what it costs

The weights are int8 everywhere, but the *activations* are f32, so a matvec dequantises each weight
row into a scratch buffer and accumulates in f32 — which is what the reference does, and why parity
holds. `--w8a8` quantises the activation to int8 as well (one symmetric per-vector scale,
`max|x|/127`), so the product accumulates in **i32** and the weight row is never expanded: 4x less
memory traffic on the side that is actually bound. It is off by default.

```
threads 4, 10-text corpus     throughput     median first audio
f32 activations (default)     6.2x realtime  0.19 s
--w8a8                        8.6x realtime  0.16 s
```

**It does not preserve parity, and it cannot.** Quantising activations changes the sampling path at
the very first frame, and the autoregressive loop amplifies that: against numpy the latents fall from
85.3 dB to 38.7 dB and the takes diverge at frame 1 instead of frame 71. That is not damage, it is a
different draw — measured against its own f32 output, `--w8a8` at the same seed tracks *closer*
(energy-envelope correlation 0.81, duration ratio 0.993) than the f32 model does against a different
seed (0.58). Same words, same length, sibling performance.

So this is a knob for throughput, not a quality setting: use it when you want 40% more speed and can
accept a different (not obviously worse) take. Whether it sounds as good is a listening judgement and
this README will not pretend to have made it. The kernel's own error is bounded by construction — a
unit test pins `int8_matvec` against the f32 path at better than 40 dB — so anything you hear is the
sampler's sensitivity, not arithmetic sloppiness.

i32 is the accumulator for a reason worth writing down: an i8×i8 product is exact and fits i16
(127×127 = 16129), but the *sum* overflows i16 after two terms. i32 holds ~133,000 maximal terms, and
the longest reduction here is 3584 — and `vpmaddubsw`/`vpdpbusd`/`SDOT` give i8→i32 multiply-accumulate
in one instruction, which is why the format is specified that way.

## Address space vs memory (`--malloc-arenas`)

`VmSize` is address space, not memory use. This process measured 1723 MiB of VmSize against 67 MiB
of RSS, and the difference is glibc: every thread that allocates gets its own malloc arena, each
reserving 64 MiB of PROT_NONE address space. 20 rayon workers + tiny_http's threads = 26 arenas
≈ 1.7 GiB of untouched space.

Glancing at VmSize is therefore misleading, but address space is not free *everywhere* — `RLIMIT_AS`,
32-bit targets, `vm.overcommit_memory=2`, and some container/Android configurations care. Hence:

```
--malloc-arenas N   # 0 = glibc default; 2 or 4 for a small address space
```

`mallopt(M_ARENA_MAX, N)`, not the `MALLOC_ARENA_MAX` environment variable — glibc reads that during
`ptmalloc_init`, which has already run by the time `main` starts.

| `--malloc-arenas` | VmSize | RSS | 10 requests (20 threads) |
|---|---|---|---|
| 0 (default) | 1723 MiB | 66.9 MiB | 22.2 s |
| 2 | **187 MiB** | 66.8 MiB | 23.5 s |
| 4 | 315 MiB | 66.8 MiB | — |

So it is a 9× address-space reduction for free in resident memory, at a possible few percent of
throughput from arena contention when 20 threads share two arenas (small sample; treat as "no clear
cost, no clear win"). Desktop: leave it. Phone or a memory-limited sandbox: set 2-4.

## KV cache: lazy growth, and what it actually saves

The model's tables are sized for `max_len = 1100` latent frames — at 25 fps that is **44 s** — while a
typical utterance is a 120-character sentence, about 5 s (125 frames). The cache starts at 128 frames
(5.1 s) and doubles on demand, rebuilding the RoPE tables only when it grows; anything longer than
`max_len` is refused rather than silently truncated.

| | VmSize | VmRSS at startup |
|---|---|---|
| eager, 1100 frames | 711 MB | 68.7 MB |
| lazy, 128 → grows | 683 MB | 68.0 MB |

Honest reading: that is **28 MB of address space and ~0.7 MB of RSS**. `Array3::zeros` does not fault
pages in until they are written, so the classic "preallocated 31.5 MB" was mostly virtual all along.
The lazy version still earns its place (no 31.5 MB over-commit, which matters on a phone), but it is
not the 24 MB resident saving the arithmetic suggests — that mistake is easy to make and I made it.

For contrast, the per-voice duplication *was* resident: caching one `Voice` per voice name cost
**42 MB of RSS each** (cloned int8 weights + a separately touched cache). Voices now cost a 28 KiB
prefix swap.

## Startup and residency (measured)

Two very different "startups" get conflated, so they are separated here:

| what | cost | resident |
|---|---|---|
| `woof-rs` exec + int8 weight load (38.7 MB, page-cached) | **0.03 s** | 133 MB peak |
| python wrapper: numpy import + normaliser construction | **0.13 s** | 41 MB |
| numpy runtime (fallback only) | — | 245 MB peak |
| spawning `woof-rs` per request (instead of the numpy path) | ~0.03 s | — |

So startup is **3–12 %** of time-to-first-audio (0.25–0.9 s); the rest is synthesis. Consequences,
which is what the wrapper does:

* **Keep the HTTP wrapper resident.** It owns the endpoint, the normaliser and the voice choice, and
  llama-swap proxies to a long-lived child by design — a spawn-per-request script would break
  `ttl`, preload and the health check.
* **Do not keep the numpy TTS weights resident** when the Rust path is available: that is what takes
  the seat from ~293 MB to ~41 MB.
* **Do not keep it resident either**: ~133 MB of RSS and a worker protocol to save ~30 ms per request
  is not a trade worth making at these latencies.

## Normaliser: ported, and measured against the reference

`src/normalise.rs` is a staged port of `underdog_woof/normalize.py` — 670 lines of rules, ~30 tables
and a 590 KB respelling lexicon, whose behaviour is defined only by that code. So the port is
measured, not asserted:

```
$ python tools/normalise_corpus.py --repo /path/to/underdog_woof   # reference -> tests/data/normalise_cases.tsv
$ python tools/gen_normalise_tables.py --repo /path/to/underdog_woof   # -> src/normalise/tables.rs
$ python tools/normalise_diff.py                                   # 317/317 cases match (100%)
$ python tools/normalise_fuzz.py --repo /path/to/underdog_woof --count 2000   # 2000/2000 (100%)
```

Behaviour is checked three ways: the fixed corpus (numbers, money, dates, times, units, codes, Roman
numerals, symbols, abbreviations, hard names and prose), a seeded fuzz generator that composes those
same atoms into sentences, and three paragraphs of real prose — all byte-identical to the Python
reference over ~7,000 cases. `--normalise "<text>"` prints the result, `--normalise-file` batches,
and `--no-normalise` reproduces the reference's `normalize=False`.

Why not transpile or embed instead? Both were checked: `py2many` is not on crates.io and its Rust
backend has no `re` support, and the reference uses 156 regex operations (63 with lookarounds), so a
transpiler cannot carry it. `rustpython-vm` 0.6.0 — the only Python-in-Rust embedder — fails to
compile against rustc 1.99 (10 errors in `rustpython-codegen` from `ruff_python_ast` churn), and
RustPython's `re` is a re-implementation, so "run the file unchanged" would not guarantee identical
behaviour anyway. Translated-with-a-differential-corpus is the route that can be *proven*.

**Status: complete.** `norm_tokens` → `norm_pre` → `fallback` → whitespace tidy → respell →
capitalisation, in the reference's order, including the bits that are easy to get wrong: Python's
`or`/`and` precedence inside `roman_sub`, the `\b` that stops the money rule eating a following
comma, and the fallback's year rule running before plain numbers (so `1204` is "twelve oh four", not
"one thousand two hundred and four").

**The tables are generated, not retyped.** `tools/gen_normalise_tables.py` walks the reference with
`ast` and emits `src/normalise/tables.rs`: 31 constants (ABBR 73 entries, CURCODE 40, SYMBOLS, UNITS,
SING, TITLES, REGNAL, ROMAN_*, DAYS, MON_ABBR, MONTHS, ORD, STREET, ID_WORDS, CODE_CTX, the long
regex alternations, …). Anything the evaluator cannot handle is *reported*, never guessed — currently
one entry (`LETTER`, an identity comprehension Rust does not need). That leaves only the 19 rule
functions as hand work, and the tables cannot drift from the reference by transcription.

The lexer-level shim that made this possible is `src/normalise/re.rs`: lookarounds through
`fancy-regex` (the reference uses 63 of them), Python replacement syntax (`\1`, `\g<n>`) expanded
locally rather than rewriting 127 call sites, a compiled-pattern cache, and `sub_with_ctx` for the
context rules that read `m.string` before and after a match.

The respell lexicon (`respell.tsv`, 24,883 entries) is embedded in the binary and matched the way the
reference does: NFKD-folded keys, a "only when capitalised" flag, longest multi-word key first, and
the hyphen rule (`Worcestershire-glazed` matches, `dove-cote` does not).

Because the binary now normalises, `woof_server.py` passes raw text through on the Rust path (no
double normalisation) and only normalises in Python for the numpy fallback.

## Benchmark

`--bench` runs a built-in corpus (short commands, medium sentences, long clauses, multi-sentence
paragraphs) through one process, so weights load once; `--bench-file lines.txt` uses your own text and
`--bench-json out.json` writes the raw numbers. Current output, desktop, 4 threads:

```
chars utts 1st.ch 1st.audio   audio.s   wall.s    RTF draws
    4    1     4     0.17s     1.24s    0.21s   5.9x     1
   38    1    38     0.16s     2.60s    0.41s   6.3x     1
  155    2    68     0.19s     9.00s    1.44s   6.2x     2
...
total: 10 texts, 15 utterances, 51.1s audio in 7.6s wall (6.7x realtime)
first audio: min 0.16s  median 0.17s  max 0.20s
```

**The `draws` column only means something with `--chunk-frames 0`.** Chunked output is single-take by
construction (audio already played cannot be redrawn), and every text's first utterance is the streamed
one, so at the default settings the bench cannot exercise the guard at all — it reports 1.00 draws and
0 bad takes no matter what. For the guard, ask for sentence-level decoding:

```
$ woof-rs --bench --text x --chunk-frames 0
first audio: min 0.26s  median 0.57s  max 0.87s
guard: 1.47 draws/utterance, 2/15 utterances kept a bad take
  < 20 chars:   3 utterances, 3.33 draws each, 67% bad
       20-49:   4 utterances, 1.00 draws each, 0% bad
       50-99:   7 utterances, 1.00 draws each, 0% bad
        100+:   1 utterances, 1.00 draws each, 0% bad
```

That table is the argument for the flags below: **short utterances can never pass the `spc` test**
(1.2 s of audio for 4 characters), so the guard redraws them three times and keeps the first take
anyway. Anything ≥ 20 characters never fired the guard once in this corpus.

## Streaming

The reference synthesises the whole text before returning audio (`Woof.__call__` splits into
utterances, then concatenates). For playback that is the wrong shape, so the crate always splits —
matching the reference — and `--stream` writes each utterance the moment it lands:

```
$ woof-rs --model /tmp/woof-i8 --text "The review was moved to Thursday at ten. Please bring the cost
projections." --out out.wav --stream
first audio after 0.24s (2 utterance(s) to synthesise)
2 utterances, 5.36s of audio, 0.82s wall (6.5x realtime)
```

**The responsive preset is the default**: `--chunk-frames 26` emits audio every 26 latent frames
(~1 s of audio) for the *first* utterance, so playback starts without waiting for the whole sentence.
Chunking takes median first audio from 0.63 s to 0.17 s and, more usefully, **bounds the worst case**
(0.89 s -> 0.20 s) so latency stops depending on sentence length or on whether the guard redraws.

The chunk size is not a free choice, and 13 (the first value tried) turned out to be the expensive
one. Until the fix below, the cost grew with the *number* of chunks — every push dequantised the whole
decoder. Measured on one 4.12 s sentence, production rate by chunk size:

| chunk frames | desktop | Tab S9 | N150 NAS |
|---|---|---|---|
| 13 | 5.4x realtime | — | 1.0x |
| **26 (default)** | **6.1x** | — | 1.1x |
| 52 | 6.3x | — | — |
| 0 (per utterance) | 7.0x | — | 1.7x |

First audio is 0.31 s at 13 frames and 0.37 s at 26, so 13 is now a reasonable choice when latency
matters more than twelve percent of throughput. The NAS column comes from a loaded box and moves by
50% between runs; the tablet was not attached for these numbers, which is the one gap worth closing.

**What the chunking cost actually was.** The previous version of this section said "the transformer is
re-run over the prefix per chunk". That was wrong by an order of magnitude, and the wrongness is the
useful part:

* Measured per push at 13 frames: **transformer 15 ms, SEANet stages 145 ms**.
* The stages dequantised **all 31M decoder weights on every push**: `Conv::taps()` was written for
  whole-utterance decoding, where it runs once per sentence (the same assumption `Linear::to_dense`
  documents), and streaming quietly turned "once per sentence" into "once per 13 frames".
* The dequantiser then made it worse than it needed to be: bounds-checked `[[row, col]]` indexing per
  element and, for the `axis != 0` layout the conv tensors use, an `IxDyn` coordinate vector built
  *per element*. Hoisting the group scale, iterating over slices, and replacing the per-element
  coordinate work with one reshape plus one axis permutation took the per-push stage cost from 145 ms
  to ~10 ms. Chunked throughput went **2.4x -> 5.4x** realtime, unchunked 5.3x -> 7.0x, with identical
  values: a unit test pins the fast kernel against the plain loop, and parity is unchanged at 85.3 dB.

**The KV cache** (`LayerCache`) keeps keys and values between pushes, so a push runs the transformer
over its own frames only. That is what the 15 ms is measured with: it removes ~10% of a push and, more
usefully, removes the latent-prefix buffer entirely. It also took streaming parity from 65 dB to
**102 dB (1 LSB)**, because cached keys are reused exactly instead of recomputed from a trimmed prefix.

What remains is proportional to the frames decoded, which is the shape you want: a 13-frame push costs
a 13-frame push.

`tools/stream_parity.py` guards the arrangement: chunked output is compared against whole-utterance
decoding for the same seed, and is **bit-identical** for every utterance shorter than the window.
Beyond it the matrix shapes differ, so the last bit does (1 LSB, ~102 dB; the tool fails below 60 dB).

The other trade is the guard: chunked audio has already been played, so it cannot be redrawn. That is
why it applies to the first utterance only, and why a buffered `--out` file is untouched by it — there,
retries still fire and the take can still be replaced. `--chunk-frames 0` restores one emission per
utterance.

`--pcm-stdout` pipes raw 16-bit PCM instead of a WAV (`- | aplay -f S16_LE -r 24000`), and
`--first-max N` shortens only the *first* utterance at a clause boundary — trading a little prosody
for latency: on a 3-sentence sample, sentence-level decoding went from 0.87 s to **0.62 s** with
`--first-max 40`, and on top of chunking from 0.26 s to 0.24 s (it mostly stops mattering once audio
starts at the first chunk). The WAV header is patched on close, so a player can start reading
immediately.

### Latency: what the two knobs buy

Benchmark over the 10-text corpus, desktop, 4 threads:

| policy | first audio (median) | first audio (worst) | throughput |
|---|---|---|---|
| sentence-level (`--chunk-frames 0`) | 0.63 s | 0.89 s | 5.7x |
| `--chunk-frames 26` (default) | **0.17 s** | **0.20 s** | **6.7x** |
| `--chunk-frames 13` | **0.13 s** | **0.15 s** | 5.7x |

The interesting column is the third one: frame-level streaming does not just lower the median, it
**bounds the worst case** — latency stops depending on sentence length or on whether the guard
redraws, and settles at one chunk (13 or 26 frames) plus one decode pass.

`--chunk-frames` applies to the *first* utterance only: incremental decoding costs throughput in
proportion to the number of chunks (the transformer is re-run per chunk; the SEANet stages are the
stateful part), which is wasted on audio nobody is waiting for. Chunked output matches whole-sentence
decoding — every stage is causal, so it is bit-identical for utterances under the attention window and
within 2 LSB beyond it (`tools/stream_parity.py` checks this) — and the guard is skipped for that
utterance by construction, since emitted audio cannot be redrawn.

## Play it through the speakers (Linux)

Pipe the PCM straight into PipeWire, PulseAudio or ALSA — `tools/say.sh` picks whichever is present:

```
tools/say.sh --text "Your order ships on March 3rd. Call 415-555-0199 with questions."
tools/say.sh --voice mira --first-max 40 --text "Longer text here. The first clause is short on purpose."
```

or by hand, each verified on a PipeWire desktop:

```
woof-rs --text "…" --pcm-stdout --stream | pw-cat --playback --raw --format=s16 --rate=24000 --channels=1 -
woof-rs --text "…" --pcm-stdout --stream | pacat --format=s16le --rate=24000 --channels=1 --latency-msec=120
woof-rs --text "…" --pcm-stdout --stream | aplay -f S16_LE -r 24000 -c 1 -q --period-size=1024 --buffer-size=4096
```

`pw-cat` needs `--raw` (otherwise it tries to sniff stdin as a file and fails with
"Format not recognised"). `--first-max N` shortens only the first utterance (when it still has a clause boundary left after
splitting), so sound starts sooner without touching the rest. `--retries N` / `--no-retry-first`
control the bad-take guard described below.

Two things worth knowing when reading the timings:

* **A real-time sink paces the writer.** Synthesis runs at 6-7× realtime, but piping into an audio
  device blocks once the playback buffer fills, so wall time tracks audio length. The
  `first audio after …` line then reports when the *first utterance finished being written*, not
  when sound started — the true latency is the speech time for that utterance (0.4-0.9 s) plus the
  player's buffer (~0.1-0.2 s). Measured without a sink (`--pcm-stdout | wc -c`): 0.55 s for a 43-char
  first sentence.
* **A growing WAV is playable.** `--stream --out file.wav` writes a header declaring an unknown
  length (`0xFFFF_FFFF`) and patches the real sizes on close, so `paplay file.wav` started mid-write
  plays what exists and stops at EOF.

## Use

```
python tools/export.py --repo /path/to/underdog_woof --out /tmp/woof-i8
cargo run --release -- --text "Your order ships on March 3rd." --out out.wav --timings
cargo run --release -- --text "Several sentences. Written as a stream." --out out.wav --stream
python tools/parity.py --model /tmp/woof-i8      # expect ~72 dB
cargo test --release
```

Useful flags: `--voice mira`, `--seed N` (same text + seed is byte-identical), `--threads N`
(default 4, `0` = auto), `--chunk-frames N` (default 26, `0` = per utterance), `--retries N`,
`--first-max N`, `--max-chars N` (0 = no limit), `--cpu-mask 3,4,5,6,7`, `--timings`,
`--verbose` (emission log), `--dump-latents`, `--dump-tensor NAME PATH`, `--bench-dot`,
`--features gpu --bench-gpu`, `--features gpu-resident --gpu-resident`, `--verify-gpu`,
`--serve --port N`, `--w8a8`, `--offline`.

Diagnostics: `WOOF_DUMP_STAGES=<dir>` writes the decoder's transformer output and every stage
output as raw f32, which is how each parity regression so far was localised; `--timings` splits each
utterance into speech time and decoder time (transformer / head / tail); `--verbose` logs every
emission with the wall time it arrived, which is how playback starvation is measured.

Three tools are the regression net: `tools/parity.py` (decoder against numpy),
`tools/normalise_diff.py` (317 recorded cases, byte-for-byte) and `tools/stream_parity.py` (chunked
output against whole-utterance decoding). All three need weights; the 47 unit tests do not.

## Measured

Same sentence (4.20 s of audio), weights int8 resident, ~120 MB peak RSS:

| machine | speech | decoder | wall | realtime |
|---|---|---|---|---|
| i5-14600KF, 4 threads | 0.37 s | 0.19 s | 0.63 s | **6.6x** |
| N150 NAS, 4 threads | — | — | 1.79 s | 2.3x |
| Galaxy Tab S9, 4 threads | — | — | — | 2.8x (measured earlier) |

Same sentence, 4.12 s of audio, weights int8 resident. The desktop split comes from `--timings`
(`speech` is the autoregressive model, `decoder` the vocoder: transformer 0.02 s, SEANet 0.17 s). The
tablet row is from an earlier round and is the one number here not re-measured since the dequantisation
fix — the device was not attached.

Parity against the numpy reference is measured on the stretch of audio where the two implementations
are comparable, and `tools/parity.py` finds that stretch itself:

```
$ python tools/parity.py --model /tmp/woof-i8
latents, first 71/137 frames: 85.3 dB  max|d| 5.43e-04
waveform, first 68160 samples: 71.8 dB  max|d| 2.73e-04   (5.48s total)
  diverged at frame 71 (per-frame max|d| > 0.001) -- chaotic amplification, not a porting
  difference; whole-take latents 37.2 dB
PASS: 71 frames agree at 85.3 dB (needs > 60 dB over >= 20 frames)
```

Why the harness does not simply report the whole take: the model's autoregressive loop is chaotic, so a
1e-7 difference amplifies until the trajectories are unrelated. The reference **disagrees with itself
at 57.6 dB** (diverging at frame 113 of 137) when only `OPENBLAS_NUM_THREADS` changes from 4 to 1 —
same code, same seed. A shorter take can stay inside the coherent window for its whole length (105 of
105 frames at 89.0 dB), which is why the `--no-normalize` control run reports no divergence at all;
that is a property of the take's length, not a better port. The reference's own `TorchRandn` stream, the
tokeniser, and `GroupedInt8`'s dequantisation are covered by unit tests.

Decoder breakdown (desktop): transformer 0.030 s, stages 0..6 0.182 s, chunked tail 0.181 s — the
SEANet stack is ~91 %, which is why GPU work targets it and not the speech loop.

## Notes worth keeping

**Quantising activations does not work** (measured, then reverted). The flow head emits a small
correction (`z = noise + head(cond, noise)`), so 8-bit activation error is the same order as the
term being computed, and 105 autoregressive steps amplify what is left:

| activations | first-frame error | waveform vs reference |
|---|---|---|
| f32 (as shipped) | 9.5e-07 | **72.2 dB** |
| int16 Q15, per-32-block | 4.9e-05 | 34.4 dB |
| int8, per-32-block | 8.3e-03 | 3.2 dB |

**big.LITTLE affinity is worth more than SIMD.** On the Tab S9 the speech pass took 2.32 s pinned
to a big core, 8.43 s on a little core, and 2.15 s with six unpinned threads — worse than one lucky
thread, because work-stealing hands a third of the rows to the 2 GHz cores. Pinning gives 0.64 s.
Two traps: `ThreadPoolBuilder::build()` does not install the pool (`build_global()` does), and
per-thread `sched_setaffinity` in `start_handler` is a no-op on Android — pin the main thread
*before* the pool exists so the workers inherit the mask.

**GPU scope.** Two GPU paths, both numerically verified against the reference:

| path | desktop decoder | Tab S9 decoder | N150 NAS iGPU | correct? |
|---|---|---|---|---|
| CPU, 4 threads (RTX 5090 host) | 0.19 s | 0.60 s | — | 72.2 dB |
| `--gpu` (per-op `MatMul`) | 0.40 s | 2.12 s | — | 72.2 dB |
| `--gpu-resident` (whole stack on device) | **0.04 s** (RTX 5090) | 1.16 s | 2.4-2.7 s | 72.2 dB |

**The NAS result is the same story as the phone.** Its ADL-N iGPU needed Mesa ≥ 24 to be recognised
at all — Ubuntu 22.04 ships 23.2, which rejects the device ID (`8086:46d4`), so the run above used
Mesa 25.2 in a container with `--device /dev/dri`, no host changes. There, `--verify-gpu` agreed on
all 15 stages (129-467 dB, float32 noise) and the resident decoder took 2.4-2.7 s for a 4.12 s
utterance whose *entire* CPU pipeline runs in 4.04 s — so the GPU is roughly 2x the cost of the CPU
decoder alone. Consistent with the tablet: mobile-class GPUs lose this workload to submit overhead and
memory-bound matvecs, and a shared-memory iGPU on a swapping box has nothing to win.

`--verify-gpu` compares every stage against the CPU runner: all 15 stages agree at 129–468 dB, i.e.
to float32 noise. Getting there needed two things beyond the kernel: staging state on the device
(conv history, convt carry) and grid-stride shader loops, because one dispatch dimension caps at
65535 workgroups and the high-rate stages exceed it.

The per-op path loses because it pays a GPU↔CPU round trip per matmul (~0.5 ms for ~20 µs of work).
The resident path wins big on a discrete GPU and **loses on the phone** (1.07 s vs 0.60 s), for two
measured reasons:

* **Submits, not FLOPs, came first**: one `queue.submit` per op cost ~0.35 s of the tablet's 1.16 s.
  Recording the whole chain into a single encoder (with a grid-stride shader loop for the 65535
  workgroup cap) took it to 0.80 s.
* **Two kernel shapes were tried and rejected on measurement**: fusing taps into one GEMM via
  im2col (0.80 → 0.91 s — the gather traffic costs more than the dispatches it saves), and a 16×64
  tile with four outputs per thread (0.80 → ~1.0 s — the decoder's `n` runs from 768 down to 1, so a
  64-wide tile wastes most of its work and starves the grid). Both are recorded in the shader
  comments so they are not re-attempted blindly.

What is left for mobile would be tiling chosen per stage shape, plus fp16 arithmetic — and fp16 is
out, because it changes the numerics the same way int16 activations did (34 dB).

**A race worth remembering**: `queue.write_buffer` is ordered against the *next submit*, not
against an encoder created earlier, so uploading the transformer output after opening the batch let
the dispatches read uninitialised memory. `Gpu::write`/`fill` now flush an open batch first.

## Not done yet

* **Android packaging**: `cargo build --target aarch64-linux-android` works today (NDK 28.2 clang as
  the linker; 4.5 MiB CPU binary with the normaliser and the lexicon). `oboe`/JNI around the existing
  core, and a `wasm32` target from the same library, are the remaining steps.
* **Mobile GPU kernels**: vectorised loads and one GEMM per convolution (the taps are accumulated as
  separate dispatches today) before the device path beats the CPU on a phone — and on the two
  mobile-class GPUs measured (Adreno Xclipse in a Tab S9, Intel ADL-N in an N150 NAS) it currently
  loses, for reasons recorded in the shader comments.
* **Tablet numbers for the current chunk costs**: the measurements above are desktop and NAS; the Tab
  S9 is the device where a marginal default actually matters, and it was not attached for the last
  round.
* **Voice conditioning beyond two voices**: the model ships Juno and Mira; there is no way to
  generate a new voice, and the port does not pretend otherwise.
