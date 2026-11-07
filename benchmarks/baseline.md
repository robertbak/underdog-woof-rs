# Baseline before the speed work

**Superseded**: current numbers are in the README section "Speed work (v0.1.3)" and in
`benchmarks/profile.md`. This file is the reference the quality proxy compares audio against.

Recorded 2026-11-07, before any optimisation, with `v0.1.2`. Every later change is measured against
these numbers on the same machine; a change that cannot beat them by 5% does not ship.

```
CPU     Intel Core i5-14600KF (20 logical CPUs, SMT, pinned to physical cores)
load    1.38 1.43 1.90   (other work on the box; re-measure, don't compare across days)
build   cargo build --release, 4 threads unless stated
```

## Throughput and latency (`--bench`, 10 texts / 15 utterances / 51.1 s of audio)

| threads | wall | throughput | first audio min / median / max |
|---|---|---|---|
| 1 | 13.7 s | 3.7x | 0.28 / 0.30 / 0.33 s |
| 2 | 9.3 s | 5.5x | 0.20 / 0.21 / 0.23 s |
| **4 (default)** | **7.5 s** | **6.8x** | **0.15 / 0.17 / 0.20 s** |

Three runs at four threads gave 7.6 / 7.5 / 7.5 s, so the run-to-run spread is under 2%.

## Where the time goes (`--timings`, one 4.12 s sentence, 103 frames, 4 threads)

| stage | time | share |
|---|---|---|
| speech (autoregressive sampler) | 0.35 s | 65% |
| decoder, total | 0.19 s | 35% |
| — latent transformer | 0.02 s | |
| — SEANet head stages | 0.06 s | |
| — SEANet tail stages | 0.11 s | |

The sampler is the majority, and the decoder sits *serially* behind it — nothing overlaps today. That
makes "hide the decoder behind the sampler" the most obviously available structural win, worth up to a
third of the wall time if it hides completely.

## A kernel fact worth carrying into the work

Micro-benchmarked separately (a 448x1344 int8 matvec, one core): **24 GMAC/s** through the int8 path
against **2.7 GMAC/s** through the default f32 path (which dequantises the row into a scratch buffer
first). A 9x kernel difference that produced only a 1.4x end-to-end difference means the big matvecs
are a minority of a frame's cost — so the remaining time is in per-frame overhead: allocations,
elementwise passes over small vectors, attention, the flow head, and rayon dispatch. Optimising the
matvec kernel further is not where the money is.

## Quality baseline

`tools/quality_proxy.py` recorded six takes (plain prose, seed 1, single take) to
`~/.cache/woof-rs/baseline`, alongside their durations, peaks and 20 ms energy envelopes.

Re-run the whole baseline like this:

```bash
M=/path/to/woof-i8
cargo build --release
for t in 1 2 4; do ./target/release/woof-rs --model $M --bench --text x --threads $t; done
./target/release/woof-rs --model $M --text "The committee met on Tuesday and agreed the budget needs review." \
    --pcm-stdout --stream --chunk-frames 0 --threads 4 --timings > /dev/null

# quality: record once, then check any candidate binary against it
./tools/quality_proxy.py --record ~/.cache/woof-rs/baseline --model $M
./tools/quality_proxy.py --against ~/.cache/woof-rs/baseline --model $M \
    --repo /path/to/underdog_woof --python /path/to/venv/bin/python
```

The proxy's audio checks are stdlib-only; the correctness tools it drives (`normalise_diff`,
`parity`, `stream_parity`) need numpy, hence `--python`. Thresholds: duration within 5%, energy
envelope correlation at least 0.80, peak within 30% — the "same words, same prosody, a different take"
contract, since the sampler is chaotic and identical audio is not a reasonable bar.
