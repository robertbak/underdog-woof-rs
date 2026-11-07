# Where a frame's 8.5 ms goes

`WOOF_PROFILE=1` instruments the sampler (and counts allocations through the global allocator). Span
classes are chosen so they *don't overlap*: the flow head and the prefix pass are counted whole rather
than as their constituent matvecs, so the columns add to the frame time. One 4.12 s sentence, 103
frames, `--retries 0`:

## Default (f32 activations)

| part | 1 thread | share | 4 threads | share | calls/frame | µs/call (1T) |
|---|---|---|---|---|---|---|
| Matvec | 662 ms | **75.8%** | 197 ms | 59.8% | 33 | 195 |
| Flow head | 115 ms | 13.1% | 40 ms | 12.1% | 2 | 556 |
| Attention | 54 ms | 6.2% | 49 ms | 14.7% | 8 | 66 |
| Prefix (one-off) | 40 ms | 4.6% | 42 ms | 12.7% | 0.1 | 5019 |
| Elementwise | 2 ms | 0.2% | 2 ms | 0.6% | 24 | 0.9 |
| **per frame** | **8.5 ms** | | **3.2 ms** | | | |

allocations: **6,880 per frame, 14.4 MiB per frame** (at 4 threads: 7,701 and 15.9 MiB).

## With `--w8a8` (int8 activations)

| part | 1 thread | share | calls/frame | µs/call |
|---|---|---|---|---|
| Matvec | 123 ms | 50.9% | 33 | **36** |
| Attention | 53 ms | 21.8% | 8 | 64 |
| Prefix (one-off) | 40 ms | 16.8% | 0.1 | 5056 |
| Flow head | 23 ms | 9.7% | 2 | 113 |
| Elementwise | 2 ms | 0.9% | 24 | 0.8 |
| **per frame** | **2.3 ms** | | | |

allocations: 6,593 per frame, 13.6 MiB per frame.

## What this says

1. **The matvec is 76% of a frame, and it is not the arithmetic.** The same weights through the int8
   kernel cost 36 µs per call; the default path costs 195 µs for the *same product*, because it first
   dequantises the row into a `vec![0f32; cols]` scratch and reads it back. That scratch allocation is
   also where most of the 6,880 allocations per frame come from.
2. **Therefore the biggest win is not w8a8 and not a better kernel**: it is dequantising *inside* the
   dot, `s_row * Σ q_i·x_i`, which is the same product up to float rounding and removes the scratch
   write, the scratch read, and the allocation. It keeps f32 activations, so the take stays as close
   to the reference as it is today, unlike `--w8a8`.
3. **Parallelism is not free either**: the f32 matvec goes 195 µs → 58 µs from 1 to 4 threads (3.3x),
   which is why the end-to-end gain from int8 activations looks like 1.4x at 4 threads but 3.7x at
   one. The int8 path is already near its per-core limit, so it scales less.
4. Attention (66 µs/call, 8 calls) and the flow head (556 µs/call, 2 calls) are the next targets, and
   both allocate per call: `to_owned()` per head per frame, `Array2::zeros` per layer, `mapv` chains.
5. Elementwise chains are free (0.2%); fusing them is not worth the churn.

Order of work implied by the numbers: attack the matvec's *scratch traffic* → then the flow head and
attention → then overlapping the decoder with the sampler.

## What was tried on the matvec, and what the measurements said

All three variants compute the same product; they differ in where the dequantised values live. One
core, 448x1344 row block, `--threads 4` end-to-end on the bench corpus:

| variant | matvec µs/call | end-to-end | verdict |
|---|---|---|---|
| dequantise the whole row into a heap scratch, then dot (original) | 195 | 6.3x | the baseline |
| dequantise inside the dot loop: `Σ f32::from(q_i)·x_i`, scale after | 213 | 5.8x (at equal load: unchanged) | **rejected**: the per-element i8→f32 widen blocks the FMA path that a contiguous f32 dot gets |
| **tiles of 256 floats into a stack buffer** (chosen) | **156** | **7.1-7.4x** | kept: the dequantised block stays in L1 instead of streaming 2.4 MB of scratch through L2 per call |
| tile sizes 128 / 512 / 1024 | — | 7.0-7.4x | indistinguishable; 256 is 1 KiB, which is the reason it is 256 |

Also measured and rejected: removing the per-call *allocation* of the scratch (a thread-local reusable
buffer). It cut allocations per frame from 6,880 to 6,551 and changed the wall time by nothing —
interleaved A/B against the previous binary gave 6.2/6.2, 6.3/6.3, 6.5/6.6. The allocator is not the
bottleneck here, so the remaining 6,551 allocations per frame are not worth chasing on their own.

Side effect worth recording: the tiled summation order is *closer* to numpy than the row-at-a-time one,
so decoder parity improved from 85.3 dB to **86.7 dB** for free.

After the tiling, the profile at one thread is 7.4 ms/frame (was 8.5), with Matvec at 156 µs/call
(was 195), Flow 470 µs/call, Attention 101 µs/call.

## After the AVX2 kernel (and what was tried next)

With the fused int8->f32 AVX2 dot in place, one 4.12 s sentence at one thread:

| part | before (tiled) | after (AVX2) | share now |
|---|---|---|---|
| Matvec | 156 µs/call | **42 µs/call** | 51% |
| Attention | 95 µs/call | 95 µs/call | 25% |
| Prefix (one-off) | 5.6 ms/utterance | 5.1 ms | 14% |
| Flow head | 470 µs/call | 125 µs/call | 9% |
| Elementwise | 1 µs/call | 1 µs/call | 0.9% |
| per frame | 7.4 ms | **2.8 ms** | |

Attention and the prefix pass are now the visible remainder. Two attempts at attention, both measured
and both reverted:

* **views instead of copies** (drop the two `to_owned()` of each head's keys and values — ~3.4 MB of
  copying per frame): 95 -> **128 µs/call**, end-to-end 8.2x -> 7.8x. `ndarray`'s `dot` reaches BLAS
  for owned arrays and takes a generic path for views, so removing the copies moved the work somewhere
  slower.
* **views + a slice softmax** (no `(1, n)` weights matrix): 95 -> 74 µs/call for the block, but no
  end-to-end change beyond noise, because the transposed dot on a view copies internally.

So attention needs a hand-written kernel (or a contiguous scratch layout that BLAS likes), not an
`ndarray` refactor. That is future work, not a win to claim now.

The prefix pass is 5.1 ms of a 0.15-0.22 s time-to-first-audio: caching its KV per (voice, text) would
save under 4%, below the 5% bar this work is measured against, so it was not implemented.
