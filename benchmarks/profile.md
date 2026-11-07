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

Order of work implied by the numbers: fuse the scale into the dot (kills most of the matvec cost and a
large share of the allocations) → remove the remaining per-frame allocations in attention/flow → then
consider overlapping the decoder with the sampler.
