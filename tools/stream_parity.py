#!/usr/bin/env python3
"""Check that chunked streaming decodes the same audio as whole-utterance decoding.

`--chunk-frames N` makes the decoder run incrementally: the transformer sees the frames emitted so
far and each SEANet stage keeps its own state. That is a lot of machinery to get exactly right, so
it is checked here rather than assumed:

    python tools/stream_parity.py --model /tmp/woof-i8

The utterances in normal text are shorter than the decoder's attention window (250 frames = 10 s),
so the transformer sees the same rows and the output is bit-identical. Longer utterances make the
streaming path drop a prefix attention cannot see, which changes the shapes handed to the matrix
multiply: the result then differs in the last bit (~1 LSB at 16 bit, 60+ dB), not in the audio.

Exit code 0 when every case passes.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

TEXTS = [
    "The committee met on Tuesday and agreed the budget needs review.",
    "Your order ships on March 3rd, tracking QX-4471.",
    "One short one. Two short ones. Three.",
    # longer than the 250-frame attention window (no sentence breaks, so it stays one utterance)
    " ".join(["the committee agreed the budget needs a careful review before the end of the quarter"] * 6),
]

# One take each (`--retries 0`): the guard redraws a rejected utterance, a redraw consumes the RNG,
# and chunked output is single-take by design (it has already been played) -- so with redraws on,
# the two runs diverge for reasons that have nothing to do with streaming. This compares the
# decoder, not the guard.


def pcm(binary, model, text, chunk, seed):
    out = subprocess.run(
        [binary, "--model", model, "--text", text, "--pcm-stdout", "--stream",
         "--chunk-frames", str(chunk), "--seed", str(seed), "--threads", "2",
         "--retries", "0"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="weights prefix, as written by tools/export.py")
    ap.add_argument("--bin", default=os.environ.get("WOOF_RS_BIN", "woof-rs"))
    ap.add_argument("--chunks", default="13,26,52")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-snr", type=float, default=60.0, help="dB, for utterances over the window")
    a = ap.parse_args()

    failures = 0
    for text in TEXTS:
        whole = pcm(a.bin, a.model, text, 0, a.seed)
        label = text if len(text) <= 46 else text[:43] + "..."
        for chunk in [int(c) for c in a.chunks.split(",")]:
            streamed = pcm(a.bin, a.model, text, chunk, a.seed)
            n = min(len(whole), len(streamed))
            if n == 0:
                print(f"FAIL  {label}  chunk {chunk}: no audio")
                failures += 1
                continue
            identical = np.array_equal(whole[:n], streamed[:n])
            diff = whole[:n] - streamed[:n]
            snr = 10 * np.log10((whole[:n] ** 2).sum() / max((diff**2).sum(), 1e-12))
            ok = identical or snr >= a.min_snr
            failures += 0 if ok else 1
            verdict = "bit-identical" if identical else f"{snr:.1f} dB, max |d| {np.abs(diff).max():.0f} LSB"
            print(f"{'ok  ' if ok else 'FAIL'}  chunk {chunk:>2}  {len(whole) / 48000:5.2f}s audio  "
                  f"{verdict:38s}  {label}")
    print()
    print("PASS" if failures == 0 else f"FAIL: {failures} case(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
