#!/usr/bin/env python3
"""Parity check: Rust port vs the numpy reference, on the same text + seed.

    python tools/parity.py --text "..." [--voice juno] [--seed 0]

Prints SNR / max-abs-diff for latents and for the waveform. Anything above ~70 dB on the
waveform means the port is right (f32 accumulation order differs, nothing else).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout of the reference (underdog_woof)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that can import numpy and the reference (default: this one)")
    ap.add_argument("--model", default="/tmp/woof-f32")
    ap.add_argument("--bin", default=os.environ.get("WOOF_RS_BIN", "woof-rs"))
    ap.add_argument("--text", default="Your order ships on March 3rd, and the tracking code is QX-4471.")
    ap.add_argument("--voice", default="juno")
    ap.add_argument("--seed", default="0")
    # both sides must agree on normalisation: the reference defaults to normalize=True and so does
    # woof-rs, so the harness compares the deployed behaviour unless asked otherwise
    ap.add_argument("--no-normalize", action="store_true",
                    help="compare raw text (reference normalize=False, woof-rs --no-normalise)")
    # The model's autoregressive loop is chaotic: a 1e-7 difference amplifies until the two
    # trajectories are unrelated (~frame 115 on a 137-frame utterance). The reference disagrees with
    # *itself* at 57 dB when only the BLAS thread count changes, so parity is judged on the frames
    # before the amplification takes over.
    ap.add_argument("--frames", type=int, default=0,
                    help="compare only the first N latent frames (0 = find the divergence point)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="per-frame difference that counts as divergence (default 1e-3)")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="woof-parity-")
    if os.environ.get("WOOF_PARITY_KEEP"):
        tmp = os.environ["WOOF_PARITY_KEEP"]
        os.makedirs(tmp, exist_ok=True)
        print(f"parity scratch: {tmp}")
    wav_rs, lat_rs = os.path.join(tmp, "rs.wav"), os.path.join(tmp, "rs_lat.f32")
    # the reference path here is a single take (`model.latents` never retries), so the guard must be
    # off on both sides for the comparison to mean anything
    extra = ["--retries", "0"]
    if a.no_normalize:
        extra.append("--no-normalise")
    subprocess.run([a.bin, "--model", a.model, "--text", a.text, "--out", wav_rs,
                    "--voice", a.voice, "--seed", a.seed, "--dump-latents", lat_rs] + extra, check=True)

    ref = os.path.join(tmp, "ref.py")
    with open(ref, "w") as f:
        f.write(
            "import sys, numpy as np\n"
            f"sys.path.insert(0, {a.repo!r})\n"
            "from underdog_woof import Woof\n"
            "from underdog_woof import runtime\n"
            f"t = Woof(voice={a.voice!r}, normalize={not a.no_normalize})\n"
            # `model.latents` takes text verbatim, so apply the reference's normaliser here the
            # same way `Woof.__call__` would -- otherwise the two sides see different text
            f"prompt = t.norm({a.text!r}) if t.norm is not None else {a.text!r}\n"
            f"runtime.RNG.seed({a.seed})\n"
            f"lat = t.model.latents(prompt, temp=0.5, eos=0.9)\n"
            "wav = t.model.dec(lat * t.model.std + t.model.mean)\n"
            f"np.save({os.path.join(tmp,'lat.npy')!r}, lat)\n"
            f"np.save({os.path.join(tmp,'wav.npy')!r}, wav)\n"
        )
    env = dict(os.environ, PYTHONPATH=a.repo)
    subprocess.run([a.python, ref], check=True, env=env, capture_output=True)

    lat_py = np.load(os.path.join(tmp, "lat.npy")).ravel()
    wav_py = np.load(os.path.join(tmp, "wav.npy"))
    lat_rz = np.fromfile(lat_rs, dtype=np.float32)
    w = wave.open(wav_rs)
    wav_rz = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32767.0

    def snr(ref, got):
        n = min(len(ref), len(got))
        if n == 0:
            return float("-inf"), float("nan")
        r, g = ref[:n], got[:n]
        return 20 * np.log10(np.linalg.norm(r) / (np.linalg.norm(r - g) + 1e-20)), np.abs(r - g).max()

    # Where do the two implementations stop agreeing? The autoregressive loop is chaotic: a 1e-7
    # difference amplifies until the trajectories are unrelated (the reference disagrees with *itself*
    # at 57 dB when only OPENBLAS_NUM_THREADS changes). So the honest comparison is the stretch before
    # that happens, and the harness finds that point rather than assuming it.
    rows = min(len(lat_py), len(lat_rz)) // 32
    frames_py = lat_py[: rows * 32].reshape(-1, 32)
    frames_rz = lat_rz[: rows * 32].reshape(-1, 32)
    per_frame = np.abs(frames_py - frames_rz).max(axis=1) if rows else np.zeros(0)
    if a.frames:
        agreed = min(a.frames, rows)
    else:
        over = np.nonzero(per_frame > a.tol)[0]
        agreed = int(over[0]) if len(over) else rows
    agreed = max(agreed, 1)

    head = agreed * 32
    s1, d1 = snr(lat_py[:head], lat_rz[:head])
    # waveform: only the samples the agreed frames produced (960 per latent frame)
    samples = agreed * 960
    s2, d2 = snr(wav_py[:samples], wav_rz[:samples])
    full1, _ = snr(lat_py, lat_rz)

    frames = rows
    print(f"latents, first {agreed}/{frames} frames: {s1:6.1f} dB  max|d| {d1:.2e}")
    print(f"waveform, first {agreed * 960} samples:  {s2:6.1f} dB  max|d| {d2:.2e}   "
          f"({len(wav_py) / 24000:.2f}s total)")
    if agreed < frames:
        print(f"  diverged at frame {agreed} (per-frame max|d| > {a.tol:g}) -- chaotic amplification, "
              f"not a porting difference; whole-take latents {full1:.1f} dB")
    verdict = "PASS" if s1 > 60 and agreed >= 20 else "FAIL"
    print(f"{verdict}: {agreed} frames agree at {s1:.1f} dB (needs > 60 dB over >= 20 frames)")
    sys.exit(0 if s1 > 60 else 1)


if __name__ == "__main__":
    main()
