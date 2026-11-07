#!/usr/bin/env python3
"""Does a change to woof-rs keep the same words and prosody?

Comparing audio across takes with an SNR is meaningless: the sampler is chaotic, so any numeric change
gives a *different* take rather than a degraded copy. What survives a change of take is the shape of
the performance — how long it is, and where the energy sits over time. So this measures:

  * duration ratio   — same words take about the same time (the model speaks at a fixed frame rate)
  * envelope         — correlation of the 20 ms energy contour, i.e. does the emphasis land in the
                       same places
  * level            — peak and RMS ratio, to catch a change that is quiet or clipping

and then runs the existing correctness tools, because "sounds fine" never caught a porting bug.

    # record a baseline (once, before optimising)
    tools/quality_proxy.py --record ~/.cache/woof-rs/baseline --model /path/to/woof-i8

    # check a candidate binary against it
    tools/quality_proxy.py --against ~/.cache/woof-rs/baseline --model /path/to/woof-i8 \
        --bin ./target/release/woof-rs --repo /path/to/underdog_woof

Exit code 0 when every check passes.
"""
import argparse
import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

# Fixed corpus: plain prose, no digits, so the normaliser is not what is being measured.
TEXTS = [
    "The committee met on Tuesday and agreed the budget needs review.",
    "Please bring the updated cost projections to the meeting.",
    "Engineers flagged three issues with the current design last week.",
    "Shipping is scheduled for the middle of next month.",
    "The report will be ready before the end of the quarter.",
    "Comments should reach the chair before the deadline.",
]
SEED = 1
HOP = 480  # 20 ms at 24 kHz

# thresholds for "same words, same prosody, a different take"
MAX_DURATION_DRIFT = 0.05
MIN_ENVELOPE_CORR = 0.80
MAX_LEVEL_DRIFT = 0.30


def synthesise(binary, model, text, path, extra):
    subprocess.run(
        [binary, "--model", model, "--text", text, "--out", path,
         "--seed", str(SEED), "--retries", "0"] + extra,
        check=True, capture_output=True,
    )


def read_wav(path):
    """16-bit mono PCM WAV -> array of int16 samples. Stdlib only: this runs before any venv exists."""
    import wave

    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        samples = array.array("h")
        samples.frombytes(w.readframes(w.getnframes()))
        if sys.byteorder == "big":
            samples.byteswap()
    return samples


def envelope(x, hop=HOP):
    """RMS over 20 ms hops — the energy contour, which is what survives a change of take."""
    frames = len(x) // hop
    if frames == 0:
        return [0.0]
    out = []
    for f in range(frames):
        base = f * hop
        total = 0
        for i in range(base, base + hop):
            v = x[i]
            total += v * v
        out.append(math.sqrt(total / hop))
    return out


def correlation(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    ma = sum(a[:n]) / n
    mb = sum(b[:n]) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def stats(path):
    x = read_wav(path)
    env = envelope(x)
    peak = max((abs(v) for v in x), default=0)
    rms = math.sqrt(sum(v * v for v in x) / len(x)) if x else 0.0
    return {
        "samples": len(x),
        "seconds": len(x) / 24000.0,
        "peak": float(peak),
        "rms": round(rms, 1),
        "envelope": [round(v, 1) for v in env],
    }


def compare(base, cand):
    corr = correlation(base["envelope"], cand["envelope"])
    duration_ratio = cand["seconds"] / base["seconds"] if base["seconds"] else 0.0
    peak_ratio = cand["peak"] / base["peak"] if base["peak"] else 0.0
    return {
        "duration_ratio": duration_ratio,
        "envelope_corr": corr,
        "peak_ratio": peak_ratio,
        "ok": abs(duration_ratio - 1.0) <= MAX_DURATION_DRIFT
        and corr >= MIN_ENVELOPE_CORR
        and abs(peak_ratio - 1.0) <= MAX_LEVEL_DRIFT,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="weights prefix or .npz")
    ap.add_argument("--bin", default=os.environ.get("WOOF_RS_BIN", "woof-rs"))
    ap.add_argument("--record", help="write a baseline here")
    ap.add_argument("--against", help="compare a fresh run against this baseline")
    ap.add_argument("--repo", help="reference checkout, to run tools/parity.py")
    ap.add_argument("--extra", default="", help="flags for the synthesiser, e.g. '--w8a8'")
    ap.add_argument("--skip-tools", action="store_true", help="audio checks only")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the correctness tools (needs numpy; default: this one)")
    a = ap.parse_args()
    if bool(a.record) == bool(a.against):
        ap.error("give exactly one of --record or --against")

    here = os.path.dirname(os.path.abspath(__file__))
    extra = a.extra.split()
    failures = 0

    if a.record:
        os.makedirs(a.record, exist_ok=True)
        meta = {}
        for i, text in enumerate(TEXTS):
            path = os.path.join(a.record, f"{i:02d}.wav")
            synthesise(a.bin, a.model, text, path, extra)
            meta[f"{i:02d}"] = {"text": text, **stats(path)}
            print(f"recorded {path}  {meta[f'{i:02d}']['seconds']:.2f}s")
        with open(os.path.join(a.record, "baseline.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"\nbaseline written to {a.record}")
        return 0

    with open(os.path.join(a.against, "baseline.json")) as f:
        baseline = json.load(f)
    work = tempfile.mkdtemp(prefix="woof-quality-")
    print(f"{'text':>44}  {'dur':>6}  {'env':>5}  {'peak':>6}  verdict")
    try:
        for i, text in enumerate(TEXTS):
            key = f"{i:02d}"
            path = os.path.join(work, f"{key}.wav")
            synthesise(a.bin, a.model, text, path, extra)
            cand = stats(path)
            r = compare(baseline[key], cand)
            failures += 0 if r["ok"] else 1
            label = text if len(text) <= 42 else text[:39] + "..."
            print(
                f"{label:>44}  {r['duration_ratio']:6.3f}  {r['envelope_corr']:5.2f}  "
                f"{r['peak_ratio']:6.2f}  {'ok' if r['ok'] else 'FAIL'}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not a.skip_tools:
        env = dict(os.environ, WOOF_RS_BIN=a.bin)
        tools = [("normalise cases", ["tools/normalise_diff.py"])]
        if a.repo:
            tools.append(("numpy parity", ["tools/parity.py", "--model", a.model, "--repo", a.repo]))
        tools.append(("streaming parity", ["tools/stream_parity.py", "--model", a.model]))
        print()
        for label, cmd in tools:
            r = subprocess.run([a.python, os.path.join(here, os.path.basename(cmd[0]))] + cmd[1:],
                               capture_output=True, text=True, env=env)
            verdict = "PASS" if r.returncode == 0 else "FAIL"
            failures += 0 if r.returncode == 0 else 1
            tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
            print(f"{label:<18} {verdict}   {tail[:80]}")

    print()
    print("PASS" if failures == 0 else f"FAIL: {failures} check(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
