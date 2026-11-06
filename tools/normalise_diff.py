#!/usr/bin/env python3
"""Differential check of the normaliser port against the reference corpus.

    python tools/normalise_diff.py            # summary
    python tools/normalise_diff.py --failures # show each mismatching case

Exit code 0 when every case matches, 1 otherwise, so CI can gate on it as the port progresses.
"""
import argparse
import os
import subprocess
import sys

CASES = "tests/data/normalise_cases.tsv"
BIN = os.environ.get("WOOF_RS_BIN", "woof-rs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--failures", action="store_true")
    a = ap.parse_args()

    total = 0
    matched = 0
    failures = []
    with open(a.cases) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            text, expected = line.split("\t")
            got = subprocess.run([a.bin, "--normalise", text],
                                 capture_output=True, text=True).stdout.rstrip("\n")
            total += 1
            if got == expected:
                matched += 1
            else:
                failures.append((text, expected, got))

    print(f"{matched}/{total} cases match ({100.0 * matched / total:.0f}%)")
    if a.failures:
        for text, expected, got in failures:
            print(f"  {text!r}\n    want {expected!r}\n    got  {got!r}")
    sys.exit(0 if matched == total else 1)


if __name__ == "__main__":
    main()
