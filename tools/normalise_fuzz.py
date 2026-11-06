#!/usr/bin/env python3
"""Randomised differential test of the normaliser port.

Builds sentences from a grammar of the shapes the reference cares about (numbers, ordinals, money,
dates, times, units, codes, Roman numerals, abbreviations, hard names) and compares
`woof-rs --normalise-file` against `underdog_woof.normalize.Normalizer(use_nemo=False)`.

    python tools/normalise_fuzz.py --repo …/underdog_woof --count 2000 [--failures N] [--seed 1]

Exit code 0 when every case matches, so it can gate the port the way the fixed corpus does.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile

ATOMS = {
    "number": ["0", "7", "42", "99", "100", "101", "999", "1000", "1204", "1600", "2009",
               "12345", "1000000", "3.5", "0.25", "1,234", "1,234,567", "12.05", "007"],
    "ordinal": ["1st", "2nd", "3rd", "4th", "11th", "21st", "22nd", "23rd", "100th"],
    "year": ["1999", "2026", "1900", "2000", "2005", "1604"],
    "money": ["$5", "$5.99", "$1,250", "$0.50", "€10", "£3.20", "USD 5", "12.99 EUR", "¥300",
              "$4.2 million", "$50-75", "£7.50", "99 cents"],
    "date": ["3/4/2026", "03/04/26", "2026-10-21", "14 March 2027", "Mar 3", "March 3rd",
             "01/10/2026", "12/25", "1968-2020", "1990-95", "3-17 November"],
    "time": ["5:30", "5:30 p.m.", "12:00", "14:00", "10:00-16:30", "9am-5pm", "2:03:45"],
    "unit": ["5km", "10 kg", "3.5mm", "2 ft", "6 in", "12 oz", "60 mph", "-5°C", "25°C",
             "10 mg/kg", "35 Mbps", "2h 15m", "45min"],
    "code": ["QX-4471", "415-555-0199", "+1 415 555 0199", "abc123", "getUserName", "user_id",
             "192.168.1.1", "v2.1.0", "PIN 4471", "room 402", "flight 229", "SW1A 1AA",
             "C6H12O6", "#4471", "AB-4471-Z", "X7K9", "AA229", "CS-2026-0813", "me@example.com",
             "www.example.com", "report.pdf", "main.rs"],
    "roman": ["Henry VIII", "World War II", "Chapter XIV", "Super Bowl LIX", "Title IX",
              "Section IV", "John Smith III", "Queen Elizabeth II"],
    "symbol": ["&", "@", "#", "%", "+", "=", "/", "etc.", "e.g.", "vs.", "approx."],
    "name": ["Siobhan", "Worcestershire", "Reykjavik", "Sao Paulo", "Lodz", "Nguyen", "Xi'an",
             "Rene Descartes", "Blaise Pascal", "Eua"],
    "word": ["alpha", "budget", "committee", "delivery", "engineer", "Friday", "garden",
             "harbour", "invoice", "kitchen", "ledger", "morning", "night", "office", "parcel"],
}
TEMPLATES = [
    "{w} {a}.",
    "{w} {a} {w}.",
    "The {w} is {a}.",
    "We need {a} by {w}.",
    "{a}, {a} and {a}.",
    "Note: {a} ({w}).",
    "{a} - {a} - {a}.",
    "Did you see {a} on {w}?",
    "{a} was sent to {w} at {a}.",
]


def build(rng, count):
    kinds = list(ATOMS)
    out = []
    for _ in range(count):
        template = rng.choice(TEMPLATES)
        text = template.format(
            w=lambda: rng.choice(ATOMS["word"]) if False else None,
        ) if False else None
        # simple sequential substitution: fill {w} with words, {a} with a random atom of any kind
        parts = template.split("{w}")
        text = rng.choice(ATOMS["word"]).join(parts)
        while "{a}" in text:
            text = text.replace("{a}", rng.choice(ATOMS[rng.choice(kinds)]), 1)
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bin", default=os.environ.get("WOOF_RS_BIN", "woof-rs"))
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--failures", type=int, default=10)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cases = build(rng, a.count)

    sys.path.insert(0, a.repo)
    from underdog_woof.normalize import Normalizer

    norm = Normalizer(use_nemo=False)
    expected = [norm(c).replace("\t", " ").replace("\n", " ") for c in cases]

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(c.replace("\n", " ") for c in cases) + "\n")
        path = f.name
    got = subprocess.run([a.bin, "--normalise-file", path], capture_output=True, text=True).stdout.split("\n")
    mismatches = [(c, e, g) for c, e, g in zip(cases, expected, got) if e != g]
    print(f"{len(cases) - len(mismatches)}/{len(cases)} cases match ({100.0 * (len(cases) - len(mismatches)) / len(cases):.1f}%)")
    for c, e, g in mismatches[: a.failures]:
        print(f"  {c!r}\n    want {e!r}\n    got  {g!r}")
    sys.exit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()
