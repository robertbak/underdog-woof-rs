#!/usr/bin/env python3
"""Generate a differential corpus for the normaliser port.

Writes `tests/data/normalise_cases.tsv` with one `input<TAB>expected` pair per case, produced by the
reference implementation (`underdog_woof.normalize.Normalizer`, NeMo disabled — the configuration
the Rust port targets). The Rust side must reproduce every line; `tools/normalise_diff.py` reports
how many it does.

    python tools/normalise_corpus.py --repo /path/to/underdog_woof
"""
import argparse
import itertools
import os
import sys


def generated():
    """Combinatorial coverage: rules interact (money + scale, date + weekday, unit + number)."""
    money = ["$5", "$5.99", "$1,250.75", "$0.50", "€10", "£3.20", "USD 5", "12.99 EUR", "¥300",
             "$4.2 million", "$50-75", "£7.50", "5 dollars", "99 cents", "50¢"]
    for m in money:
        yield m
        yield f"Cost is {m} today."
    numbers = ["0", "7", "42", "99", "100", "101", "999", "1000", "1204", "1600", "2009", "12345",
               "1000000", "3.5", "0.25", "1,234", "1,234,567", "12.05", "007", "402", "229",
               "1st", "2nd", "3rd", "4th", "11th", "21st", "22nd", "23rd", "100th", "1,000,000",
               "1999", "2026", "1900", "2000", "2005", "2.5%", "100%", "-7", "3/4", "1 1/2"]
    for n in numbers:
        yield n
        yield f"The value is {n}."
    dates = ["5:30", "5:30 p.m.", "12:00", "14:00", "3/4/2026", "03/04/26", "2026-10-21",
             "14 March 2027", "3 March 2026", "Mar 3", "March 3rd", "01/10/2026", "12/25",
             "24/7", "9/11", "1968-2020", "1990-95", "3-17 November", "10:00-16:30", "9am-5pm"]
    for d in dates:
        yield d
        yield f"Meet on {d}."
    units = ["5km", "10 kg", "3.5mm", "2 ft", "6 in", "1/2 cup", "3/4 tsp", "12 oz", "60 mph",
             "-5°C", "25°C", "3x4", "2^10", "1 kg", "10 mg/kg", "35 Mbps", "2h 15m", "45min"]
    for u in units:
        yield u
        yield f"It measures {u}."
    codes = ["QX-4471", "415-555-0199", "+1 415 555 0199", "abc123", "getUserName", "user_id",
             "OpenAI", "HTTP", "https://example.com/path", "192.168.1.1", "v2.1.0",
             "PIN 4471", "code QX-4471", "room 402", "flight 229", "Suite 1204", "SW1A 1AA",
             "C6H12O6", "#4471", "AB-4471-Z", "X7K9", "AA229", "A380", "CS-2026-0813",
             "me@example.com", "www.example.com", "localhost:8080", "report.pdf", "main.rs",
             "@/usr/local/bin", r"C:\Users\me"]
    for c in codes:
        yield c
        yield f"Reference {c} please."
    roman = ["Henry VIII", "World War II", "Chapter XIV", "Super Bowl LIX", "Title IX",
             "Section IV", "John Smith III", "Frozen II", "Queen Elizabeth II", "part IV"]
    for r in roman:
        yield r
        yield f"I read {r}."
    symbols = ["&", "@", "#", "%", "+", "=", "/", "A&B", "etc.", "e.g.", "i.e.", "vs.",
               "approx.", "Dr. Smith", "Mr. Jones", "St. Louis", "5 Elm St", "742 Evergreen Terrace"]
    for s_ in symbols:
        yield s_
        yield f"Ask {s_} later."
    prose = [
        "The committee agreed that the budget needs review.",
        "Please call me back at 415-555-0199 before 5:30 p.m. on Friday.",
        "Revenue rose 12% in Q3 - driven by the enterprise tier - while churn held at 2.1%.",
        # respelling territory: hard names and foreign words
        "Siobhan and Worcestershire went to Reykjavik.",
        "Sao Paulo, Lodz and Nguyen are in the notes.",
        "The 'Eua ferry leaves at dawn.",
        "Worcestershire-glazed carrots, please.",
        "Xi'an and Qiqihar are both listed.",
        "Rene Descartes wrote to Blaise Pascal.",
    ]
    for p_ in prose:
        yield p_
    yield from [
        # plain numbers and decimals
        "0", "7", "42", "99", "100", "101", "999", "1000", "1204", "1600", "2009",
        "12345", "1000000", "3.5", "0.25", "1,234", "1,234,567", "12.05", "007",
        # ordinals and years
        "1st", "2nd", "3rd", "4th", "11th", "21st", "22nd", "23rd", "100th",
        "1999", "2026", "1900", "2000", "2005",
        # money
        "$5", "$5.99", "$1,250", "$0.50", "€10", "£3.20", "USD 5", "5 dollars", "99 cents",
        # times and dates
        "5:30", "5:30 p.m.", "12:00", "3/4/2026", "03/04/26", "March 3rd", "3 March 2026",
        "Mar 3", "Mon", "Tues", "Friday",
        # units and measures
        "5km", "10 kg", "3.5mm", "2 ft", "6 in", "1/2 cup", "3/4 tsp", "12 oz", "60 mph",
        "-5°C", "25°C", "100%", "3x4", "2^10",
        # codes, identifiers, URLs, phones
        "QX-4471", "415-555-0199", "+1 415 555 0199", "abc123", "getUserName", "user_id",
        "OpenAI", "HTTP", "https://example.com/path", "192.168.1.1", "v2.1.0",
        "PIN 4471", "code QX-4471", "room 402", "flight 229", "Suite 1204",
        # roman numerals
        "Henry VIII", "World War II", "Chapter XIV", "Super Bowl LIX", "Title IX", "Section IV",
        # symbols and abbreviation
        "&", "@", "#", "%", "+", "=", "/", "A&B", "etc.", "e.g.", "i.e.", "vs.", "approx.",
        # ordinary prose that must pass through untouched
        "The committee agreed that the budget needs review.",
        "Please call me back at 415-555-0199 before 5:30 p.m. on Friday.",
        "Revenue rose 12% in Q3 - driven by the enterprise tier - while churn held at 2.1%.",
    ]


def all_cases():
    seen = set()
    out = []
    for text in generated():
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="tests/data/normalise_cases.tsv")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from underdog_woof.normalize import Normalizer

    norm = Normalizer(use_nemo=False)
    seen = set()
    lines = ["# input<TAB>expected — generated by tools/normalise_corpus.py from the reference"]
    for text in all_cases():
        if text in seen:
            continue
        seen.add(text)
        expected = norm(text).replace("\t", " ").replace("\n", " ")
        lines.append(f"{text}\t{expected}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} cases -> {a.out}")


if __name__ == "__main__":
    main()
