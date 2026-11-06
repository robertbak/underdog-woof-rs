//! Money, units, powers and currencies: `$4.99`, `10 kg`, `x^2`, `12 million`.

use super::*;

/// Money, ranges, durations, decades, Roman numerals, powers and bare scales — the second half of
/// `norm_tokens`, split out so each half stays reviewable against the reference.
pub fn norm_money_and_units(t: &str) -> String {
    let mut t = t.to_string();
    let num = r"(\d[\d,]*(?:\.\d+)?)";
    let sc = r"(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)?";

    // currency codes / symbols before the number: $4.2 million, EUR 12.99, ¥300
    let first: &'static str =
        Box::leak(format!(r"(?<![\w$])({})\s?{num}{sc}", currency_codes()).into_boxed_str());
    t = sub_with(first, &t, |c| {
        currency(re::group(c, 1), re::group(c, 2), re::group(c, 3))
    });
    // number then currency code: 12.99 EUR
    let second: &'static str = Box::leak(
        format!(
            r"{num}{sc}\s?(USD|EUR|GBP|JPY|CNY|INR|CAD|AUD|CHF|NZD|HKD|SGD|KRW|MXN|BRL|RUB|SEK|NOK|DKK|ZAR|BTC|ETH)\b"
        )
        .into_boxed_str(),
    );
    t = sub_with(second, &t, |c| {
        currency(re::group(c, 3), re::group(c, 1), re::group(c, 2))
    });
    t = re::sub(r"(?<!\w)(\d+)\s?¢", r"\1 cents", &t);

    // ranges: $50-75, €10–€20
    let range: &'static str =
        Box::leak(format!(r"([$€£]){num}{sc}\s?[-–]\s?[$€£]?{num}{sc}").into_boxed_str());
    t = sub_with(range, &t, |c| {
        let (sym, a, s1, b, s2) = (
            re::group(c, 1),
            re::group(c, 2),
            re::group(c, 3),
            re::group(c, 4),
            re::group(c, 5),
        );
        let cents_pair = s1.is_empty()
            && s2.is_empty()
            && (re::is_match(r"\.\d\d\b", a) || re::is_match(r"\.\d\d\b", b));
        if cents_pair {
            return format!("{sym}{a} to {sym}{b}");
        }
        let s1w = if s1.is_empty() {
            String::new()
        } else {
            format!(" {}", scale_word(s1))
        };
        let s2w = if s2.is_empty() {
            String::new()
        } else {
            format!(" {}", scale_word(s2))
        };
        let unit = match sym {
            "$" => "dollars",
            "€" => "euros",
            _ => "pounds",
        };
        let first_suffix = if s1w == s2w { "" } else { s1w.as_str() };
        format!("{a}{first_suffix} to {b}{s2w} {unit}")
    });
    // $4.2 million
    let scaled: &'static str = Box::leak(
        format!(
            r"([$€£]){num}(\s?(?:thousand|million|billion|trillion)\b|(?:k|K|mn|m|M|bn|b|B|tn|T)\b)"
        )
        .into_boxed_str(),
    );
    t = sub_with(scaled, &t, |c| {
        let unit = match re::group(c, 1) {
            "$" => "dollars",
            "€" => "euros",
            _ => "pounds",
        };
        format!("{} {} {unit}", re::group(c, 2), scale_word(re::group(c, 3)))
    });

    // durations: 2h 15m, 1h30, 6m 30s
    t = sub_with(
        r"\b(\d+)\s?h(?:rs?)?\s?(\d{1,2})(?:\s?m(?:ins?)?)?\b",
        &t,
        |c| {
            format!(
                "{} {}",
                pl(re::group(c, 1), "hour"),
                pl(re::group(c, 2), "minute")
            )
        },
    );
    t = sub_with(
        r"\b(\d+)\s?m(?:ins?)?\s?(\d{1,2})\s?s(?:ecs?)?\b",
        &t,
        |c| {
            format!(
                "{} {}",
                pl(re::group(c, 1), "minute"),
                pl(re::group(c, 2), "second")
            )
        },
    );
    t = sub_with(
        r"(\b(?:for|last|lasts|lasted|lasting|takes?|took|taking|every|within|wait|waited|remaining|runtime|ETA|blocked)\s+(?:\w+\s+){0,2}?)(\d{1,3})m\b(?!\s*(?:dash|sprint|race|swim|hurdles|pool|freestyle|relay))",
        &t,
        |c| format!("{}{}", re::group(c, 1), pl(re::group(c, 2), "minute")),
    );

    // decades: '80s, 1990s, the 2000s; '98 -> ninety eight
    t = sub_with(
        r"(?<![\w$€£.,'])(')?(\d{2}|1\d{3}|20\d{2})'?s\b",
        &t,
        |c| {
            let text = re::group(c, 2);
            let n: u64 = text.parse().unwrap_or(0);
            if !n.is_multiple_of(10) {
                return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            let words = if text.chars().count() == 4 {
                year(n)
            } else if n == 0 && !re::group(c, 1).is_empty() {
                "two thousand".to_string()
            } else if n == 0 {
                String::new()
            } else {
                cardinal(n)
            };
            if words.is_empty() {
                return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            if let Some(stem) = words.strip_suffix('y') {
                format!("{stem}ies")
            } else {
                format!("{words}s")
            }
        },
    );
    t = sub_with(r"(?:(?<=\s)|^)'(\d{2})\b(?!['\w])", &t, |c| {
        let two = re::group(c, 1);
        if two.starts_with('0') {
            let second: usize = two.strip_prefix('0').unwrap_or(two).parse().unwrap_or(0);
            format!("oh {}", ONES[second])
        } else {
            two.to_string()
        }
    });

    // Roman numerals in context
    t = sub_with(r"\bWW\s?(II|I|2|1)\b", &t, |c| {
        format!(
            "World War {}",
            if matches!(re::group(c, 1), "II" | "2") {
                "two"
            } else {
                "one"
            }
        )
    });
    let roman_pattern = roman_re();
    t = re::sub_with_ctx(roman_pattern, &t, |c, haystack, start| {
        let prev = re::group(c, 1);
        let gap = re::group(c, 2);
        let numeral = re::group(c, 3);
        let before = &haystack[..start];
        let words: Vec<&str> = before.split_whitespace().collect();
        let prev2 = words
            .last()
            .filter(|w| !re::is_match(r#"[.,;:!?)"]$"#, w))
            .copied()
            .unwrap_or("");
        roman_sub(prev, gap, numeral, before, prev2)
    });
    t = sub_with(r"(?<=[a-z])\((I|II|III|IV|V|VI)\)", &t, |c| {
        format!(" {}", cardinal(roman(re::group(c, 1)) as u64))
    });

    // powers: x^2, 10^23, 6.02 x 10^23
    t = re::sub(r"(\d)\s?[x×*]\s?(?=10\s?\^)", r"\1 times ", &t);
    t = sub_with(
        r"(\d(?:[\d,]*\d)?(?:\.\d+)?)\s?(km|m|cm|mm|ft|mi|yd|in)\^([23])\b",
        &t,
        |c| {
            let unit = lookup(tables::UNITS, re::group(c, 2)).unwrap_or(re::group(c, 2));
            let shape = if re::group(c, 3) == "2" {
                "square"
            } else {
                "cubic"
            };
            format!("{} {shape} {unit}", re::group(c, 1))
        },
    );
    t = sub_with(
        r"(\b[A-Za-z]{1,3}|\b\d+(?:\.\d+)?|\))\s?\^\s?\(?([+-]?(?:\d+|[a-z]))\)?",
        &t,
        |c| {
            let base = re::group(c, 1);
            let exponent = re::group(c, 2);
            let negative = exponent.starts_with('-');
            let digits_only = exponent.trim_start_matches(['+', '-']);
            let base = if base.chars().all(|ch| ch.is_alphabetic()) && base.chars().count() > 1 {
                base.chars().map(String::from).collect::<Vec<_>>().join(" ")
            } else {
                base.to_string()
            };
            let Ok(n) = digits_only.parse::<u64>() else {
                return format!("{base} to the {digits_only}");
            };
            if n == 2 && !negative {
                format!("{base} squared")
            } else if n == 3 && !negative {
                format!("{base} cubed")
            } else {
                format!(
                    "{base} to the {}{}",
                    if negative { "minus " } else { "" },
                    ordinal(n)
                )
            }
        },
    );
    // an equation: 3x - 2 = 11 -> 3 x minus 2
    if re::is_match(r"[=^+]|\bsolve\b|\bequation", &t) {
        t = re::sub(r"(?<=[\w)])\s+[-–]\s+(?=[\w(])", " minus ", &t);
        t = re::sub(r"\b(\d+)([a-fi-jn-rt-z])\b", r"\1 \2", &t);
    }
    t = re::sub(
        r"\b(\d+(?:\.\d+)?)x(?=\s+(?:faster|slower|more|less|larger|smaller|bigger|higher|lower|cheaper|better|worse|longer|shorter|the|as|over|speed|zoom)\b)",
        r"\1 times",
        &t,
    );
    t = re::sub(
        r"(\d)\s+[x×*]\s+(\d+(?:\.\d+)?)(?=\s*=)",
        r"\1 times \2",
        &t,
    );

    // bare scales: 4.5M visitors -> 4.5 million visitors
    t = sub_with(
        r"(?<![\w$€£.])(\d+\.\d+|\d+(?=\s?\w+\s+[a-z]{3,}s\b))\s?(M|bn|mn|tn)\b(?=\s+[a-z]{3,})",
        &t,
        |c| format!("{} {}", re::group(c, 1), scale_word(re::group(c, 2))),
    );
    t
}

/// `SCALE` lookup for the money/duration rules.
fn scale_word(token: &str) -> String {
    lookup(tables::SCALE, token.trim())
        .unwrap_or("")
        .to_string()
}

/// The `cur()` handler: currency code/symbol plus a number, with optional scale word.
fn currency(symbol: &str, number: &str, scale: &str) -> String {
    let Some(name) = lookup(tables::CURCODE, symbol) else {
        return format!("{symbol}{number}{scale}");
    };
    if name.chars().count() == 1 {
        return format!("{name}{number}{scale}");
    }
    let scale = scale.trim();
    let (whole, cents) = match number.split_once('.') {
        Some((w, c)) => (w, c),
        None => (number, ""),
    };
    if scale.is_empty() && cents.len() == 2 && name.ends_with("dollars") {
        let cents_value: u64 = cents.parse().unwrap_or(0);
        return if cents_value == 0 {
            format!("{whole} {name}")
        } else {
            format!("{whole} {name} and {cents_value} cents")
        };
    }
    let suffix = if scale.is_empty() {
        String::new()
    } else {
        format!(" {}", scale_word(scale))
    };
    format!("{number}{suffix} {name}")
}
