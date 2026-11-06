//! Numbers, times, phones, units, identifiers, codes, durations, symbols.
//!
//! One function per stage of `norm_pre`, in the reference's order; `prepass` calls them in
//! sequence. Each is a pure String -> String transform, so order is the only state.

use super::*;

/// Mixed number and fraction: `1 1/2` -> "1 and a half".
pub(super) fn mixed_fractions(mut t: String) -> String {
    t = re::sub_with(
        r"(?<![\d/.,])\b(\d+)\s+(\d{1,2})/(\d{1,2})\b(?!/\d)",
        &t,
        |c| {
            let (whole, num, den) = (
                re::group(c, 1),
                re::group(c, 2).parse::<u64>().unwrap_or(0),
                re::group(c, 3).parse::<u64>().unwrap_or(0),
            );
            if num > 0 && num < den && tables::FRAC_DEN.contains(&(den as u32)) {
                format!("{whole} and {}", frac_words(num, den, true))
            } else {
                c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
            }
        },
    );
    t
}

/// Multi-dot versions: `4.0.1(-beta)`.
pub(super) fn versions(mut t: String) -> String {
    t = re::sub_with(
        r"\b(\d+(?:\.\d+){2,})(?:-(alpha|beta|rc\d*|preview))?\b",
        &t,
        |c| {
            let parts: Vec<&str> = re::group(c, 1).split('.').collect();
            let suffix = if re::group(c, 2).is_empty() {
                String::new()
            } else {
                format!(" {}", re::group(c, 2))
            };
            format!("{}{suffix}", parts.join(" point "))
        },
    );
    t
}

/// `2:03:45` -> "two hours three minutes forty five seconds".
pub(super) fn clock_times(mut t: String) -> String {
    t = re::sub_with_ctx(
        r"\b(\d{1,2}):(\d{2}):(\d{2})\b(?!:)",
        &t,
        |c, haystack, start| {
            let before = &haystack[..start];
            let after_clock = re::is_match(r"\b(at|by|until|from)\s+$", before);
            if after_clock {
                return format!(
                    "{}:{} and {}",
                    re::group(c, 1),
                    re::group(c, 2),
                    pl(re::group(c, 3), "second")
                );
            }
            let units = ["hour", "minute", "second"];
            let parts: Vec<String> = (1..=3)
                .filter_map(|i| {
                    let raw = re::group(c, i);
                    let n: i64 = raw.parse().unwrap_or(0);
                    if n != 0 || units[i - 1] == "second" {
                        Some(pl(raw, units[i - 1]))
                    } else {
                        None
                    }
                })
                .collect();
            parts.join(", ")
        },
    );
    t
}

/// Time ranges, ratios, letter-led codes and phone-like groups.
pub(super) fn ranges_and_ratios(mut t: String) -> String {
    t = re::sub(
        r"\b(\d{1,2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\s?[-–]\s?(\d{1,2}(?::\d{2})?\s?[AaPp]\.?[Mm]\b\.?)",
        r"\1 to \2",
        &t,
    );
    t = re::sub(
        r"\b(\d{1,2}:\d{2})\s?[-–]\s?(\d{1,2}:\d{2})\b",
        r"\1 to \2",
        &t,
    );
    t = re::sub(r"\b(\d{1,2}):(\d)\b(?!\d)", r"\1 to \2", &t);
    t = re::sub(
        r"(?<=[Rr]atio of |[Rr]atio is )(\d{1,2}):(\d{1,2})\b",
        r"\1 to \2",
        &t,
    );
    t = re::sub_with(
        r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})*-\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b",
        &t,
        |c| code_read(re::group(c, 0)),
    );
    t
}

/// Phone numbers: seven or more digits in groups.
pub(super) fn phones(mut t: String) -> String {
    t = re::sub_with(
        r"(?<![\w.,])\+?\(?\d{2,4}\)?(?:[\s.-]\(?\d{2,4}\)?){1,4}(?![\w.,]*\d)",
        &t,
        |c| {
            let text = re::group(c, 0);
            if text.chars().filter(|ch| ch.is_ascii_digit()).count() < 7 {
                return text.to_string();
            }
            let lead = if text.trim_start().starts_with('+') {
                "plus "
            } else {
                ""
            };
            let groups: Vec<String> = numbers_in(text).iter().map(|g| digits(g)).collect();
            format!(" {lead}{} ", groups.join(", "))
        },
    );
    t = re::sub_with(
        r"\b((?:[Cc]all|[Dd]ial|[Tt]ext|[Rr]ing)\s+)(\d{3,6})\b",
        &t,
        |c| format!("{}{}", re::group(c, 1), digits(re::group(c, 2))),
    );
    t
}

/// Units that follow a number.
pub(super) fn units_after_numbers(mut t: String) -> String {
    t = re::sub_with(r"\b(\d+)\s?ft\.?\s?(\d{1,2})\s?in\b\.?", &t, |c| {
        let (feet, inches) = (re::group(c, 1), re::group(c, 2));
        format!(
            "{feet} {} {inches} {}",
            if feet == "1" { "foot" } else { "feet" },
            if inches == "1" { "inch" } else { "inches" }
        )
    });
    t = re::sub(r#"\b(\d)'\s?(\d{1,2})(?:\"|'')"#, r"\1 foot \2", &t);
    let units = unit_keys();
    let per_pat = leak(format!(
        r"{}\s?({})/({}|s|in)(?![\w/])",
        tables::UNIT_PRE,
        units,
        units
    ));
    t = re::sub_with(per_pat, &t, |c| {
        if re::group(c, 3) == "in" {
            return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        }
        let per = lookup(tables::UNITS, re::group(c, 3)).unwrap_or("seconds");
        format!(
            "{} {} per {}",
            re::group(c, 1),
            unit(re::group(c, 2), re::group(c, 1)),
            sing(per)
        )
    });
    let plain_units = leak(format!(r"{}\s?({})(?![\w/'])", tables::UNIT_PRE, units));
    t = re::sub_with(plain_units, &t, |c| {
        format!(
            "{} {}",
            re::group(c, 1),
            unit(re::group(c, 2), re::group(c, 1))
        )
    });
    let after_power = leak(format!(
        r"(\b(?:squared|cubed|to the (?:minus )?(?:\w+ )?\w+(?:th|st|nd|rd))\s)({})(?:/({}|s))?(?![\w/'])",
        units, units
    ));
    t = re::sub_with(after_power, &t, |c| {
        let base = lookup(tables::UNITS, re::group(c, 2)).unwrap_or(re::group(c, 2));
        let per = if re::group(c, 3).is_empty() {
            String::new()
        } else {
            format!(
                " per {}",
                sing(lookup(tables::UNITS, re::group(c, 3)).unwrap_or("seconds"))
            )
        };
        format!("{}{base}{per}", re::group(c, 1))
    });
    t
}

/// Identifiers read in pairs.
pub(super) fn identifiers(mut t: String) -> String {
    let id_pat = leak(format!(
        r"\b({})(\s+(?:No\.?\s*|#\s*)?)([A-Z]{{1,3}}\s?)?(\d{{2,4}})([A-Z]{{0,2}})\b(?![.,]\d|\s?[-–]\s?\d)",
        tables::ID_WORDS
    ));
    t = re::sub_with(id_pat, &t, |c| {
        let mut out = format!("{}{}", re::group(c, 1), re::group(c, 2));
        if !re::group(c, 3).is_empty() {
            out.push_str(&format!("{} ", spell(re::group(c, 3))));
        }
        out.push_str(&nread(re::group(c, 4)));
        if !re::group(c, 5).is_empty() {
            out.push_str(&format!(" {}", spell(re::group(c, 5))));
        }
        out
    });
    t = re::sub_with(r"\b([A-Z]{2,3})\s(\d{3,4})\b(?![.,:]\d|\s\d)", &t, |c| {
        format!("{} {}", re::group(c, 1), nread(re::group(c, 2)))
    });
    t
}

/// Letter/digit codes, chemical formulas and hash numbers.
pub(super) fn letter_digit_codes(mut t: String) -> String {
    let code_ctx = tables::CODE_CTX;
    t = re::sub_with_ctx(
        r"(?<![-/.\w])([A-Z]{1,3})(\d{1,4})([A-Z]{0,2})(?![-/\w])",
        &t,
        |c, haystack, start| {
            let before = &haystack[..start];
            let before_tail = if before.len() > 40 {
                &before[before.len() - 40..]
            } else {
                before
            };
            let not_in_code_context = !re::is_match_leaked(code_ctx, before_tail);
            let digits_part = re::group(c, 2);
            let letters = re::group(c, 3);
            let qualifies = not_in_code_context
                && (digits_part.chars().count() <= 2
                    || (re::group(c, 1).chars().count() == 2 && letters.is_empty()));
            if qualifies {
                let parts: Vec<String> =
                    [spell(re::group(c, 1)), nread(digits_part), spell(letters)]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect();
                parts.join(" ")
            } else {
                c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
            }
        },
    );
    t = re::sub_with_ctx(
        r"(?<![-/.\w])(\d{1,4})([A-Z]{1,2})(?![-/\w])",
        &t,
        |c, haystack, start| {
            let before = &haystack[..start];
            let before_tail = if before.len() > 40 {
                &before[before.len() - 40..]
            } else {
                before
            };
            if re::is_match_leaked(code_ctx, before_tail) {
                return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            format!("{} {}", nread(re::group(c, 1)), spell(re::group(c, 2)))
        },
    );
    t = re::sub_with_ctx(
        r"(?<![-/.\w])(?=[A-Za-z]*\d)(?:[A-Z][a-z]?\d{0,3}){2,}(?![-/\w])",
        &t,
        |c, haystack, start| {
            let before = &haystack[..start];
            let before_tail = if before.len() > 40 {
                &before[before.len() - 40..]
            } else {
                before
            };
            if re::is_match_leaked(code_ctx, before_tail) {
                return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            let formula = re::group(c, 0);
            let elements = all_elements(formula);
            if !elements {
                return formula.to_string();
            }
            let mut parts: Vec<String> = Vec::new();
            let mut rest = formula;
            while !rest.is_empty() {
                let mut chars = rest.chars();
                let Some(first) = chars.next() else { break };
                let mut element = first.to_string();
                if let Some(second) = chars.clone().next() {
                    if second.is_lowercase() {
                        element.push(second);
                        chars.next();
                    }
                }
                let digits: String = chars.clone().take_while(|c| c.is_ascii_digit()).collect();
                let consumed = element.len() + digits.len();
                parts.push(spell(&element));
                if !digits.is_empty() {
                    parts.push(cardinal(digits.parse().unwrap_or(0)));
                }
                rest = &rest[consumed..];
            }
            parts.join(" ")
        },
    );
    t = re::sub_with(r"#(\d+(?:-\d+)*)\b", &t, |c| {
        let groups: Vec<String> = re::group(c, 1)
            .split('-')
            .map(|g| {
                if g.chars().count() <= 3 {
                    g.to_string()
                } else {
                    digits(g)
                }
            })
            .collect();
        format!("number {}", groups.join(", "))
    });
    t
}

/// Codes and random mixed ids.
pub(super) fn mixed_ids(mut t: String) -> String {
    t = re::sub_with(
        r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b",
        &t,
        |c| code_read(re::group(c, 0)),
    );
    t = re::sub_with(r"\b(?=\w*\d)(?=\w*[A-Z])[A-Z0-9]{3,}\b", &t, |c| {
        spell(re::group(c, 0))
    });
    t = re::sub_with(
        r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{4,}\b",
        &t,
        |c| {
            let token = re::group(c, 0);
            let unit_like = re::is_match(
                r"(?i)\d+(?:st|nd|rd|th|am|pm|s|kg|km|mm|cm|ml|mg|gb|mb|tb|kb|ghz|mhz|kw|kwh|mah|hrs?|mins?|h|m|k|x|g|l|w|v)",
                token,
            );
            if unit_like {
                token.to_string()
            } else {
                spell(token)
            }
        },
    );
    t = re::sub_with(r"(?<![\d$€£.,])\b(\d{5,})\b(?![.,]\d)", &t, |c| {
        digits(re::group(c, 1))
    });
    t = re::sub(
        r"(?<![\d/:.-])\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b(?![/:.-]\d)",
        r"\1 to \2",
        &t,
    );
    t
}

/// Durations.
pub(super) fn durations(mut t: String) -> String {
    t = re::sub_with(r"\b(\d+)\s*h\s*(\d{1,2})\b", &t, |c| {
        format!(
            "{} hour{} {}",
            re::group(c, 1),
            if re::group(c, 1) == "1" { "" } else { "s" },
            re::group(c, 2)
        )
    });
    t = re::sub_with(r"\b(\d+)\s*hrs?\b", &t, |c| {
        format!(
            "{} hour{}",
            re::group(c, 1),
            if re::group(c, 1) == "1" { "" } else { "s" }
        )
    });
    t = re::sub_with(r"\b(\d+)\s*mins?\b", &t, |c| {
        format!(
            "{} minute{}",
            re::group(c, 1),
            if re::group(c, 1) == "1" { "" } else { "s" }
        )
    });
    t
}

/// Dimensions, symbols, pounds and pence, dashes.
pub(super) fn dimensions_and_symbols(mut t: String) -> String {
    t = re::sub(r"\s*×\s*", " by ", &t);
    t = re::sub(r"(?<=\d)\s+x\s+(?=\d)", " by ", &t);
    for (pattern, replacement) in tables::SYMBOLS {
        let pat = leak(pattern.to_string());
        t = re::sub(pat, replacement, &t);
    }
    t = re::sub_with(r"£(\d[\d,]*)\.(\d{2})\b", &t, |c| {
        if re::group(c, 2) == "00" {
            format!("{} pounds", re::group(c, 1))
        } else {
            let pence: u64 = re::group(c, 2).parse().unwrap_or(0);
            format!("{} pounds {pence}", re::group(c, 1))
        }
    });
    t = re::sub(r"\s+–\s+", " — ", &t).replace('–', "-");
    collapse_spaces(&t)
}
