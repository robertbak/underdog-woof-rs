//! Every date form the reference reads: slashed, ISO, day/month, ranges, years.
//!
//! One function per stage of `norm_pre`, in the reference's order; `prepass` calls them in
//! sequence. Each is a pure String -> String transform, so order is the only state.

use super::*;

/// `12-25-1999` and `14.03.2027` are dates, not ranges: give them the slashed form the later date rules read.
pub(super) fn slashed_dates(mut t: String) -> String {
    t = re::sub(r"\b(\d{1,2})-(\d{1,2})-((?:19|20)\d\d)\b", r"\1/\2/\3", &t);
    t = re::sub_with(r"\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d\d)\b", &t, |c| {
        let (day, month) = (
            re::group(c, 1).parse::<usize>().unwrap_or(0),
            re::group(c, 2).parse::<usize>().unwrap_or(0),
        );
        if (1..=31).contains(&day) && (1..=12).contains(&month) {
            format!(
                "the {} of {} {}",
                ordinal(day as u64),
                tables::MONTHS[month - 1],
                re::group(c, 3)
            )
        } else {
            c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
        }
    });
    t
}

/// Day ranges and weekday abbreviations.
pub(super) fn day_ranges(mut t: String) -> String {
    let days = days_alternation();
    let day_range = leak(format!(r"\b({days})\.?\s*[-–]\s*({days})\.?"));
    t = re::sub_with(day_range, &t, |c| {
        let from = lookup(tables::DAYS, re::group(c, 1)).unwrap_or(re::group(c, 1));
        let to = lookup(tables::DAYS, re::group(c, 2)).unwrap_or(re::group(c, 2));
        format!("{from} to {to}")
    });
    let day_dot = leak(format!(r"(?<!\bthe )(?<!\bThe )\b({days})\.(?=[\s,]|$)"));
    t = re::sub_with_ctx(day_dot, &t, |c, haystack, start| {
        let full = lookup(tables::DAYS, re::group(c, 1)).unwrap_or(re::group(c, 1));
        let after = &haystack[start + c.get(0).map(|m| m.end() - m.start()).unwrap_or(0)..];
        let keep_dot = match after.chars().next() {
            None => true,
            Some(ch) => {
                ch.is_whitespace() && {
                    let rest = after.trim_start();
                    rest.is_empty() || rest.chars().next().is_some_and(|c| c.is_uppercase())
                }
            }
        };
        format!("{full}{}", if keep_dot { "." } else { "" })
    });
    let day_phrase = leak(format!(
        r"\b(on|by|until|till|every|from|next|this|last) ({days})\b(?!')"
    ));
    t = re::sub_with(day_phrase, &t, |c| {
        format!(
            "{} {}",
            re::group(c, 1),
            lookup(tables::DAYS, re::group(c, 2)).unwrap_or(re::group(c, 2))
        )
    });
    let mon = mon_abbr_alternation();
    let mon_dot = leak(format!(r"\b({mon})\.(?=\s*\d)"));
    t = re::sub_with(mon_dot, &t, |c| {
        lookup(tables::MON_ABBR, re::group(c, 1))
            .unwrap_or(re::group(c, 1))
            .to_string()
    });
    let mon_num = leak(format!(r"\b({mon})\b(?=\s+\d{{1,4}}\b(?!:))"));
    t = re::sub_with(mon_num, &t, |c| {
        lookup(tables::MON_ABBR, re::group(c, 1))
            .unwrap_or(re::group(c, 1))
            .to_string()
    });
    let num_mon = leak(format!(r"(?<=\d )({mon})\b\.?"));
    t = re::sub_with(num_mon, &t, |c| {
        lookup(tables::MON_ABBR, re::group(c, 1))
            .unwrap_or(re::group(c, 1))
            .to_string()
    });
    t
}

/// ISO dates: `2026-10-21` -> "October twenty first, 2026".
pub(super) fn iso_dates(mut t: String) -> String {
    t = re::sub_with(r"\b(\d{4})-(\d{2})-(\d{2})\b", &t, |c| {
        let month = re::group(c, 2).parse::<usize>().unwrap_or(0);
        if (1..=12).contains(&month) {
            let day = re::group(c, 3).parse::<u64>().unwrap_or(1);
            format!("{} {day}, {}", tables::MONTHS[month - 1], re::group(c, 1))
        } else {
            c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
        }
    });
    t
}

/// `14 March 2027` -> "the fourteenth of March 2027"; `3-17 November` -> "the third to the seventeenth of November".
pub(super) fn day_month_year(mut t: String) -> String {
    let months = month_alternation();
    let month_range = leak(format!(
        r"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*[-–]\s*(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\b"
    ));
    t = re::sub_with(month_range, &t, |c| {
        let (from, to) = (
            re::group(c, 1).parse::<u64>().unwrap_or(0),
            re::group(c, 2).parse::<u64>().unwrap_or(0),
        );
        if from < to && to <= 31 {
            format!(
                "the {} to the {} of {}",
                ordinal(from),
                ordinal(to),
                re::group(c, 3)
            )
        } else {
            c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
        }
    });
    let day_month = leak(format!(
        r"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months})\b"
    ));
    t = re::sub_with_ctx(day_month, &t, |c, haystack, start| {
        let day = re::group(c, 1).parse::<u64>().unwrap_or(0);
        if !(1..=31).contains(&day) {
            return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        }
        let before = &haystack[..start];
        let already_the = re::is_match(r"(?i)\bthe\s+$", before);
        format!(
            "{}{} of {}",
            if already_the { "" } else { "the " },
            ordinal(day),
            re::group(c, 2)
        )
    });
    t
}

/// Year ranges and month/year, before the phone rule can read them as digits.
pub(super) fn year_ranges(mut t: String) -> String {
    t = re::sub_with(
        r"\b(1[5-9]\d\d|20\d\d)\s*[-–]\s*(1[5-9]\d\d|20\d\d|\d\d)\b(?![-/.]\d)",
        &t,
        |c| {
            let to = re::group(c, 2);
            let four_digit = to.chars().count() == 4;
            let later =
                to.parse::<u64>().unwrap_or(0) > re::group(c, 1)[2..].parse::<u64>().unwrap_or(0);
            if four_digit || later {
                format!("{} to {to}", re::group(c, 1))
            } else {
                c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
            }
        },
    );
    t = re::sub_with(r"(?<![\d/])\b(\d{1,2})/((?:19|20)\d\d)\b", &t, |c| {
        let month = re::group(c, 1).parse::<usize>().unwrap_or(0);
        if (1..=12).contains(&month) {
            format!("{} {}", tables::MONTHS[month - 1], re::group(c, 2))
        } else {
            c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
        }
    });
    t = re::sub_with(r"(?<![\d/])\b(24/7|50/50|9/11)\b(?!/)", &t, |c| {
        match re::group(c, 1) {
            "24/7" => "24 7",
            "50/50" => "fifty fifty",
            _ => "nine eleven",
        }
        .to_string()
    });
    t
}

/// Day/month without a year.
pub(super) fn day_month(mut t: String) -> String {
    t = re::sub_with_ctx(
        r"(?<![\d/])\b(\d{1,2})/(\d{1,2})\b(?!/\d)",
        &t,
        |c, haystack, start| {
            let a = re::group(c, 1).parse::<u64>().unwrap_or(0);
            let b = re::group(c, 2).parse::<u64>().unwrap_or(0);
            let whole = c.get(0).map(|m| m.end() - m.start()).unwrap_or(0);
            let after = haystack.get(start + whole..).unwrap_or("");
            let before = &haystack[..start];
            let before_tail = if before.len() > 16 {
                &before[before.len() - 16..]
            } else {
                before
            };

            if re::is_match(
                r"\s*(stars?|points?|marks?|votes?)\b",
                &after.to_lowercase(),
            ) || re::is_match(r"(?i)(rated|scored?|rating|got)\s*$", before_tail)
            {
                return format!("{a} out of {b}");
            }
            let no_leading_zero =
                !re::group(c, 1).starts_with('0') && !re::group(c, 2).starts_with('0');
            let fraction_shape = a > 0
                && a < b
                && tables::FRAC_DEN.contains(&(b as u32))
                && no_leading_zero
                && (!re::is_match_leaked(tables::DATE_BEFORE, before_tail)
                    || re::is_match(tables::FRAC_AFTER, after));
            if fraction_shape {
                return frac_words(a, b, false);
            }
            let ordn = |n: u64| -> String {
                let suffix = if (10..=20).contains(&(n % 100)) {
                    "th"
                } else {
                    match n % 10 {
                        1 => "st",
                        2 => "nd",
                        3 => "rd",
                        _ => "th",
                    }
                };
                format!("{n}{suffix}")
            };
            let the = if re::is_match(r"(?i)\bthe\s+$", before) {
                ""
            } else {
                "the "
            };
            if (1..=12).contains(&b) && (1..=31).contains(&a) {
                return format!("{the}{} of {}", ordn(a), tables::MONTHS[(b - 1) as usize]);
            }
            if (1..=12).contains(&a) && (1..=31).contains(&b) {
                return format!("{} {}", tables::MONTHS[(a - 1) as usize], ordn(b));
            }
            c.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
        },
    );
    t
}
