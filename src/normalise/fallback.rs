//! `fallback` — the verbaliser the reference uses when NeMo is absent, which is the configuration
//! the packaged server runs.
//!
//! Order matters here as everywhere else: money before the clock (both match digits), the year rule
//! before plain numbers (that is what makes `1204` "twelve oh four" rather than "one thousand two
//! hundred and four"), and the symbol pass last because it rewrites `&`, `+`, `=` and `@` to words.

use super::prepass::collapse_spaces;
use super::pretokens::frac_words;
use super::re;
use super::tables;
use super::{cardinal, digits};

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

/// `number()`: cardinal, or "point" digits after a decimal point.
pub fn number(s: &str) -> String {
    let s = s.replace(',', "");
    match s.split_once('.') {
        Some((whole, frac)) => {
            let whole_words = if whole.is_empty() {
                "zero".to_string()
            } else {
                cardinal(whole.parse().unwrap_or(0))
            };
            format!("{whole_words} point {}", digits(frac))
        }
        None => cardinal(s.parse().unwrap_or(0)),
    }
}

/// `year()` as the fallback uses it (2000-2009 plain, otherwise split at the century).
fn year_words(n: u64) -> String {
    if (2000..=2009).contains(&n) {
        return cardinal(n);
    }
    let (hi, lo) = (n / 100, n % 100);
    if lo == 0 {
        format!("{} hundred", cardinal(hi))
    } else if lo < 10 {
        format!("{} oh {}", cardinal(hi), ONES[lo as usize])
    } else {
        format!("{} {}", cardinal(hi), cardinal(lo))
    }
}

/// `ordinal()` for a bare number.
fn ordinal_words(n: u64) -> String {
    let mut words: Vec<String> = cardinal(n).split(' ').map(str::to_string).collect();
    if let Some(last) = words.pop() {
        words.push(super::ordinal_word(&last));
    }
    words.join(" ")
}

/// Currency names for the fallback's money rule: `(one, many, one cent, many cents)`.
fn currency_names(
    symbol: &str,
) -> Option<(&'static str, &'static str, &'static str, &'static str)> {
    Some(match symbol {
        "$" => ("dollar", "dollars", "cent", "cents"),
        "€" => ("euro", "euros", "cent", "cents"),
        "£" => ("pound", "pounds", "penny", "pence"),
        _ => return None,
    })
}

/// `fallback`.
pub fn fallback(text: &str) -> String {
    let mut t = text.to_string();

    // money: $5, $5.99, $1,250, $0.50
    t = re::sub_with(r"([$€£])(\d[\d,]*)(?:\.(\d+))?\b", &t, |c| {
        let symbol = re::group(c, 1);
        let Some((one, many, cent_one, cent_many)) = currency_names(symbol) else {
            return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        };
        let fraction = re::group(c, 3);
        let whole: u64 = re::group(c, 2).replace(',', "").parse().unwrap_or(0);
        if !fraction.is_empty() && fraction.chars().count() != 2 {
            // $1.5 -> one point five dollars
            return format!(
                "{} {many}",
                number(&format!("{}.{}", re::group(c, 2), fraction))
            );
        }
        let cents: u64 = if fraction.is_empty() {
            0
        } else {
            fraction.parse().unwrap_or(0)
        };
        if whole == 0 && cents > 0 {
            return format!(
                "{} {}",
                cardinal(cents),
                if cents == 1 { cent_one } else { cent_many }
            );
        }
        let mut out = format!(
            "{} {}",
            cardinal(whole),
            if whole == 1 { one } else { many }
        );
        if cents > 0 {
            out.push_str(&format!(
                " and {} {}",
                cardinal(cents),
                if cents == 1 { cent_one } else { cent_many }
            ));
        }
        out
    });

    // clock times: 5:30, 14:00, 5:30 p.m.
    t = re::sub_with_ctx(
        r"\b(\d{1,2}):(\d{2})\s*([aApP]\.?[mM])?(?:\.(?=\s+[a-z]))?(?=\W|$)",
        &t,
        |c, haystack, start| {
            let hour: u64 = re::group(c, 1).parse().unwrap_or(0);
            let minute: u64 = re::group(c, 2).parse().unwrap_or(0);
            let meridiem = re::group(c, 3);
            let whole = c.get(0).map(|m| m.end() - m.start()).unwrap_or(0);
            let after = haystack.get(start + whole..).unwrap_or("");
            if minute == 0 && meridiem.is_empty() && re::is_match(r"\s*(noon|midnight)", after) {
                return cardinal(hour);
            }
            if minute == 0 && meridiem.is_empty() && (hour > 12 || hour == 0) {
                // 14:00 -> fourteen hundred
                return format!(
                    "{} hundred",
                    if hour == 0 {
                        "zero".to_string()
                    } else {
                        cardinal(hour)
                    }
                );
            }
            let mut out = cardinal(hour);
            if minute == 0 {
                if meridiem.is_empty() {
                    out.push_str(" o'clock");
                }
            } else if minute < 10 {
                out.push_str(&format!(" oh {}", ONES[minute as usize]));
            } else {
                out.push_str(&format!(" {}", cardinal(minute)));
            }
            if !meridiem.is_empty() {
                out.push_str(&format!(" {}", meridiem.replace('.', "").to_uppercase()));
            }
            out
        },
    );
    t = re::sub_with(
        r"\b(\d{1,2})\s*([aApP])\.?[mM](?:\.(?=\s+[a-z]))?(?=\W|$)",
        &t,
        |c| {
            format!(
                "{} {}M",
                cardinal(re::group(c, 1).parse().unwrap_or(0)),
                re::group(c, 2).to_uppercase()
            )
        },
    );

    // dates read month first: 01/10/2026, 1/10/26
    t = re::sub_with(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", &t, |c| {
        let (month, day) = (
            re::group(c, 1).parse::<usize>().unwrap_or(0),
            re::group(c, 2).parse::<usize>().unwrap_or(0),
        );
        let raw_year = re::group(c, 3);
        let year: u64 = raw_year.parse::<u64>().unwrap_or(0)
            + if raw_year.chars().count() == 2 {
                2000
            } else {
                0
            };
        if (13..=31).contains(&month) && (1..=12).contains(&day) {
            return format!(
                "the {} of {} {}",
                ordinal_words(month as u64),
                tables::MONTHS[day - 1],
                year_words(year)
            );
        }
        if !((1..=12).contains(&month) && (1..=31).contains(&day)) {
            return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        }
        format!(
            "{} {} {}",
            tables::MONTHS[month - 1],
            ordinal_words(day as u64),
            year_words(year)
        )
    });

    // "March 3" -> "March third"
    let months = tables::MONTHS.join("|");
    let month_day = Box::leak(format!(r"\b({months})\.? (\d{{1,2}})\b(?!:)").into_boxed_str());
    t = re::sub_with(month_day, &t, |c| {
        format!(
            "{} {}",
            re::group(c, 1),
            ordinal_words(re::group(c, 2).parse().unwrap_or(0))
        )
    });

    // temperatures, slashes between letters, ordinals, years
    t = re::sub(r"(\d)\s?°\s?C\b", r"\1 degrees Celsius", &t);
    t = re::sub(r"(\d)\s?°\s?F\b", r"\1 degrees Fahrenheit", &t);
    t = re::sub(r"(\d)\s?°", r"\1 degrees", &t);
    t = re::sub(r"(?<=\w)/(?=\w)", " slash ", &t);
    t = re::sub_with(r"\b(\d+)(st|nd|rd|th)\b", &t, |c| {
        ordinal_words(re::group(c, 1).parse().unwrap_or(0))
    });
    // the year rule: 1204 -> twelve oh four, 1999 -> nineteen ninety nine
    t = re::sub_with(r"\b(1[1-9]\d\d|20\d\d)\b(?![.,]\d)", &t, |c| {
        year_words(re::group(c, 1).parse().unwrap_or(0))
    });
    t = re::sub(r"(?<![\w-])-(?=\d)", "minus ", &t);
    t = re::sub(r"(?<=[A-Za-z])(?=\d)", " ", &t); // v2 -> v two
    t = re::sub_with(r"(\d+(?:\.\d+)?)\s?%", &t, |c| {
        format!("{} percent", number(re::group(c, 1)))
    });
    t = re::sub_with(
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b",
        &t,
        |c| number(re::group(c, 0)),
    );
    t = t
        .replace('&', " and ")
        .replace('+', " plus ")
        .replace('=', " equals ")
        .replace('@', " at ")
        .replace('#', " number ");

    // the reference does not collapse whitespace here; `__call__` does
    let _ = collapse_spaces;
    let _ = frac_words;
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_reads_dollars_and_cents() {
        assert_eq!(fallback("$5"), "five dollars");
        assert_eq!(fallback("$5.99"), "five dollars and ninety nine cents");
        assert_eq!(fallback("$0.50"), "fifty cents");
        assert_eq!(
            fallback("$1,250"),
            "one thousand two hundred and fifty dollars"
        );
    }

    #[test]
    fn clock_times() {
        assert_eq!(fallback("5:30"), "five thirty");
        assert_eq!(fallback("14:00"), "fourteen hundred");
        assert_eq!(fallback("5:30 p.m."), "five thirty PM.");
    }

    #[test]
    fn years_before_plain_numbers() {
        assert_eq!(fallback("1204"), "twelve oh four");
        assert_eq!(fallback("1600"), "sixteen hundred");
        assert_eq!(fallback("1999"), "nineteen ninety nine");
        assert_eq!(fallback("42"), "forty two");
    }

    #[test]
    fn dates_and_temperatures() {
        assert_eq!(fallback("01/10/2026"), "January tenth twenty twenty six");
        assert_eq!(fallback("March 3"), "March third");
        // the reference verbalises the number too: five degrees Celsius
        assert_eq!(fallback("5°C"), "five degrees Celsius");
    }
}
