//! House numbers, street types and the abbreviation table.
//!
//! One function per stage of `norm_pre`, in the reference's order; `prepass` calls them in
//! sequence. Each is a pure String -> String transform, so order is the only state.

use super::*;

/// House numbers and street types.
pub(super) fn house_numbers(mut t: String) -> String {
    let street_pat = leak(format!(
        r"\b(\d{{1,4}})([A-Za-z]?)\s+((?:(?:North|South|East|West|N\.?|S\.?|E\.?|W\.?)\s+)?(?:[A-Z][\w']*|\d+(?:st|nd|rd|th))(?:\s+[A-Z][\w']*){{0,2}})\s+{}",
        street_re()
    ));
    t = re::sub_with_ctx(street_pat, &t, |c, haystack, start| {
        let whole = c.get(0).map(|m| m.end() - m.start()).unwrap_or(0);
        let after = &haystack[start + whole..];
        let tail = if !re::group(c, 5).is_empty()
            && (after.trim_start().is_empty()
                || after
                    .trim_start()
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_uppercase()))
        {
            "."
        } else {
            ""
        };
        let letter = re::group(c, 2);
        let kind = lookup(tables::STREET, re::group(c, 4)).unwrap_or(re::group(c, 4));
        format!(
            "{}{} {} {kind}{tail}",
            nread(re::group(c, 1)),
            if letter.is_empty() {
                String::new()
            } else {
                format!(" {}", letter.to_uppercase())
            },
            re::group(c, 3)
        )
    });
    t = re::sub_with_ctx(
        r"(?<!Mount )(?<!Port )(?<!Fort )(?<!Lake )(?<!Cape )\b([A-Z][a-z]+|\d+(?:st|nd|rd|th))\s+(St|Rd|Ave|Blvd|Ln|Hwy|Pkwy|Cres|Terr?)\b(\.?)",
        &t,
        |c, haystack, start| {
            let whole = c.get(0).map(|m| m.end() - m.start()).unwrap_or(0);
            let after = &haystack[start + whole..];
            let first = re::group(c, 1);
            let kind = re::group(c, 2);
            let stop_words = ["on", "at", "visit", "see"];
            if tables::ROMAN_STOP.contains(&first.to_lowercase().as_str())
                || stop_words.contains(&first.to_lowercase().as_str())
            {
                return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            if kind == "St"
                && after
                    .trim_start()
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_uppercase())
            {
                return c.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            let tail = if !re::group(c, 3).is_empty() && after.trim_start().is_empty() {
                "."
            } else {
                ""
            };
            format!(
                "{first} {}{tail}",
                lookup(tables::STREET, kind).unwrap_or(kind)
            )
        },
    );
    t = re::sub_with(
        r"\b((?:I|US|SR|Route|Highway|Hwy|Interstate)[- ]\d+)\s([NSEW])\b",
        &t,
        |c| {
            let direction = match re::group(c, 2) {
                "N" => "North",
                "S" => "South",
                "E" => "East",
                _ => "West",
            };
            format!("{} {direction}", re::group(c, 1))
        },
    );
    t
}

/// The abbreviation table (patterns literal, replacements in the reference's style).
pub(super) fn abbreviations(mut t: String) -> String {
    for (pattern, replacement) in tables::ABBR {
        let pat = leak(pattern.to_string());
        t = re::sub(pat, replacement, &t);
    }
    t
}
