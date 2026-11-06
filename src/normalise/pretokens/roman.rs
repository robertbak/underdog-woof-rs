//! Roman numerals: `IV` -> four, `Henry VIII` -> Henry the eighth, and the `i`/`v`/`x` ambiguity the
//! reference resolves in `roman_sub`.

use super::*;

/// `roman()`: additive/subtractive value of a numeral.
pub fn roman(s: &str) -> u32 {
    let value = |c: char| match c {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        'M' => 1000,
        _ => 0,
    };
    let chars: Vec<char> = s.chars().collect();
    chars
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let next = chars.get(i + 1).map(|n| value(*n)).unwrap_or(0);
            if next > value(*c) {
                -value(*c)
            } else {
                value(*c)
            }
        })
        .sum::<i32>() as u32
}

/// The reference's `ROMAN_RE`, composed from the base alternation.
pub(crate) fn roman_re() -> &'static str {
    static PATTERN: OnceLock<&'static str> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Box::leak(
            format!(
                r"\b([A-Za-z][A-Za-z']*\.?)(\s+)({})\b(?!'\w|-\d)",
                r"(?=[MDCLXVI]+\b)M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
            )
            .into_boxed_str(),
        )
    })
}

/// `roman_sub`: decide, from the preceding word, whether a numeral is a count, an ordinal or nothing.
pub(crate) fn roman_sub(prev: &str, gap: &str, numeral: &str, before: &str, prev2: &str) -> String {
    let key = prev.to_lowercase();
    let key = key.trim_end_matches('.').to_string();
    let n = roman(numeral);
    let title = |w: &str| {
        let mut chars = w.chars();
        matches!(
            (chars.next(), chars.next(), chars.next()),
            (Some(a), Some(b), None) if a.is_uppercase() && b.is_lowercase()
        ) || (w.len() >= 2
            && w.chars().next().is_some_and(|c| c.is_uppercase())
            && w.chars().skip(1).all(|c| c.is_lowercase()))
    };
    let in_card = tables::ROMAN_CARD.contains(&key.as_str());
    // "through XIV", "to XIV" are ranges, not counts - unless a card word precedes
    if ["through", "thru", "to", "and", "or"].contains(&key.as_str()) {
        let has_card_before = tables::ROMAN_CARD
            .iter()
            .any(|w| before.to_lowercase().contains(w));
        if !has_card_before {
            return format!("{prev}{gap}{numeral}");
        }
    }
    if in_card
        && (numeral.chars().count() > 1
            || (numeral == "I" && prev.chars().next().is_some_and(|c| c.is_uppercase()))
            || ("VX".contains(numeral) && !tables::ROMAN_LABEL.contains(&key.as_str())))
    {
        return format!("{prev}{gap}{}", cardinal(n as u64));
    }
    if !numeral.chars().all(|c| "IVXL".contains(c)) {
        return format!("{prev}{gap}{numeral}");
    }
    if tables::REGNAL.contains(&prev)
        || (tables::TITLES.contains(&prev2) && title(prev) && numeral != "I")
    {
        return format!("{prev}{gap}the {}", ordinal(n as u64));
    }
    let has_l = numeral.contains('L');
    if numeral.chars().count() > 1 && title(prev) && !has_l {
        let full = title(prev2) && !tables::ROMAN_STOP.contains(&prev2.to_lowercase().as_str());
        return format!(
            "{prev}{gap}{}",
            if full {
                format!("the {}", ordinal(n as u64))
            } else {
                cardinal(n as u64)
            }
        );
    }
    format!("{prev}{gap}{numeral}")
}

/// `handle`: `@openai` -> `open A I`, `@Jane_Doe` -> `Jane underscore Doe`.
pub(crate) fn handle(h: &str) -> String {
    let parts: Vec<String> = h
        .split('_')
        .map(|part| {
            if let Some(groups) = super::re::groups_opt(r"([a-z]{3,}[nlyex])(ai|hq)", part) {
                let stem = groups.get(1).cloned().flatten().unwrap_or_default();
                let tail = groups.get(2).cloned().flatten().unwrap_or_default();
                format!("{stem} {}", spell(&tail))
            } else {
                split_ident(part)
            }
        })
        .filter(|s| !s.is_empty())
        .collect();
    parts.join(" underscore ")
}
