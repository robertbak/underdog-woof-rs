//! Helpers shared by the stage functions: table lookups, alternations, pluralisation,
//! space collapsing and the element check.

use super::*;

pub(crate) fn lookup(
    table: &'static [(&'static str, &'static str)],
    key: &str,
) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// `sing`: drop a plural from the last word (`feet` -> `foot` via the table).
pub(crate) fn sing(w: &str) -> String {
    let (head, sep, rest) = match w.split_once(" per ") {
        Some((h, r)) => (h, " per ", r),
        None => (w, "", ""),
    };
    if head.contains(" of ") {
        return w.to_string();
    }
    let mut words: Vec<String> = head.split(' ').map(str::to_string).collect();
    if let Some(last) = words.pop() {
        let singular = lookup(tables::SING, &last)
            .map(str::to_string)
            .unwrap_or_else(|| {
                if last.ends_with('s') {
                    last[..last.len() - 1].to_string()
                } else {
                    last.clone()
                }
            });
        words.push(singular);
    }
    format!("{}{sep}{rest}", words.join(" "))
}

/// `unit`: the unit's words, singular when the number is exactly 1.
pub(crate) fn unit(key: &str, n: &str) -> String {
    let words = lookup(tables::UNITS, key).unwrap_or(key);
    if n == "1" {
        sing(words)
    } else {
        words.to_string()
    }
}

/// The unit-name alternation, longest first (the reference builds it from `UNITS`).
pub(crate) fn unit_keys() -> &'static str {
    static KEYS: OnceLock<&'static str> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut keys: Vec<&str> = tables::UNITS
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| *k != "in" && *k != "s")
            .collect();
        keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let escaped: Vec<String> = keys.iter().map(|k| regex_escape(k)).collect();
        Box::leak(escaped.join("|").into_boxed_str())
    })
}

/// The street-type alternation (`STREET_RE`), from the reference's literal plus its table's keys.
pub(crate) fn street_re() -> &'static str {
    static PATTERN: OnceLock<&'static str> = OnceLock::new();
    PATTERN.get_or_init(|| Box::leak(tables::STREET_RE.to_string().into_boxed_str()))
}

pub(crate) fn month_alternation() -> &'static str {
    static MONTHS: OnceLock<&'static str> = OnceLock::new();
    MONTHS.get_or_init(|| Box::leak(tables::MONTHS.join("|").into_boxed_str()))
}

pub(crate) fn days_alternation() -> &'static str {
    static DAYS: OnceLock<&'static str> = OnceLock::new();
    DAYS.get_or_init(|| {
        let keys: Vec<&str> = tables::DAYS.iter().map(|(k, _)| *k).collect();
        Box::leak(keys.join("|").into_boxed_str())
    })
}

pub(crate) fn mon_abbr_alternation() -> &'static str {
    static ABBR: OnceLock<&'static str> = OnceLock::new();
    ABBR.get_or_init(|| {
        let keys: Vec<&str> = tables::MON_ABBR.iter().map(|(k, _)| *k).collect();
        Box::leak(keys.join("|").into_boxed_str())
    })
}

pub(crate) fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

pub(crate) fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub(crate) fn all_elements(formula: &str) -> bool {
    let mut rest = formula;
    let mut any = false;
    while !rest.is_empty() {
        let mut chars = rest.chars();
        let Some(first) = chars.next() else { break };
        if !first.is_uppercase() {
            return false;
        }
        let mut element = first.to_string();
        if let Some(second) = chars.clone().next() {
            if second.is_lowercase() {
                element.push(second);
                chars.next();
            }
        }
        let digits: String = chars.clone().take_while(|c| c.is_ascii_digit()).collect();
        if !tables::ELEMENTS.contains(&element.as_str()) {
            return false;
        }
        any = true;
        rest = &rest[element.len() + digits.len()..];
    }
    any
}

pub(crate) fn numbers_in(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            current.push(c);
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

pub(crate) fn pl(n: &str, word: &str) -> String {
    let count: i64 = n.parse().unwrap_or(0);
    format!("{count} {word}{}", if count == 1 { "" } else { "s" })
}

/// `re.sub(r"\s{2,}", " ", t).strip()`
pub fn collapse_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !space {
                out.push(' ');
                space = true;
            }
        } else {
            out.push(c);
            space = false;
        }
    }
    out.trim().to_string()
}
