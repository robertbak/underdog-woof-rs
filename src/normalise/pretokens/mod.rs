//! `norm_tokens` — the rules that must see the raw string before the general passes.
//!
//! Straight port of the reference's `norm_tokens` and `norm_money_and_units`, in the reference's rule
//! order: money has to be recognised before the phone rule can eat its digits, Roman numerals before
//! the abbreviation table, and the toll-free rule before the general phone rule.
//!
//! Split by topic because the two entry points are independent — [`norm_tokens`] is the front door,
//! [`norm_money_and_units`] the pass it delegates currency and units to — and both are called from
//! `prepass`. The shared vocabulary helpers live here.
//!
//! Patterns the reference builds at import time from its tables (`codes`, the day/month
//! alternations, the Roman numeral matcher) are composed once through [`crate::normalise::re`]'s
//! cache, so they cost nothing after the first call.

mod money;
mod roman;
mod tokens;

pub use money::norm_money_and_units;
pub use roman::roman;
pub(super) use roman::{handle, roman_re, roman_sub};
pub use tokens::{frac_words, norm_tokens};

// one import surface for the child modules: `use super::*` in each of them
pub(super) use std::sync::OnceLock;

pub(super) use crate::normalise::re;
pub(super) use crate::normalise::re::sub_with;
pub(super) use crate::normalise::tables;
pub(super) use crate::normalise::{cardinal, code_read, digits, ordinal, spell, split_ident, year};

/// `spell`/`digits`/`cardinal` etc. live in the parent module; this module adds the rule helpers.
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

/// Look up a value in one of the generated two-column tables.
fn lookup(table: &'static [(&'static str, &'static str)], key: &str) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// `pl(n, w)`: pluralise a unit.
fn pl(n: &str, word: &str) -> String {
    let count: i64 = n.parse().unwrap_or(0);
    format!("{count} {word}{}", if count == 1 { "" } else { "s" })
}

/// The currency alternation the reference builds from `CURCODE`, longest code first.
fn currency_codes() -> &'static str {
    static CODES: OnceLock<&'static str> = OnceLock::new();
    CODES.get_or_init(|| {
        let mut keys: Vec<&str> = tables::CURCODE.iter().map(|(k, _)| *k).collect();
        keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let escaped: Vec<String> = keys.iter().map(|k| regex_escape(k)).collect();
        Box::leak(escaped.join("|").into_boxed_str())
    })
}

pub(super) fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_and_tech_tokens() {
        assert_eq!(norm_tokens("C#"), "C sharp");
        assert_eq!(norm_tokens("C++"), "C plus plus");
        assert_eq!(norm_tokens(".NET"), "dot net");
        assert_eq!(norm_tokens("getUserName()"), "get User Name");
    }

    #[test]
    fn money_and_scales() {
        let out = norm_money_and_units("$4.2 million");
        assert!(
            out.contains("million") && out.contains("dollars"),
            "got {out}"
        );
        // the reference leaves a bare euro sign alone (only multi-character codes are money)
        assert_eq!(norm_money_and_units("€10"), "€10");
    }

    #[test]
    fn durations_and_decades() {
        assert_eq!(norm_money_and_units("2h 15m"), "2 hours 15 minutes");
        assert_eq!(norm_money_and_units("the 1990s"), "the nineteen nineties");
        assert_eq!(norm_money_and_units("'80s"), "eighties");
    }

    #[test]
    fn roman_in_context() {
        assert_eq!(norm_money_and_units("Henry VIII"), "Henry the eighth");
        assert_eq!(norm_money_and_units("Chapter XIV"), "Chapter fourteen");
        assert_eq!(norm_money_and_units("World War II"), "World War two");
    }

    #[test]
    fn powers() {
        assert_eq!(norm_money_and_units("x^2"), "x squared");
        assert_eq!(norm_money_and_units("10^23"), "10 to the twenty third");
    }
}
