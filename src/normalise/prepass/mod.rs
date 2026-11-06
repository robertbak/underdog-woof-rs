//! `norm_pre` — everything the reference does before the fallback verbaliser.
//!
//! `norm_tokens` first (see [`super::pretokens`]), then port/host/URL/e-mail rules, file names, the
//! several date and month forms, house numbers and street types, the abbreviation table, ISO dates,
//! ranges, fractions, day/month pairs, versions, clock times, phones, units, identifiers, chemical
//! formulas, hash numbers, codes, long digit strings, durations, dimensions, symbols and
//! pounds/pence.
//!
//! The stages live in sibling modules by topic, one function per stage, called here in the
//! reference's order — that order is the only state the pipeline carries. Context-sensitive rules
//! (`street`, `dm`, `pair`, the day/month forms) use [`crate::normalise::re::sub_with_ctx`] because
//! the reference inspects `m.string` before and after each match.

mod addresses;
mod dates;
mod numbers;
mod support;
mod urls;

pub use support::collapse_spaces;

// one import surface for the stage modules: `use super::*` in each of them
pub(super) use std::sync::OnceLock;

pub(super) use super::pretokens::{code_read, frac_words, norm_money_and_units, norm_tokens};
pub(super) use super::re;
pub(super) use super::tables;
pub(super) use super::{cardinal, digits, nread, ordinal, pronounceable, spell};

pub(super) use support::{
    all_elements, days_alternation, leak, lookup, mon_abbr_alternation, month_alternation,
    numbers_in, pl, sing, street_re, unit, unit_keys,
};

/// `norm_pre`.
pub fn norm_pre(text: &str) -> String {
    let mut t = norm_money_and_units(&norm_tokens(text));
    t = urls::host_ports(t);
    t = urls::schemes(t);
    t = urls::emails_and_domains(t);
    t = urls::file_names(t);
    t = dates::slashed_dates(t);
    t = dates::day_ranges(t);
    t = addresses::house_numbers(t);
    t = addresses::abbreviations(t);
    t = dates::iso_dates(t);
    t = dates::day_month_year(t);
    t = dates::year_ranges(t);
    t = numbers::mixed_fractions(t);
    t = dates::day_month(t);
    t = numbers::versions(t);
    t = numbers::clock_times(t);
    t = numbers::ranges_and_ratios(t);
    t = numbers::phones(t);
    t = numbers::units_after_numbers(t);
    t = numbers::identifiers(t);
    t = numbers::letter_digit_codes(t);
    t = numbers::mixed_ids(t);
    t = numbers::durations(t);
    t = numbers::dimensions_and_symbols(t);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_emails_and_schemes() {
        assert!(norm_pre("https://example.com/path").contains("dot"));
        assert_eq!(norm_pre("me@example.com"), "me at example dot com");
        assert_eq!(norm_pre("localhost:8080"), "localhost colon eighty eighty");
    }

    #[test]
    fn dates_and_months() {
        assert_eq!(norm_pre("2026-10-21"), "October 21, 2026");
        assert_eq!(norm_pre("14 March 2027"), "the fourteenth of March 2027");
        assert_eq!(norm_pre("3 March 2026"), "the third of March 2026");
    }

    #[test]
    fn units_keep_the_reference_shape() {
        assert_eq!(norm_pre("5km"), "5 kilometers");
        assert_eq!(norm_pre("1 kg"), "1 kilogram");
        // the reference leaves 3x4 alone at this stage (the times rule needs a following word)
        assert_eq!(norm_pre("3x4"), "3x4");
    }

    #[test]
    fn phones_and_codes() {
        assert_eq!(
            norm_pre("415-555-0199"),
            "four one five, five five five, zero one nine nine"
        );
        assert_eq!(norm_pre("QX-4471"), "Q X four four seven one");
    }

    #[test]
    fn street_names_and_abbreviations() {
        assert_eq!(
            norm_pre("742 Evergreen Terrace"),
            "seven forty two Evergreen Terrace"
        );
        // the abbreviation table drops the dot
        assert_eq!(norm_pre("etc."), "et cetera");
    }
}
