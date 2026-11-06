//! English text normalisation: raw text -> the spoken form the model reads.
//!
//! A port of `underdog_woof/normalize.py` (which itself layers its own rules on top of NeMo when
//! NeMo is installed; the Rust port targets the NeMo-free configuration, which is what the packaged
//! server runs).
//!
//! # Porting strategy
//!
//! The reference is ~670 lines of rules plus ~30 lookup tables plus a 590 KB respelling lexicon, and
//! its behaviour is only defined by that code. A hand-written "equivalent" would diverge silently, so
//! the port is staged and measured against a differential corpus generated from the reference
//! (`tools/normalise_corpus.py` → `tests/data/normalise_cases.tsv`); `tools/normalise_diff.py`
//! reports how many cases match. Each stage below is expected to raise that number:
//!
//! 1. **numbers and identifiers** (this stage): cardinals, ordinals, years, decimals, digit strings,
//!    spelled codes, camelCase/underscore identifiers.
//! 2. money, times, dates, phone numbers, ranges, decades, powers, filenames, street names.
//! 3. Roman numerals in context (chapter/super-bowl/regnal rules), abbreviations, symbols, units.
//! 4. the respelling lexicon (`respell.tsv`), which needs its own case/accent/hyphen matcher.
//!
//! Anything not yet ported falls through unchanged, which is also what the reference does with text
//! it does not recognise.

/// `fallback`: the verbaliser used when NeMo is absent.
pub mod fallback;
/// `norm_pre`: everything before the fallback verbaliser.
pub mod prepass;
/// `norm_tokens`: rules that must see the raw string before the general passes.
pub mod pretokens;
/// Python-`re` compatibility (lookarounds, `\1` replacement syntax, pattern cache).
pub mod re;
/// The respelling lexicon (`respell.tsv`, embedded).
pub mod respell;
/// Literal tables extracted from the reference (`tools/gen_normalise_tables.py`).
pub mod tables;

/// One to nineteen, the reference's `ONES`.
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

/// Multiples of ten from twenty, the reference's `TENS` (underscores hold the empty slots).
const TENS: [&str; 10] = [
    "_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Scale names by magnitude, the reference's `SCALES`.
const SCALES: [(u64, &str); 4] = [
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1000, "thousand"),
];

/// Irregular ordinals, the reference's `ORD`.
pub(super) fn ordinal_word(last: &str) -> String {
    match last {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        other if other.ends_with('y') => format!("{}ieth", &other[..other.len() - 1]),
        other => format!("{other}th"),
    }
}

/// Cardinal of a non-negative integer, exactly as `cardinal()` builds it — note the British "and".
pub fn cardinal(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let (t, o) = (n / 10, n % 10);
        return if o == 0 {
            TENS[t as usize].to_string()
        } else {
            format!("{} {}", TENS[t as usize], ONES[o as usize])
        };
    }
    if n < 1000 {
        let (h, rest) = (n / 100, n % 100);
        return if rest == 0 {
            format!("{} hundred", ONES[h as usize])
        } else {
            format!("{} hundred and {}", ONES[h as usize], cardinal(rest))
        };
    }
    for (scale, name) in SCALES {
        if n >= scale {
            let rest = n % scale;
            let tail = if rest == 0 {
                String::new()
            } else if rest < 100 {
                format!(" and {}", cardinal(rest))
            } else {
                format!(" {}", cardinal(rest))
            };
            return format!("{} {}{}", cardinal(n / scale), name, tail);
        }
    }
    n.to_string()
}

/// Ordinal of a non-negative integer (`ordinal()`).
pub fn ordinal(n: u64) -> String {
    let mut words: Vec<String> = cardinal(n).split(' ').map(str::to_string).collect();
    if let Some(last) = words.pop() {
        words.push(ordinal_word(&last));
    }
    words.join(" ")
}

/// Year reading (`year()`): 2000-2009 are plain cardinals, otherwise split at the century.
pub fn year(n: u64) -> String {
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

/// Digits of a numeric string, one word each (`digits()`).
pub fn digits(s: &str) -> String {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| ONES[d as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Spell a chunk as letters, with digits spoken (`spell()`).
pub fn spell(s: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            out.push(ONES[d as usize].to_string());
        } else if c.is_alphabetic() {
            out.extend(c.to_uppercase().map(String::from));
        }
    }
    out.join(" ")
}

/// All-caps chunk that reads as a word (SAVE, WINTER) rather than as letters (ORD, XT): `pronounceable()`.
pub fn pronounceable(word: &str) -> bool {
    let w = word.to_lowercase();
    if w.chars().count() < 4 || !w.chars().any(|c| "aeiouy".contains(c)) {
        return false;
    }
    let has_common_cluster = has_run(&w, 3)
        && [
            "str", "sch", "thr", "chr", "spr", "scr", "ght", "nds", "nts",
        ]
        .iter()
        .any(|c| w.contains(c));
    if has_run(&w, 3) && !has_common_cluster {
        return false;
    }
    // (?:[^aeiouy]{0,3}[aeiouy]{1,2})+[^aeiouy]{0,3}
    let bytes: Vec<char> = w.chars().collect();
    let is_vowel = |c: char| "aeiouy".contains(c);
    let mut i = 0usize;
    let mut groups = 0usize;
    while i < bytes.len() {
        let start = i;
        while i < bytes.len() && !is_vowel(bytes[i]) && i - start < 3 {
            i += 1;
        }
        if i < bytes.len() && !is_vowel(bytes[i]) {
            return false; // more than three consonants in a row
        }
        if i >= bytes.len() {
            break; // trailing consonant run is allowed
        }
        let vowel_start = i;
        while i < bytes.len() && is_vowel(bytes[i]) && i - vowel_start < 2 {
            i += 1;
        }
        if i < bytes.len() && is_vowel(bytes[i]) {
            return false; // more than two vowels in a row
        }
        groups += 1;
    }
    groups > 0 && bytes.iter().rev().take_while(|c| !is_vowel(**c)).count() <= 3
}

/// Whether `s` contains a run of `n` consecutive characters that are not vowels.
fn has_run(s: &str, n: usize) -> bool {
    let mut run = 0usize;
    for c in s.chars() {
        if "aeiouy".contains(c) {
            run = 0;
        } else {
            run += 1;
            if run >= n {
                return true;
            }
        }
    }
    false
}

/// Split an identifier on `-` and `/` and read each part (`code_read()`).
pub fn code_read(s: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for part in s.split(['-', '/']) {
        if part.is_empty() {
            continue;
        }
        let part_string = part.to_string();
        if part.chars().all(|c| c.is_ascii_digit()) {
            let short = part.len() <= 2 && !part.starts_with('0');
            out.push(if short { part_string } else { digits(part) });
        } else if part.chars().all(|c| c.is_alphabetic())
            && part.chars().all(|c| c.is_lowercase())
            && part.chars().count() > 3
            && pronounceable(part)
        {
            out.push(part_string);
        } else if part.chars().all(|c| c.is_alphabetic()) && pronounceable(part) {
            let mut chars = part.chars();
            let mut capitalised = String::new();
            if let Some(first) = chars.next() {
                capitalised.extend(first.to_uppercase());
                capitalised.push_str(&chars.as_str().to_lowercase());
            }
            out.push(capitalised);
        } else {
            out.push(spell(part));
        }
    }
    out.join(" ")
}

/// Numbers in identifiers, read like room and flight numbers (`nread()`).
pub fn nread(s: &str) -> String {
    let Ok(n) = s.parse::<u64>() else {
        return digits(s);
    };
    let leading_zero = s.starts_with('0') && s.chars().count() > 1;
    if leading_zero || s.chars().count() > 4 {
        return digits(s);
    }
    if n < 100 || (n % 100 == 0 && s.chars().count() == 3) || n % 1000 == 0 {
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

/// `getUserName` -> `get user name`, `user_id` -> `user id`, `OpenAI` -> `Open A I` (`split_ident()`).
pub fn split_ident(s: &str) -> String {
    let replaced = s.replace('_', " ");
    let chars: Vec<char> = replaced.chars().collect();
    let mut out = String::with_capacity(replaced.len() + 8);
    for (i, c) in chars.iter().enumerate() {
        let prev = i.checked_sub(1).map(|p| chars[p]);
        let next = chars.get(i + 1).copied();
        let next_next = chars.get(i + 2).copied();
        let boundary = match prev {
            Some(p) if p.is_lowercase() || p.is_ascii_digit() => c.is_uppercase(),
            Some(p) if p.is_uppercase() => {
                c.is_uppercase() && next.is_some_and(|n| n.is_lowercase())
            }
            _ => false,
        };
        if boundary && !out.ends_with(' ') {
            out.push(' ');
        }
        out.push(*c);
        let _ = next_next;
    }
    out.split_whitespace()
        .map(|word| {
            if word.chars().count() <= 4 && word.chars().all(|c| c.is_uppercase()) {
                spell(word)
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Integer value of a Roman numeral (`roman()`), or `None` if it is not well formed.
pub fn roman_value(s: &str) -> Option<u32> {
    fn value(c: char) -> Option<u32> {
        Some(match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        })
    }
    let values: Vec<u32> = s.chars().map(value).collect::<Option<_>>()?;
    let mut total = 0u32;
    for (i, v) in values.iter().enumerate() {
        if values.get(i + 1).is_some_and(|next| next > v) {
            total -= v;
        } else {
            total += v;
        }
    }
    Some(total)
}

/// Stage 1 of the port: numeric tokens, ordinals, years and identifiers.
///
/// Money, times, dates, phones, Roman numerals, abbreviations, units and the respelling lexicon are
/// still handled by the reference implementation (stage 2-4, tracked by the diff corpus).
/// The reference's `Normalizer.__call__`, minus NeMo (the configuration the package ships).
pub fn normalise(text: &str) -> String {
    let mut t = prepass::norm_pre(text);
    // only verbalise when something is left that needs it, exactly as the reference checks
    if re::is_match(r"[0-9#%$€£@&/_+=<>±≥≤×°]", &t) {
        t = fallback::fallback(&t);
    }
    t = re::sub(r"\s+([.,!?;:])", r"\1", &t);
    t = prepass::collapse_spaces(&t);
    // names and foreign words last: no rule above should see a respelling
    t = respell::respell(&t);

    // capitalisation is decided by the *input's* first character
    if text.chars().next().is_some_and(|c| c.is_uppercase()) {
        let mut chars = t.chars();
        if let Some(first) = chars.next() {
            t = first.to_uppercase().collect::<String>() + chars.as_str();
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals_match_the_reference_wording() {
        assert_eq!(cardinal(0), "zero");
        assert_eq!(cardinal(42), "forty two");
        assert_eq!(cardinal(101), "one hundred and one");
        assert_eq!(cardinal(1000), "one thousand");
        assert_eq!(
            cardinal(1_250_000),
            "one million two hundred and fifty thousand"
        );
    }

    #[test]
    fn ordinals_replace_only_the_last_word() {
        assert_eq!(ordinal(1), "first");
        assert_eq!(ordinal(22), "twenty second");
        assert_eq!(ordinal(100), "one hundredth");
    }

    #[test]
    fn years_split_at_the_century() {
        assert_eq!(year(1999), "nineteen ninety nine");
        // 2000-2009 go through cardinal(), which says "and" (reference behaviour)
        assert_eq!(year(2005), "two thousand and five");
        assert_eq!(year(1900), "nineteen hundred");
        assert_eq!(year(1604), "sixteen oh four");
    }

    #[test]
    fn pronounceable_distinguishes_words_from_letter_runs() {
        assert!(pronounceable("SAVE"));
        assert!(pronounceable("WINTER"));
        assert!(!pronounceable("ORD"));
        assert!(!pronounceable("XT"));
        assert!(!pronounceable("CS"));
    }

    #[test]
    fn identifiers_are_split_and_spelled() {
        // the function's own semantics; the pipeline applies it only in identifier contexts
        assert_eq!(split_ident("getUserName"), "get User Name");
        assert_eq!(split_ident("user_id"), "user id");
        assert_eq!(split_ident("OpenAI"), "Open A I");
    }

    #[test]
    fn roman_values_are_subtractive() {
        assert_eq!(roman_value("VIII"), Some(8));
        assert_eq!(roman_value("XIV"), Some(14));
        assert_eq!(roman_value("LIX"), Some(59));
        assert_eq!(roman_value("XYZ"), None);
    }

    #[test]
    fn normalise_translates_numbers_and_leaves_prose_alone() {
        assert_eq!(normalise("42"), "forty two");
        assert_eq!(normalise("1999"), "nineteen ninety nine");
        assert_eq!(normalise("3rd"), "third");
        assert_eq!(normalise("3.5"), "three point five");
        assert_eq!(normalise("The committee agreed."), "The committee agreed.");
    }
}
