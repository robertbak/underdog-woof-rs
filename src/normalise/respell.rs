//! The respelling lexicon: hard names and foreign words rewritten to how the model reads them.
//!
//! A port of `underdog_woof/respell.py` over `respell.tsv` (24,883 entries), embedded in the binary
//! at build time. Matching is deliberately fuzzy on the *key* side and strict on the *text* side:
//!
//! * keys fold case, accents, apostrophes and hyphens (`fold`), and the lexicon's "c" flag means
//!   "only when capitalised";
//! * a word hyphen-joined to another word only matches when it is a capitalised name next to lower
//!   case (`Worcestershire-glazed` yes, `dove-cote` no);
//! * the longest multi-word key wins, so `'O sole mio` is not read as three separate lookups.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The lexicon, embedded so the binary is self-contained (it is ~590 KB).
const LEXICON: &str = include_str!("../../data/respell.tsv");

struct Lexicon {
    entries: HashMap<String, (String, bool)>,
    max_words: usize,
}

fn lexicon() -> &'static Lexicon {
    static LEX: OnceLock<Lexicon> = OnceLock::new();
    LEX.get_or_init(|| {
        let mut entries = HashMap::new();
        let mut max_words = 1usize;
        for line in LEXICON.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.trim_end().split('\t');
            let written = fields.next().unwrap_or("");
            let respelling = fields.next().unwrap_or("");
            let flags = fields.next().unwrap_or("");
            if written.is_empty() || respelling.is_empty() {
                continue;
            }
            let key = fold(written);
            max_words = max_words.max(key.matches(' ').count() + 1);
            entries.insert(key, (respelling.to_string(), flags.contains('c')));
        }
        Lexicon { entries, max_words }
    })
}

/// Lookup key: no accents, lower case, straight apostrophes, hyphens read as spaces.
///
/// Mirrors the reference's `fold` exactly: `unicodedata.normalize("NFKD", …)`, drop combining
/// marks, lower case, then the letters NFKD does not decompose (`ł`, `ø`, `æ`, `œ`, `ß`, `đ`).
pub fn fold(s: &str) -> String {
    use unicode_normalization::char::is_combining_mark;
    use unicode_normalization::UnicodeNormalization;

    let replaced: String = s
        .chars()
        .map(|c| match c {
            '’' | '‘' => '\'',
            other => other,
        })
        .collect::<String>()
        .to_lowercase();
    let mut folded = String::with_capacity(replaced.len());
    for c in replaced.nfkd() {
        if is_combining_mark(c) {
            continue;
        }
        match c {
            'ł' | 'ĺ' | 'ļ' | 'ľ' => folded.push('l'),
            'ø' => folded.push('o'),
            'æ' => folded.push_str("ae"),
            'œ' => folded.push_str("oe"),
            'ß' => folded.push_str("ss"),
            'đ' | 'ð' => folded.push('d'),
            'þ' => folded.push_str("th"),
            other => folded.push(other),
        }
    }
    let dashed: String = folded
        .chars()
        .map(|c| {
            if matches!(c, '-' | '‐' | '–' | '—') {
                ' '
            } else {
                c
            }
        })
        .collect();
    dashed.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A word token: letters plus internal apostrophes (`[^\W\d_]+(?:'[^\W\d_]+)*`).
fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let (start, c) = chars[i];
        if c.is_alphabetic() {
            let mut end = start + c.len_utf8();
            let mut j = i + 1;
            while j < chars.len() {
                let (pos, ch) = chars[j];
                if ch.is_alphabetic() {
                    end = pos + ch.len_utf8();
                    j += 1;
                } else if ch == '\'' && j + 1 < chars.len() && chars[j + 1].1.is_alphabetic() {
                    end = chars[j + 1].0 + chars[j + 1].1.len_utf8();
                    j += 2;
                } else {
                    break;
                }
            }
            spans.push((start, end));
            i = j;
        } else {
            i += 1;
        }
    }
    spans
}

/// `hyphen_ok`: a hyphen-joined word matches alone only when it is a capitalised name beside lower
/// case text.
fn hyphen_ok(text: &str, start: usize, end: usize, src: &str) -> bool {
    let before = &text[..start];
    let after = &text[end..];
    let left_joined = before.ends_with('-')
        && before[..before.len() - 1]
            .chars()
            .last()
            .is_some_and(|c| c.is_alphabetic());
    let right_joined =
        after.starts_with('-') && after[1..].chars().next().is_some_and(|c| c.is_alphabetic());
    let capitalised = src.chars().next().is_some_and(|c| c.is_uppercase());
    let mut others: Vec<&str> = Vec::new();
    if left_joined {
        others.push(
            before[..before.len() - 1]
                .split_whitespace()
                .last()
                .unwrap_or(""),
        );
    }
    if right_joined {
        others.push(after[1..].split_whitespace().next().unwrap_or(""));
    }
    for other in others {
        let other_lower = other.chars().next().is_some_and(|c| c.is_lowercase());
        if !(capitalised && other_lower) {
            return false;
        }
    }
    true
}

/// Apply the lexicon.
pub fn respell(text: &str) -> String {
    let lex = lexicon();
    if lex.entries.is_empty() {
        return text.to_string();
    }
    let text = text.replace('’', "'");
    let spans = word_spans(&text);
    let mut out = String::with_capacity(text.len());
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < spans.len() {
        let mut matched = false;
        let max_n = lex.max_words.min(spans.len() - i);
        for n in (1..=max_n).rev() {
            let window = &spans[i..i + n];
            // words must be separated only by single spaces or hyphens
            let contiguous = window
                .windows(2)
                .all(|pair| matches!(&text[pair[0].1..pair[1].0], " " | "-"));
            if !contiguous {
                continue;
            }
            let (start, end) = (window[0].0, window[n - 1].1);
            let src = &text[start..end];
            if !hyphen_ok(&text, start, end, src) {
                continue;
            }
            let key = fold(src);
            let hit = lex.entries.get(&key).cloned().or_else(|| {
                // possessive: "Siobhan's" -> "Siobhan" + "'s"
                key.strip_suffix("'s").and_then(|stem| {
                    lex.entries
                        .get(stem)
                        .map(|(rep, caps)| (format!("{rep}{}", &src[src.len() - 2..]), *caps))
                })
            });
            let Some((replacement, caps_only)) = hit else {
                continue;
            };
            if caps_only
                && !window.iter().all(|(a, b)| {
                    text[*a..*b]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_uppercase())
                })
            {
                continue;
            }
            let mut replacement = replacement;
            if src.chars().next().is_some_and(|c| c.is_uppercase())
                && replacement.chars().next().is_some_and(|c| c.is_lowercase())
            {
                replacement = replacement
                    .chars()
                    .next()
                    .unwrap()
                    .to_uppercase()
                    .collect::<String>()
                    + &replacement[1..];
            }
            out.push_str(&text[pos..start]);
            out.push_str(&replacement);
            pos = end;
            i += n;
            matched = true;
            break;
        }
        if !matched {
            i += 1;
        }
    }
    out.push_str(&text[pos..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_strips_accents_case_and_hyphens() {
        assert_eq!(fold("São-Paulo"), "sao paulo");
        assert_eq!(fold("’O sole mio"), "'o sole mio");
        assert_eq!(fold("Łódź"), "lodz");
        assert_eq!(fold("Nguyễn"), "nguyen");
    }

    #[test]
    fn lexicon_is_loaded() {
        assert!(
            lexicon().entries.len() > 20_000,
            "got {}",
            lexicon().entries.len()
        );
        assert!(lexicon().max_words >= 2);
    }

    #[test]
    fn respell_replaces_hard_names() {
        // expectations taken from the reference Respeller
        assert_eq!(respell("'Eua and Worcestershire"), "'Eua and Wuster-sher");
        assert_eq!(respell("Siobhan spoke"), "Shivawn spoke");
        assert_eq!(respell("Reykjavik and Lodz"), "Rayk-yuh-veek and Lodz");
        // hyphen-joined names keep their hyphen and get replaced (capitalised name beside lower case)
        assert_eq!(
            respell("Worcestershire-glazed carrots"),
            "Wuster-sher-glazed carrots"
        );
    }

    #[test]
    fn ordinary_words_survive() {
        assert_eq!(respell("The committee agreed."), "The committee agreed.");
    }
}
