//! Sentence and clause splitting, so audio can start before the whole text is synthesised.
//!
//! A port of `underdog_woof/textsplit.py`. The rules are the reference's, and they matter for
//! streaming latency: every sentence becomes its own utterance, and a sentence longer than
//! [`CLAUSE_MAX`] is split at clause punctuation, merging clauses greedily. A piece that is still
//! too long and has no clause punctuation is split once near its middle, before a conjunction.
//!
//! `FIRST_MAX` in the reference shortens only the *first* piece (trading prosody for latency); it is
//! off by default there and exposed here as [`Options::first_max`].

/// A sentence longer than this is split at clause boundaries.
pub const CLAUSE_MAX: usize = 120;
/// Never produce a piece shorter than this when splitting at a conjunction, or when shortening the
/// first utterance.
const MIN_PIECE: usize = 30;
const FIRST_MIN: usize = 20;

const CONJUNCTIONS: &[&str] = &[
    "and", "but", "or", "nor", "that", "which", "who", "whom", "whose", "when", "where", "while",
    "because", "although", "though", "so", "if", "as", "than", "until",
];

/// Splitting behaviour.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Maximum characters per utterance before splitting at clause boundaries.
    pub clause_max: usize,
    /// If non-zero, shorten the first utterance to `first_max` characters so playback can start
    /// sooner (the reference tests 70; default off).
    pub first_max: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            clause_max: CLAUSE_MAX,
            first_max: 0,
        }
    }
}

/// Split `text` into utterances ready for synthesis.
pub fn split_text(text: &str, options: Options) -> Vec<String> {
    let mut pieces = split_sentences(text, options.clause_max);
    if options.first_max > 0 && !pieces.is_empty() {
        let first = shorten_first(&pieces[0], options.first_max);
        pieces.splice(0..1, first);
    }
    pieces
}

fn split_sentences(text: &str, clause_max: usize) -> Vec<String> {
    let mut out = Vec::new();
    for sentence in sentence_bounds(text.trim()) {
        if sentence.chars().count() <= clause_max {
            out.push(sentence);
            continue;
        }
        // greedy merge of clauses, so no piece exceeds clause_max where avoidable
        let mut merged: Vec<String> = Vec::new();
        let mut current = String::new();
        for clause in clause_bounds(&sentence) {
            let too_long = !current.is_empty()
                && current.chars().count() + 1 + clause.chars().count() > clause_max;
            if too_long {
                merged.push(std::mem::take(&mut current));
                current = clause;
            } else if current.is_empty() {
                current = clause;
            } else if current.ends_with('—') || current.ends_with("--") {
                current.push_str(&clause);
            } else {
                current.push(' ');
                current.push_str(&clause);
            }
        }
        if !current.is_empty() {
            merged.push(current);
        }
        for piece in merged {
            out.extend(split_at_conjunction(&piece, clause_max));
        }
    }
    out
}

/// Sentence boundaries: after `.`, `!`, `?`, `…` followed by whitespace.
fn sentence_bounds(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (i, &(pos, ch)) in chars.iter().enumerate() {
        if matches!(ch, '.' | '!' | '?' | '…') {
            let next = chars.get(i + 1);
            match next {
                Some(&(_, n)) if n.is_whitespace() => {
                    let end = pos + ch.len_utf8();
                    out.push(text[start..end].trim().to_string());
                    start = next.map(|(p, _)| *p).unwrap_or(text.len());
                }
                None if i + 1 == chars.len() => {
                    out.push(text[start..].trim().to_string());
                    start = text.len();
                }
                _ => {}
            }
        }
    }
    if start < text.len() {
        let tail = text[start..].trim();
        if !tail.is_empty() {
            out.push(tail.to_string());
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Clause boundaries: after `,`, `;`, `:` followed by whitespace, or around `—`/`--`.
fn clause_bounds(sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = sentence.as_bytes();
    for (i, ch) in sentence.char_indices() {
        let boundary = match ch {
            ',' | ';' | ':' => bytes
                .get(i + ch.len_utf8())
                .is_some_and(|b| b.is_ascii_whitespace()),
            '—' => true,
            '-' if bytes.get(i + 1) == Some(&b'-') => true,
            _ => false,
        };
        if boundary {
            let end = i + ch.len_utf8();
            let piece = sentence[start..end].trim();
            if !piece.is_empty() {
                out.push(piece.to_string());
            }
            start = end;
        }
    }
    let tail = sentence[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Split before a conjunction near the middle, recursively, while the piece is too long.
fn split_at_conjunction(piece: &str, clause_max: usize) -> Vec<String> {
    if piece.chars().count() <= clause_max {
        return vec![piece.to_string()];
    }
    let mut candidates: Vec<usize> = Vec::new();
    for (i, _) in piece.match_indices(|c: char| c.is_whitespace()) {
        let rest = &piece[i + 1..];
        let word: String = rest.chars().take_while(|c| c.is_alphanumeric()).collect();
        if CONJUNCTIONS.contains(&word.to_ascii_lowercase().as_str())
            && i >= MIN_PIECE
            && piece.len() - i >= MIN_PIECE
        {
            candidates.push(i);
        }
    }
    if candidates.is_empty() {
        return vec![piece.to_string()];
    }
    let middle = piece.len() / 2;
    let cut = *candidates
        .iter()
        .min_by_key(|i| (**i as isize - middle as isize).unsigned_abs())
        .expect("non-empty");
    let mut out = split_at_conjunction(piece[..cut].trim(), clause_max);
    out.extend(split_at_conjunction(piece[cut..].trim(), clause_max));
    out
}

/// Shorten the first utterance at the last clause boundary within `first_max`, or the first boundary
/// after it. Boundary cuts next to a quote mark are skipped, matching the reference.
fn shorten_first(piece: &str, first_max: usize) -> Vec<String> {
    if piece.chars().count() <= first_max {
        return vec![piece.to_string()];
    }
    let quotes: Vec<usize> = piece
        .char_indices()
        .filter(|(_, c)| matches!(c, '"' | '“' | '”'))
        .map(|(i, _)| i)
        .collect();
    let mut cuts: Vec<usize> = Vec::new();
    for (i, c) in piece.char_indices() {
        let is_cut = match c {
            ',' | ';' | ':' => piece[i + c.len_utf8()..].starts_with(char::is_whitespace),
            '—' => true,
            _ => false,
        };
        if !is_cut || i < FIRST_MIN || piece.len() - i < FIRST_MIN {
            continue;
        }
        if quotes.iter().any(|q| (*q as isize - i as isize).abs() < 12) {
            continue;
        }
        cuts.push(i + c.len_utf8());
    }
    if cuts.is_empty() {
        return vec![piece.to_string()];
    }
    let inside: Vec<usize> = cuts.iter().copied().filter(|c| *c <= first_max).collect();
    let cut = inside
        .last()
        .copied()
        .unwrap_or_else(|| *cuts.first().expect("non-empty"));
    vec![
        piece[..cut].trim().to_string(),
        piece[cut..].trim().to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_are_separate_utterances() {
        let pieces = split_text("One. Two! Three?", Options::default());
        assert_eq!(pieces, vec!["One.", "Two!", "Three?"]);
    }

    #[test]
    fn long_sentence_splits_at_clauses() {
        let text = "The committee, which had met every Tuesday since the spring of 2019, finally \
                    agreed that the budget would need a second review before anyone could sign it, \
                    and that the review should be public.";
        let pieces = split_text(text, Options::default());
        assert!(pieces.len() >= 2, "got {pieces:?}");
        assert!(
            pieces.iter().all(|p| p.chars().count() <= CLAUSE_MAX + 8),
            "got {pieces:?}"
        );
    }

    #[test]
    fn first_max_only_shortens_the_first_piece() {
        let text = "A sentence that is long enough to be cut, with a clause boundary in the middle. Second one.";
        let pieces = split_text(
            text,
            Options {
                clause_max: 120,
                first_max: 40,
            },
        );
        assert!(pieces[0].chars().count() <= 45, "got {pieces:?}");
        assert!(pieces.len() >= 2);
    }
}
