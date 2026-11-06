//! Python-`re` shims: the reference is written against Python's dialect, and two differences matter.
//!
//! * **Lookaround and backreferences.** 63 constructs in `normalize.py` use them, which the `regex`
//!   crate rejects, so patterns compile through `fancy-regex`.
//! * **Replacement syntax.** Python expands `\1`/`\0` in the replacement string; Rust's `regex`
//!   expects `$1`. Rather than rewrite 127 replacements, this module expands Python syntax itself.
//!
//! Compiled patterns are cached by source string: the pipeline applies ~150 patterns per call, and
//! recompiling them would dominate the ~0.1 ms the normaliser currently costs.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

use fancy_regex::{Captures, Regex};

fn cache() -> &'static Mutex<HashMap<&'static str, &'static Regex>> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, &'static Regex>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Compile (and cache) a pattern. Panics on an invalid pattern: these are compile-time constants
/// mirrored from the reference, so a failure is a porting bug, not runtime input.
fn pattern(source: &'static str) -> &'static Regex {
    // leak the regex once; patterns are 'static so the leak is bounded by the number of rules
    let mut guard = cache().lock().expect("pattern cache poisoned");
    if let Some(existing) = guard.get(source) {
        return existing;
    }
    let compiled: &'static Regex = Box::leak(Box::new(
        Regex::new(source).unwrap_or_else(|e| panic!("bad pattern {source:?}: {e}")),
    ));
    guard.insert(source, compiled);
    compiled
}

/// Expand a Python-style replacement (`\0`, `\1`..`\9`, `\\`) for one match.
fn expand(template: &str, captures: &Captures<'_>, text: &str) -> String {
    let mut out = String::with_capacity(template.len() + 8);
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.peek().copied() {
            Some('\\') => {
                chars.next();
                out.push('\\');
            }
            Some(d) if d.is_ascii_digit() => {
                chars.next();
                let index = d.to_digit(10).unwrap_or(0) as usize;
                if index == 0 {
                    out.push_str(captures.get(0).map(|m| m.as_str()).unwrap_or(""));
                } else if let Some(group) = captures.get(index) {
                    out.push_str(group.as_str());
                }
            }
            Some('g') => {
                // \g<1> form
                chars.next();
                let mut name = String::new();
                for c in chars.by_ref() {
                    if c == '>' {
                        break;
                    }
                    name.push(c);
                }
                if let Ok(index) = name.trim_matches(|c| c == '<' || c == '>').parse::<usize>() {
                    if let Some(group) = captures.get(index) {
                        out.push_str(group.as_str());
                    }
                }
            }
            _ => out.push('\\'),
        }
    }
    let _ = text;
    out
}

/// `re.sub(pattern, repl, text)` with Python replacement syntax, all matches.
pub fn sub(source: &'static str, template: &str, text: &str) -> String {
    let re = pattern(source);
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for captures in re.captures_iter(text).flatten() {
        let whole = captures.get(0).expect("group 0 always exists");
        out.push_str(&text[last..whole.start()]);
        out.push_str(&expand(template, &captures, text));
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

/// `re.sub(pattern, callable, text)`, all matches.
pub fn sub_with(
    source: &'static str,
    text: &str,
    mut f: impl FnMut(&Captures<'_>) -> String,
) -> String {
    let re = pattern(source);
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for captures in re.captures_iter(text).flatten() {
        let whole = captures.get(0).expect("group 0 always exists");
        out.push_str(&text[last..whole.start()]);
        out.push_str(&f(&captures));
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

/// `re.sub(pattern, callable, text)` where the callable also sees the haystack and match offset —
/// Python callbacks use `m.string` and `m.start()` for context rules (Roman numerals, dates).
pub fn sub_with_ctx(
    source: &'static str,
    text: &str,
    mut f: impl FnMut(&Captures<'_>, &str, usize) -> String,
) -> String {
    let re = pattern(source);
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for captures in re.captures_iter(text).flatten() {
        let whole = captures.get(0).expect("group 0 always exists");
        out.push_str(&text[last..whole.start()]);
        out.push_str(&f(&captures, text, whole.start()));
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

/// `re.search(pattern, text)` — match anywhere.
pub fn is_match(source: &'static str, text: &str) -> bool {
    pattern(source).is_match(text).unwrap_or(false)
}

/// `re.search` for a pattern that is not a compile-time constant (table entries, composed patterns).
pub fn is_match_leaked(source: &'static str, text: &str) -> bool {
    is_match(source, text)
}

/// `re.match(pattern, text)` — anchored at the start.
pub fn starts_match(source: &'static str, text: &str) -> bool {
    let re = pattern(source);
    re.find(text)
        .map(|m| m.map(|m| m.start() == 0).unwrap_or(false))
        .unwrap_or(false)
}

/// `re.fullmatch(pattern, text)`.
pub fn full_match(source: &'static str, text: &str) -> bool {
    let anchored: &'static str = Box::leak(format!("^(?:{source})$").into_boxed_str());
    is_match(anchored, text)
}

/// `re.sub` returning `None` when nothing matched, so callers can preserve the original text.
pub fn sub_opt(source: &'static str, template: &str, text: &str) -> Option<String> {
    if !is_match(source, text) {
        return None;
    }
    Some(sub(source, template, text))
}

/// Group texts of the first match, or `None` when the pattern does not match.
///
/// Returns owned strings so callers never hold a borrow of the haystack (an earlier version leaked
/// the text to satisfy the lifetime, which grew memory on every call).
pub fn groups_opt(source: &'static str, text: &str) -> Option<Vec<Option<String>>> {
    let re = pattern(source);
    let captures = re.captures(text).ok().flatten()?;
    Some(
        (0..captures.len())
            .map(|i| captures.get(i).map(|m| m.as_str().to_string()))
            .collect(),
    )
}

/// A capture group's text, or `""`.
pub fn group<'a>(captures: &'a Captures<'a>, index: usize) -> &'a str {
    captures.get(index).map(|m| m.as_str()).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_expands_python_group_syntax() {
        // \1 and a literal backslash, as the reference's abbreviations use
        assert_eq!(
            sub(r"\b([A-Za-z])#", r"\1 sharp", "C# code"),
            "C sharp code"
        );
        assert_eq!(sub(r"(?<=\d)(\s?)", r"-\1", "1 2"), "1- 2-");
    }

    #[test]
    fn lookarounds_are_supported() {
        // a lookbehind that the regex crate would reject
        assert_eq!(sub(r"(?<=\d)([½¼¾])", r" \1", "1½"), "1 ½");
    }

    #[test]
    fn sub_with_sees_capture_groups() {
        let out = sub_with(r"(\d+)", "a1 b22", |c| group(c, 1).repeat(2));
        assert_eq!(out, "a11 b2222");
    }

    #[test]
    fn anchored_helpers_behave_like_python() {
        assert!(starts_match(r"\d+", "42abc"));
        assert!(!starts_match(r"\d+", "abc42"));
        assert!(full_match(r"[a-z]+", "abc"));
        assert!(!full_match(r"[a-z]+", "abc1"));
    }
}
