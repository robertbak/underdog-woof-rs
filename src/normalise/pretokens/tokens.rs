//! `norm_tokens`: the first pass over raw text, and the fractions it reads.

use super::*;

/// `norm_tokens`.
pub fn norm_tokens(t: &str) -> String {
    // typographic characters -> ASCII, so the later rules only see one form
    let mut t: String = t
        .chars()
        .map(|c| match c {
            '−' => '-',
            '’' | '‘' => '\'',
            '“' | '”' => '"',
            other => other,
        })
        .collect();

    // vulgar fractions: "1½" -> "1 ½", then to words
    t = sub_with(
        r"(?<=\d)([\u{00BD}\u{00BC}\u{00BE}\u{2153}\u{2154}\u{215B}])",
        &t,
        |c| format!(" {}", re::group(c, 1)),
    );
    t = sub_with(
        r"[\u{00BD}\u{00BC}\u{00BE}\u{2153}\u{2154}\u{215B}]",
        &t,
        |c| {
            match re::group(c, 1) {
                "½" => "1/2",
                "¼" => "1/4",
                "¾" => "3/4",
                "⅓" => "1/3",
                "⅔" => "2/3",
                _ => "1/8",
            }
            .to_string()
        },
    );
    // superscripts: 10² -> 10^2
    t = sub_with(
        r"[\u{2070}\u{00B9}\u{00B2}\u{00B3}\u{2074}-\u{2079}\u{207B}]+",
        &t,
        |c| {
            let mapped: String = re::group(c, 1)
                .chars()
                .map(|ch| match ch {
                    '⁰' => '0',
                    '¹' => '1',
                    '²' => '2',
                    '³' => '3',
                    '⁴' => '4',
                    '⁵' => '5',
                    '⁶' => '6',
                    '⁷' => '7',
                    '⁸' => '8',
                    '⁹' => '9',
                    '⁻' => '-',
                    other => other,
                })
                .collect();
            format!("^{mapped}")
        },
    );

    // tech tokens: .NET, C#, C++, getUserName()
    t = sub_with(r"(?<![\w.])([A-Za-z]*)\.NET\b", &t, |c| {
        let prefix = re::group(c, 1);
        if prefix.is_empty() {
            "dot net".to_string()
        } else {
            format!("{} dot net", spell(prefix))
        }
    });
    t = re::sub(r"\b([A-Za-z])#(?!\w)", r"\1 sharp", &t);
    t = re::sub(r"\b([A-Za-z])\+\+(?!\w)", r"\1 plus plus", &t);
    t = sub_with(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)", &t, |c| {
        split_ident(re::group(c, 1))
    });
    t = sub_with(r"\b[a-z]{2,}(?:[A-Z][a-z]+){2,}\b", &t, |c| {
        split_ident(re::group(c, 0))
    });
    t = sub_with(r"\bv(\d+(?:\.\d+)+)\b", &t, |c| {
        format!(
            "version {}",
            re::group(c, 1)
                .split('.')
                .collect::<Vec<_>>()
                .join(" point ")
        )
    });

    // hashtags and handles
    t = sub_with(r"(?<![\w&#])#([A-Za-z][A-Za-z_]*)\b(?![\d-])", &t, |c| {
        format!("hashtag {}", split_ident(re::group(c, 1)))
    });
    t = sub_with(r"(?<![\w.@])@([A-Za-z_][A-Za-z0-9_]{1,30})\b", &t, |c| {
        format!("at {}", handle(re::group(c, 1)))
    });

    // UK and Canadian postcodes
    t = sub_with(
        r"\b([A-Z]{1,2}\d[A-Z\d]?|[A-Z]\d[A-Z])\s(\d[A-Z]{2}|\d[A-Z]\d)\b",
        &t,
        |c| format!("{}, {}", spell(re::group(c, 1)), spell(re::group(c, 2))),
    );

    // digits read one by one after a signalling word
    t = sub_with(
        r"\b((?:ending|ends) in|last (?:four|4)(?: digits)?(?: of)?|PIN|[Pp]in|passcode|OTP|[Cc]ode)(\s+)(\d{3,8}(?:\s\d{2,4})*)\b(?![.,]\d|-\w)",
        &t,
        |c| {
            let groups: Vec<String> = re::group(c, 3).split(' ').map(digits).collect();
            format!(
                "{}{}{}",
                re::group(c, 1),
                re::group(c, 2),
                groups.join(", ")
            )
        },
    );

    // paths and code
    t = re::sub(r"\b([A-Z]):\\", r"\1 colon backslash ", &t);
    t = t.replace('\\', " backslash ");
    t = re::sub(r"(?<=\s)\*(?=\s)(?!\s*\d)", "star", &t);
    t = re::sub(
        r"(?<![\w.])\.(env|gitignore|bashrc|zshrc|npmrc|htaccess|dockerignore)\b",
        r"dot \1",
        &t,
    );
    t = sub_with(
        r"\b(?=\w*_)(?!\w*\.\w)[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b",
        &t,
        |c| {
            re::group(c, 0)
                .split('_')
                .map(|p| {
                    if p.chars().any(|ch| ch.is_ascii_digit())
                        && p.chars().any(|ch| ch.is_alphabetic())
                    {
                        spell(p)
                    } else {
                        p.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(" underscore ")
        },
    );

    t
}

/// `frac_words`: 3/4 -> three quarters, 1/2 -> a half with `lead`.
///
/// Used by the fraction and date rules of `norm_pre` (stage 3); kept with the tokens rules because
/// the money/fraction wording shares the same helpers.
pub fn frac_words(a: u64, b: u64, lead: bool) -> String {
    let name = if b == 2 {
        if a == 1 {
            "half".to_string()
        } else {
            "halves".to_string()
        }
    } else {
        let base = if b == 4 {
            "quarter".to_string()
        } else {
            ordinal(b)
        };
        if a == 1 {
            base
        } else {
            format!("{base}s")
        }
    };
    let lead_word = if lead && a == 1 {
        "a".to_string()
    } else {
        cardinal(a)
    };
    format!("{lead_word} {name}")
}
