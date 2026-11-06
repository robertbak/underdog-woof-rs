//! URLs, hosts, e-mail addresses and file names.
//!
//! One function per stage of `norm_pre`, in the reference's order; `prepass` calls them in
//! sequence. Each is a pure String -> String transform, so order is the only state.

use super::*;

/// Host ports: `localhost:8080` -> "localhost colon eighty eighty".
pub(super) fn host_ports(mut t: String) -> String {
    t = re::sub_with(
        r"\b(localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}):(\d{2,5})\b",
        &t,
        |c| format!("{} colon {}", re::group(c, 1), nread(re::group(c, 2))),
    );
    t
}

/// URL schemes: `https://` -> "h t t p s colon slash slash".
pub(super) fn schemes(mut t: String) -> String {
    let say = |s: &str| -> String {
        lookup(tables::SCHEME_SAY, &s.to_lowercase())
            .map(str::to_string)
            .unwrap_or_else(|| {
                if pronounceable(s) {
                    s.to_lowercase()
                } else {
                    spell(s)
                }
            })
    };
    t = re::sub_with(
        r#"(?<![\w.+-])([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\s,.;:!?)"']|$)"#,
        &t,
        |c| format!("{} colon slash slash", say(re::group(c, 1))),
    );
    t = re::sub_with(
        r"(?<![\w.+-])(?!https?://)([A-Za-z][A-Za-z0-9+.-]{1,9})://(?=[\w/@\[])((?![\w.-]*@)[A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
        &t,
        |c| {
            let rest = re::group(c, 2).replace('.', " dot ").replace('-', " dash ");
            format!("{} colon slash slash {rest}", say(re::group(c, 1)))
        },
    );
    t = re::sub(r"\bmailto:(?=[\w.+-]+@)", "mail to ", &t);
    t
}

/// E-mail addresses and bare domains.
pub(super) fn emails_and_domains(mut t: String) -> String {
    t = re::sub_with(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b", &t, |c| {
        let local = re::group(c, 1)
            .replace('.', " dot ")
            .replace('_', " underscore ")
            .replace('-', " dash ");
        let host = re::group(c, 2).replace('.', " dot ").replace('-', " dash ");
        format!("{local} at {host}")
    });
    t = re::sub_with(
        r"\bhttps?://(?![\w.-]*[:@])([A-Za-z][\w-]*(?:\.[\w-]+)+\b)?",
        &t,
        |c| {
            let host = re::group(c, 1);
            if host.is_empty() {
                return String::new();
            }
            let known = re::is_match(
                r"\.(?:com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz)$",
                host,
            );
            if known {
                return host.to_string();
            }
            let spaced = host.replace('.', " dot ").replace('-', " dash ");
            re::sub(r"^www ", "w w w ", &spaced)
        },
    );
    t = re::sub(r"\bwww\.", "w w w dot ", &t);
    t = re::sub_with(
        r"\b((?:[\w-]+\.)+)(com|org|net|io|ai|co|uk|dev|app|gov|edu|ly|me|tv|us|eu|de|fr|info|biz)\b",
        &t,
        |c| {
            format!(
                "{}{}",
                re::group(c, 1).replace('.', " dot ").replace('-', " dash "),
                re::group(c, 2)
            )
        },
    );
    t
}

/// File names: `report.pdf` -> "report dot p d f".
pub(super) fn file_names(mut t: String) -> String {
    t = re::sub_with(
        r"\b([\w-]+)\.(pdf|png|jpe?g|gif|docx?|xlsx?|csv|txt|zip|mp3|mp4|wav|json|py|js|html?|gradle|ipynb|rtf|pptx?|webm|bin|dat|bat|ps1|vue|jsx|kts|cs|dart|lua|r)\b",
        &t,
        |c| {
            format!(
                "{} dot {}",
                re::group(c, 1).replace('_', " underscore "),
                re::group(c, 2)
            )
        },
    );
    t = re::sub_with(
        r"(?<![\w@])((?:/?[\w-]+/)*[\w-]*[a-z_][\w-]*)\.(conf|log|cfg|ini|yaml|yml|toml|md|sh|ts|tsx|css|xml|sql|db|bak|tmp|exe|dmg|apk|iso|tar|gz|rar|7z|svg|webp|heic|mov|avi|mkv|flac|ogg|m4a|key|pem|crt|env|lock|rs|go|java|kt|swift|rb|php|c|cpp|h|git)\b",
        &t,
        |c| {
            let base = re::group(c, 1)
                .replace('_', " underscore ")
                .replace('/', " slash ")
                .replace('-', " dash ")
                .replace('.', " dot ");
            format!("{base} dot {}", re::group(c, 2))
        },
    );
    t = re::sub_with(r"(?<![\w])/((?:[\w.-]+/)*[\w.-]+)", &t, |c| {
        format!(" slash {}", re::group(c, 1).replace('/', " slash "))
    });
    t
}
