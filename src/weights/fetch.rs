//! Where the weights come from when nobody said.
//!
//! They are not in this repository and not in the release archives: they belong to Underdog and are
//! published by them. So `--model` is optional, and this fetches the reference's own `woof_int8.npz`
//! from upstream's Hugging Face repository, caches it under the user's cache directory, and verifies
//! the SHA-256 before the loader ever sees it.
//!
//! Deliberately not an HTTP client: pulling in rustls to fetch one file would cost more binary than
//! the whole synthesiser. `curl` or `wget` does it, and saying so is more honest than a hidden network
//! dependency. `--offline` turns the whole path off, and `--model` skips it entirely.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::Result;

/// Upstream's weights, as published.
pub const URL: &str =
    "https://huggingface.co/ConwayResearch/Underdog-Woof/resolve/main/woof_int8.npz";
/// Pinned so a changed file upstream is reported rather than silently adopted.
pub const SHA256: &str = "d895fbbe84b8f4f461e3e94ab48b1164353b1988ce8d365f5aa8263d0de1203e";
pub const BYTES: u64 = 40_595_426;

/// `$XDG_CACHE_HOME/woof-rs`, else `~/.cache/woof-rs`, else the temp directory.
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("woof-rs")
}

/// Default weights file inside the cache.
pub fn cached_path() -> PathBuf {
    cache_dir().join("woof_int8.npz")
}

fn size_of(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

/// SHA-256 of a file, read in chunks.
pub fn digest(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// A weights path to run with: the cached download, fetched first if it is not there.
///
/// Refuses on a hash mismatch instead of silently accepting whatever upstream served: a changed file
/// means the model changed, and that should be a decision, not a surprise.
pub fn ensure_npz(offline: bool) -> Result<PathBuf> {
    let path = cached_path();
    if size_of(&path).unwrap_or(0) == BYTES {
        let seen = digest(&path)?;
        if seen == SHA256 {
            return Ok(path);
        }
        if offline {
            return Err(format!(
                "{}: cached weights do not match the expected SHA-256\n  expected {SHA256}\n  found    {seen}\n\
                 delete it, or pass --model explicitly",
                path.display()
            )
            .into());
        }
        eprintln!("woof: cached weights do not match the expected hash; downloading again");
    } else if offline {
        return Err(format!(
            "no weights: {}\n\
             --offline was given, so nothing was downloaded. Fetch them yourself:\n\
             \n  curl -L -o {} {URL}\n\n\
             or pass --model /path/to/woof_int8.npz",
            path.display(),
            path.display()
        )
        .into());
    }

    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let part = dir.join("woof_int8.npz.part");
    let _ = std::fs::remove_file(&part);

    eprintln!(
        "woof: no weights yet; downloading {:.1} MB from upstream\n  {URL}",
        BYTES as f64 / (1024.0 * 1024.0)
    );
    let target = part
        .to_str()
        .ok_or("cache path is not valid UTF-8")?
        .to_string();
    let (program, args) = if which("curl") {
        (
            "curl",
            vec!["-L", "--fail", "--progress-bar", "-o", &target, URL],
        )
    } else if which("wget") {
        ("wget", vec!["-q", "--show-progress", "-O", &target, URL])
    } else {
        return Err(format!(
            "no weights and neither curl nor wget is available.\n  \
             fetch {URL}\n  and pass it as --model, or put it at {}",
            cached_path().display()
        )
        .into());
    };
    let status = std::process::Command::new(program)
        .args(&args)
        .status()
        .map_err(|e| format!("could not run {program}: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{program} failed ({status}); nothing was installed").into());
    }

    let size = size_of(&part)?;
    if size != BYTES {
        let _ = std::fs::remove_file(&part);
        return Err(
            format!("download is {size} bytes, expected {BYTES}; nothing was installed").into(),
        );
    }
    let seen = digest(&part)?;
    if seen != SHA256 {
        let _ = std::fs::remove_file(&part);
        return Err(format!(
            "downloaded weights do not match the pinned SHA-256\n  expected {SHA256}\n  found    {seen}\n\
             upstream may have replaced the file; pass --model explicitly if you trust it"
        )
        .into());
    }
    std::fs::rename(&part, &path)?;
    eprintln!("woof: weights verified and cached at {}", path.display());
    Ok(path)
}

fn which(program: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(program);
                std::fs::metadata(&candidate)
                    .map(|m| m.is_file())
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_matches_a_known_value() {
        // "abc" is the canonical SHA-256 test vector
        let dir = std::env::temp_dir().join("woof-rs-hash-test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abc.txt");
        std::fs::write(&file, b"abc").unwrap();
        assert_eq!(
            digest(&file).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn cache_dir_is_absolute_and_named() {
        let dir = cache_dir();
        assert!(dir.is_absolute(), "{}", dir.display());
        assert_eq!(dir.file_name().and_then(|n| n.to_str()), Some("woof-rs"));
    }
}
