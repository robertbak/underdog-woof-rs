//! Audio constants, the character vocabulary, and WAV output.
//!
//! Mirrors `underdog_woof/runtime.py`: `CHARS`/`VOCAB`, `encode_text`, `write_wav`.

/// Sample rate of the model's output (24 kHz mono).
pub const SAMPLE_RATE: u32 = 24000;

/// Vocabulary, exactly as the reference orders it; ids 0..2 are reserved
/// (0 = pad, 1 = BOS, 2 = unknown).
const CHARS: &str =
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()&/%$€£@#+=*_<>[]…—";

/// Tokenise text the way the reference does: `[BOS] + per-character ids`, unknown -> 2.
pub fn encode_text(text: &str) -> Vec<usize> {
    let chars: Vec<char> = CHARS.chars().collect();
    let mut ids = Vec::with_capacity(text.chars().count() + 1);
    ids.push(1);
    for c in text.chars() {
        ids.push(match chars.iter().position(|&x| x == c) {
            Some(i) => i + 3,
            None => 2,
        });
    }
    ids
}

/// A WAV file written as audio arrives: the header is emitted up front with placeholder sizes and
/// patched on [`StreamingWav::finish`], so a player can start reading while synthesis continues.
pub struct StreamingWav {
    file: std::fs::File,
    samples: u32,
}

impl StreamingWav {
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Write as _;
        let mut file = std::fs::File::create(path)?;
        // 0xFFFF_FFFF is the streaming convention for "length not known yet": a player that opens
        // the file mid-write then reads until EOF instead of stopping at a zero-length data chunk.
        file.write_all(&header_with(u32::MAX))?;
        file.flush()?;
        Ok(StreamingWav { file, samples: 0 })
    }

    /// Append samples (clamped and rounded exactly like [`write_wav`]).
    pub fn write(&mut self, samples: &[f32]) -> std::io::Result<()> {
        use std::io::Write as _;
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for v in samples {
            let pcm = (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16;
            buf.extend_from_slice(&pcm.to_le_bytes());
        }
        self.file.write_all(&buf)?;
        self.file.flush()?; // a reader may be following the file
        self.samples += samples.len() as u32;
        Ok(())
    }

    /// Patch the header with the final sizes.
    pub fn finish(mut self) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom, Write as _};
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header(self.samples))?;
        self.file.flush()
    }
}

/// Raw 16-bit mono PCM, for piping into a player.
pub fn pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for v in samples {
        let pcm = (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    out
}

/// 44-byte RIFF/WAVE header for `samples` interleaved mono 16-bit frames.
fn header(samples: u32) -> Vec<u8> {
    header_with(samples.saturating_mul(2))
}

/// Header whose `data` chunk declares `data_bytes` (0xFFFF_FFFF means "streaming, unknown").
fn header_with(data_bytes: u32) -> Vec<u8> {
    let data_len = data_bytes;
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(data_len.wrapping_add(36)).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    h.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    h
}

/// The voices the model ships with, in the reference's order.
pub const VOICES: [&str; 2] = ["juno", "mira"];

/// Voice names as owned strings, for JSON.
pub fn voices() -> Vec<String> {
    VOICES.iter().map(|v| (*v).to_string()).collect()
}

/// A complete WAV file in memory (same bytes as [`write_wav`], for HTTP responses).
pub fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = header(samples.len() as u32);
    out.reserve(samples.len() * 2);
    for v in samples {
        let pcm = (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    out
}

/// Write 16-bit mono PCM. Rounding is `round_ties_even`, matching numpy's `.round()`
/// (and therefore `soundfile`) so the output is bit-comparable with the reference.
pub fn write_wav(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    use std::io::Write as _;
    let data_len = samples.len() * 2;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&((36 + data_len) as u32).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&SAMPLE_RATE.to_le_bytes())?;
    f.write_all(&(SAMPLE_RATE * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&(data_len as u32).to_le_bytes())?;
    for v in samples {
        let pcm = (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16;
        f.write_all(&pcm.to_le_bytes())?;
    }
    f.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_text_matches_reference_layout() {
        // "a" -> BOS, then 'a' at index 1 of CHARS -> id 4
        assert_eq!(encode_text("a"), vec![1, 4]);
        // space is CHARS[0] -> 3; unknown characters collapse to 2
        assert_eq!(encode_text(" \u{4e00}"), vec![1, 3, 2]);
    }
}
