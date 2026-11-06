//! A repeatable benchmark: latency, throughput, and how often the bad-take guard fires.
//!
//! Run it with `woof-rs --bench` (or `--bench-file corpus.txt` for your own lines). It reports, per
//! text, and then in aggregate:
//!
//! * **time to first samples** — the latency a player actually feels, before its own buffer,
//! * **realtime factor** — audio seconds produced per wall second,
//! * **draws per utterance** — the guard's firing rate, bucketed by utterance length, which is the
//!   number that decides whether `--retries N` buys anything on a given kind of text.
//!
//! Everything is measured on one process with the weights loaded once, so the numbers do not
//! include model loading.

use std::time::Instant;

use crate::{audio, textsplit, Mt19937, Result, Utterance, Voice, Weights};

/// Built-in corpus: short commands, medium sentences, long clauses, multi-sentence paragraphs —
/// the shapes whose latency and guard behaviour differ.
pub const CORPUS: &[&str] = &[
    "Yes.",
    "Order shipped.",
    "Please call me back at 5:30 on Friday.",
    "Your order ships on March 3rd, and the tracking code is QX-4471.",
    "The committee, which had met every Tuesday since the spring of 2019, agreed the budget needs review.",
    "The committee, which had met every Tuesday since the spring of 2019, finally agreed that the \
     budget would need a second review before anyone could sign it.",
    "Two sentences here. The second one is a little longer, with a number: 42.",
    "First, check the logs. Then restart the service, wait thirty seconds, and confirm the health \
     endpoint answers. If it still fails, escalate.",
    "Is the backup finished? The last run took four hours, so it may still be writing.",
    "Revenue rose 12% in Q3 — driven by the enterprise tier — while churn held flat at 2.1%.",
];

/// Per-text result.
pub struct Row {
    pub text: String,
    pub chars: usize,
    pub utterances: usize,
    pub first_utterance_chars: usize,
    pub first_samples_s: f32,
    pub audio_s: f32,
    pub wall_s: f32,
    pub draws: usize,
    pub bad_utterances: usize,
}

impl Row {
    pub fn realtime(&self) -> f32 {
        self.audio_s / self.wall_s.max(1e-6)
    }
}

/// Aggregated numbers over the corpus.
pub struct Summary {
    pub rows: Vec<Row>,
    /// draws per utterance, indexed by bucket of utterance length
    pub guard: Vec<GuardBucket>,
}

/// Guard behaviour for a range of utterance lengths.
pub struct GuardBucket {
    pub label: String,
    pub utterances: usize,
    pub draws: usize,
    pub bad: usize,
}

impl GuardBucket {
    pub fn draws_per_utterance(&self) -> f32 {
        self.draws as f32 / self.utterances.max(1) as f32
    }
    pub fn bad_rate(&self) -> f32 {
        self.bad as f32 / self.utterances.max(1) as f32
    }
}

const BUCKETS: &[(&str, usize, usize)] = &[
    ("< 20 chars", 0, 19),
    ("20-49", 20, 49),
    ("50-99", 50, 99),
    ("100+", 100, usize::MAX),
];

/// Run the corpus (or `texts`) through one process.
pub fn run(
    voice: &mut Voice,
    weights: &Weights,
    rng: &mut Mt19937,
    texts: &[String],
    split: textsplit::Options,
) -> Result<Summary> {
    let mut rows = Vec::with_capacity(texts.len());
    let mut buckets: Vec<GuardBucket> = BUCKETS
        .iter()
        .map(|(label, _, _)| GuardBucket {
            label: label.to_string(),
            utterances: 0,
            draws: 0,
            bad: 0,
        })
        .collect();

    for text in texts {
        let started = Instant::now();

        // measure first samples from inside the sink: that is when audio becomes available
        let mut first_seen: Option<f32> = None;
        let mut first_chars = 0usize;
        let mut recorded: Vec<Utterance> = Vec::new();
        {
            let mut sink = |samples: &[f32]| -> Result<()> {
                if first_seen.is_none() && !samples.is_empty() {
                    first_seen = Some(started.elapsed().as_secs_f32());
                    first_chars = recorded.last().map(|u| u.text.chars().count()).unwrap_or(0);
                }
                Ok(())
            };
            recorded = voice.render(text, weights, rng, split, &mut sink)?;
        }

        let mut utterances = 0usize;
        let mut audio_s = 0.0f32;
        let mut draws = 0usize;
        let mut bad = 0usize;
        for utterance in &recorded {
            utterances += 1;
            audio_s += utterance.duration;
            draws += utterance.draws;
            if utterance.bad {
                bad += 1;
            }
            let len = utterance.text.chars().count();
            if let Some(idx) = BUCKETS
                .iter()
                .position(|(_, lo, hi)| len >= *lo && len <= *hi)
            {
                let bucket = &mut buckets[idx];
                bucket.utterances += 1;
                bucket.draws += utterance.draws;
                bucket.bad += usize::from(utterance.bad);
            }
        }
        let first_samples = first_seen.unwrap_or(0.0);
        let first_chars = if first_chars == 0 {
            recorded
                .first()
                .map(|u| u.text.chars().count())
                .unwrap_or(0)
        } else {
            first_chars
        };

        rows.push(Row {
            text: text.clone(),
            chars: text.chars().count(),
            utterances,
            first_utterance_chars: first_chars,
            first_samples_s: first_samples,
            audio_s,
            wall_s: started.elapsed().as_secs_f32(),
            draws,
            bad_utterances: bad,
        });
    }
    Ok(Summary {
        rows,
        guard: buckets,
    })
}

/// Human-readable report.
pub fn report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>5} {:>4} {:>5} {:>9} {:>9} {:>8} {:>6} {:>5}\n",
        "chars", "utts", "1st.ch", "1st.audio", "audio.s", "wall.s", "RTF", "draws"
    ));
    for row in &summary.rows {
        out.push_str(&format!(
            "{:>5} {:>4} {:>5} {:>8.2}s {:>8.2}s {:>7.2}s {:>5.1}x {:>5}\n",
            row.chars,
            row.utterances,
            row.first_utterance_chars,
            row.first_samples_s,
            row.audio_s,
            row.wall_s,
            row.realtime(),
            row.draws
        ));
    }

    let firsts: Vec<f32> = summary.rows.iter().map(|r| r.first_samples_s).collect();
    let audio: f32 = summary.rows.iter().map(|r| r.audio_s).sum();
    let wall: f32 = summary.rows.iter().map(|r| r.wall_s).sum();
    let draws: usize = summary.rows.iter().map(|r| r.draws).sum();
    let utterances: usize = summary.rows.iter().map(|r| r.utterances).sum();
    let bad: usize = summary.rows.iter().map(|r| r.bad_utterances).sum();

    out.push_str(&format!(
        "\ntotal: {} texts, {} utterances, {:.1}s audio in {:.1}s wall ({:.1}x realtime)\n",
        summary.rows.len(),
        utterances,
        audio,
        wall,
        audio / wall.max(1e-6)
    ));
    out.push_str(&format!(
        "first audio: min {:.2}s  median {:.2}s  max {:.2}s\n",
        firsts.iter().cloned().fold(f32::INFINITY, f32::min),
        median(&firsts),
        firsts.iter().cloned().fold(0.0, f32::max)
    ));
    out.push_str(&format!(
        "guard: {:.2} draws/utterance, {}/{} utterances kept a bad take\n",
        draws as f32 / utterances.max(1) as f32,
        bad,
        utterances
    ));
    for bucket in &summary.guard {
        if bucket.utterances > 0 {
            out.push_str(&format!(
                "  {:>10}: {:>3} utterances, {:.2} draws each, {:.0}% bad\n",
                bucket.label,
                bucket.utterances,
                bucket.draws_per_utterance(),
                bucket.bad_rate() * 100.0
            ));
        }
    }
    out
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

/// JSON, for scripts.
pub fn to_json(summary: &Summary) -> serde_json::Value {
    let rows: Vec<serde_json::Value> = summary
        .rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "text": r.text,
                "chars": r.chars,
                "utterances": r.utterances,
                "first_utterance_chars": r.first_utterance_chars,
                "first_samples_s": r.first_samples_s,
                "audio_s": r.audio_s,
                "wall_s": r.wall_s,
                "realtime": r.realtime(),
                "draws": r.draws,
                "bad_utterances": r.bad_utterances,
            })
        })
        .collect();
    let guard: Vec<serde_json::Value> = summary
        .guard
        .iter()
        .map(|b| {
            serde_json::json!({
                "bucket": b.label,
                "utterances": b.utterances,
                "draws_per_utterance": b.draws_per_utterance(),
                "bad_rate": b.bad_rate(),
            })
        })
        .collect();
    serde_json::json!({
        "sample_rate": audio::SAMPLE_RATE,
        "rows": rows,
        "guard": guard,
    })
}
