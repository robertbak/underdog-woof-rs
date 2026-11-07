//! Opt-in profiling: where a generated frame's time actually goes.
//!
//! Enabled by `WOOF_PROFILE=1`; with it off every span is a load of one relaxed atomic and a branch,
//! so the measurement does not perturb what it measures. Spans are *non-overlapping*: the flow head
//! and the prefix pass are counted whole rather than as their constituent matvecs, so the columns add
//! up to the frame time instead of double counting.
//!
//! An allocation counter rides along (`Counting` in `main.rs`), because the hypothesis this was
//! written to test is that the per-frame cost is dominated by allocations and small temporaries
//! rather than by the matvecs.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// the big weight-times-vector products (qkv, out, ff, latent-in)
    Matvec,
    /// scaled dot-product attention over the frame's keys/values
    Attention,
    /// RMS scaling, RoPE, silu gating and the other elementwise chains
    Elementwise,
    /// the flow head (`Speech.flow`), counted whole
    Flow,
    /// the voice+text prefix pass
    Prefix,
}

const PARTS: [Part; 5] = [
    Part::Matvec,
    Part::Attention,
    Part::Elementwise,
    Part::Flow,
    Part::Prefix,
];

fn index(part: Part) -> usize {
    match part {
        Part::Matvec => 0,
        Part::Attention => 1,
        Part::Elementwise => 2,
        Part::Flow => 3,
        Part::Prefix => 4,
    }
}

static NANOS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static HITS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static FRAMES: AtomicU64 = AtomicU64::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Read the environment once, before any synthesis. Cheap to call more than once.
pub fn init_from_env() {
    if std::env::var_os("WOOF_PROFILE").is_some() {
        ENABLED.store(true, Ordering::Relaxed);
    }
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Count one generation frame, so the report can normalise per frame.
pub fn frame() {
    if enabled() {
        FRAMES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Called from the global allocator; one relaxed load when profiling is off.
#[inline]
pub fn count_alloc(bytes: usize) {
    if ENABLED.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// A timed region. Does nothing (and costs one load) when profiling is off.
pub struct Span(Option<(usize, Instant)>);

#[inline]
pub fn span(part: Part) -> Span {
    if enabled() {
        Span(Some((index(part), Instant::now())))
    } else {
        Span(None)
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        if let Some((slot, started)) = self.0 {
            NANOS[slot].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            HITS[slot].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Print the split, sorted by time. No-op unless `WOOF_PROFILE` was set.
pub fn report() {
    if !enabled() {
        return;
    }
    let frames = FRAMES.load(Ordering::Relaxed).max(1);
    let total: u64 = PARTS
        .iter()
        .map(|p| NANOS[index(*p)].load(Ordering::Relaxed))
        .sum();
    if total == 0 {
        return;
    }
    let mut rows: Vec<(Part, u64, u64)> = PARTS
        .iter()
        .map(|p| {
            (
                *p,
                NANOS[index(*p)].load(Ordering::Relaxed),
                HITS[index(*p)].load(Ordering::Relaxed),
            )
        })
        .collect();
    rows.sort_by_key(|(_, nanos, _)| std::cmp::Reverse(*nanos));

    eprintln!(
        "\nwoof profile: {} frame(s), {:.1} ms accounted per frame",
        frames,
        total as f64 / frames as f64 / 1e6
    );
    eprintln!(
        "  {:<12} {:>10} {:>8} {:>10} {:>12}",
        "part", "total ms", "share", "calls/frm", "µs/call"
    );
    for (part, nanos, hits) in rows {
        if nanos == 0 {
            continue;
        }
        eprintln!(
            "  {:<12} {:>10.1} {:>7.1}% {:>10.1} {:>12.1}",
            format!("{part:?}"),
            nanos as f64 / 1e6,
            100.0 * nanos as f64 / total as f64,
            hits as f64 / frames as f64,
            nanos as f64 / hits.max(1) as f64 / 1e3,
        );
    }
    let allocs = ALLOCS.load(Ordering::Relaxed);
    let bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    eprintln!(
        "  allocations: {} total, {:.1} per frame, {:.1} KiB per frame",
        allocs,
        allocs as f64 / frames as f64,
        bytes as f64 / frames as f64 / 1024.0
    );
}
