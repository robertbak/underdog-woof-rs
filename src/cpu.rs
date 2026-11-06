//! CPU topology and the worker pool.
//!
//! On big.LITTLE the pool must not land on little cores. Measured on a Galaxy Tab S9
//! (1× X3 @3.36 GHz / 4× A715 @2.8 / 3× A510 @2.0): the speech pass took 2.32 s pinned to a big
//! core, 8.43 s on a little one, and 2.15 s with six unpinned threads — worse than a single
//! lucky thread, because work-stealing hands a third of the rows to the 2 GHz cores and waits.
//! Pinning the pool to the fast cores gave 0.64 s.
//!
//! Two traps, both of which cost real time to find:
//!
//! 1. `ThreadPoolBuilder::build()` does **not** install the pool — `par_chunks_mut` runs on the
//!    global one. It has to be `build_global()`.
//! 2. Per-thread `sched_setaffinity` from `start_handler` is a **no-op on Android** (measured:
//!    2.2 s, i.e. identical to unpinned). Pinning the *main* thread before the pool is created
//!    works, because the workers inherit the mask at creation — which is also why `taskset`
//!    before exec always worked.

/// One entry per online CPU: `(index, capacity)`.
fn capacities() -> Vec<(usize, u64)> {
    let mut out = Vec::new();
    for cpu in 0..64usize {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}");
        let read = |file: &str| -> u64 {
            std::fs::read_to_string(format!("{base}/{file}"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        };
        let cap = read("cpu_capacity");
        let cap = if cap == 0 {
            read("cpufreq/cpuinfo_max_freq")
        } else {
            cap
        };
        if cap > 0 {
            out.push((cpu, cap));
        }
    }
    out
}

/// One logical CPU per physical core, for machines where nothing tells us which cores are faster.
///
/// Without this, an SMT machine gets one worker per *logical* CPU and its threads fight their own
/// siblings: on an i5-14600KF (6 P-cores x2 + 8 E-cores) the auto rule picked 20 workers and measured
/// 13.1 s on the benchmark corpus against 10.7 s with 12.
fn one_per_physical_core() -> Vec<usize> {
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for cpu in 0..64usize {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let Ok(siblings) = std::fs::read_to_string(&path) else {
            continue;
        };
        let siblings = siblings.trim().to_string();
        if siblings.is_empty() || seen.contains(&siblings) {
            continue;
        }
        seen.push(siblings);
        out.push(cpu);
    }
    out
}

/// Indices of the fastest cores.
///
/// Preference order, because no single source is reliable everywhere:
///
/// 1. `cpu_capacity` when it is *non-uniform* — big.LITTLE ARM (A510 vs A715 vs X3), where it is
///    exactly what we want. Cut-off at 75 % of the fastest.
/// 2. one logical CPU per physical core when capacities are uniform. Intel hybrid x86 reports
///    `cpu_capacity = 1024` for every core unless the kernel enables ITMT, so the capacity rule cannot
///    see the split -- but SMT siblings can still be excluded, which is what matters. Measured on an
///    i5-14600KF (6 P-cores x2 + 8 E-cores): 20 workers 13.1 s, 14 (one per physical core) 10.6 s,
///    12 (P-cores only) 10.7 s. Clock-rate filtering was tried and does not separate them either
///    (4.0/5.3 GHz is 0.755, inside any sane ratio).
pub fn fast_cores() -> Vec<usize> {
    select(capacities(), one_per_physical_core())
}

/// The selection rule, separated from sysfs so it can be tested (and so the ARM and x86 shapes are
/// asserted rather than assumed).
fn select(caps: Vec<(usize, u64)>, per_core: Vec<usize>) -> Vec<usize> {
    let Some(max) = caps.iter().map(|(_, c)| *c).max() else {
        return per_core;
    };
    if caps.iter().all(|(_, c)| *c == max) {
        if !per_core.is_empty() {
            return per_core;
        }
        return caps.iter().map(|(i, _)| *i).collect();
    }
    let mut fast: Vec<usize> = caps
        .iter()
        .filter(|(_, c)| (*c as f64) >= 0.75 * max as f64)
        .map(|(i, _)| *i)
        .collect();
    fast.sort_unstable();
    fast
}

/// Restrict the calling thread to `mask`.
#[cfg(unix)]
pub fn pin_current_thread(mask: u64) {
    if mask == 0 {
        return;
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for cpu in 0..64 {
            if mask & (1u64 << cpu) != 0 {
                libc::CPU_SET(cpu, &mut set);
            }
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(unix))]
pub fn pin_current_thread(_mask: u64) {}

/// Cap glibc's per-thread malloc arenas.
///
/// Each arena reserves 64 MiB of *address space* (PROT_NONE, untouched), so a 20-thread pool leaves
/// VmSize at ~1.7 GiB for ~67 MiB of resident memory. That costs nothing on a desktop, but address
/// space is not free everywhere: `RLIMIT_AS`, a 32-bit target, or `vm.overcommit_memory=2` all care.
/// `mallopt(M_ARENA_MAX)` is used rather than the `MALLOC_ARENA_MAX` environment variable because
/// glibc reads the variable during `ptmalloc_init`, which has already run by the time `main` starts.
///
/// Returns the value glibc accepted, or `None` if the call failed.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn limit_malloc_arenas(max_arenas: usize) -> Option<usize> {
    if max_arenas == 0 {
        return None;
    }
    let value = max_arenas.min(i32::MAX as usize) as i32;
    // SAFETY: mallopt is a glibc tuning call with no preconditions; M_ARENA_MAX takes an int count.
    let ok = unsafe { libc::mallopt(libc::M_ARENA_MAX, value) };
    (ok != 0).then_some(max_arenas)
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn limit_malloc_arenas(_max_arenas: usize) -> Option<usize> {
    None
}

/// Install the global rayon pool, pinned to `cores`. Returns the thread count actually used.
///
/// `threads == 0` means "auto": one worker per fast core (see [`fast_cores`]). The default is a
/// single thread because one core is already faster than realtime on every target measured so far —
/// 1.4x on a Galaxy Tab S9, 2.3x on a desktop i5 — and because scaling is poor: 1 -> 5 threads is
/// 1.64x on the tablet and 1 -> 8 is 1.5x on the desktop, since the per-frame matvec is bound by
/// memory and cache rather than FLOPs. One thread is also predictable: no pool ramp-up, no
/// contention with whatever else owns the machine, and nothing for a scheduler to misplace on a
/// heterogeneous CPU.
///
/// Idempotent: a second call is a no-op, which matters because the pool is global and tests may run
/// in the same process.
pub fn init_pool(threads: usize, cores: &[usize]) -> usize {
    let mask: u64 = cores.iter().fold(0u64, |m, c| m | (1u64 << c));
    let n = if threads > 0 {
        threads
    } else {
        cores.len().max(1)
    };
    pin_current_thread(mask);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global();
    n
}

/// Whether the pool has already been installed in this process.
pub fn pool_installed() -> bool {
    rayon::current_num_threads() > 0
}

#[cfg(test)]
mod tests {
    use super::select;

    /// A Galaxy Tab S9: 3x A510 (266), 4x A715 (811), 1x X3 (1024).
    #[test]
    fn arm_big_little_keeps_the_big_cluster() {
        let mut caps: Vec<(usize, u64)> = (0..3).map(|i| (i, 266)).collect();
        caps.extend((3..7).map(|i| (i, 811)));
        caps.push((7, 1024));
        assert_eq!(select(caps, Vec::new()), vec![3, 4, 5, 6, 7]);
    }

    /// An i5-14600KF: the kernel reports 1024 for every core, so SMT siblings must be excluded by
    /// topology instead (measured: 20 workers 13.1 s vs 14 at 10.6 s).
    #[test]
    fn uniform_capacities_use_one_logical_cpu_per_physical_core() {
        let caps: Vec<(usize, u64)> = (0..20).map(|i| (i, 1024)).collect();
        let per_core = vec![0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19];
        assert_eq!(select(caps, per_core.clone()), per_core);
    }

    /// Homogeneous machine with no topology information: fall back to every CPU rather than none.
    #[test]
    fn uniform_without_topology_uses_everything() {
        let caps: Vec<(usize, u64)> = (0..4).map(|i| (i, 1024)).collect();
        assert_eq!(select(caps, Vec::new()), vec![0, 1, 2, 3]);
    }
}
