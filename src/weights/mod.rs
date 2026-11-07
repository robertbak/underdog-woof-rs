//! Model weights: the flat blob produced by `tools/export.py` plus the two quantisation layouts
//! the model actually uses.
//!
//! The reference (`underdog_woof/runtime.py`) dequantises everything to f32 at load, which is
//! where its ~205 MB of resident weights come from. Here the quantised form stays resident
//! (~39 MB) and is expanded per use:
//!
//! * [`Linear`] — the speech model's `lin_w` layout: int8 rows `(out, in)` with one f32 scale per
//!   *output row*, so `w[o][i] = q[o][i] * s[o]`.
//! * [`GroupedInt8`] — the decoder's `q8` layout: int8 `(rows, n)` with scales `(rows, groups)`
//!   shared by `group` consecutive inputs along one axis, plus the axis to move back into place.
//!   Reproducing `deq()` exactly, `moveaxis` included, is what keeps parity at 72 dB.

use std::collections::HashMap;

use ndarray::{Array1, Array2, Array3, ArrayD, Ix1, Ix2, Ix3};

use crate::nn::{dequantise_row, dot};
use crate::Result;

/// Shape metadata for a grouped-int8 tensor.
#[derive(Clone, Debug)]
pub struct Q8Meta {
    /// target shape *before* the axis move, as stored in the file
    pub shape: Vec<usize>,
    /// axis the tensor was quantised along
    pub axis: usize,
    /// number of consecutive inputs sharing one scale
    pub group: usize,
    pub rows: usize,
    pub n: usize,
    pub groups: usize,
}

/// All tensors, keyed by name.
pub mod fetch;
mod npz;

pub struct Weights {
    f32s: HashMap<String, ArrayD<f32>>,
    int8s: HashMap<String, Array2<i8>>,
    row_scales: HashMap<String, Array1<f32>>,
    group_scales: HashMap<String, Array2<f32>>,
    meta: HashMap<String, Q8Meta>,
}

impl Weights {
    pub(crate) fn empty() -> Self {
        Weights {
            f32s: HashMap::new(),
            int8s: HashMap::new(),
            row_scales: HashMap::new(),
            group_scales: HashMap::new(),
            meta: HashMap::new(),
        }
    }

    /// Load either form: the reference's `.npz` directly (no Python needed), or the exported
    /// `<prefix>.json` + `<prefix>.bin`. Returns the weights and the `cfg` the model runs with.
    pub fn load_any(path: &std::path::Path) -> Result<(Self, serde_json::Value)> {
        let is_npz = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("npz"))
            .unwrap_or(false);
        if is_npz {
            return npz::load(path);
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?;
        Ok((Self::load(path)?, manifest["cfg"].clone()))
    }

    /// Load `<prefix>.json` + `<prefix>.bin` as written by `tools/export.py`.
    pub fn load(prefix: &std::path::Path) -> Result<Self> {
        let manifest_path = prefix.with_extension("json");
        let blob_path = prefix.with_extension("bin");
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let blob = std::fs::read(&blob_path)?;

        let mut weights = Weights::empty();

        let tensors = manifest["tensors"]
            .as_object()
            .ok_or_else(|| format!("{}: missing `tensors`", manifest_path.display()))?;
        for (name, entry) in tensors {
            // grouped-int8 tensors carry metadata only; their payload is "<name>.q"
            let Some(offset) = entry["offset"].as_u64().map(|v| v as usize) else {
                if entry["dtype"].as_str() == Some("q8") {
                    weights.meta.insert(name.clone(), read_q8_meta(entry)?);
                }
                continue;
            };
            let shape = read_shape(entry)?;
            let count: usize = shape.iter().product();
            match entry["dtype"].as_str().unwrap_or("f32") {
                "i8" => {
                    let bytes = blob
                        .get(offset..offset + count)
                        .ok_or_else(|| format!("{name}: blob is too short"))?;
                    let data: Vec<i8> = bytes.iter().map(|b| *b as i8).collect();
                    weights.int8s.insert(
                        name.clone(),
                        Array2::from_shape_vec((shape[0], shape[1]), data)?,
                    );
                }
                "f32" => {
                    let bytes = blob
                        .get(offset..offset + count * 4)
                        .ok_or_else(|| format!("{name}: blob is too short"))?;
                    let (words, _) = bytes.as_chunks::<4>();
                    let data: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
                    weights
                        .f32s
                        .insert(name.clone(), ArrayD::from_shape_vec(shape, data)?);
                }
                other => return Err(format!("{name}: unknown dtype {other:?}").into()),
            }
        }

        // second pass: scales live as "<stem>.s" beside their quantised payload
        let scale_names: Vec<String> = weights
            .f32s
            .keys()
            .filter(|k| weights.has_quantised_payload(k))
            .cloned()
            .collect();
        for name in scale_names {
            let stem = name.trim_end_matches(".s").to_string();
            let array = weights.f32s.remove(&name).expect("filtered above");
            if array.ndim() == 2 {
                weights
                    .group_scales
                    .insert(stem, array.into_dimensionality::<Ix2>()?);
            } else {
                weights
                    .row_scales
                    .insert(stem, array.into_dimensionality::<Ix1>()?);
            }
        }
        Ok(weights)
    }

    fn has_quantised_payload(&self, scale_name: &str) -> bool {
        let stem = scale_name.trim_end_matches(".s");
        self.int8s.contains_key(stem) || self.int8s.contains_key(&format!("{stem}.q"))
    }

    /// Whether a dense tensor or a quantised linear of this name exists.
    pub fn has(&self, name: &str) -> bool {
        self.f32s.contains_key(name) || self.int8s.contains_key(&format!("{name}.q"))
    }

    /// A dense vector, e.g. `sp.bos`, `mean`, `dec.inp_b`.
    pub fn vector(&self, name: &str) -> Result<Array1<f32>> {
        self.dense(name)?
            .clone()
            .into_dimensionality::<Ix1>()
            .map_err(|_| format!("{name}: expected a 1-D tensor").into())
    }

    /// A dense vector if present, `None` otherwise.
    pub fn vector_opt(&self, name: &str) -> Option<Array1<f32>> {
        self.vector(name).ok()
    }

    /// A dense matrix, e.g. `dec.inp_w`, `sp.text_emb`.
    pub fn matrix(&self, name: &str) -> Result<Array2<f32>> {
        self.dense(name)?
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|_| format!("{name}: expected a 2-D tensor").into())
    }

    fn dense(&self, name: &str) -> Result<&ArrayD<f32>> {
        self.f32s
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}").into())
    }

    /// A speech-model linear: `<name>.q` int8 + `<name>.s` per-row scales, or `<name>.w` f32.
    pub fn linear(&self, name: &str) -> Result<Linear> {
        if let Some(q) = self.int8s.get(&format!("{name}.q")) {
            let scales = self
                .row_scales
                .get(name)
                .ok_or_else(|| format!("{name}: missing per-row scales"))?
                .clone();
            Ok(Linear::Int8 {
                q: q.clone(),
                scales,
            })
        } else {
            Ok(Linear::F32(self.matrix(&format!("{name}.w"))?))
        }
    }

    /// A voice's prefix embedding (`sp.voice.<name>`, 16x448 f32 — 28 KiB).
    ///
    /// This is the *entire* per-voice state: everything else in the speech model is shared, which is
    /// why one `Speech` can serve every voice by swapping this.
    pub fn voice_prefix(&self, name: &str) -> Result<Array2<f32>> {
        self.matrix(&format!("sp.voice.{name}"))
    }

    /// Voice names present in the file.
    pub fn voice_names(&self) -> Vec<String> {
        self.f32s
            .keys()
            .filter_map(|k| k.strip_prefix("sp.voice.").map(str::to_string))
            .collect()
    }

    /// A grouped-int8 decoder tensor.
    pub fn grouped(&self, name: &str) -> Result<GroupedInt8<'_>> {
        let payload = self
            .int8s
            .get(&format!("{name}.q"))
            .ok_or_else(|| format!("{name}: missing int8 payload"))?;
        let scales = self
            .group_scales
            .get(name)
            .ok_or_else(|| format!("{name}: missing group scales"))?;
        let meta = self
            .meta
            .get(name)
            .ok_or_else(|| format!("{name}: missing metadata"))?;
        Ok(GroupedInt8 {
            payload,
            scales,
            meta,
        })
    }
}

fn read_shape(entry: &serde_json::Value) -> Result<Vec<usize>> {
    entry["shape"]
        .as_array()
        .ok_or_else(|| "tensor entry without `shape`".into())
        .map(|dims| {
            dims.iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect()
        })
}

fn read_q8_meta(entry: &serde_json::Value) -> Result<Q8Meta> {
    Ok(Q8Meta {
        shape: read_shape(entry)?,
        axis: entry["axis"].as_u64().unwrap_or(0) as usize,
        group: entry["g"].as_u64().unwrap_or(1) as usize,
        rows: entry["rows"].as_u64().unwrap_or(0) as usize,
        n: entry["n"].as_u64().unwrap_or(0) as usize,
        groups: entry["ng"].as_u64().unwrap_or(0) as usize,
    })
}

/// A speech-model linear layer. `w[o][i] = q[o][i] * s[o]`.
#[derive(Clone)]
pub enum Linear {
    F32(Array2<f32>),
    Int8 { q: Array2<i8>, scales: Array1<f32> },
}

impl Linear {
    pub fn rows(&self) -> usize {
        match self {
            Linear::F32(w) => w.nrows(),
            Linear::Int8 { q, .. } => q.nrows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Linear::F32(w) => w.ncols(),
            Linear::Int8 { q, .. } => q.ncols(),
        }
    }

    /// Expand one row into `out` (length `cols()`).
    pub fn expand_row(&self, row: usize, out: &mut [f32]) {
        match self {
            Linear::F32(w) => out.copy_from_slice(w.row(row).as_slice().expect("dense row")),
            Linear::Int8 { q, scales } => {
                dequantise_row(q.row(row).as_slice().expect("dense row"), scales[row], out)
            }
        }
    }

    /// The int8 payload of one output row (the caller only uses this on `Int8`).
    fn row_slice(&self, row: usize) -> &[i8] {
        match self {
            Linear::Int8 { q, .. } => {
                // the payload is loaded contiguous (from_shape_vec), so slice it directly rather than
                // going through a row view whose lifetime ends with the temporary
                let cols = q.ncols();
                let flat = q.as_slice().expect("payload is contiguous");
                &flat[row * cols..(row + 1) * cols]
            }
            Linear::F32(_) => unreachable!("row_slice on f32 weights"),
        }
    }

    /// The per-output-row scale of an int8 row.
    fn row_scale(&self, row: usize) -> f32 {
        match self {
            Linear::Int8 { scales, .. } => scales[row],
            Linear::F32(_) => unreachable!("row_scale on f32 weights"),
        }
    }

    /// Materialise the matrix; the batched path hands this to sgemm, which is worth the copy
    /// because it runs once per sentence.
    pub fn to_dense(&self) -> Array2<f32> {
        match self {
            Linear::F32(w) => w.clone(),
            Linear::Int8 { q, scales } => {
                let mut dense = Array2::<f32>::zeros(q.dim());
                for (row, scale) in scales.iter().enumerate() {
                    let mut dst = dense.row_mut(row);
                    dequantise_row(
                        q.row(row).as_slice().expect("dense row"),
                        *scale,
                        dst.as_slice_mut().expect("dense row"),
                    );
                }
                dense
            }
        }
    }

    /// `W · x`, parallel over output rows (they are independent). Small layers run sequentially:
    /// a rayon task costs more than it saves below ~200k multiply-adds.
    pub fn matvec(&self, x: &Array1<f32>) -> Array1<f32> {
        use rayon::prelude::*;

        if let Linear::F32(w) = self {
            return w.dot(x);
        }
        if crate::nn::w8a8() {
            if let Linear::Int8 { q, scales } = self {
                return int8_matvec(q, scales, x);
            }
        }

        // int8 weights against f32 activations, dequantised a tile at a time into a stack buffer.
        //
        // Three variants were measured on this shape (448x1344, one core):
        //   * dequantise the whole row into a heap scratch, then dot  — 195 us, and the scratch wrote
        //     2.4 MB to L2 per call
        //   * dequantise inside the dot (`f32::from(q[i]) * x[i]`)    — 213 us: the per-element widen
        //     blocks the FMA path the contiguous f32 dot gets
        //   * this: 256-float tiles, so the dequantised block stays in L1 and still feeds vectorised
        //     FMA — see benchmarks/profile.md for the numbers
        const TILE: usize = 256;
        let (rows, cols) = (self.rows(), self.cols());
        let activation = x.as_slice().expect("dense activation");
        let mut out = vec![0f32; rows];
        let simd = crate::nn::simd_dot_available();
        let row_dot = |row: usize| -> f32 {
            let w = self.row_slice(row);
            let scale = self.row_scale(row);
            if simd {
                return crate::nn::dot_i8_scaled(w, activation, scale);
            }
            let mut tile = [0f32; TILE];
            let mut acc = 0f32;
            let mut offset = 0;
            while offset < cols {
                let n = TILE.min(cols - offset);
                let q = &w[offset..offset + n];
                for (slot, value) in tile[..n].iter_mut().zip(q.iter()) {
                    *slot = f32::from(*value) * scale;
                }
                acc += dot(&tile[..n], &activation[offset..offset + n]);
                offset += n;
            }
            acc
        };

        if rows * cols < 200_000 {
            for (row, slot) in out.iter_mut().enumerate() {
                *slot = row_dot(row);
            }
        } else {
            out.par_chunks_mut(32)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        *slot = row_dot(chunk_index * 32 + k);
                    }
                });
        }
        Array1::from(out)
    }

    /// `x · Wᵀ` for a batch (the prefix pass).
    pub fn matmul_t(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.to_dense().t())
    }
}

thread_local! {
    /// One dequantisation buffer per worker thread, grown as needed and never freed until the thread
    /// ends. A matvec runs tens of times per frame, so allocating this per call was most of the
    /// per-frame allocation churn.
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// W8A8: quantise the activation once, then accumulate each row as i32 and scale once at the end.
///
/// `sum(q_w * q_x)` is exact in i32 (the longest reduction here is 3584, the largest product
/// 127*127), so the only error is the two quantisations. The weight row is read as int8 and never
/// expanded to f32, which is where the f32 path loses: it wrote `cols` floats per output row into
/// scratch and read them straight back.
fn int8_matvec(q: &Array2<i8>, scales: &Array1<f32>, x: &Array1<f32>) -> Array1<f32> {
    use rayon::prelude::*;

    let (rows, cols) = q.dim();
    let activation = x.as_slice().expect("dense activation");
    let (qx, a_scale) = crate::nn::quantise_vector(activation);
    let mut out = vec![0f32; rows];

    let row_dot = |row: usize| -> f32 {
        let w = q.row(row);
        let w = w.as_slice().expect("dense row");
        let mut acc = 0i32;
        for (a, b) in w.iter().zip(qx.iter()) {
            acc += i32::from(*a) * i32::from(*b);
        }
        acc as f32 * scales[row] * a_scale
    };

    if rows * cols < 200_000 {
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = row_dot(row);
        }
    } else {
        out.par_chunks_mut(32)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                for (k, slot) in chunk.iter_mut().enumerate() {
                    *slot = row_dot(chunk_index * 32 + k);
                }
            });
    }
    Array1::from(out)
}

/// A grouped-int8 decoder tensor: `(rows, n)` int8 with `(rows, groups)` scales over `group`
/// consecutive inputs along the quantised axis.
pub struct GroupedInt8<'a> {
    payload: &'a Array2<i8>,
    scales: &'a Array2<f32>,
    meta: &'a Q8Meta,
}

impl<'a> GroupedInt8<'a> {
    pub fn meta(&self) -> &Q8Meta {
        self.meta
    }

    /// Reproduce `runtime.deq()`: expand the groups, reshape, then `moveaxis(0, axis)`.
    /// Materialise the tensor as f32.
    ///
    /// Two things matter here, both learned from a profile rather than by reading it:
    ///
    /// * the scale is constant over `group` consecutive values, so the inner loop is per group with
    ///   the scale hoisted, on slices rather than through `[[row, col]]` indexing (which is a bounds
    ///   check and a stride multiply *per element*);
    /// * the `axis != 0` layout used by the convolutional tensors is a plain C-order array with the
    ///   quantised axis moved, so it can be built with one reshape and one axis permutation instead
    ///   of composing an `IxDyn` coordinate vector per element.
    ///
    /// The values produced are identical to the straightforward loop (same product, same order), so
    /// this is a pure speed change.
    pub fn dequantise(&self) -> ArrayD<f32> {
        let m = self.meta;
        let payload = self
            .payload
            .as_slice()
            .expect("payload is loaded contiguous");
        let scales = self
            .scales
            .as_slice()
            .expect("scales are loaded contiguous");
        let mut flat = vec![0f32; m.rows * m.n];
        for row in 0..m.rows {
            let q = &payload[row * m.n..(row + 1) * m.n];
            let sc = &scales[row * m.groups..(row + 1) * m.groups];
            let out = &mut flat[row * m.n..(row + 1) * m.n];
            for (group, scale) in sc.iter().enumerate() {
                let from = group * m.group;
                let to = ((group + 1) * m.group).min(m.n);
                for (o, &v) in out[from..to].iter_mut().zip(&q[from..to]) {
                    *o = f32::from(v) * scale;
                }
            }
        }
        if m.axis == 0 {
            return ArrayD::from_shape_vec(m.shape.clone(), flat).expect("shape validated on load");
        }

        // (rows, rest...) in C order, then move `rows` back to its axis
        let rest: Vec<usize> = m
            .shape
            .iter()
            .enumerate()
            .filter(|(d, _)| *d != m.axis)
            .map(|(_, v)| *v)
            .collect();
        let mut dims = Vec::with_capacity(rest.len() + 1);
        dims.push(m.rows);
        dims.extend(rest.iter().copied());
        let rows_first = ArrayD::from_shape_vec(dims, flat).expect("shape validated on load");
        let mut order: Vec<usize> = (1..=rest.len()).collect();
        order.insert(m.axis.min(rest.len()), 0);
        rows_first.permuted_axes(order)
    }

    pub fn dequantise_3d(&self) -> Array3<f32> {
        self.dequantise()
            .into_dimensionality::<Ix3>()
            .expect("SEANet weights are 3-D")
    }

    /// The decoder transformer's 2-D matrices.
    pub fn dequantise_2d(&self) -> Array2<f32> {
        self.dequantise()
            .into_dimensionality::<Ix2>()
            .expect("transformer weights are 2-D")
    }

    /// Raw quantised payload, for back-ends that expand on device.
    pub fn payload(&self) -> &Array2<i8> {
        self.payload
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward implementation, kept as the specification for the fast one.
    fn dequantise_reference(
        payload: &crate::ndarray_compat::Array2<i8>,
        scales: &crate::ndarray_compat::Array2<f32>,
        m: &Q8Meta,
    ) -> Vec<f32> {
        let mut flat = vec![0f32; m.rows * m.n];
        for row in 0..m.rows {
            for i in 0..m.n {
                flat[row * m.n + i] = f32::from(payload[[row, i]]) * scales[[row, i / m.group]];
            }
        }
        flat
    }

    #[test]
    fn grouped_dequantise_matches_reference_layout() {
        // 2 rows x 6 values, groups of 3, one scale per (row, group)
        let payload =
            Array2::from_shape_vec((2, 6), vec![1i8, -2, 3, 4, -5, 6, -1, 2, -3, -4, 5, -6])
                .unwrap();
        let scales = Array2::from_shape_vec((2, 2), vec![0.5f32, 2.0, 0.25, 1.0]).unwrap();
        let meta = Q8Meta {
            shape: vec![2, 6],
            axis: 0,
            group: 3,
            rows: 2,
            n: 6,
            groups: 2,
        };
        let view = GroupedInt8 {
            payload: &payload,
            scales: &scales,
            meta: &meta,
        };
        let dense = view.dequantise_2d();
        assert_eq!(dense[[0, 0]], 0.5);
        assert_eq!(dense[[0, 3]], 8.0);
        assert_eq!(dense[[1, 0]], -0.25);
        assert_eq!(dense[[1, 5]], -6.0);
    }

    #[test]
    fn w8a8_matvec_tracks_the_f32_one() {
        let (rows, cols) = (64usize, 512usize);
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let q = Array2::from_shape_fn((rows, cols), |_| (next() * 100.0) as i8);
        let scales = Array1::from_shape_fn(rows, |_| 0.01 + 0.01 * next().abs());
        let x = Array1::from_shape_fn(cols, |_| next());
        let exact = Linear::Int8 {
            q: q.clone(),
            scales: scales.clone(),
        }
        .matvec(&x);
        let quantised = int8_matvec(&q, &scales, &x);
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in exact.iter().zip(quantised.iter()) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        let snr = 10.0 * (den / num.max(1e-12)).log10();
        assert!(snr > 40.0, "W8A8 vs f32 matvec: {snr:.1} dB");
    }

    #[test]
    fn fast_dequantise_matches_the_plain_loop() {
        // rows = 5, n = 7, group = 3 -> three groups, for both axes the loader produces
        for axis in [0usize, 1] {
            let shape = if axis == 0 { vec![5, 7] } else { vec![7, 5] };
            let payload = crate::ndarray_compat::Array2::from_shape_fn((5, 7), |(r, c)| {
                ((r * 7 + c) as i32 % 61 - 30) as i8
            });
            let scales = crate::ndarray_compat::Array2::from_shape_fn((5, 3), |(r, g)| {
                0.25 + 0.5 * ((r + g) as f32)
            });
            let meta = Q8Meta {
                shape: shape.clone(),
                axis,
                group: 3,
                rows: 5,
                n: 7,
                groups: 3,
            };
            let expected = dequantise_reference(&payload, &scales, &meta);
            let tensor = GroupedInt8 {
                payload: &payload,
                scales: &scales,
                meta: &meta,
            };
            let got: Vec<f32> = tensor.dequantise().iter().copied().collect();
            if axis == 0 {
                assert_eq!(got, expected, "axis 0 layout");
            } else {
                // rows-first (5, 7) -> (7, 5): element [r, i] lands at [i, r]
                let mut want = vec![0f32; expected.len()];
                for r in 0..5 {
                    for i in 0..7 {
                        want[i * 5 + r] = expected[r * 7 + i];
                    }
                }
                assert_eq!(got, want, "axis 1 layout");
            }
        }
    }
}
