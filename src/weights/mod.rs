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

use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};

use crate::Result;

pub mod fetch;
mod grouped;
mod linear;
mod npz;

pub use grouped::{GroupedInt8, Q8Meta};
pub use linear::Linear;

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
