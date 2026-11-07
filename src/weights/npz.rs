//! Load the reference's `.npz` directly, so that running the model needs no Python.
//!
//! `woof_int8.npz` is self-describing, which is what makes this possible:
//!
//! * it is a ZIP of `.npy` files, all of them **stored** rather than compressed, so the payload can be
//!   read without a decompressor;
//! * `__tree__` is a single JSON string holding the nested structure — configuration numbers, which
//!   node is a grouped-int8 tensor (`shape`/`axis`/`g` live in the tree, not in arrays), and a
//!   `{"@": "<key>"}` reference wherever an array belongs;
//! * the arrays themselves are int8 (`|i1`), float32 (`<f4`) and float16 (`<f2`); the reference casts
//!   the half-precision ones to f32 on load, and so does this.
//!
//! So the export step (`tools/export.py`) is only about the *shape of the file*: one flat blob plus a
//! small manifest loads in 0.03 s and can be `mmap`ed, where the npz costs a little more to walk. Both
//! paths produce the same weights; `--model` accepts either.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};
use serde_json::Value;

use super::{Q8Meta, Weights};
use crate::Result;

/// One array from the archive.
enum Npy {
    Int8(Array2<i8>),
    Float(ArrayD<f32>),
    /// the UTF-32 payload of `__tree__`
    Text(String),
}

/// Parse one `.npy` file: magic, version, header dict, then the payload in C order.
fn parse_npy(bytes: &[u8], name: &str) -> Result<Npy> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if !bytes.starts_with(MAGIC) || bytes.len() < 10 {
        return Err(format!("{name}: not an .npy (bad magic)").into());
    }
    let (major, header_start) = match bytes[6] {
        1 => (1u8, 10usize),
        2 | 3 => (bytes[6], 12usize),
        other => return Err(format!("{name}: unknown .npy version {other}").into()),
    };
    let len_bytes = if major == 1 {
        &bytes[8..10]
    } else {
        &bytes[8..12]
    };
    let mut header_len = 0usize;
    for (i, byte) in len_bytes.iter().enumerate() {
        header_len |= (*byte as usize) << (8 * i);
    }
    let header_end = header_start + header_len;
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_end)
            .ok_or_else(|| format!("{name}: truncated header"))?,
    )?;
    let descr = field(header, "descr")?;
    let fortran = field(header, "fortran_order")?;
    if fortran != "False" {
        return Err(format!("{name}: Fortran-ordered arrays are not expected here").into());
    }
    let shape_text = field(header, "shape")?;
    let shape: Vec<usize> = shape_text
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let payload = bytes
        .get(header_end..)
        .ok_or_else(|| format!("{name}: truncated payload"))?;
    // `<U72` is a fixed-width UTF-32 string: four bytes per code unit, `width` units per element
    let utf32_width = descr
        .strip_prefix("<U")
        .map(|w| w.parse::<usize>())
        .transpose()?;
    let item_size: usize = if let Some(width) = utf32_width {
        4 * width
    } else {
        match descr.as_str() {
            "|i1" | "<i1" | "|b1" => 1,
            "<f4" => 4,
            "<f2" => 2,
            other => return Err(format!("{name}: unsupported dtype {other:?}").into()),
        }
    };
    let count: usize = shape.iter().product();
    let need = count * item_size;
    if payload.len() < need {
        return Err(format!(
            "{name}: payload is {} bytes, header promises {need}",
            payload.len()
        )
        .into());
    }

    if let Some(width) = utf32_width {
        // UTF-32LE, fixed width; used only for `__tree__`, where one element holds the whole string
        let units = width * count;
        let mut text = String::with_capacity(units);
        for i in 0..units {
            let start = i * 4;
            let word = u32::from_le_bytes(payload[start..start + 4].try_into()?);
            if let Some(c) = char::from_u32(word) {
                if c != '\0' {
                    text.push(c);
                }
            }
        }
        return Ok(Npy::Text(text));
    }

    match item_size {
        1 => {
            let data: Vec<i8> = payload[..need].iter().map(|b| *b as i8).collect();
            match shape.len() {
                2 => Ok(Npy::Int8(Array2::from_shape_vec(
                    (shape[0], shape[1]),
                    data,
                )?)),
                other => Err(format!("{name}: int8 array has {other} dimensions").into()),
            }
        }
        4 => {
            let (words, _) = payload[..need].as_chunks::<4>();
            let data: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
            Ok(Npy::Float(ArrayD::from_shape_vec(shape, data)?))
        }
        2 => {
            let (words, _) = payload[..need].as_chunks::<2>();
            let data: Vec<f32> = words
                .iter()
                .map(|w| f32::from(half::f16::from_bits(u16::from_le_bytes(*w))))
                .collect();
            Ok(Npy::Float(ArrayD::from_shape_vec(shape, data)?))
        }
        _ => unreachable!("item_size is one of the four values above"),
    }
}

/// Pull one key out of the `.npy` header dictionary.
fn field(header: &str, key: &str) -> Result<String> {
    let start = header
        .find(&format!("'{key}'"))
        .ok_or_else(|| format!(".npy header has no {key:?}"))?
        + key.len()
        + 2;
    let rest = &header[start..];
    let rest = rest
        .split_once(':')
        .map(|(_, v)| v.trim_start())
        .ok_or_else(|| format!(".npy header has no value for {key:?}"))?;
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped
            .find('"')
            .ok_or("unterminated string in .npy header")?;
        Ok(stripped[..end].to_string())
    } else if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped
            .find('\'')
            .ok_or("unterminated string in .npy header")?;
        Ok(stripped[..end].to_string())
    } else if let Some(stripped) = rest.strip_prefix('(') {
        // a tuple: ((a, b, ...)), possibly with a trailing comma for one element
        let end = stripped
            .find(')')
            .ok_or("unterminated tuple in .npy header")?;
        Ok(stripped[..end].to_string())
    } else {
        let end = rest
            .find([',', '}'])
            .ok_or("unterminated value in .npy header")?;
        Ok(rest[..end].trim().to_string())
    }
}

fn read_archive(path: &Path) -> Result<HashMap<String, Npy>> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(file)?;
    let mut out = HashMap::with_capacity(zip.len());
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        let Some(name) = entry.name().strip_suffix(".npy").map(str::to_string) else {
            continue;
        };
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes)?;
        out.insert(name.clone(), parse_npy(&bytes, &name)?);
    }
    Ok(out)
}

/// Walk the tree exactly as `tools/export.py` does and fill the same maps the blob loader fills.
pub fn load(path: &Path) -> Result<(Weights, Value)> {
    let arrays = read_archive(path)?;
    let tree: Value = serde_json::from_str(match arrays.get("__tree__") {
        Some(Npy::Text(t)) => t,
        _ => return Err(format!("{}: no __tree__ string", path.display()).into()),
    })?;

    let mut weights = Weights::empty();

    let sp = tree.get("speech").ok_or("__tree__ has no `speech`")?;
    // speech linears are int8 + per-row scale (+ optional bias)
    {
        let mut put_lin = |name: &str, node: &Value| -> Result<()> {
            if let Some((q, s)) = grouped_linear(&arrays, node)? {
                weights.int8s.insert(format!("{name}.q"), q);
                weights.row_scales.insert(name.to_string(), s);
            } else if let Some(w) = node.get("w").and_then(|_| array_f32(&arrays, node, "w")) {
                weights.f32s.insert(format!("{name}.w"), w);
            } else {
                return Err(format!("{name}: neither q/s nor w").into());
            }
            if let Some(b) = node.get("b").and_then(|_| array_f32(&arrays, node, "b")) {
                weights.f32s.insert(format!("{name}.b"), b);
            }
            Ok(())
        };
        for (i, layer) in array_of(sp, "layers")?.iter().enumerate() {
            for (k, node) in layer
                .as_array()
                .ok_or("layer is not a list")?
                .iter()
                .enumerate()
            {
                put_lin(&format!("sp.layers.{i}.{k}"), node)?;
            }
        }
        for nm in ["ce", "ada", "lat_in", "h_inp", "h_out"] {
            if let Some(node) = sp.get(nm) {
                put_lin(&format!("sp.{nm}"), node)?;
            }
        }
        for (j, block) in sp
            .get("hblocks")
            .and_then(Value::as_array)
            .unwrap_or(&vec![])
            .iter()
            .enumerate()
        {
            for (k, node) in block
                .as_array()
                .ok_or("hblock is not a list")?
                .iter()
                .enumerate()
            {
                put_lin(&format!("sp.hblocks.{j}.{k}"), node)?;
            }
        }
        for nm in ["text_emb", "bos"] {
            let v =
                array_f32(&arrays, sp, nm).ok_or_else(|| format!("speech.{nm} is not an array"))?;
            weights.f32s.insert(format!("sp.{nm}"), v);
        }
    }
    for voice in ["juno", "mira"] {
        let v = tree
            .get("voices")
            .and_then(|v| array_f32(&arrays, v, voice))
            .ok_or_else(|| format!("voices.{voice} is not an array"))?;
        weights.f32s.insert(format!("sp.voice.{voice}"), v);
    }
    for nm in ["mean", "std"] {
        let v = array_f32(&arrays, &tree, nm).ok_or_else(|| format!("{nm} is not an array"))?;
        weights.f32s.insert(nm.to_string(), v);
    }

    // decoder: grouped-int8 throughout, with the shape/axis/group metadata inline in the tree
    let dec = tree.get("decoder").ok_or("__tree__ has no `decoder`")?;
    {
        let inp = dec
            .get("inp")
            .and_then(Value::as_array)
            .ok_or("decoder.inp")?;
        let w = value_f32(&arrays, inp.first().ok_or("decoder.inp is empty")?)?;
        let w = w.into_dimensionality::<Ix2>()?;
        // the reference stores it transposed
        weights
            .f32s
            .insert("dec.inp_w".into(), w.t().to_owned().into_dyn());
        weights
            .f32s
            .insert("dec.inp_b".into(), value_f32(&arrays, &inp[1])?);
        weights.f32s.insert(
            "dec.inv_freq".into(),
            value_f32(&arrays, dec.get("inv_freq").ok_or("decoder.inv_freq")?)?,
        );
    }
    let mut seanet = Vec::new();
    for (i, layer) in dec
        .get("tf")
        .and_then(Value::as_array)
        .ok_or("decoder.tf")?
        .iter()
        .enumerate()
    {
        for nm in ["qkv", "o", "fc1", "fc2"] {
            put_q8(
                &mut weights,
                &arrays,
                &format!("dec.tf.{i}.{nm}"),
                &layer[nm],
            )?;
        }
        for ln in ["ln1", "ln2"] {
            let pair = layer.get(ln).and_then(Value::as_array).ok_or("ln pair")?;
            for (j, node) in pair.iter().enumerate() {
                weights
                    .f32s
                    .insert(format!("dec.tf.{i}.{ln}.{j}"), value_f32(&arrays, node)?);
            }
        }
    }
    // the SEANet is a list of [kind, params] pairs, as `unflatten` leaves it
    for (idx, entry) in dec
        .get("seanet")
        .and_then(Value::as_array)
        .ok_or("decoder.seanet")?
        .iter()
        .enumerate()
    {
        let pair = entry
            .as_array()
            .ok_or("seanet entry is not a [kind, params] pair")?;
        let kind = pair
            .first()
            .and_then(Value::as_str)
            .ok_or("seanet entry has no kind")?;
        let params = pair.get(1).cloned().unwrap_or(Value::Null);
        match kind {
            "conv" | "convt" => {
                let shape = put_q8(
                    &mut weights,
                    &arrays,
                    &format!("dec.seanet.{idx}.w"),
                    &params["w"],
                )?;
                put_bias(
                    &mut weights,
                    &format!("dec.seanet.{idx}.b"),
                    &params["b"],
                    &arrays,
                )?;
                let mut rec = serde_json::json!({ "kind": kind });
                if kind == "conv" {
                    rec["cout"] = shape[0].into();
                    rec["cin"] = shape[1].into();
                    rec["k"] = shape[2].into();
                } else {
                    rec["cin"] = shape[0].into();
                    rec["cout"] = shape[1].into();
                    rec["kin"] = shape[2].into();
                    rec["s"] = params["stride"].as_u64().unwrap_or(1).into();
                }
                seanet.push(rec);
            }
            "res" => {
                let subs = params.as_array().ok_or("res.sub")?;
                let mut recs = Vec::new();
                for (sub_i, sub) in subs.iter().enumerate() {
                    let shape = put_q8(
                        &mut weights,
                        &arrays,
                        &format!("dec.seanet.{idx}.{sub_i}.w"),
                        &sub["w"],
                    )?;
                    put_bias(
                        &mut weights,
                        &format!("dec.seanet.{idx}.{sub_i}.b"),
                        &sub["b"],
                        &arrays,
                    )?;
                    recs.push(serde_json::json!({
                        "cout": shape[0], "cin": shape[1], "k": shape[2]
                    }));
                }
                seanet.push(serde_json::json!({ "kind": "res", "sub": recs }));
            }
            other => seanet.push(serde_json::json!({ "kind": other })),
        }
    }

    // configuration, derived from the tree exactly as export.py derives it
    let sp_cfg = sp.get("cfg").ok_or("speech.cfg")?;
    let geti = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0) as usize;
    let dec_cfg = dec.get("cfg").ok_or("decoder.cfg")?;
    let (heads, head_dim) = (geti(dec_cfg, "heads"), geti(dec_cfg, "head_dim"));
    let cfg = serde_json::json!({
        "sr": 24000,
        "sp": {
            "L": sp.get("layers").and_then(Value::as_array).map(|v| v.len()).unwrap_or(0),
            "d": geti(sp_cfg, "d"),
            "H": geti(sp_cfg, "H"),
            "ff": geti(sp_cfg, "ff"),
            "hw": geti(sp_cfg, "hw"),
            "depth": geti(sp_cfg, "depth"),
            "latent": geti(sp_cfg, "latent"),
            "Dh": geti(sp_cfg, "d") / geti(sp_cfg, "H").max(1),
            "max_len": 1100,
        },
        "dec": {
            "nh": heads,
            "hd": head_dim,
            "window": geti(dec_cfg, "window"),
            "eps": dec_cfg.get("eps").and_then(Value::as_f64).unwrap_or(1e-5),
            "chunk": 8,
            "split": 6,
            "D": heads * head_dim,
        },
        "seanet": seanet,
    });

    Ok((weights, cfg))
}

/// Resolve a node that may be an `{"@": key}` reference or a literal (nested) number list.
fn value_f32(arrays: &HashMap<String, Npy>, node: &Value) -> Result<ArrayD<f32>> {
    if let Some(key) = node.get("@").and_then(Value::as_str) {
        return match arrays.get(key) {
            Some(Npy::Float(f)) => Ok(f.clone()),
            Some(_) => Err(format!("{key}: not a float array").into()),
            None => Err(format!("__tree__ references missing array {key:?}").into()),
        };
    }
    match node {
        Value::Array(items) if items.iter().all(|v| v.is_number()) => Ok(Array1::from_iter(
            items.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32),
        )
        .into_dyn()),
        Value::Array(items) => {
            let rows = items.len();
            let cols = items
                .first()
                .and_then(Value::as_array)
                .map(|r| r.len())
                .unwrap_or(0);
            let mut flat = Vec::with_capacity(rows * cols);
            for row in items {
                for v in row.as_array().ok_or("ragged matrix in __tree__")? {
                    flat.push(v.as_f64().unwrap_or(0.0) as f32);
                }
            }
            Ok(Array2::from_shape_vec((rows, cols), flat)?.into_dyn())
        }
        Value::Number(n) => Ok(ndarray::arr0(n.as_f64().unwrap_or(0.0) as f32).into_dyn()),
        other => Err(format!("expected a float array or a reference, found {other}").into()),
    }
}

/// `{"q": …, "s": …, "b": …}` speech node -> (int8 payload, per-row scales).
fn grouped_linear(
    arrays: &HashMap<String, Npy>,
    node: &Value,
) -> Result<Option<(Array2<i8>, Array1<f32>)>> {
    let (Some(q), Some(s)) = (node.get("q"), node.get("s")) else {
        return Ok(None);
    };
    let q = match array_key(arrays, q)? {
        Npy::Int8(m) => m.clone(),
        _ => return Err("speech `q` is not int8".into()),
    };
    let s = match array_key(arrays, s)? {
        Npy::Float(f) => f.clone().into_dimensionality::<Ix1>()?,
        _ => return Err("speech `s` is not float".into()),
    };
    Ok(Some((q, s)))
}

/// A grouped-int8 decoder tensor: payload + scales + metadata, recorded under `name`.
fn put_q8(
    weights: &mut Weights,
    arrays: &HashMap<String, Npy>,
    name: &str,
    node: &Value,
) -> Result<Vec<usize>> {
    let q = match array_key(
        arrays,
        node.get("q8").ok_or_else(|| format!("{name}: no q8"))?,
    )? {
        Npy::Int8(m) => m.clone(),
        _ => return Err(format!("{name}: q8 is not int8").into()),
    };
    let s = match array_key(
        arrays,
        node.get("s").ok_or_else(|| format!("{name}: no s"))?,
    )? {
        Npy::Float(f) => f.clone().into_dimensionality::<Ix2>()?,
        _ => return Err(format!("{name}: s is not float").into()),
    };
    let shape: Vec<usize> = node
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{name}: no shape"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let meta = Q8Meta {
        shape: shape.clone(),
        axis: node.get("axis").and_then(Value::as_u64).unwrap_or(0) as usize,
        group: node.get("g").and_then(Value::as_u64).unwrap_or(32) as usize,
        rows: q.nrows(),
        n: q.ncols(),
        groups: s.ncols(),
    };
    // same convention as the blob loader: the payload lives under "<name>.q", scales and metadata
    // under the bare name
    weights.int8s.insert(format!("{name}.q"), q);
    weights.group_scales.insert(name.to_string(), s);
    weights.meta.insert(name.to_string(), meta);
    Ok(shape)
}

fn put_bias(
    weights: &mut Weights,
    name: &str,
    node: &Value,
    arrays: &HashMap<String, Npy>,
) -> Result<()> {
    weights
        .f32s
        .insert(name.to_string(), value_f32(arrays, node)?);
    Ok(())
}

/// Resolve a `{"@": key}` reference (or a literal list) into a float array.
fn array_f32(arrays: &HashMap<String, Npy>, node: &Value, key: &str) -> Option<ArrayD<f32>> {
    let entry = node.get(key)?;
    let resolved = array_key(arrays, entry).ok()?;
    match resolved {
        Npy::Float(f) => Some(f.clone()),
        _ => None,
    }
}

fn array_key<'a>(arrays: &'a HashMap<String, Npy>, node: &Value) -> Result<&'a Npy> {
    if let Some(key) = node.get("@").and_then(Value::as_str) {
        return arrays
            .get(key)
            .ok_or_else(|| format!("__tree__ references missing array {key:?}").into());
    }
    Err("expected a {\"@\": key} reference".into())
}

fn array_of<'a>(node: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    node.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {key}").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(dtype: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.as_bytes());
        out.extend(payload);
        out
    }

    #[test]
    fn reads_int8_and_f32_arrays() {
        let i8s = npy("|i1", "(2, 3)", &[1u8, 2, 3, 250, 251, 252]);
        match parse_npy(&i8s, "t").unwrap() {
            Npy::Int8(m) => {
                assert_eq!(m.dim(), (2, 3));
                assert_eq!(m[[0, 0]], 1);
                assert_eq!(m[[1, 0]], -6); // 250 as i8
            }
            _ => panic!("expected int8"),
        }
        let f = npy(
            "<f4",
            "(4,)",
            &[
                1.0f32.to_le_bytes(),
                2.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
                0.0f32.to_le_bytes(),
            ]
            .concat(),
        );
        match parse_npy(&f, "t").unwrap() {
            Npy::Float(a) => {
                assert_eq!(a.shape(), &[4]);
                assert_eq!(a[[1]], 2.0);
            }
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn reads_float16_as_float32() {
        // 0.5, 1.0, -2.0, 65504.0 (the largest finite half)
        let halves = [0x3800u16, 0x3c00, 0xc000, 0x7bff];
        let payload: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let bytes = npy("<f2", "(4,)", &payload);
        match parse_npy(&bytes, "t").unwrap() {
            Npy::Float(a) => {
                assert_eq!(a[[0]], 0.5);
                assert_eq!(a[[1]], 1.0);
                assert_eq!(a[[2]], -2.0);
                assert_eq!(a[[3]], 65504.0);
            }
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn reads_the_utf32_tree_string() {
        let text = "hi";
        let mut payload = Vec::new();
        for c in text.chars() {
            payload.extend((c as u32).to_le_bytes());
        }
        let bytes = npy("<U2", "()", &payload);
        match parse_npy(&bytes, "__tree__").unwrap() {
            Npy::Text(t) => assert_eq!(t, "hi"),
            _ => panic!("expected text"),
        }
    }
}
