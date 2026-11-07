//! The `.npy` half of the reader: magic, header dictionary, payload.
//!
//! A `.npy` file is a two-line header (`{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`)
//! followed by the raw little-endian payload in C order. That is the whole format, so it is parsed by
//! hand rather than with a dependency: the dtypes the reference archive uses are int8, float32,
//! float16 and `<U…` (UTF-32, for `__tree__`).

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use ndarray::{Array2, ArrayD};

use crate::Result;

pub(super) enum Npy {
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

pub(super) fn read_archive(path: &Path) -> Result<HashMap<String, Npy>> {
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
