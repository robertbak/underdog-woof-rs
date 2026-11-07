//! A speech-model linear layer: either dense f32, or int8 with one scale per output row.
//!
//! The int8 form is the reason this crate exists — the reference dequantises every weight to f32 at
//! load (~205 MB resident), while keeping the quantised form is 38.7 MB.

use ndarray::{Array1, Array2};

use crate::nn::{dequantise_row, dot};

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
