//! Grouped-int8 decoder tensors: `(rows, n)` int8 with `(rows, groups)` scales over `group`
//! consecutive inputs along the quantised axis, plus the metadata needed to reshape it.

use ndarray::{Array2, Array3, ArrayD, Ix2, Ix3};

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

thread_local! {
    /// One dequantisation buffer per worker thread, grown as needed and never freed until the thread
    /// ends. A matvec runs tens of times per frame, so allocating this per call was most of the
    /// per-frame allocation churn.
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A grouped-int8 decoder tensor: `(rows, n)` int8 with `(rows, groups)` scales over `group`
/// consecutive inputs along the quantised axis.
pub struct GroupedInt8<'a> {
    pub(super) payload: &'a Array2<i8>,
    pub(super) scales: &'a Array2<f32>,
    pub(super) meta: &'a Q8Meta,
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
