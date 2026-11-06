//! Matrix multiplication back-ends.
//!
//! The decoder's stages are batched (up to ~5000 rows), which is the only shape where a GPU pays
//! off here: the speech model's per-frame matvecs are ~1 MFLOP each, so thousands of tiny
//! dispatches would be pure launch overhead. Hence a trait with two implementations and no
//! knowledge of the model in either.

use ndarray::Array2;

/// `C = A · B` for dense row-major matrices.
pub trait MatMul: Sync {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    /// Short name for reporting.
    fn name(&self) -> &str;
}

/// `ndarray`'s `matrixmultiply` sgemm — the reference-quality path.
pub struct CpuMatMul;

impl MatMul for CpuMatMul {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(b)
    }

    fn name(&self) -> &str {
        "cpu/ndarray"
    }
}
