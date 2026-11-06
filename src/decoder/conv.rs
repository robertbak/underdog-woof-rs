//! SEANet primitives: causal convolution, transposed convolution, and the residual wrapper.
//!
//! All three keep streaming state, because the reference runs the decoder's high-rate half in
//! chunks and the state has to survive the chunk boundary. The layout of the weights is the
//! reference's: a grouped-int8 `(Cout, Cin, K)` tensor for convolutions, `(Cin, Cout, K)` for
//! transposed ones, dequantised only for the duration of a call.

use ndarray::{s, Array1, Array2};

use crate::weights::{GroupedInt8, Weights};
use crate::Result;

/// Causal 1-D convolution with `k` taps and a `(k-1, Cin)` history.
pub struct Conv {
    /// weight tensor name, e.g. `dec.seanet.0.w`
    pub source: String,
    pub bias: Array1<f32>,
    pub taps: usize,
    pub cin: usize,
    pub cout: usize,
    pub state: Array2<f32>,
}

impl Conv {
    pub fn new(source: String, bias: Array1<f32>, cout: usize, cin: usize, taps: usize) -> Self {
        Conv {
            source,
            bias,
            taps,
            cin,
            cout,
            state: Array2::zeros((taps - 1, cin)),
        }
    }

    pub fn reset(&mut self) {
        self.state = Array2::zeros((self.taps - 1, self.cin));
    }

    /// Per-tap matrices `(Cin, Cout)`, i.e. `weight[:, :, j].T` as the reference builds them.
    pub fn taps(&self, weights: &Weights) -> Result<Vec<Array2<f32>>> {
        let full = weights.grouped(&self.source)?.dequantise_3d();
        Ok((0..self.taps)
            .map(|j| full.slice(s![.., .., j]).t().to_owned())
            .collect())
    }

    /// `out[t] = bias + (res) + Σ_j x[t + j - (k-1)] · tap[j]`
    pub fn run(
        &mut self,
        x: &Array2<f32>,
        residual: Option<&Array2<f32>>,
        weights: &Weights,
    ) -> Result<Array2<f32>> {
        let taps = self.taps(weights)?;
        let rows = x.nrows();
        let extended = if self.taps > 1 {
            let mut e = Array2::<f32>::zeros((self.taps - 1 + rows, self.cin));
            e.slice_mut(s![..self.taps - 1, ..]).assign(&self.state);
            e.slice_mut(s![self.taps - 1.., ..]).assign(x);
            self.state = e.slice(s![rows.., ..]).to_owned();
            e
        } else {
            x.clone()
        };

        let mut out = extended.slice(s![0..rows, ..]).dot(&taps[0]);
        out += &self.bias;
        if let Some(r) = residual {
            out += r;
        }
        for (j, tap) in taps.iter().enumerate().skip(1) {
            out += &extended.slice(s![j..j + rows, ..]).dot(tap);
        }
        Ok(out)
    }
}

/// Strided transposed convolution, implemented the reference's way: one matrix multiply into
/// `(rows, k*Cout)`, then an interleave that folds in the bias and the one-frame carry.
pub struct ConvT {
    pub source: String,
    /// tiled bias, length `stride * cout`
    pub bias: Array1<f32>,
    pub stride: usize,
    pub cout: usize,
    pub carry: Array1<f32>,
}

impl ConvT {
    pub fn new(source: String, bias: Array1<f32>, stride: usize, cout: usize) -> Self {
        let mut tiled = Array1::<f32>::zeros(stride * cout);
        for (i, v) in tiled.iter_mut().enumerate() {
            *v = bias[i % cout];
        }
        ConvT {
            source,
            bias: tiled,
            stride,
            cout,
            carry: Array1::zeros(stride * cout),
        }
    }

    pub fn reset(&mut self) {
        self.carry = Array1::zeros(self.bias.len());
    }

    /// `(Cin, k*Cout)` weight matrix, exactly as `_ConvT._weights` reshapes it.
    pub fn matrix(&self, weights: &Weights) -> Result<Array2<f32>> {
        let full = weights.grouped(&self.source)?.dequantise_3d();
        let (cin, cout, k) = full.dim();
        let mut m = Array2::<f32>::zeros((cin, k * cout));
        for i in 0..cin {
            for j in 0..k {
                for o in 0..cout {
                    m[[i, j * cout + o]] = full[[i, o, j]];
                }
            }
        }
        Ok(m)
    }

    /// `x (rows, Cin)` -> `(rows * stride, Cout)`.
    pub fn run(&mut self, x: &Array2<f32>, weights: &Weights) -> Result<Array2<f32>> {
        let matrix = self.matrix(weights)?;
        let rows = x.nrows();
        let stride = self.stride;
        let cout = self.cout;
        let half = stride * cout;
        let z = x.dot(&matrix); // (rows, k*Cout), k == 2*stride

        let mut out = Array2::<f32>::zeros((rows, half));
        for t in 0..rows {
            for j in 0..half {
                out[[t, j]] = z[[t, j]] + self.bias[j];
            }
        }
        for j in 0..half {
            out[[0, j]] += self.carry[j];
        }
        for t in 1..rows {
            for j in 0..half {
                out[[t, j]] += z[[t - 1, half + j]];
            }
        }
        self.carry = z.slice(s![rows - 1, half..]).to_owned();

        let mut interleaved = Array2::<f32>::zeros((rows * stride, cout));
        for t in 0..rows {
            for j in 0..half {
                interleaved[[t * stride + j / cout, j % cout]] = out[[t, j]];
            }
        }
        Ok(interleaved)
    }
}

/// One decoder stage. `Res` mirrors the reference's `op[1](elu(op[0](elu(x))), res=x)`.
pub enum Stage {
    Conv(Conv),
    ConvT(ConvT),
    Res(Conv, Conv),
    Elu,
}

impl Stage {
    pub fn reset(&mut self) {
        match self {
            Stage::Conv(c) => c.reset(),
            Stage::ConvT(c) => c.reset(),
            Stage::Res(a, b) => {
                a.reset();
                b.reset();
            }
            Stage::Elu => {}
        }
    }
}

pub fn build_stage(weights: &Weights, cfg: &serde_json::Value, index: usize) -> Result<Stage> {
    let kind = cfg["kind"].as_str().ok_or("stage without `kind`")?;
    let name = |suffix: &str| format!("dec.seanet.{index}{suffix}");
    Ok(match kind {
        "conv" => Stage::Conv(Conv::new(
            name(".w"),
            weights.vector(&name(".b"))?,
            cfg["cout"].as_u64().unwrap_or(0) as usize,
            cfg["cin"].as_u64().unwrap_or(0) as usize,
            cfg["k"].as_u64().unwrap_or(1) as usize,
        )),
        "convt" => Stage::ConvT(ConvT::new(
            name(".w"),
            weights.vector(&name(".b"))?,
            cfg["s"].as_u64().unwrap_or(1) as usize,
            cfg["cout"].as_u64().unwrap_or(0) as usize,
        )),
        "res" => Stage::Res(
            Conv::new(
                name(".0.w"),
                weights.vector(&name(".0.b"))?,
                cfg["sub"][0]["cout"].as_u64().unwrap_or(0) as usize,
                cfg["sub"][0]["cin"].as_u64().unwrap_or(0) as usize,
                cfg["sub"][0]["k"].as_u64().unwrap_or(1) as usize,
            ),
            Conv::new(
                name(".1.w"),
                weights.vector(&name(".1.b"))?,
                cfg["sub"][1]["cout"].as_u64().unwrap_or(0) as usize,
                cfg["sub"][1]["cin"].as_u64().unwrap_or(0) as usize,
                cfg["sub"][1]["k"].as_u64().unwrap_or(1) as usize,
            ),
        ),
        "elu" => Stage::Elu,
        other => return Err(format!("unknown SEANet stage {other:?}").into()),
    })
}

/// Read-only view of a grouped-int8 tensor, useful for reporting shapes.
pub fn describe(weights: &Weights, name: &str) -> Result<String> {
    let g: GroupedInt8<'_> = weights.grouped(name)?;
    Ok(format!(
        "{:?} group={} axis={}",
        g.meta().shape,
        g.meta().group,
        g.meta().axis
    ))
}
