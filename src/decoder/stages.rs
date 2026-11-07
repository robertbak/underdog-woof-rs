//! Driving the SEANet stack: the stages themselves live in `conv`, this runs them and can dump what
//! each one produced, which is how a parity regression gets localised.

use ndarray::Array2;

use super::conv::Stage;
use super::Decoder;
use crate::weights::Weights;
use crate::Result;

pub(super) fn run_stages_dumped(
    mut x: Array2<f32>,
    stages: &mut [Stage],
    weights: &Weights,
    offset: usize,
) -> Result<Array2<f32>> {
    for (i, stage) in stages.iter_mut().enumerate() {
        let kind = match stage {
            Stage::Elu => "elu",
            Stage::Conv(_) => "conv",
            Stage::ConvT(_) => "convt",
            Stage::Res(..) => "res",
        };
        x = run_stages(x, std::slice::from_mut(stage), weights)?;
        Decoder::dump(&format!("{:02}_{}", offset + i, kind), &x);
    }
    Ok(x)
}

/// Apply a run of stages; `Elu` is in-place, everything else replaces the buffer.
pub fn run_stages(
    mut x: Array2<f32>,
    stages: &mut [Stage],
    weights: &Weights,
) -> Result<Array2<f32>> {
    for stage in stages {
        x = match stage {
            Stage::Elu => {
                x.mapv_inplace(crate::nn::elu);
                x
            }
            Stage::Conv(conv) => conv.run(&x, None, weights)?,
            Stage::ConvT(convt) => convt.run(&x, weights)?,
            Stage::Res(a, b) => {
                let mut t = x.mapv(crate::nn::elu);
                t = a.run(&t, None, weights)?;
                t.mapv_inplace(crate::nn::elu);
                b.run(&t, Some(&x), weights)?
            }
        };
    }
    Ok(x)
}
