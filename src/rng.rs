//! The reference sampler's RNG.
//!
//! `underdog_woof.runtime.TorchRandn` reproduces `torch.manual_seed(seed); torch.randn(n)` with
//! numpy: `MT19937._legacy_seeding(seed)` (i.e. the classic `init_genrand`) followed by a
//! Box-Muller draw per 16 samples. There is no SeedSequence in play, which is why this is a
//! straight port rather than a bit-twiddling exercise.

use ndarray::Array1;
use std::f32::consts::PI;

/// Mersenne Twister, seeded and tempered exactly like `torch.manual_seed`.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let mut rng = Mt19937 { state, index: 624 };
        rng.twist();
        rng.index = 0;
        rng
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = next;
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `TorchRandn.randn(n)`: `n` 24-bit uniforms, Box-Muller in groups of 16.
    ///
    /// # Panics
    /// If `n` is not a multiple of 16 (the caller always asks for the 32-wide latent).
    pub fn randn(&mut self, n: usize) -> Array1<f32> {
        assert_eq!(n % 16, 0, "randn expects a multiple of 16, got {n}");
        let mut u = vec![0f64; n];
        for v in u.iter_mut() {
            *v = f64::from(self.next_u32() & 0x00ff_ffff) * 2.0f64.powi(-24);
        }
        let mut out = vec![0f32; n];
        for block in 0..n / 16 {
            for j in 0..8 {
                let u1 = 1.0 - u[block * 16 + j];
                let u2 = u[block * 16 + 8 + j];
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * PI as f64 * u2;
                out[block * 16 + j] = (r * theta.cos()) as f32;
                out[block * 16 + 8 + j] = (r * theta.sin()) as f32;
            }
        }
        Array1::from(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_stream_matches_numpy_mt19937() {
        // numpy: np.random.MT19937(0); _legacy_seeding(0); random_raw(4)
        let mut rng = Mt19937::new(0);
        let words: Vec<u32> = (0..4).map(|_| rng.next_u32()).collect();
        assert_eq!(
            words,
            vec![0x8c7f_0aac, 0x97c4_aa2f, 0xb716_a675, 0xd821_ccc0]
        );
    }

    #[test]
    fn randn_matches_reference_box_muller() {
        // reference: underdog_woof.runtime.TorchRandn(0).randn(16)[:4]
        let mut rng = Mt19937::new(0);
        let v = rng.randn(16);
        let want = [-1.125_839_8f32, -1.152_360_2, -0.250_578_58, -0.433_878_84];
        for (i, w) in want.iter().enumerate() {
            assert!(
                (v[i] - w).abs() < 1e-6,
                "sample {i}: got {}, want {w}",
                v[i]
            );
        }
    }
}
