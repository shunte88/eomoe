//! ggml block formats -> f32 (ports of ggml-quants.c dequantize_row_*), plus a
//! row-parallel matmul over quantized weights for the CPU reference path.

use crate::gguf::GgmlType;
use half::{bf16, f16};
use rayon::prelude::*;

const QK_K: usize = 256;

#[inline]
fn h(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

#[inline]
pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantize one row (`y.len()` elements) of type `ty` from its raw bytes.
pub fn dequant_row(ty: GgmlType, x: &[u8], y: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, c) in y.iter_mut().zip(x.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in y.iter_mut().zip(x.as_chunks::<2>().0) {
                *o = f16::from_le_bytes(*c).to_f32();
            }
        }
        GgmlType::BF16 => {
            for (o, c) in y.iter_mut().zip(x.as_chunks::<2>().0) {
                *o = bf16::from_le_bytes(*c).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            for (yb, b) in y.as_chunks_mut::<32>().0.iter_mut().zip(x.as_chunks::<34>().0) {
                let d = h(b, 0);
                for (o, &q) in yb.iter_mut().zip(&b[2..]) {
                    *o = d * (q as i8) as f32;
                }
            }
        }
        GgmlType::Q4_K => {
            for (yb, b) in y.as_chunks_mut::<QK_K>().0.iter_mut().zip(x.as_chunks::<144>().0) {
                let (d, min) = (h(b, 0), h(b, 2));
                let (scales, qs) = (&b[4..16], &b[16..144]);
                for j in 0..4 {
                    let (sc1, m1) = scale_min_k4(2 * j, scales);
                    let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
                    let (d1, m1, d2, m2) = (d * sc1 as f32, min * m1 as f32, d * sc2 as f32, min * m2 as f32);
                    let q = &qs[32 * j..32 * j + 32];
                    let o = &mut yb[64 * j..64 * j + 64];
                    for l in 0..32 {
                        o[l] = d1 * (q[l] & 0xF) as f32 - m1;
                        o[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                }
            }
        }
        GgmlType::Q5_K => unimplemented!("Q5_K dequant"),
        GgmlType::Q6_K => {
            for (yb, b) in y.as_chunks_mut::<QK_K>().0.iter_mut().zip(x.as_chunks::<210>().0) {
                let d = h(b, 208);
                for n in 0..2 {
                    let ql = &b[64 * n..64 * n + 64];
                    let qh = &b[128 + 32 * n..128 + 32 * n + 32];
                    let sc = &b[192 + 8 * n..192 + 8 * n + 8];
                    let o = &mut yb[128 * n..128 * n + 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 accumulators so LLVM vectorizes without needing fast-math.
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

/// out[t][r] = sum_k W[r][k] * x[t][k] for `n_rows` rows of length `k` starting at `w`
/// (raw bytes of consecutive rows of type `ty`), `x` is T x k, `out` is T x n_rows.
/// Each row is dequantized once and reused for all T inputs.
pub fn matmul(ty: GgmlType, w: &[u8], k: usize, x: &[f32], out: &mut [f32]) {
    let (be, bb) = ty.block();
    let row_bytes = k / be as usize * bb as usize;
    let n_rows = w.len() / row_bytes;
    let t = x.len() / k;
    debug_assert_eq!(out.len(), t * n_rows);
    // Compute row-major into a scratch (n_rows x T) for parallel writes, then transpose.
    let mut tmp = vec![0f32; n_rows * t];
    tmp.par_chunks_mut(t)
        .zip(w.par_chunks_exact(row_bytes))
        .for_each_init(
            || vec![0f32; k],
            |row, (o, wb)| {
                dequant_row(ty, wb, row);
                for (ti, ov) in o.iter_mut().enumerate() {
                    *ov = dot(row, &x[ti * k..(ti + 1) * k]);
                }
            },
        );
    for r in 0..n_rows {
        for ti in 0..t {
            out[ti * n_rows + r] = tmp[r * t + ti];
        }
    }
}
