//! CPU expert FFN for the hybrid MoE path: experts that miss the GPU cache are computed here,
//! straight from the (pinned) host copy, while the GPU runs the resident ones. RAM bandwidth
//! (~40 GB/s) beats uploading the weights over PCIe (~14 GB/s).
//!
//! Activations stay f32 (like the GPU kernels), so results match the GPU to rounding.

use crate::gguf::GgmlType;
use crate::quant::dequant_row;
use rayon::prelude::*;

/// One expert's weights (gate, up, down rows; types) and the rows routed to it.
pub struct ExpertJob<'a> {
    pub w: [&'a [u8]; 3],
    pub ty: [GgmlType; 3],
    /// gpt-oss: f32 biases of gate, up, down (raw bytes).
    pub bias: Option<[&'a [u8]; 3]>,
    /// (row of `h`, routing weight)
    pub rows: Vec<(usize, f32)>,
}

/// Rows of matvec work per rayon task.
const ROWS_PER_TASK: usize = 64;

/// out[r] += sum over jobs routed to row r of weight * down(glu(gate h[r], up h[r])), biases
/// added after each matvec when the job has them (`qwen3moe::glu`, `oai` for gpt-oss).
/// `h` and `out` are rows of `e`; experts have hidden size `ff`. Contributions are summed in
/// job order, so the result does not depend on threading.
pub fn moe(jobs: &[ExpertJob], h: &[f32], e: usize, ff: usize, oai: bool, out: &mut [f32]) {
    pool().install(|| moe_in_pool(jobs, h, e, ff, oai, out))
}

/// f32 `i` of raw little-endian bias bytes, 0 without a bias.
fn bias(b: Option<&[u8]>, i: usize) -> f32 {
    b.map_or(0.0, |b| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// Physical cores only: the work streams weights from RAM, and SMT siblings just contend
/// (bench_moe_vs_bandwidth: 16 threads 27 GB/s, 32 threads 24 GB/s).
fn pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        // ponytail: assumes 2-way SMT; read the topology if a non-SMT host shows up.
        let n = std::thread::available_parallelism().map_or(1, |n| n.get()).div_ceil(2);
        rayon::ThreadPoolBuilder::new().num_threads(n).build().expect("cpu_moe pool")
    })
}

fn moe_in_pool(jobs: &[ExpertJob], h: &[f32], e: usize, ff: usize, oai: bool, out: &mut [f32]) {
    // gate | up | SwiGLU fused per (job, block of ROWS_PER_TASK hidden units): act^T [unit][token].
    let mut act_t: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; j.rows.len() * ff]).collect();
    let tasks: Vec<(usize, usize, &mut [f32])> = act_t
        .iter_mut()
        .enumerate()
        .flat_map(|(ji, a)| {
            let n = jobs[ji].rows.len();
            a.chunks_mut(ROWS_PER_TASK * n).enumerate().map(move |(b, c)| (ji, b * ROWS_PER_TASK, c))
        })
        .collect();
    tasks.into_par_iter().for_each(|(ji, i0, o)| {
        let j = &jobs[ji];
        let n = j.rows.len();
        let (rg, ru) = (j.w[0].len() / ff, j.w[1].len() / ff);
        for (k, ov) in o.chunks_mut(n).enumerate() {
            let i = i0 + k;
            let (gr, ur) = (&j.w[0][i * rg..(i + 1) * rg], &j.w[1][i * ru..(i + 1) * ru]);
            for (t, v) in ov.iter_mut().enumerate() {
                let x = &h[j.rows[t].0 * e..(j.rows[t].0 + 1) * e];
                let g = dot(j.ty[0], gr, x) + bias(j.bias.map(|b| b[0]), i);
                let u = dot(j.ty[1], ur, x) + bias(j.bias.map(|b| b[1]), i);
                *v = crate::qwen3moe::glu(oai, g, u);
            }
        }
    });
    // Down wants each token's activations contiguous: act[t][unit].
    let acts: Vec<Vec<f32>> = jobs
        .iter()
        .zip(&act_t)
        .map(|(j, a)| {
            let n = j.rows.len();
            let mut r = vec![0f32; n * ff];
            for i in 0..ff {
                for t in 0..n {
                    r[t * ff + i] = a[i * n + t];
                }
            }
            r
        })
        .collect();
    // down: per (job, row block), [weight row][token].
    let mut dn: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; j.rows.len() * e]).collect();
    let tasks: Vec<(usize, usize, &mut [f32])> = dn
        .iter_mut()
        .enumerate()
        .flat_map(|(ji, d)| {
            let n = jobs[ji].rows.len();
            d.chunks_mut(ROWS_PER_TASK * n).enumerate().map(move |(b, c)| (ji, b * ROWS_PER_TASK, c))
        })
        .collect();
    tasks.into_par_iter().for_each(|(ji, r0, o)| {
        let j = &jobs[ji];
        let n = j.rows.len();
        let rb = j.w[2].len() / e;
        for (i, ov) in o.chunks_mut(n).enumerate() {
            let row = &j.w[2][(r0 + i) * rb..(r0 + i + 1) * rb];
            for (t, v) in ov.iter_mut().enumerate() {
                *v = dot(j.ty[2], row, &acts[ji][t * ff..(t + 1) * ff]) + bias(j.bias.map(|b| b[2]), r0 + i);
            }
        }
    });
    for (j, d) in jobs.iter().zip(&dn) {
        let n = j.rows.len();
        for (t, &(r, wgt)) in j.rows.iter().enumerate() {
            let o = &mut out[r * e..(r + 1) * e];
            for (i, ov) in o.iter_mut().enumerate() {
                *ov += wgt * d[i * n + t];
            }
        }
    }
}

/// Dot product of one quantized weight row with f32 `x`.
pub fn dot(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        match ty {
            GgmlType::Q4_K => return unsafe { avx2::dot_q4_k(row, x) },
            GgmlType::Q6_K => return unsafe { avx2::dot_q6_k(row, x) },
            _ => {}
        }
    }
    let mut w = vec![0f32; x.len()];
    dequant_row(ty, row, &mut w);
    crate::quant::dot(&w, x)
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[inline]
    fn f16(b: &[u8], i: usize) -> f32 {
        half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// 8 bytes at p, zero-extended to 8 x i32.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn u8x8(p: *const u8) -> __m256i {
        _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_k(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let m4 = _mm256_set1_epi32(0xF);
        for (sb, b) in row.as_chunks::<144>().0.iter().enumerate() {
            let (d, dmin) = (f16(b, 0), f16(b, 2));
            let (sc, qs) = (&b[4..16], &b[16..144]);
            let xb = x.as_ptr().add(sb * 256);
            for j in 0..4 {
                let (s1, m1) = crate::quant::scale_min_k4(2 * j, sc);
                let (s2, m2) = crate::quant::scale_min_k4(2 * j + 1, sc);
                let (mut lo, mut hi) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                let mut xs = _mm256_setzero_ps();
                for g in 0..4 {
                    let v = u8x8(qs.as_ptr().add(32 * j + 8 * g));
                    let (xl, xh) = (_mm256_loadu_ps(xb.add(64 * j + 8 * g)), _mm256_loadu_ps(xb.add(64 * j + 32 + 8 * g)));
                    lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m4)), xl, lo);
                    hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(v, 4)), xh, hi);
                    xs = _mm256_fmadd_ps(_mm256_set1_ps(dmin * m1 as f32), xl, xs);
                    xs = _mm256_fmadd_ps(_mm256_set1_ps(dmin * m2 as f32), xh, xs);
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * s1 as f32), lo, acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d * s2 as f32), hi, acc);
                acc = _mm256_sub_ps(acc, xs);
            }
        }
        hsum(acc)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q6_k(row: &[u8], x: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let (m4, m3, k32) = (_mm256_set1_epi32(0xF), _mm256_set1_epi32(3), _mm256_set1_epi32(32));
        for (sb, b) in row.as_chunks::<210>().0.iter().enumerate() {
            let d = f16(b, 208);
            for n in 0..2 {
                let (ql, qh, sc) = (b.as_ptr().add(64 * n), b.as_ptr().add(128 + 32 * n), &b[192 + 8 * n..200 + 8 * n]);
                let xb = x.as_ptr().add(sb * 256 + 128 * n);
                for l in (0..32).step_by(8) {
                    let is = l / 16;
                    let (a, c, h) = (u8x8(ql.add(l)), u8x8(ql.add(l + 32)), u8x8(qh.add(l)));
                    let q = [
                        _mm256_or_si256(_mm256_and_si256(a, m4), _mm256_slli_epi32(_mm256_and_si256(h, m3), 4)),
                        _mm256_or_si256(_mm256_and_si256(c, m4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 2), m3), 4)),
                        _mm256_or_si256(_mm256_srli_epi32(a, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(h, 4), m3), 4)),
                        _mm256_or_si256(_mm256_srli_epi32(c, 4), _mm256_slli_epi32(_mm256_srli_epi32(h, 6), 4)),
                    ];
                    for (k, qk) in q.iter().enumerate() {
                        let qf = _mm256_cvtepi32_ps(_mm256_sub_epi32(*qk, k32));
                        let p = _mm256_mul_ps(qf, _mm256_loadu_ps(xb.add(l + 32 * k)));
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * (sc[is + 2 * k] as i8) as f32), p, acc);
                    }
                }
            }
        }
        hsum(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes with every f16 scale field set to a sane value.
    fn blocks(ty: GgmlType, n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut b: Vec<u8> = (0..n_blocks * if ty == GgmlType::Q4_K { 144 } else { 210 })
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        let d = half::f16::from_f32(0.0123).to_le_bytes();
        for blk in b.chunks_mut(if ty == GgmlType::Q4_K { 144 } else { 210 }) {
            if ty == GgmlType::Q4_K {
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&half::f16::from_f32(0.0071).to_le_bytes());
            } else {
                blk[208..210].copy_from_slice(&d);
            }
        }
        b
    }

    #[test]
    fn dots_match_dequant_reference() {
        for ty in [GgmlType::Q4_K, GgmlType::Q6_K] {
            let row = blocks(ty, 8, 7);
            let x: Vec<f32> = (0..2048).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let mut w = vec![0f32; 2048];
            dequant_row(ty, &row, &mut w);
            let want: f64 = w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
            let got = dot(ty, &row, &x) as f64;
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "{ty:?}: {got} vs {want}");
        }
    }

    #[test]
    fn moe_matches_reference_ffn() {
        let (e, ff) = (256, 512);
        let gw = blocks(GgmlType::Q4_K, ff, 1);
        let uw = blocks(GgmlType::Q4_K, ff, 2);
        let dw = blocks(GgmlType::Q6_K, e * 2, 3);
        let h: Vec<f32> = (0..3 * e).map(|i| ((i * 13 % 29) as f32 - 14.0) / 14.0).collect();
        let job = ExpertJob { w: [&gw, &uw, &dw], ty: [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K], bias: None, rows: vec![(0, 0.5), (2, 0.25)] };
        let mut out = vec![1f32; 3 * e];
        moe(&[job], &h, e, ff, false, &mut out);
        // Reference: dequantize everything, plain f64 loops.
        let deq = |ty, w: &[u8], rows: usize, k: usize| {
            let mut m = vec![0f32; rows * k];
            for (r, o) in m.chunks_mut(k).enumerate() {
                dequant_row(ty, &w[r * w.len() / rows..(r + 1) * w.len() / rows], o);
            }
            m
        };
        let (g, u, d) = (deq(GgmlType::Q4_K, &gw, ff, e), deq(GgmlType::Q4_K, &uw, ff, e), deq(GgmlType::Q6_K, &dw, e, ff));
        for (r, wgt) in [(0usize, 0.5f64), (1, 0.0), (2, 0.25)] {
            let x = &h[r * e..(r + 1) * e];
            let mv = |m: &[f32], i: usize, k: usize, v: &[f64]| (0..k).map(|c| m[i * k + c] as f64 * v[c]).sum::<f64>();
            let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let a: Vec<f64> = (0..ff).map(|i| { let (gt, ut) = (mv(&g, i, e, &xd), mv(&u, i, e, &xd)); gt / (1.0 + (-gt).exp()) * ut }).collect();
            for i in 0..e {
                let want = 1.0 + wgt * mv(&d, i, ff, &a);
                let got = out[r * e + i] as f64;
                assert!((got - want).abs() <= 1e-3 * want.abs().max(1.0), "row {r} col {i}: {got} vs {want}");
            }
        }
    }

    /// `cargo test --release --lib cpu_moe::tests::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_moe_vs_bandwidth() {
        let (e, ff, n_ex, per) = (2048, 768, 64, 10);
        let ws: Vec<[Vec<u8>; 3]> = (0..n_ex as u64)
            .map(|k| [blocks(GgmlType::Q4_K, ff * e / 256, 3 * k + 1), blocks(GgmlType::Q4_K, ff * e / 256, 3 * k + 2), blocks(GgmlType::Q6_K, e * ff / 256, 3 * k + 3)])
            .collect();
        let h: Vec<f32> = (0..4 * e).map(|i| ((i * 13 % 29) as f32 - 14.0) / 140.0).collect();
        let mut out = vec![0f32; 4 * e];
        for rows in [1usize, 2] {
            let t0 = std::time::Instant::now();
            let iters = 60;
            for it in 0..iters {
                let jobs: Vec<ExpertJob> = (0..per)
                    .map(|j| {
                        let w = &ws[(it * per + j) % n_ex];
                        ExpertJob { w: [&w[0], &w[1], &w[2]], ty: [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K], bias: None, rows: (0..rows).map(|r| ((j + r) % 4, 0.1)).collect() }
                    })
                    .collect();
                moe(&jobs, &h, e, ff, false, &mut out);
            }
            let us = t0.elapsed().as_micros() as f64 / iters as f64;
            let bytes: usize = ws[0].iter().map(|b| b.len()).sum::<usize>() * per;
            println!("moe {per} experts x {rows} rows: {us:.0} us/call, {:.1} GB/s", bytes as f64 / us / 1e3);
        }
        let t0 = std::time::Instant::now();
        let mut tot = 0u64;
        for it in 0..60 {
            tot += (0..per).into_par_iter().map(|j| ws[(it * per + j) % n_ex].iter().map(|b| b.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).fold(0u64, u64::wrapping_add)).fold(0, u64::wrapping_add)).reduce(|| 0, u64::wrapping_add);
        }
        let us = t0.elapsed().as_micros() as f64 / 60.0;
        let bytes: usize = ws[0].iter().map(|b| b.len()).sum::<usize>() * per;
        println!("read {per} experts: {us:.0} us/call, {:.1} GB/s ({tot})", bytes as f64 / us / 1e3);
    }
}
