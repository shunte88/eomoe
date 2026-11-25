// eomoe CUDA kernels. Every kernel is a plain extern "C" __global__ taking only pointers
// and plain values, loaded by name from the fatbin (see src/gpu.rs), so an all-Rust
// (rust-cuda) module exporting the same names can replace this file.
// Numerics mirror the CPU reference (src/cpu.rs, src/quant.rs): f32 activations,
// weights dequantized on the fly from ggml blocks.

#include <stdint.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float ld_f16(const uint8_t* p) {
    return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

// Sum across the block (blockDim.x a multiple of 32, <= 1024); result valid in all threads.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();  // part[] may still be read by a previous call
    if ((threadIdx.x & 31) == 0) part[w] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? part[threadIdx.x] : 0.0f;
    if (w == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) part[0] = v;
    __syncthreads();
    return part[0];
}

// y[t][r] = sum_i W[r][i] * x[t][i] for f32 W. grid (n_rows, T), block multiple of 32.
extern "C" __global__ void mm_f32(const uint8_t* w, unsigned long long row_bytes, int k,
                                  const float* x, float* y, int n_rows) {
    const int r = blockIdx.x, t = blockIdx.y;
    const float* row = (const float*)(w + (size_t)r * row_bytes);
    const float* xt = x + (size_t)t * k;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) acc += row[i] * xt[i];
    acc = block_sum(acc);
    if (threadIdx.x == 0) y[(size_t)t * n_rows + r] = acc;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Quantized matvec: one warp per row, MM_ROWS rows per block; grid (ceil(n_rows/MM_ROWS), T),
// block 32 * MM_ROWS. Each lane decodes 8 elements of every 256-element super-block.
// ponytail: weights re-read per token (grid.y = T); tile tokens for fast prefill.
#define MM_ROWS 4

// Q4_K: 144 bytes: d f16, dmin f16, scales[12] (6-bit packed), qs[128]. Lane l: chunk
// c = l/8 covers elements 64c..64c+63; it takes qs bytes 32c+4s..+3 (s = l%8): low
// nibbles -> elements 64c+4s..+3 (scale 2c), high nibbles -> 64c+32+4s..+3 (scale 2c+1).
extern "C" __global__ void mm_q4_k(const uint8_t* w, unsigned long long row_bytes, int k,
                                   const float* x, float* y, int n_rows) {
    const int r = blockIdx.x * MM_ROWS + (threadIdx.x >> 5), t = blockIdx.y, lane = threadIdx.x & 31;
    if (r >= n_rows) return;
    const uint8_t* row = w + (size_t)r * row_bytes;
    const float* xt = x + (size_t)t * k;
    const int c = lane >> 3, s = lane & 7;
    float acc = 0.0f;
    for (int sb = 0; sb < k / 256; sb++) {
        const uint8_t* b = row + sb * 144;
        const uint8_t* sc = b + 4;
        const int is = 2 * c;  // is and is + 1, both < 8
        int s1, m1, s2, m2;
        if (is < 4) {
            s1 = sc[is] & 63; m1 = sc[is + 4] & 63;
            s2 = sc[is + 1] & 63; m2 = sc[is + 5] & 63;
        } else {
            s1 = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m1 = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
            s2 = (sc[is + 5] & 0xF) | ((sc[is - 3] >> 6) << 4); m2 = (sc[is + 5] >> 4) | ((sc[is + 1] >> 6) << 4);
        }
        const float d = ld_f16(b), dmin = ld_f16(b + 2);
        const float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
        const unsigned q = *(const unsigned*)(b + 16 + 32 * c + 4 * s);
        const float4 xl = *(const float4*)(xt + sb * 256 + 64 * c + 4 * s);
        const float4 xh = *(const float4*)(xt + sb * 256 + 64 * c + 32 + 4 * s);
        acc += (d1 * (q & 0xF) - mm1) * xl.x + (d1 * ((q >> 8) & 0xF) - mm1) * xl.y
             + (d1 * ((q >> 16) & 0xF) - mm1) * xl.z + (d1 * ((q >> 24) & 0xF) - mm1) * xl.w
             + (d2 * ((q >> 4) & 0xF) - mm2) * xh.x + (d2 * ((q >> 12) & 0xF) - mm2) * xh.y
             + (d2 * ((q >> 20) & 0xF) - mm2) * xh.z + (d2 * (q >> 28) - mm2) * xh.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t)t * n_rows + r] = acc;
}

// Q6_K: 210 bytes: ql[128], qh[64], scales[16] i8, d f16 at 208. Lane l takes position l
// of each of the 4 quadrants (32 elements) in both 128-element halves.
extern "C" __global__ void mm_q6_k(const uint8_t* w, unsigned long long row_bytes, int k,
                                   const float* x, float* y, int n_rows) {
    const int r = blockIdx.x * MM_ROWS + (threadIdx.x >> 5), t = blockIdx.y, l = threadIdx.x & 31;
    if (r >= n_rows) return;
    const uint8_t* row = w + (size_t)r * row_bytes;
    const float* xt = x + (size_t)t * k;
    float acc = 0.0f;
    for (int sb = 0; sb < k / 256; sb++) {
        const uint8_t* b = row + sb * 210;
        const float d = ld_f16(b + 208);
        for (int n = 0; n < 2; n++) {
            const uint8_t* ql = b + 64 * n;
            const int h = b[128 + 32 * n + l];
            const int8_t* sc = (const int8_t*)(b + 192 + 8 * n) + (l >> 4);
            const float* xv = xt + sb * 256 + 128 * n + l;
            const int a = ql[l], c = ql[l + 32];
            acc += d * sc[0] * (float)(((a & 0xF) | ((h & 3) << 4)) - 32) * xv[0]
                 + d * sc[2] * (float)(((c & 0xF) | (((h >> 2) & 3) << 4)) - 32) * xv[32]
                 + d * sc[4] * (float)(((a >> 4) | (((h >> 4) & 3) << 4)) - 32) * xv[64]
                 + d * sc[6] * (float)(((c >> 4) | (((h >> 6) & 3) << 4)) - 32) * xv[96];
        }
    }
    acc = warp_sum(acc);
    if (l == 0) y[(size_t)t * n_rows + r] = acc;
}

// Row-wise RMSNorm, y may alias x. grid rows.
extern "C" __global__ void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    const float* xr = x + (size_t)blockIdx.x * n;
    float* yr = y + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float s = 1.0f / sqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * s * w[i];
}

// Per-head RMSNorm (q_norm/k_norm) then NEOX rope in place. v is T x heads x hd,
// grid T * heads, block hd. Pairs (i, i + hd/2), theta_i = base^(-2i/hd), pos = pos0 + t.
extern "C" __global__ void qk_norm_rope(float* v, const float* w, int heads, int hd, int pos0,
                                        float eps, float base) {
    __shared__ float s[1024];
    float* hv = v + (size_t)blockIdx.x * hd;
    const int i = threadIdx.x;
    const float a = hv[i];
    const float sc = 1.0f / sqrtf(block_sum(a * a) / hd + eps);
    s[i] = a * sc * w[i];
    __syncthreads();
    const int half = hd / 2;
    const float pos = (float)(pos0 + blockIdx.x / heads);
    const int j = i < half ? i : i - half;
    const float th = pos * powf(base, -2.0f * j / hd);
    const float sn = sinf(th), cs = cosf(th);
    hv[i] = i < half ? s[i] * cs - s[i + half] * sn : s[i - half] * sn + s[i] * cs;
}

// Causal GQA attention with online softmax. q: T x nh x hd, caches: n_ctx x nkv x hd,
// out: T x nh x hd. grid (nh, T), block hd. Token t sees positions [0, n_past + t].
// ponytail: one position per step with a block reduction; tile K/V (flash-style) for long contexts.
extern "C" __global__ void attention(const float* q, const float* kc, const float* vc, float* out,
                                     int nh, int nkv, int hd, int n_past, float scale) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int hk = h / (nh / nkv);
    const float qi = q[((size_t)t * nh + h) * hd + i];
    const int n_ctx = n_past + t + 1;
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int p = 0; p < n_ctx; p++) {
        const size_t off = ((size_t)p * nkv + hk) * hd + i;
        const float s = block_sum(qi * kc[off]) * scale;
        const float mn = fmaxf(m, s);
        const float c = expf(m - mn), e = expf(s - mn);
        l = l * c + e;
        acc = acc * c + e * vc[off];
        m = mn;
    }
    out[((size_t)t * nh + h) * hd + i] = acc / l;
}

extern "C" __global__ void add_inplace(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// dst[j] = src[idx[j]] for rows of length n. grid rows.
extern "C" __global__ void gather_rows(float* dst, const float* src, const int* idx, int n) {
    const float* s = src + (size_t)idx[blockIdx.x] * n;
    float* d = dst + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

// g = silu(g) * u
extern "C" __global__ void swiglu(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

// x[t] += sum_k w[t][k] * d[rows[t][k]] (k ascending by expert id, as the CPU reference).
// grid T.
extern "C" __global__ void moe_combine(float* x, const float* d, const int* rows, const float* w,
                                       int k_used, int n) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < k_used; k++)
            acc += w[t * k_used + k] * d[(size_t)rows[t * k_used + k] * n + i];
        x[(size_t)t * n + i] += acc;
    }
}
