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
#define MM_ROWS 4

// Each dot computes TT tokens at once (x[j] = xt + j*k, j < n): the weights of a super-block
// are decoded once and reused across the tile. Per token the arithmetic is identical for
// every TT, so tiled and untiled launches give bit-identical results.

// Q4_K: 144 bytes: d f16, dmin f16, scales[12] (6-bit packed), qs[128]. Lane l: chunk
// c = l/8 covers elements 64c..64c+63; it takes qs bytes 32c+4s..+3 (s = l%8): low
// nibbles -> elements 64c+4s..+3 (scale 2c), high nibbles -> 64c+32+4s..+3 (scale 2c+1).
template <int TT>
__device__ __forceinline__ void dot_q4_k(const uint8_t* row, const float* xt, int k, int lane, int n, float* out) {
    const int c = lane >> 3, s = lane & 7;
    float acc[TT];
#pragma unroll
    for (int j = 0; j < TT; j++) acc[j] = 0.0f;
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
        const float w0 = d1 * (q & 0xF) - mm1, w1 = d1 * ((q >> 8) & 0xF) - mm1;
        const float w2 = d1 * ((q >> 16) & 0xF) - mm1, w3 = d1 * ((q >> 24) & 0xF) - mm1;
        const float w4 = d2 * ((q >> 4) & 0xF) - mm2, w5 = d2 * ((q >> 12) & 0xF) - mm2;
        const float w6 = d2 * ((q >> 20) & 0xF) - mm2, w7 = d2 * (q >> 28) - mm2;
#pragma unroll
        for (int j = 0; j < TT; j++) {
            if (j < n) {
                const float* xj = xt + (size_t)j * k + sb * 256 + 64 * c + 4 * s;
                const float4 xl = *(const float4*)xj, xh = *(const float4*)(xj + 32);
                acc[j] += w0 * xl.x + w1 * xl.y + w2 * xl.z + w3 * xl.w + w4 * xh.x + w5 * xh.y + w6 * xh.z + w7 * xh.w;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TT; j++) out[j] = warp_sum(acc[j]);
}

// Q6_K: 210 bytes: ql[128], qh[64], scales[16] i8, d f16 at 208. Lane l takes position l
// of each of the 4 quadrants (32 elements) in both 128-element halves.
template <int TT>
__device__ __forceinline__ void dot_q6_k(const uint8_t* row, const float* xt, int k, int l, int n, float* out) {
    float acc[TT];
#pragma unroll
    for (int j = 0; j < TT; j++) acc[j] = 0.0f;
    for (int sb = 0; sb < k / 256; sb++) {
        const uint8_t* b = row + sb * 210;
        const float d = ld_f16(b + 208);
        for (int h2 = 0; h2 < 2; h2++) {
            const uint8_t* ql = b + 64 * h2;
            const int h = b[128 + 32 * h2 + l];
            const int8_t* sc = (const int8_t*)(b + 192 + 8 * h2) + (l >> 4);
            const int a = ql[l], c = ql[l + 32];
            const float w0 = d * sc[0] * (float)(((a & 0xF) | ((h & 3) << 4)) - 32);
            const float w1 = d * sc[2] * (float)(((c & 0xF) | (((h >> 2) & 3) << 4)) - 32);
            const float w2 = d * sc[4] * (float)(((a >> 4) | (((h >> 4) & 3) << 4)) - 32);
            const float w3 = d * sc[6] * (float)(((c >> 4) | (((h >> 6) & 3) << 4)) - 32);
#pragma unroll
            for (int j = 0; j < TT; j++) {
                if (j < n) {
                    const float* xv = xt + (size_t)j * k + sb * 256 + 128 * h2 + l;
                    acc[j] += w0 * xv[0] + w1 * xv[32] + w2 * xv[64] + w3 * xv[96];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TT; j++) out[j] = warp_sum(acc[j]);
}

// mm_X: y[t][r] = W[r] . x[t], one weight matrix. grid (ceil(n_rows / MM_ROWS), ceil(T / TT)).
// mm_X_ids: the weights of token t start at wp[t] + off (grouped experts: one launch per
// projection for all the experts of a layer, each gathered row reading its own expert).
// A tile whose tokens straddle two experts falls back to one token at a time.
// _t8 variants tile 8 tokens (prefill); the plain ones take one token per grid row (decode).
template <int TT, void (*DOT)(const uint8_t*, const float*, int, int, int, float*)>
__device__ __forceinline__ void mm_tile(const uint8_t* w, unsigned long long row_bytes, int k,
                                        const float* x, float* y, int n_rows, int T) {
    const int r = blockIdx.x * MM_ROWS + (threadIdx.x >> 5), t0 = blockIdx.y * TT;
    if (r >= n_rows) return;
    float acc[TT];
    const int n = min(TT, T - t0);
    DOT(w + (size_t)r * row_bytes, x + (size_t)t0 * k, k, threadIdx.x & 31, n, acc);
    if ((threadIdx.x & 31) == 0)
        for (int j = 0; j < n; j++) y[(size_t)(t0 + j) * n_rows + r] = acc[j];
}
template <int TT, void (*DOT1)(const uint8_t*, const float*, int, int, int, float*),
          void (*DOT)(const uint8_t*, const float*, int, int, int, float*)>
__device__ __forceinline__ void mm_tile_ids(const unsigned long long* wp, unsigned long long off,
                                            unsigned long long row_bytes, int k, const float* x,
                                            float* y, int n_rows, int T) {
    const int r = blockIdx.x * MM_ROWS + (threadIdx.x >> 5), t0 = blockIdx.y * TT;
    if (r >= n_rows) return;
    const int n = min(TT, T - t0);
    bool same = true;
    for (int j = 1; j < n; j++) same &= wp[t0 + j] == wp[t0];
    float acc[TT];
    if (same) {
        DOT((const uint8_t*)(wp[t0] + off) + (size_t)r * row_bytes, x + (size_t)t0 * k, k, threadIdx.x & 31, n, acc);
    } else {
        for (int j = 0; j < n; j++)
            DOT1((const uint8_t*)(wp[t0 + j] + off) + (size_t)r * row_bytes, x + (size_t)(t0 + j) * k, k,
                 threadIdx.x & 31, 1, acc + j);
    }
    if ((threadIdx.x & 31) == 0)
        for (int j = 0; j < n; j++) y[(size_t)(t0 + j) * n_rows + r] = acc[j];
}
#define MM_QUANT(name, TT, sfx)                                                                    \
    extern "C" __global__ void mm_##name##sfx(const uint8_t* w, unsigned long long row_bytes, int k, \
                                              const float* x, float* y, int n_rows, int T) {        \
        mm_tile<TT, dot_##name<TT>>(w, row_bytes, k, x, y, n_rows, T);                             \
    }                                                                                              \
    extern "C" __global__ void mm_##name##_ids##sfx(const unsigned long long* wp, unsigned long long off, \
                                                    unsigned long long row_bytes, int k, const float* x, \
                                                    float* y, int n_rows, int T) {                 \
        mm_tile_ids<TT, dot_##name<1>, dot_##name<TT>>(wp, off, row_bytes, k, x, y, n_rows, T);    \
    }
MM_QUANT(q4_k, 1, )
MM_QUANT(q6_k, 1, )
MM_QUANT(q4_k, 8, _t8)
MM_QUANT(q6_k, 8, _t8)

// ---- Prefill GEMM: y[t][r] = W[r] . x[t] for many tokens, f32 SIMT, tiles in shared memory.
// Block: 128 rows x 32 tokens, 128 threads, each 8 rows x 4 tokens. K advances one 32-element
// sub-block per step: thread i dequantizes row r0+i's sub-block into ws[k][i] (k-major), the
// block loads x[t0..t0+31][k0..k0+31] into xs. f32 throughout (sum order: k ascending).
#define GB_R 128
#define GB_T 32

// Element 32j + l of a Q4_K row (l = 0..31) for every l; out[l] .
__device__ __forceinline__ void deq32_q4_k(const uint8_t* row, int j, float* out) {
    const uint8_t* b = row + (j >> 3) * 144;
    const int is = j & 7;
    const uint8_t* sc = b + 4;
    int sv, mv;
    if (is < 4) { sv = sc[is] & 63; mv = sc[is + 4] & 63; }
    else { sv = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); mv = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
    const float d = ld_f16(b) * sv, m = ld_f16(b + 2) * mv;
    const uint8_t* qs = b + 16 + 32 * (is >> 1);
    const int sh = (is & 1) * 4;
#pragma unroll
    for (int l = 0; l < 32; l += 4) {
        const unsigned q = *(const unsigned*)(qs + l);
        out[l] = d * ((q >> sh) & 0xF) - m;
        out[l + 1] = d * ((q >> (8 + sh)) & 0xF) - m;
        out[l + 2] = d * ((q >> (16 + sh)) & 0xF) - m;
        out[l + 3] = d * ((q >> (24 + sh)) & 0xF) - m;
    }
}

// Q6_K: chunk j of a super-block is quadrant qd = j%4 of half n = (j/4)%2.
__device__ __forceinline__ void deq32_q6_k(const uint8_t* row, int j, float* out) {
    const uint8_t* b = row + (j >> 3) * 210;
    const int n = (j >> 2) & 1, qd = j & 3;
    const uint8_t* ql = b + 64 * n + 32 * (qd & 1);
    const uint8_t* qh = b + 128 + 32 * n;
    const int8_t* sc = (const int8_t*)(b + 192 + 8 * n) + 2 * qd;
    const float d = ld_f16(b + 208);
    const float d0 = d * sc[0], d1 = d * sc[1];
    const int sl = (qd >> 1) * 4, sh = 2 * qd;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        const int v = ((ql[l] >> sl) & 0xF) | (((qh[l] >> sh) & 3) << 4);
        out[l] = (l < 16 ? d0 : d1) * (float)(v - 32);
    }
}

template <void (*DEQ)(const uint8_t*, int, float*)>
__device__ __forceinline__ void gemm_tile(const uint8_t* w, unsigned long long row_bytes, int k, const float* x,
                                          float* y, int n_rows, int r0, int t0, int ts, int te, float (*acc)[4],
                                          float (*ws)[GB_R + 4], float (*xs)[GB_T + 1]) {
    // Accumulates tokens [t0+ts, t0+te) of the tile (others get +0) using weights w.
    const int tid = threadIdx.x, rg = tid & 15, tg = tid >> 4;
    float buf[32];
    for (int j = 0; j < k / 32; j++) {
        __syncthreads();
        if (r0 + tid < n_rows) {
            DEQ(w + (size_t)(r0 + tid) * row_bytes, j, buf);
#pragma unroll
            for (int l = 0; l < 32; l++) ws[l][tid] = buf[l];
        } else {
#pragma unroll
            for (int l = 0; l < 32; l++) ws[l][tid] = 0.0f;
        }
        for (int i = tid; i < 32 * GB_T; i += 128) {
            const int t = i >> 5, l = i & 31;
            xs[l][t] = (t >= ts && t < te) ? x[(size_t)(t0 + t) * k + 32 * j + l] : 0.0f;
        }
        __syncthreads();
#pragma unroll 4
        for (int l = 0; l < 32; l++) {
            const float4 wa = *(const float4*)&ws[l][8 * rg], wb = *(const float4*)&ws[l][8 * rg + 4];
            const float wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const float xv = xs[l][4 * tg + c];
#pragma unroll
                for (int r = 0; r < 8; r++) acc[r][c] += wv[r] * xv;
            }
        }
    }
}

__device__ __forceinline__ void gemm_store(float* y, int n_rows, int r0, int t0, int T, float (*acc)[4]) {
    const int rg = threadIdx.x & 15, tg = threadIdx.x >> 4;
    for (int c = 0; c < 4; c++) {
        const int t = t0 + 4 * tg + c;
        if (t >= T) continue;
        for (int r = 0; r < 8; r++) {
            const int row = r0 + 8 * rg + r;
            if (row < n_rows) y[(size_t)t * n_rows + row] = acc[r][c];
        }
    }
}

// grid (ceil(n_rows / 128), ceil(T / 32)), block 128.
#define GEMM_QUANT(name)                                                                           \
    extern "C" __global__ void gemm_##name(const uint8_t* w, unsigned long long row_bytes, int k,  \
                                           const float* x, float* y, int n_rows, int T) {          \
        __shared__ __align__(16) float ws[32][GB_R + 4];                                           \
        __shared__ float xs[32][GB_T + 1];                                                         \
        const int r0 = blockIdx.x * GB_R, t0 = blockIdx.y * GB_T;                                  \
        float acc[8][4] = {};                                                                      \
        gemm_tile<deq32_##name>(w, row_bytes, k, x, y, n_rows, r0, t0, 0, min(GB_T, T - t0), acc, ws, xs); \
        gemm_store(y, n_rows, r0, t0, T, acc);                                                     \
    }                                                                                              \
    /* Gathered rows grouped by expert: each run of equal wp in the tile is one pass. */           \
    extern "C" __global__ void gemm_##name##_ids(const unsigned long long* wp, unsigned long long off, \
                                                 unsigned long long row_bytes, int k, const float* x, \
                                                 float* y, int n_rows, int T) {                    \
        __shared__ __align__(16) float ws[32][GB_R + 4];                                           \
        __shared__ float xs[32][GB_T + 1];                                                         \
        const int r0 = blockIdx.x * GB_R, t0 = blockIdx.y * GB_T, n = min(GB_T, T - t0);           \
        float acc[8][4] = {};                                                                      \
        for (int s = 0; s < n;) {                                                                  \
            int e = s + 1;                                                                         \
            while (e < n && wp[t0 + e] == wp[t0 + s]) e++;                                         \
            gemm_tile<deq32_##name>((const uint8_t*)(wp[t0 + s] + off), row_bytes, k, x, y, n_rows, r0, t0, s, e, acc, ws, xs); \
            s = e;                                                                                 \
        }                                                                                          \
        gemm_store(y, n_rows, r0, t0, T, acc);                                                     \
    }
GEMM_QUANT(q4_k)
GEMM_QUANT(q6_k)

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
// out: T x nh x hd. grid (nh, T), block 32 * ATT_WARPS; hd a multiple of 32, <= 256.
// Token t sees positions [0, n_past + t]. Each warp takes positions w, w + nw, ... with its own
// softmax state (lane owns dims lane + 32d), then the warps merge.
// The caches are f16 (llama.cpp's default KV type), so 32K ctx costs what 16K did in f32.
// ponytail: K/V re-read per query head (no GQA sharing); tile for long contexts.
#define ATT_WARPS 8
extern "C" __global__ void attention(const float* q, const __half* kc, const __half* vc, float* out,
                                     int nh, int nkv, int hd, int n_past, float scale) {
    __shared__ float sm[ATT_WARPS], sl[ATT_WARPS], sacc[ATT_WARPS][256];
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int hk = h / (nh / nkv), nd = hd / 32;
    const float* qp = q + ((size_t)t * nh + h) * hd;
    float qv[8], acc[8];
    for (int d = 0; d < nd; d++) { qv[d] = qp[lane + 32 * d]; acc[d] = 0.0f; }
    const int n_ctx = n_past + t + 1;
    float m = -INFINITY, l = 0.0f;
    for (int p = w; p < n_ctx; p += ATT_WARPS) {
        const size_t off = ((size_t)p * nkv + hk) * hd;
        float s = 0.0f;
        for (int d = 0; d < nd; d++) s += qv[d] * __half2float(kc[off + lane + 32 * d]);
        s = warp_sum(s) * scale;
        const float mn = fmaxf(m, s);
        const float c = expf(m - mn), e = expf(s - mn);
        l = l * c + e;
        for (int d = 0; d < nd; d++) acc[d] = acc[d] * c + e * __half2float(vc[off + lane + 32 * d]);
        m = mn;
    }
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    for (int d = 0; d < nd; d++) sacc[w][lane + 32 * d] = acc[d];
    __syncthreads();
    if (w != 0) return;
    float M = sm[0];
    for (int i = 1; i < ATT_WARPS; i++) M = fmaxf(M, sm[i]);
    float L = 0.0f;
    for (int i = 0; i < ATT_WARPS; i++) L += sl[i] * expf(sm[i] - M);
    for (int d = 0; d < nd; d++) {
        float o = 0.0f;
        for (int i = 0; i < ATT_WARPS; i++) o += sacc[i][lane + 32 * d] * expf(sm[i] - M);
        out[((size_t)t * nh + h) * hd + lane + 32 * d] = o / L;
    }
}

// Decode attention (T <= 4), flash style. A block owns one KV head and 8 query rows: the G =
// nh/nkv heads sharing it, for 8/G consecutive tokens. Tiles of FA_KT positions of K and V are
// staged once in shared memory for all rows (GQA sharing). Warp w owns row w (one token's G = 8
// heads keep all 8 warps busy); scores: lane j = position j of the tile; output: lane owns dims
// 4*lane..+3. R rows per warp is kept as a constant: the loops are written for R > 1.
// Requires hd == 128 and 8 % G == 0 (checked on the host). grid (nkv, ceil(T*G/8), S), block 256.
// Split-K (decode): with S > 1, block z covers its S-th of the positions and writes its softmax
// state (m, l, unnormalized acc[128]) to part[(z * T * nh + row) * 130]; attn_merge combines.
#define FA_KT 32
extern "C" __global__ void attention_dec(const float* q, const __half* kc, const __half* vc, float* out,
                                         int nh, int nkv, int n_past, int T, float scale, float* part) {
    constexpr int R = 1, NR = 8 * R;
    __shared__ float sq[NR][128];
    __shared__ __half2 sk[FA_KT][65];  // padded: lane j reads row j conflict-free
    __shared__ __half2 sv[FA_KT][64];
    __shared__ float sp[NR][FA_KT];
    const int G = nh / nkv, QT = NR / G, hk = blockIdx.x, t0 = blockIdx.y * QT;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < NR * 128; i += 256) {
        const int r = i >> 7, d = i & 127, t = min(t0 + r / G, T - 1);
        sq[r][d] = q[((size_t)t * nh + hk * G + r % G) * 128 + d];
    }
    int tr[R];
    float m[R], l[R], acc[R][4];
    for (int i = 0; i < R; i++) {
        tr[i] = n_past + min(t0 + (R * w + i) / G, T - 1);  // last visible position
        m[i] = -INFINITY; l[i] = 0.0f;
        for (int d = 0; d < 4; d++) acc[i][d] = 0.0f;
    }
    const int n_pos = n_past + min(t0 + QT, T);
    const int span = (n_pos + FA_KT * gridDim.z - 1) / (FA_KT * gridDim.z) * FA_KT;
    const int pb = blockIdx.z * span, pe = min(n_pos, pb + span);
    const bool live = t0 + R * w / G < T;  // warp has at least one real row
    for (int p0 = pb; p0 < pe; p0 += FA_KT) {
        __syncthreads();  // previous tile fully consumed (and sq written, first time)
        for (int i = threadIdx.x; i < FA_KT * 16; i += 256) {  // 16 x 8 halves per row
            const int j = i >> 4, c = i & 15, p = p0 + j;
            uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
            if (p < pe) {
                const size_t off = ((size_t)p * nkv + hk) * 128 + 8 * c;
                kv4 = *(const uint4*)(kc + off);
                vv4 = *(const uint4*)(vc + off);
            }
            const __half2* k2 = (const __half2*)&kv4;
            for (int e = 0; e < 4; e++) sk[j][4 * c + e] = k2[e];
            *(uint4*)&sv[j][4 * c] = vv4;
        }
        __syncthreads();
        if (!live) continue;
        float s[R] = {};
        for (int d2 = 0; d2 < 64; d2++) {
            const float2 kf = __half22float2(sk[lane][d2]);
            for (int i = 0; i < R; i++) {
                const float2 qf = *(const float2*)&sq[R * w + i][2 * d2];
                s[i] += qf.x * kf.x + qf.y * kf.y;
            }
        }
        const int p = p0 + lane;
        for (int i = 0; i < R; i++) {
            const float si = p <= tr[i] ? s[i] * scale : -INFINITY;
            float mt = si;
            for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, o));
            const float mn = fmaxf(m[i], mt);  // -inf only if no position seen yet (split-K)
            const float c = mn == -INFINITY ? 1.0f : expf(m[i] - mn), e = si == -INFINITY ? 0.0f : expf(si - mn);
            l[i] = l[i] * c + warp_sum(e);
            m[i] = mn;
            for (int d = 0; d < 4; d++) acc[i][d] *= c;
            sp[R * w + i][lane] = e;
        }
        __syncwarp();
        for (int j = 0; j < FA_KT; j++) {
            const float2 v01 = __half22float2(sv[j][2 * lane]), v23 = __half22float2(sv[j][2 * lane + 1]);
            for (int i = 0; i < R; i++) {
                const float e = sp[R * w + i][j];
                acc[i][0] += e * v01.x; acc[i][1] += e * v01.y; acc[i][2] += e * v23.x; acc[i][3] += e * v23.y;
            }
        }
    }
    for (int i = 0; i < R; i++) {
        const int r = R * w + i, t = t0 + r / G;
        if (t >= T) continue;
        const size_t row = (size_t)t * nh + hk * G + r % G;
        if (gridDim.z > 1) {
            float* pp = part + ((size_t)blockIdx.z * T * nh + row) * 130;
            if (lane == 0) { pp[0] = m[i]; pp[1] = l[i]; }
            for (int d = 0; d < 4; d++) pp[2 + 4 * lane + d] = acc[i][d];
            continue;
        }
        float4 o = make_float4(acc[i][0] / l[i], acc[i][1] / l[i], acc[i][2] / l[i], acc[i][3] / l[i]);
        *(float4*)(out + ((size_t)t * nh + hk * G + r % G) * 128 + 4 * lane) = o;
    }
}


// Tensor-core attention (prefill), FlashAttention-2 layout with mma.sync m16n8k16 (f16 in, f32
// accumulate). A block owns one KV head and 64 query rows (64/G tokens x the G heads sharing it);
// warp w owns rows 16w..16w+15. K tiles of TC_KT positions are staged in shared memory (V
// transposed, so both B fragments are half2 loads). The online softmax runs on the accumulator
// registers: thread (g = lane/4, tq = lane%4) holds rows g and g+8, columns 8n + 2tq + {0,1}.
// Q and P are fed as f16 hi + lo pairs (two mmas each): ~f32 accuracy; K and V are exact f16.
// Requires hd == 128 and 64 % G == 0. grid (nkv, ceil(T*G/64), S), block 128. Split-K as above.
#define TC_KT 64
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void split_h2(float x, float y, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(x, y);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(x - hf.x, y - hf.y);
    hi = *(const uint32_t*)&h;
    lo = *(const uint32_t*)&l;
}
extern "C" __global__ void __launch_bounds__(128) attention_tc(const float* q, const __half* kc, const __half* vc, float* out,
                                                              int nh, int nkv, int n_past, int T, float scale, float* part) {
    __shared__ __align__(16) __half sk[TC_KT][136];
    __shared__ __align__(16) __half svt[128][TC_KT + 8];
    const int G = nh / nkv, QT = 64 / G, hk = blockIdx.x, t0 = blockIdx.y * QT;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const int rr[2] = {16 * w + g, 16 * w + g + 8};
    int tr[2];
    const float* qp[2];
    for (int i = 0; i < 2; i++) {
        const int t = min(t0 + rr[i] / G, T - 1);
        tr[i] = n_past + t;  // last visible position
        qp[i] = q + ((size_t)t * nh + hk * G + rr[i] % G) * 128;
    }
    uint32_t qh[8][4], ql[8][4];
    for (int kk = 0; kk < 8; kk++) {
        const int c = 16 * kk + 2 * tq;
        split_h2(qp[0][c], qp[0][c + 1], qh[kk][0], ql[kk][0]);
        split_h2(qp[1][c], qp[1][c + 1], qh[kk][1], ql[kk][1]);
        split_h2(qp[0][c + 8], qp[0][c + 9], qh[kk][2], ql[kk][2]);
        split_h2(qp[1][c + 8], qp[1][c + 9], qh[kk][3], ql[kk][3]);
    }
    float o[16][4];
    for (int n = 0; n < 16; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    const int n_pos = n_past + min(t0 + QT, T);
    const int span = (n_pos + TC_KT * gridDim.z - 1) / (TC_KT * gridDim.z) * TC_KT;
    const int pb = blockIdx.z * span, pe = min(n_pos, pb + span);
    for (int p0 = pb; p0 < pe; p0 += TC_KT) {
        __syncthreads();  // previous tile consumed
        for (int i = threadIdx.x; i < TC_KT * 16; i += 128) {
            const int j = i >> 4, c = i & 15, p = p0 + j;
            uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
            if (p < pe) {
                const size_t off = ((size_t)p * nkv + hk) * 128 + 8 * c;
                kv4 = *(const uint4*)(kc + off);
                vv4 = *(const uint4*)(vc + off);
            }
            *(uint4*)&sk[j][8 * c] = kv4;
            const __half* vh = (const __half*)&vv4;
            for (int e = 0; e < 8; e++) svt[8 * c + e][j] = vh[e];
        }
        __syncthreads();
        float sc[8][4];
        for (int n = 0; n < 8; n++) {
            sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
            for (int kk = 0; kk < 8; kk++) {
                const __half* kr = &sk[8 * n + g][16 * kk + 2 * tq];
                const uint32_t b0 = *(const uint32_t*)kr, b1 = *(const uint32_t*)(kr + 8);
                mma16816(sc[n], qh[kk], b0, b1);
                mma16816(sc[n], ql[kk], b0, b1);
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
        for (int n = 0; n < 8; n++)
            for (int e = 0; e < 4; e++) {
                const int r = e >> 1, p = p0 + 8 * n + 2 * tq + (e & 1);
                const float v = p <= tr[r] ? sc[n][e] * scale : -INFINITY;
                sc[n][e] = v;
                mx[r] = fmaxf(mx[r], v);
            }
        float c[2], sum[2] = {0.0f, 0.0f};
        for (int r = 0; r < 2; r++) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);  // -inf only if no position seen yet
            c[r] = mn == -INFINITY ? 1.0f : expf(m[r] - mn);
            m[r] = mn;
        }
        for (int n = 0; n < 8; n++)
            for (int e = 0; e < 4; e++) {
                const int r = e >> 1;
                const float v = sc[n][e] == -INFINITY ? 0.0f : expf(sc[n][e] - m[r]);
                sc[n][e] = v;
                sum[r] += v;
            }
        for (int r = 0; r < 2; r++) {
            sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 1);
            sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 2);
            l[r] = l[r] * c[r] + sum[r];
        }
        for (int n = 0; n < 16; n++) { o[n][0] *= c[0]; o[n][1] *= c[0]; o[n][2] *= c[1]; o[n][3] *= c[1]; }
        for (int kk = 0; kk < TC_KT / 16; kk++) {
            uint32_t ph[4], pl[4];
            split_h2(sc[2 * kk][0], sc[2 * kk][1], ph[0], pl[0]);
            split_h2(sc[2 * kk][2], sc[2 * kk][3], ph[1], pl[1]);
            split_h2(sc[2 * kk + 1][0], sc[2 * kk + 1][1], ph[2], pl[2]);
            split_h2(sc[2 * kk + 1][2], sc[2 * kk + 1][3], ph[3], pl[3]);
            for (int n = 0; n < 16; n++) {
                const __half* vr = &svt[8 * n + g][16 * kk + 2 * tq];
                const uint32_t b0 = *(const uint32_t*)vr, b1 = *(const uint32_t*)(vr + 8);
                mma16816(o[n], ph, b0, b1);
                mma16816(o[n], pl, b0, b1);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        const int r = rr[i], t = t0 + r / G;
        if (t >= T) continue;
        const size_t row = (size_t)t * nh + hk * G + r % G;
        if (gridDim.z > 1) {
            float* pp = part + ((size_t)blockIdx.z * T * nh + row) * 130;
            if (tq == 0) { pp[0] = m[i]; pp[1] = l[i]; }
            for (int n = 0; n < 16; n++) { pp[2 + 8 * n + 2 * tq] = o[n][2 * i]; pp[3 + 8 * n + 2 * tq] = o[n][2 * i + 1]; }
            continue;
        }
        for (int n = 0; n < 16; n++)
            *(float2*)(out + row * 128 + 8 * n + 2 * tq) = make_float2(o[n][2 * i] / l[i], o[n][2 * i + 1] / l[i]);
    }
}

// Combine split-K partials of attention_dec / attention_tc. grid T * nh, block 128 (= hd).
extern "C" __global__ void attn_merge(const float* part, float* out, int rows, int S) {
    const int r = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int z = 0; z < S; z++) M = fmaxf(M, part[((size_t)z * rows + r) * 130]);
    float L = 0.0f, o = 0.0f;
    for (int z = 0; z < S; z++) {
        const float* pp = part + ((size_t)z * rows + r) * 130;
        if (pp[1] == 0.0f) continue;  // split saw no visible position
        const float c = expf(pp[0] - M);
        L += pp[1] * c;
        o += pp[2 + d] * c;
    }
    out[(size_t)r * 128 + d] = o / L;
}

// KV cache store: dst[i] = (f16) src[i].
extern "C" __global__ void store_f16(__half* dst, const float* src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
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

// out[j][i] = silu(gu[j][i]) * gu[j][ff + i] for n rows of [gate | up] (2 * ff wide).
extern "C" __global__ void swiglu(float* out, const float* gu, int ff, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * ff) return;
    const int j = i / ff, c = i % ff;
    const float g = gu[(size_t)j * 2 * ff + c], u = gu[(size_t)j * 2 * ff + ff + c];
    out[i] = g / (1.0f + expf(-g)) * u;
}

// x[t] += sum_{j in off[t]..off[t+1]} w[j] * d[rows[j]]: one token per block (CSR lists, in
// expert order). Called once per expert wave; a token with no rows in the wave gets += 0.
extern "C" __global__ void moe_combine(float* x, const float* d, const int* off, const int* rows,
                                       const float* w, int n) {
    const int t = blockIdx.x, j0 = off[t], j1 = off[t + 1];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = j0; j < j1; j++) acc += w[j] * d[(size_t)rows[j] * n + i];
        x[(size_t)t * n + i] += acc;
    }
}
