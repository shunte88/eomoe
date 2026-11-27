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

// Q8_0: 34 bytes: d f16, qs[32] i8. Lane l takes element l of every block.
template <int TT>
__device__ __forceinline__ void dot_q8_0(const uint8_t* row, const float* xt, int k, int l, int n, float* out) {
    float acc[TT];
#pragma unroll
    for (int j = 0; j < TT; j++) acc[j] = 0.0f;
    for (int b = 0; b < k / 32; b++) {
        const uint8_t* blk = row + b * 34;
        const float w = ld_f16(blk) * (float)(int8_t)blk[2 + l];
#pragma unroll
        for (int j = 0; j < TT; j++)
            if (j < n) acc[j] += w * xt[(size_t)j * k + b * 32 + l];
    }
#pragma unroll
    for (int j = 0; j < TT; j++) out[j] = warp_sum(acc[j]);
}

// MXFP4: 17 bytes: e8m0 scale, qs[16] e2m1 nibbles (low: elements 0..15, high: 16..31). Lane l
// takes element l of every block. Values are doubled e2m1 (0 1 2 3 4 6 8 12, sign bit 3) times
// half the e8m0 scale, 2^(e - 128), as ggml.
template <int TT>
__device__ __forceinline__ void dot_mxfp4(const uint8_t* row, const float* xt, int k, int l, int n, float* out) {
    float acc[TT];
#pragma unroll
    for (int j = 0; j < TT; j++) acc[j] = 0.0f;
    for (int b = 0; b < k / 32; b++) {
        const uint8_t* blk = row + b * 17;
        const unsigned e = blk[0];
        const float d = __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
        const int q = l < 16 ? blk[1 + l] & 15 : blk[1 + l - 16] >> 4, m = q & 7;
        const float v = (float)(m < 4 ? m : m < 6 ? 2 * m - 4 : 4 * m - 16);
        const float w = d * (q & 8 ? -v : v);
#pragma unroll
        for (int j = 0; j < TT; j++)
            if (j < n) acc[j] += w * xt[(size_t)j * k + b * 32 + l];
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
MM_QUANT(q8_0, 1, )
MM_QUANT(q8_0, 8, _t8)
MM_QUANT(mxfp4, 1, )
MM_QUANT(mxfp4, 8, _t8)

// Per-head matvec (MLA: q_nope through attn_k_b, the latent output through attn_v_b): head h's
// weights are the n_rows rows at w + h * w_hs; y[t * y_ts + h * y_hs + r] = W_h[r] . (the k
// floats at x + t * x_ts + h * x_hs). grid (ceil(n_rows / MM_ROWS), T, heads), block 32 * MM_ROWS.
// ponytail: one token per warp; tile tokens (strided x) if MLA prefill shows up in profiles.
template <void (*DOT)(const uint8_t*, const float*, int, int, int, float*)>
__device__ __forceinline__ void mm_heads_tile(const uint8_t* w, unsigned long long row_bytes, unsigned long long w_hs,
                                              int k, const float* x, int x_ts, int x_hs, float* y, int y_ts,
                                              int y_hs, int n_rows) {
    const int r = blockIdx.x * MM_ROWS + (threadIdx.x >> 5), t = blockIdx.y, h = blockIdx.z;
    if (r >= n_rows) return;
    float acc;
    DOT(w + h * w_hs + (size_t)r * row_bytes, x + (size_t)t * x_ts + (size_t)h * x_hs, k, threadIdx.x & 31, 1, &acc);
    if ((threadIdx.x & 31) == 0) y[(size_t)t * y_ts + (size_t)h * y_hs + r] = acc;
}
#define MM_HEADS(name)                                                                              \
    extern "C" __global__ void mm_heads_##name(const uint8_t* w, unsigned long long row_bytes,     \
                                               unsigned long long w_hs, int k, const float* x, int x_ts, \
                                               int x_hs, float* y, int y_ts, int y_hs, int n_rows) { \
        mm_heads_tile<dot_##name<1>>(w, row_bytes, w_hs, k, x, x_ts, x_hs, y, y_ts, y_hs, n_rows);  \
    }
MM_HEADS(q4_k)
MM_HEADS(q6_k)
MM_HEADS(q8_0)

// ---- Prefill GEMM: y[t][r] = W[r] . x[t] for many tokens, on tensor cores (mma.sync m16n8k16,
// f16 in, f32 accumulate), MMQ-style. Block: 128 rows x NT tokens (GB_T = 128 dense, GB_TI = 64
// experts), 4 warps; warp w owns rows 32w..32w+31 x all NT tokens (2 x NT/8 mma tiles). K advances one 32-element sub-block per
// step: thread i unpacks row r0+i's quants as exact f16 integers plus per-16 scale and min; the
// block splits x[t0..][k0..k0+31] into f16 hi + lo and sums each 16. Per 16 of K: p = q.x (two
// mmas), acc += sc * p - mn * sum(x), in f32. Weights are exact, activations ~f32. Row strides of
// 40 halves keep the fragment loads bank-conflict free.
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
#define GB_R 128
#define GB_T 128  // tokens per tile: dense projections
#define GB_TI 64  // expert (_ids) GEMM tile

// Sub-block j (elements 32j..32j+31) of a quantized row as exact small integers q[l] (f16) and
// per-16 scales: element 32j + l = sc[l / 16] * q[l] - mn[l / 16].
__device__ __forceinline__ uint32_t h2u(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t*)&h;
}
__device__ __forceinline__ void qdeq32_q4_k(const uint8_t* row, int j, uint32_t* q, float* sc, float* mn) {
    const uint8_t* b = row + (j >> 3) * 144;
    const int is = j & 7;
    const uint8_t* s = b + 4;
    int sv, mv;
    if (is < 4) { sv = s[is] & 63; mv = s[is + 4] & 63; }
    else { sv = (s[is + 4] & 0xF) | ((s[is - 4] >> 6) << 4); mv = (s[is + 4] >> 4) | ((s[is] >> 6) << 4); }
    sc[0] = sc[1] = ld_f16(b) * sv;
    mn[0] = mn[1] = ld_f16(b + 2) * mv;
    const uint8_t* qs = b + 16 + 32 * (is >> 1);
    const int sh = (is & 1) * 4;
    // 16-byte loads: Q4_K blocks are 144 bytes, so qs is 16-aligned (rows are 144-byte multiples).
    const uint4 u0 = *(const uint4*)qs, u1 = *(const uint4*)(qs + 16);
    const unsigned w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned v = w[i] >> sh;
        q[2 * i] = h2u(v & 0xF, (v >> 8) & 0xF);
        q[2 * i + 1] = h2u((v >> 16) & 0xF, (v >> 24) & 0xF);
    }
}

// Q6_K: chunk j of a super-block is quadrant qd = j%4 of half n = (j/4)%2; q - 32 in [-32, 31].
__device__ __forceinline__ void qdeq32_q6_k(const uint8_t* row, int j, uint32_t* q, float* sc, float* mn) {
    const uint8_t* b = row + (j >> 3) * 210;
    const int n = (j >> 2) & 1, qd = j & 3;
    const uint8_t* ql = b + 64 * n + 32 * (qd & 1);
    const uint8_t* qh = b + 128 + 32 * n;
    const int8_t* s = (const int8_t*)(b + 192 + 8 * n) + 2 * qd;
    const float d = ld_f16(b + 208);
    sc[0] = d * s[0];
    sc[1] = d * s[1];
    mn[0] = mn[1] = 0.0f;
    const int sl = (qd >> 1) * 4, sh = 2 * qd;
#pragma unroll
    for (int l = 0; l < 32; l += 4) {
        // 16-bit loads: Q6_K blocks are 210 bytes, 2- but not 4-aligned.
        const unsigned lo = (*(const unsigned short*)(ql + l) | (*(const unsigned short*)(ql + l + 2) << 16)) >> sl;
        const unsigned hi = (*(const unsigned short*)(qh + l) | (*(const unsigned short*)(qh + l + 2) << 16)) >> sh;
        int v[4];
#pragma unroll
        for (int c = 0; c < 4; c++) v[c] = (int)(((lo >> 8 * c) & 0xF) | (((hi >> 8 * c) & 3) << 4)) - 32;
        q[l / 2] = h2u(v[0], v[1]);
        q[l / 2 + 1] = h2u(v[2], v[3]);
    }
}

#define GB_KS 40
template <void (*DEQ)(const uint8_t*, int, uint32_t*, float*, float*), bool PER32, int NT>
__device__ __forceinline__ void gemm_tile(const uint8_t* w, unsigned long long row_bytes, int k, const float* x,
                                          int n_rows, int r0, int t0, int ts, int te, float (*acc)[(NT / 8)][4],
                                          __half (*wq)[GB_KS], float (*wsc)[2], float (*wmn)[2],
                                          __half (*xh)[GB_KS], __half (*xl)[GB_KS], float (*xs)[2]) {
    // Accumulates tokens [t0+ts, t0+te) of the tile (others get +0) using weights w.
    const int tid = threadIdx.x, lane = tid & 31, wr = (tid >> 5) * 32, g = lane >> 2, tq = lane & 3;
    for (int j = 0; j < k / 32; j++) {
        __syncthreads();
        uint32_t q[16] = {};
        float sc2[2] = {0.0f, 0.0f}, mn2[2] = {0.0f, 0.0f};
        if (r0 + tid < n_rows) DEQ(w + (size_t)(r0 + tid) * row_bytes, j, q, sc2, mn2);
#pragma unroll
        for (int l = 0; l < 16; l += 4) *(uint4*)&wq[tid][2 * l] = make_uint4(q[l], q[l + 1], q[l + 2], q[l + 3]);
        wsc[tid][0] = sc2[0]; wsc[tid][1] = sc2[1];
        wmn[tid][0] = mn2[0]; wmn[tid][1] = mn2[1];
        // x: 4 lanes per (token, 16-half); each loads 4 floats, splits hi/lo, sums the half.
        for (int i = tid; i < 8 * NT; i += 128) {
            const int t = i >> 3, l = 4 * (i & 7);
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t >= ts && t < te) v = *(const float4*)&x[(size_t)(t0 + t) * k + 32 * j + l];
            split_h2(v.x, v.y, *(uint32_t*)&xh[t][l], *(uint32_t*)&xl[t][l]);
            split_h2(v.z, v.w, *(uint32_t*)&xh[t][l + 2], *(uint32_t*)&xl[t][l + 2]);
            float sum = (v.x + v.y) + (v.z + v.w);
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            if ((i & 3) == 0) xs[t][(i >> 2) & 1] = sum;
        }
        __syncthreads();
        // Q4_K's scale covers all 32 (PER32: one fold per 32); Q6_K's changes every 16.
        uint32_t a[2][2][4];
        float sc[2][2][2], mn[2][2][2];
#pragma unroll
        for (int hk = 0; hk < 2; hk++)
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const int r = wr + 16 * m + g, kk = 16 * hk + 2 * tq;
                a[hk][m][0] = *(const uint32_t*)&wq[r][kk];
                a[hk][m][1] = *(const uint32_t*)&wq[r + 8][kk];
                a[hk][m][2] = *(const uint32_t*)&wq[r][kk + 8];
                a[hk][m][3] = *(const uint32_t*)&wq[r + 8][kk + 8];
                sc[hk][m][0] = wsc[r][hk]; sc[hk][m][1] = wsc[r + 8][hk];
                mn[hk][m][0] = wmn[r][hk]; mn[hk][m][1] = wmn[r + 8][hk];
            }
#pragma unroll
        for (int n = 0; n < (NT / 8); n++) {
            const int t = 8 * n + g, tc = 8 * n + 2 * tq;
            uint32_t h[2][2], lo[2][2];
            float xsum[2][2];
#pragma unroll
            for (int hk = 0; hk < 2; hk++) {
                const int kk = 16 * hk + 2 * tq;
                h[hk][0] = *(const uint32_t*)&xh[t][kk]; h[hk][1] = *(const uint32_t*)&xh[t][kk + 8];
                lo[hk][0] = *(const uint32_t*)&xl[t][kk]; lo[hk][1] = *(const uint32_t*)&xl[t][kk + 8];
                xsum[hk][0] = xs[tc][hk]; xsum[hk][1] = xs[tc + 1][hk];
            }
#pragma unroll
            for (int m = 0; m < 2; m++) {
                float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    mma16816(p, a[hk][m], lo[hk][0], lo[hk][1]);
                    mma16816(p, a[hk][m], h[hk][0], h[hk][1]);
                    if (!PER32 || hk == 1) {
                        const float s0 = PER32 ? xsum[0][0] + xsum[1][0] : xsum[hk][0];
                        const float s1 = PER32 ? xsum[0][1] + xsum[1][1] : xsum[hk][1];
                        acc[m][n][0] += sc[hk][m][0] * p[0] - mn[hk][m][0] * s0;
                        acc[m][n][1] += sc[hk][m][0] * p[1] - mn[hk][m][0] * s1;
                        acc[m][n][2] += sc[hk][m][1] * p[2] - mn[hk][m][1] * s0;
                        acc[m][n][3] += sc[hk][m][1] * p[3] - mn[hk][m][1] * s1;
                        p[0] = p[1] = p[2] = p[3] = 0.0f;
                    }
                }
            }
        }
    }
}

template <int NT>
__device__ __forceinline__ void gemm_store(float* y, int n_rows, int r0, int t0, int T, float (*acc)[(NT / 8)][4]) {
    const int lane = threadIdx.x & 31, wr = (threadIdx.x >> 5) * 32, g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int n = 0; n < (NT / 8); n++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int row = r0 + wr + 16 * m + g + 8 * (c >> 1), t = t0 + 8 * n + 2 * tq + (c & 1);
                if (row < n_rows && t < T) y[(size_t)t * n_rows + row] = acc[m][n][c];
            }
}

// grid (ceil(n_rows / 128), ceil(T / GB_T)) (_ids: ceil(T / GB_TI)), block 128.
#define GEMM_QUANT(name, per32)                                                                          \
    extern "C" __global__ void gemm_##name(const uint8_t* w, unsigned long long row_bytes, int k,  \
                                           const float* x, float* y, int n_rows, int T) {          \
        __shared__ __align__(16) __half wq[GB_R][GB_KS], xh[GB_T][GB_KS], xl[GB_T][GB_KS];                   \
        __shared__ float wsc[GB_R][2], wmn[GB_R][2], xs[GB_T][2];                                  \
        const int r0 = blockIdx.x * GB_R, t0 = blockIdx.y * GB_T;                                  \
        float acc[2][GB_T / 8][4] = {};                                                                \
        gemm_tile<qdeq32_##name, per32, GB_T>(w, row_bytes, k, x, n_rows, r0, t0, 0, min(GB_T, T - t0), acc, wq, wsc, wmn, xh, xl, xs); \
        gemm_store<GB_T>(y, n_rows, r0, t0, T, acc);                                                     \
    }                                                                                              \
    /* Gathered rows grouped by expert: each run of equal wp in the tile is one pass. */           \
    extern "C" __global__ void gemm_##name##_ids(const unsigned long long* wp, unsigned long long off, \
                                                 unsigned long long row_bytes, int k, const float* x, \
                                                 float* y, int n_rows, int T) {                    \
        __shared__ __align__(16) __half wq[GB_R][GB_KS], xh[GB_TI][GB_KS], xl[GB_TI][GB_KS];                   \
        __shared__ float wsc[GB_R][2], wmn[GB_R][2], xs[GB_TI][2];                                  \
        const int r0 = blockIdx.x * GB_R, t0 = blockIdx.y * GB_TI, n = min(GB_TI, T - t0);           \
        float acc[2][GB_TI / 8][4] = {};                                                                 \
        for (int s = 0; s < n;) {                                                                  \
            int e = s + 1;                                                                         \
            while (e < n && wp[t0 + e] == wp[t0 + s]) e++;                                         \
            gemm_tile<qdeq32_##name, per32, GB_TI>((const uint8_t*)(wp[t0 + s] + off), row_bytes, k, x, n_rows, r0, t0, s, e, acc, wq, wsc, wmn, xh, xl, xs); \
            s = e;                                                                                 \
        }                                                                                          \
        gemm_store<GB_TI>(y, n_rows, r0, t0, T, acc);                                                     \
    }
GEMM_QUANT(q4_k, true)
GEMM_QUANT(q6_k, false)

// Row-wise RMSNorm, y may alias x. grid rows.
__device__ __forceinline__ void rmsnorm_row(float* yr, const float* xr, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float s = 1.0f / sqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * s * w[i];
}
extern "C" __global__ void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    rmsnorm_row(y + (size_t)blockIdx.x * n, x + (size_t)blockIdx.x * n, w, n, eps);
}
// The first n of rows `stride` apart (MLA: the latent half of each kv_a row).
extern "C" __global__ void rmsnorm_strided(float* y, const float* x, const float* w, int n, int stride, float eps) {
    rmsnorm_row(y + (size_t)blockIdx.x * stride, x + (size_t)blockIdx.x * stride, w, n, eps);
}

// NORM rope (adjacent pairs (2i, 2i+1), theta_i = pos * base^(-2i/d)) of d-wide row pieces: row r
// reads src + r * s_stride, writes dst + r * d_stride (may alias); pos = pos0 + r / rows_per_tok.
// grid rows, block d / 2.
extern "C" __global__ void rope_norm(float* dst, int d_stride, const float* src, int s_stride,
                                     int rows_per_tok, int pos0, float base, int d) {
    const int r = blockIdx.x, i = threadIdx.x;
    const float* sp = src + (size_t)r * s_stride;
    float* dp = dst + (size_t)r * d_stride;
    const float th = (float)(pos0 + r / rows_per_tok) * powf(base, -2.0f * i / d);
    const float sn = sinf(th), cs = cosf(th), a = sp[2 * i], b = sp[2 * i + 1];
    dp[2 * i] = a * cs - b * sn;
    dp[2 * i + 1] = a * sn + b * cs;
}

// Per-head RMSNorm (q_norm/k_norm; none if w is null) then NEOX rope in place. v is
// T x heads x hd, grid T * heads, block hd. Pairs (i, i + hd/2), pos = pos0 + t, theta_i =
// pos * freq[i] (YaRN table) or pos * base^(-2i/hd) if freq is null; cos/sin times mscale.
extern "C" __global__ void qk_norm_rope(float* v, const float* w, int heads, int hd, int pos0,
                                        float eps, float base, const float* freq, float mscale) {
    __shared__ float s[1024];
    float* hv = v + (size_t)blockIdx.x * hd;
    const int i = threadIdx.x;
    const float a = hv[i];
    const float sc = 1.0f / sqrtf(block_sum(a * a) / hd + eps);
    s[i] = w ? a * sc * w[i] : a;
    __syncthreads();
    const int half = hd / 2;
    const float pos = (float)(pos0 + blockIdx.x / heads);
    const int j = i < half ? i : i - half;
    const float th = pos * (freq ? freq[j] : powf(base, -2.0f * j / hd));
    const float sn = sinf(th) * mscale, cs = cosf(th) * mscale;
    hv[i] = i < half ? s[i] * cs - s[i + half] * sn : s[i - half] * sn + s[i] * cs;
}

// Causal GQA attention with online softmax. q: T x nh x hd, caches: n_ctx x nkv x hd,
// out: T x nh x hd. grid (nh, T), block 32 * ATT_WARPS; hd a multiple of 32, <= 256.
// Token t sees positions [0, n_past + t], or the last `window` of them if window > 0. With
// sinks (one logit per head, or null) the sink joins the softmax denominator, no value row.
// Each warp takes positions w, w + nw, ... with its own
// softmax state (lane owns dims lane + 32d), then the warps merge.
// The caches are f16 (llama.cpp's default KV type), so 32K ctx costs what 16K did in f32.
// ponytail: K/V re-read per query head (no GQA sharing); tile for long contexts.
#define ATT_WARPS 8
// Paged KV: a sequence's position p lives at slot pt[p / KV_PAGE] * KV_PAGE + p % KV_PAGE of the
// layer's K / V pool (pages shared by all sequences; tiles never straddle a page).
#define KV_PAGE 256
__device__ __forceinline__ size_t kv_slot(const int* pt, int p) {
    return (size_t)pt[p / KV_PAGE] * KV_PAGE + p % KV_PAGE;
}
extern "C" __global__ void attention(const float* q, const __half* kc, const __half* vc, const int* pt, float* out,
                                     int nh, int nkv, int hd, int n_past, float scale,
                                     const float* sinks, int window) {
    __shared__ float sm[ATT_WARPS], sl[ATT_WARPS], sacc[ATT_WARPS][256];
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int hk = h / (nh / nkv), nd = hd / 32;
    const float* qp = q + ((size_t)t * nh + h) * hd;
    float qv[8], acc[8];
    for (int d = 0; d < nd; d++) { qv[d] = qp[lane + 32 * d]; acc[d] = 0.0f; }
    const int n_ctx = n_past + t + 1, p0 = window > 0 ? max(0, n_ctx - window) : 0;
    float m = -INFINITY, l = 0.0f;
    for (int p = p0 + w; p < n_ctx; p += ATT_WARPS) {
        const size_t off = (kv_slot(pt, p) * nkv + hk) * hd;
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
    float M = sinks ? sinks[h] : -INFINITY;
    for (int i = 0; i < ATT_WARPS; i++) M = fmaxf(M, sm[i]);
    float L = sinks ? expf(sinks[h] - M) : 0.0f;
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
extern "C" __global__ void attention_dec(const float* q, const __half* kc, const __half* vc, const int* pt, float* out,
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
                const size_t off = (kv_slot(pt, p) * nkv + hk) * 128 + 8 * c;
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
extern "C" __global__ void __launch_bounds__(128) attention_tc(const float* q, const __half* kc, const __half* vc, const int* pt, float* out,
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
                const size_t off = (kv_slot(pt, p) * nkv + hk) * 128 + 8 * c;
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

// Combine split-K partials (m, l, acc[hd]) of attention_dec / attention_tc / attention_mla.
// grid T * nh, block hd.
extern "C" __global__ void attn_merge(const float* part, float* out, int rows, int S) {
    const int r = blockIdx.x, d = threadIdx.x, hd = blockDim.x;
    float M = -INFINITY;
    for (int z = 0; z < S; z++) M = fmaxf(M, part[((size_t)z * rows + r) * (hd + 2)]);
    float L = 0.0f, o = 0.0f;
    for (int z = 0; z < S; z++) {
        const float* pp = part + ((size_t)z * rows + r) * (hd + 2);
        if (pp[1] == 0.0f) continue;  // split saw no visible position
        const float c = expf(pp[0] - M);
        L += pp[1] * c;
        o += pp[2 + d] * c;
    }
    out[(size_t)r * hd + d] = o / L;
}

// MLA attention (GLM, src/glm.rs) over the shared latent cache: one f16 row of MLA_ROW per
// position (latent | k_pe); V is its first MLA_LAT. q: T x nh x MLA_ROW (absorbed q_nope | roped
// q_pe), out: T x nh x MLA_LAT. grid (T, S), block 32 * nh: warp h is head h, every head reads the
// same staged tile of MLA_KT cache rows. Lane owns dims lane + 32d. Split z covers positions
// [z * per, (z + 1) * per) of the n_past + t + 1 visible; with S > 1 it writes (m, l, acc) to
// part[((z * T + t) * nh + h) * (MLA_LAT + 2)] for attn_merge.
// ponytail: CUDA cores, one position at a time per warp; a long prefill wants a tensor-core tile.
#define MLA_ROW 576
#define MLA_LAT 512
#define MLA_KT 16
extern "C" __global__ void attention_mla(const float* q, const __half* kc, const int* pt, float* out,
                                         int nh, int n_past, int T, float scale, int per, float* part) {
    __shared__ __align__(16) __half ks[MLA_KT][MLA_ROW];
    const int t = blockIdx.x, z = blockIdx.y, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p0 = z * per, p1 = min(n_past + t + 1, p0 + per);
    const float* qp = q + ((size_t)t * nh + h) * MLA_ROW;
    float qv[MLA_ROW / 32], acc[MLA_LAT / 32];
#pragma unroll
    for (int d = 0; d < MLA_ROW / 32; d++) qv[d] = qp[lane + 32 * d];
#pragma unroll
    for (int d = 0; d < MLA_LAT / 32; d++) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int b = p0; b < p1; b += MLA_KT) {
        const int n = min(MLA_KT, p1 - b);
        __syncthreads();
        for (int i = threadIdx.x; i < n * (MLA_ROW / 8); i += blockDim.x) {
            const int j = i / (MLA_ROW / 8), c = i % (MLA_ROW / 8);
            ((uint4*)ks[j])[c] = ((const uint4*)(kc + kv_slot(pt, b + j) * MLA_ROW))[c];
        }
        __syncthreads();
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < MLA_ROW / 32; d++) s += qv[d] * __half2float(ks[j][lane + 32 * d]);
            s = warp_sum(s) * scale;
            const float mn = fmaxf(m, s), c = expf(m - mn), e = expf(s - mn);
            l = l * c + e;
#pragma unroll
            for (int d = 0; d < MLA_LAT / 32; d++) acc[d] = acc[d] * c + e * __half2float(ks[j][lane + 32 * d]);
            m = mn;
        }
    }
    const size_t row = (size_t)t * nh + h;
    if (gridDim.y == 1) {
#pragma unroll
        for (int d = 0; d < MLA_LAT / 32; d++) out[row * MLA_LAT + lane + 32 * d] = acc[d] / l;
    } else {
        float* pp = part + ((size_t)z * T * nh + row) * (MLA_LAT + 2);
        if (lane == 0) { pp[0] = m; pp[1] = l; }
#pragma unroll
        for (int d = 0; d < MLA_LAT / 32; d++) pp[2 + lane + 32 * d] = acc[d];
    }
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

// gpt-oss clamped SwiGLU: g = min(g, 7), u = clamp(u, -7, 7), out = g * sigmoid(1.702 g) * (u + 1).
extern "C" __global__ void swiglu_oai(float* out, const float* gu, int ff, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * ff) return;
    const int j = i / ff, c = i % ff;
    const float g = fminf(gu[(size_t)j * 2 * ff + c], 7.0f);
    const float u = fminf(fmaxf(gu[(size_t)j * 2 * ff + ff + c], -7.0f), 7.0f);
    out[i] = g / (1.0f + expf(-1.702f * g)) * (u + 1.0f);
}

// y[i] += b[i % n_cols] for n elements (rows of n_cols).
extern "C" __global__ void add_bias(float* y, const float* b, int n_cols, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % n_cols];
}

// Gathered expert rows: y[j][c] += the n_cols f32 at wp[j] + off (the bias in the expert's slot).
// grid rows.
extern "C" __global__ void bias_ids(float* y, const unsigned long long* wp, unsigned long long off, int n_cols) {
    const float* b = (const float*)(wp[blockIdx.x] + off);
    float* yr = y + (size_t)blockIdx.x * n_cols;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) yr[c] += b[c];
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
