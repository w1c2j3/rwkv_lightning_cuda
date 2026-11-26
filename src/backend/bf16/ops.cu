// Training-only BF16 copies of the inference linear/norm/mix semantics.
#ifdef RWKV_USE_HIP
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#endif
#include "rwkv/bf16_training/ops.hpp"
#ifdef RWKV_USE_HIP
#else
#include <cublasLt.h>
#include <cublas_v2.h>
#include <map>
#include <tuple>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
namespace rwkv_bf16_training {
namespace {
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}
#ifdef RWKV_USE_HIP
__device__ __forceinline__ float wave64_sum(float v) {
  v += __shfl_xor(v, 32, 64);
  v += __shfl_xor(v, 16, 64);
  v += __shfl_xor(v, 8, 64);
  v += __shfl_xor(v, 4, 64);
  v += __shfl_xor(v, 2, 64);
  v += __shfl_xor(v, 1, 64);
  return v;
}
struct Bf16x8 { bf16 v[8]; };
__device__ __forceinline__ Bf16x8 ld8(const bf16 *p) {
  union { uint4 u; Bf16x8 b; } r;
  r.u = *reinterpret_cast<const uint4 *>(p);
  return r.b;
}
__device__ __forceinline__ void st8(bf16 *p, Bf16x8 b) {
  union { uint4 u; Bf16x8 b; } r;
  r.b = b;
  *reinterpret_cast<uint4 *>(p) = r.u;
}
__global__ void element_v8(const bf16 *__restrict__ x,
                           const bf16 *__restrict__ y,
                           bf16 *__restrict__ z, long long n8,
                           int C8, int op) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n8) return;
  Bf16x8 xv = ld8(x + i * 8);
  Bf16x8 yv;
  if (op == 0) yv = ld8(y + i * 8);
  else if (op == 4) yv = ld8(y + (i % C8) * 8);
  Bf16x8 out;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float v = to_float(xv.v[k]);
    if (op == 0 || op == 4) v += to_float(yv.v[k]);
    else if (op == 1) v = tanhf(v);
    else if (op == 2) v = 1.0f / (1.0f + expf(-v));
    else { float r = fmaxf(v, 0.0f); v = r * r; }
    out.v[k] = to_bf16(v);
  }
  st8(z + i * 8, out);
}
__global__ void mix_v8(int B, int T, int C, const int *offsets,
                       const std::uint8_t *doc_start, const bf16 *__restrict__ x,
                       const bf16 *__restrict__ shift, const bf16 *__restrict__ w,
                       bf16 *__restrict__ y) {
  const int C8 = C >> 3;
  const size_t i8 = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i8 >= size_t(offsets ? 1 : B) * T * C8) return;
  const int c8 = int(i8 % C8);
  const long long row = static_cast<long long>(i8 / C8);
  const LaneRow at = lane_row(offsets, B, T, row);
  const Bf16x8 xv = ld8(x + i8 * 8);
  const Bf16x8 wv = ld8(w + c8 * 8);
  const bool reset = (doc_start && doc_start[row]);
  Bf16x8 pv;
  if (!reset) {
    pv = !at.first ? ld8(x + (i8 - C8) * 8) : ld8(shift + (size_t(at.lane) * C8 + c8) * 8);
  }
  Bf16x8 out;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float v = to_float(xv.v[k]);
    float p = reset ? 0.0f : to_float(pv.v[k]);
    out.v[k] = to_bf16(v + (p - v) * to_float(wv.v[k]));
  }
  st8(y + i8 * 8, out);
}
__global__ void mix6_v8(int B, int T, int C, const int *offsets,
                        const std::uint8_t *doc_start, const bf16 *__restrict__ x,
                        const bf16 *__restrict__ shift,
                        const bf16 *__restrict__ wr, const bf16 *__restrict__ ww,
                        const bf16 *__restrict__ wk, const bf16 *__restrict__ wv,
                        const bf16 *__restrict__ wa, const bf16 *__restrict__ wg,
                        bf16 *__restrict__ yr, bf16 *__restrict__ yw,
                        bf16 *__restrict__ yk, bf16 *__restrict__ yv,
                        bf16 *__restrict__ ya, bf16 *__restrict__ yg) {
  const int C8 = C >> 3;
  const size_t i8 = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i8 >= size_t(offsets ? 1 : B) * T * C8) return;
  const int c8 = int(i8 % C8);
  const long long row = static_cast<long long>(i8 / C8);
  const LaneRow at = lane_row(offsets, B, T, row);
  const Bf16x8 xv = ld8(x + i8 * 8);
  const bool reset = (doc_start && doc_start[row]);
  Bf16x8 pv;
  if (!reset) {
    pv = !at.first ? ld8(x + (i8 - C8) * 8) : ld8(shift + (size_t(at.lane) * C8 + c8) * 8);
  }
  const Bf16x8 mr = ld8(wr + c8 * 8), mw = ld8(ww + c8 * 8), mk = ld8(wk + c8 * 8);
  const Bf16x8 mv = ld8(wv + c8 * 8), ma = ld8(wa + c8 * 8), mg = ld8(wg + c8 * 8);
  Bf16x8 or_, ow_, ok_, ov_, oa_, og_;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float v = to_float(xv.v[k]);
    float p = reset ? 0.0f : to_float(pv.v[k]);
    float d = p - v;
    or_.v[k] = to_bf16(v + d * to_float(mr.v[k]));
    ow_.v[k] = to_bf16(v + d * to_float(mw.v[k]));
    ok_.v[k] = to_bf16(v + d * to_float(mk.v[k]));
    ov_.v[k] = to_bf16(v + d * to_float(mv.v[k]));
    oa_.v[k] = to_bf16(v + d * to_float(ma.v[k]));
    og_.v[k] = to_bf16(v + d * to_float(mg.v[k]));
  }
  st8(yr + i8 * 8, or_); st8(yw + i8 * 8, ow_); st8(yk + i8 * 8, ok_);
  st8(yv + i8 * 8, ov_); st8(ya + i8 * 8, oa_); st8(yg + i8 * 8, og_);
}
__global__ __launch_bounds__(256) void kk_w64(
    int H, int total_heads, const bf16 *__restrict__ k, const bf16 *__restrict__ kw,
    const bf16 *__restrict__ a0, const bf16 *__restrict__ a12,
    const bf16 *__restrict__ ka, bf16 *__restrict__ nk, bf16 *__restrict__ neg,
    bf16 *__restrict__ kka, bf16 *__restrict__ alpha_out) {
  const int bth = blockIdx.x * 4 + (threadIdx.x >> 6);
  if (bth >= total_heads) return;
  const int j = threadIdx.x & 63;
  const int c = (bth % H) * 64 + j;
  const size_t i = size_t(bth) * 64 + j;
  const float v = to_float(k[i]), u = v * to_float(kw[c]);
  const float inv = 1.0f / fmaxf(sqrtf(wave64_sum(u * u)), 1e-12f);
  const float a = 1.0f / (1.0f + expf(-(to_float(a0[c]) + to_float(a12[i]))));
  if (alpha_out) alpha_out[i] = to_bf16(a);
  const float b = to_float(ka[c]);
  nk[i] = to_bf16(v * (1.0f - b + a * b));
  neg[i] = to_bf16(-u * inv);
  kka[i] = to_bf16(u * inv * a);
}
__global__ __launch_bounds__(256) void post_w64(
    int H, int total_heads, const bf16 *__restrict__ x, const bf16 *__restrict__ r,
    const bf16 *__restrict__ k, const bf16 *__restrict__ v,
    const bf16 *__restrict__ rk, const bf16 *__restrict__ w,
    const bf16 *__restrict__ b, const bf16 *__restrict__ g,
    bf16 *__restrict__ y, bf16 *__restrict__ gn_out) {
  const int bth = blockIdx.x * 4 + (threadIdx.x >> 6);
  if (bth >= total_heads) return;
  const int j = threadIdx.x & 63;
  const int c = (bth % H) * 64 + j;
  const size_t i = size_t(bth) * 64 + j;
  const float a = to_float(x[i]);
  const float q_piece = to_float(r[i]) * to_float(k[i]) * to_float(rk[c]);
  const float mean = wave64_sum(a) * (1.0f / 64.0f);
  const float q = wave64_sum(q_piece);
  const float d = a - mean;
  const float inv = rsqrtf(wave64_sum(d * d) * (1.0f / 64.0f) + 64e-5f);
  const float gn = d * inv * to_float(w[c]) + to_float(b[c]);
  if (gn_out) gn_out[i] = to_bf16(gn);
  y[i] = to_bf16((gn + q * to_float(v[i])) * to_float(g[i]));
}
__device__ __forceinline__ float block_sum_256_hip(float v, float *s) {
#if defined(__GFX8__) || defined(__GFX9__)
  v = wave64_sum(v);
  const int lane = threadIdx.x & 63;
  const int wid = threadIdx.x >> 6;
  if (lane == 0) s[wid] = v;
  __syncthreads();
  float tot = s[0] + s[1] + s[2] + s[3];
  __syncthreads();
  return tot;
#else
  v = warp_sum(v);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) s[wid] = v;
  __syncthreads();
  float tot = (s[0] + s[1]) + (s[2] + s[3]) + (s[4] + s[5]) + (s[6] + s[7]);
  __syncthreads();
  return tot;
#endif
}
__global__ void norm_cached_v8(int C, const bf16 *__restrict__ x,
                               const bf16 *__restrict__ residual,
                               bf16 *__restrict__ sum_out,
                               const bf16 *__restrict__ w,
                               const bf16 *__restrict__ b,
                               bf16 *__restrict__ y, float eps) {
  __shared__ float s[8];
  const int C8 = C >> 3;
  const size_t off8 = size_t(blockIdx.x) * C8;
  float reg[32];
  float v = 0.0f;
  int n = 0;
  for (int i8 = threadIdx.x; i8 < C8; i8 += 256) {
    const Bf16x8 xv = ld8(x + (off8 + i8) * 8);
    if (residual) {
      const Bf16x8 rv = ld8(residual + (off8 + i8) * 8);
      Bf16x8 sv;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const bf16 added = to_bf16(to_float(xv.v[k]) + to_float(rv.v[k]));
        sv.v[k] = added;
        const float val = to_float(added);
        reg[n++] = val;
        v += val;
      }
      if (sum_out)
        st8(sum_out + (off8 + i8) * 8, sv);
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float val = to_float(xv.v[k]);
        reg[n++] = val;
        v += val;
      }
    }
  }
  const float mean = block_sum_256_hip(v, s) / C;
  v = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float d = reg[k] - mean;
    v += d * d;
  }
  const float inv = rsqrtf(block_sum_256_hip(v, s) / C + eps);
  n = 0;
  for (int i8 = threadIdx.x; i8 < C8; i8 += 256) {
    const Bf16x8 wv = ld8(w + i8 * 8);
    const Bf16x8 bv = ld8(b + i8 * 8);
    Bf16x8 yv;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      yv.v[k] = to_bf16((reg[n++] - mean) * inv * to_float(wv.v[k]) +
                        to_float(bv.v[k]));
    }
    st8(y + (off8 + i8) * 8, yv);
  }
}
static inline bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}
#endif
__device__ __forceinline__ float sum64(float v, float *s) {
  __syncthreads();
  int t = threadIdx.x;
  if (t >= 32)
    s[t - 32] = v;
  __syncthreads();
  if (t < 32) {
    v = warp_sum(v + s[t]);
    if (t == 0)
      s[0] = v;
  }
  __syncthreads();
  return s[0];
}
__device__ __forceinline__ float sum256(float v, float *s) {
  __syncthreads();
  int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  if (t < 128)
    s[t] += s[t + 128];
  __syncthreads();
  if (t < 64)
    s[t] += s[t + 64];
  __syncthreads();
  if (t < 32) {
    v = warp_sum(s[t] + s[t + 32]);
    if (t == 0)
      s[0] = v;
  }
  __syncthreads();
  return s[0];
}
__global__ void element(const bf16 *x, const bf16 *y, bf16 *z, long long n,
                        int C, int op) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = to_float(x[i]);
  if (op == 0)
    v += to_float(y[i]);
  else if (op == 1)
    v = tanhf(v);
  else if (op == 2)
    v = 1 / (1 + expf(-v));
  else if (op == 3)
    v = fmaxf(v, 0) * fmaxf(v, 0);
  else
    v += to_float(y[i % C]);
  z[i] = to_bf16(v);
}
__global__ void norm(int C, const bf16 *x, const bf16 *w, const bf16 *b,
                     bf16 *y, float eps) {
  __shared__ float s[256];
  size_t off = size_t(blockIdx.x) * C;
  float v = 0;
  for (int i = threadIdx.x; i < C; i += blockDim.x)
    v += to_float(x[off + i]);
  float mean = sum256(v, s) / C;
  v = 0;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float d = to_float(x[off + i]) - mean;
    v += d * d;
  }
  float inv = rsqrtf(sum256(v, s) / C + eps);
  for (int i = threadIdx.x; i < C; i += blockDim.x)
    y[off + i] = to_bf16((to_float(x[off + i]) - mean) * inv * to_float(w[i]) +
                         to_float(b[i]));
}
// Register-cached LayerNorm with optional fused residual add (sum_out = x + residual)
// for C <= 8192 (<= 32 elements per thread at blockDim=256). Bit-identical to
// separate add + norm while eliminating two extra global-memory passes over x.
__global__ void norm_cached(int C, const bf16 *__restrict__ x,
                            const bf16 *__restrict__ residual,
                            bf16 *__restrict__ sum_out,
                            const bf16 *__restrict__ w,
                            const bf16 *__restrict__ b,
                            bf16 *__restrict__ y, float eps) {
  __shared__ float s[256];
  const size_t off = size_t(blockIdx.x) * C;
  float reg[32];
  float v = 0.0f;
  int n = 0;
  for (int i = threadIdx.x; i < C; i += 256, ++n) {
    float val = to_float(x[off + i]);
    if (residual) {
      const bf16 added = to_bf16(val + to_float(residual[off + i]));
      if (sum_out)
        sum_out[off + i] = added;
      val = to_float(added);
    }
    reg[n] = val;
    v += val;
  }
  const float mean = sum256(v, s) / C;
  v = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float d = reg[k] - mean;
    v += d * d;
  }
  const float inv = rsqrtf(sum256(v, s) / C + eps);
  n = 0;
  for (int i = threadIdx.x; i < C; i += 256, ++n) {
    y[off + i] =
        to_bf16((reg[n] - mean) * inv * to_float(w[i]) + to_float(b[i]));
  }
}
// Shift is updated by a separate kernel, after all t=0 readers finish.
// Packed calls pass offsets with B lanes and T total rows.
__global__ void mix(int B, int T, int C, const int *offsets,
                    const std::uint8_t *doc_start, const bf16 *x,
                    const bf16 *shift, const bf16 *w, bf16 *y) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(offsets ? 1 : B) * T * C)
    return;
  int c = i % C;
  long long row = static_cast<long long>(i / C);
  LaneRow at = lane_row(offsets, B, T, row);
  float v = to_float(x[i]);
  float p = (doc_start && doc_start[row])
                ? 0.0f
                : to_float(!at.first ? x[i - C] : shift[at.lane * C + c]);
  y[i] = to_bf16(v + (p - v) * to_float(w[c]));
}
__global__ void mix6(int B, int T, int C, const int *offsets,
                     const std::uint8_t *doc_start, const bf16 *x,
                     const bf16 *shift, const bf16 *wr, const bf16 *ww,
                     const bf16 *wk, const bf16 *wv, const bf16 *wa,
                     const bf16 *wg, bf16 *yr, bf16 *yw, bf16 *yk, bf16 *yv,
                     bf16 *ya, bf16 *yg) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(offsets ? 1 : B) * T * C)
    return;
  int c = i % C;
  long long row = static_cast<long long>(i / C);
  LaneRow at = lane_row(offsets, B, T, row);
  float v = to_float(x[i]);
  float p = (doc_start && doc_start[row])
                ? 0.0f
                : to_float(!at.first ? x[i - C] : shift[at.lane * C + c]);
  float d = p - v;
  yr[i] = to_bf16(v + d * to_float(wr[c]));
  yw[i] = to_bf16(v + d * to_float(ww[c]));
  yk[i] = to_bf16(v + d * to_float(wk[c]));
  yv[i] = to_bf16(v + d * to_float(wv[c]));
  ya[i] = to_bf16(v + d * to_float(wa[c]));
  yg[i] = to_bf16(v + d * to_float(wg[c]));
}
// An empty packed lane keeps its incoming shift.
__global__ void shift_last(int B, int T, int C, const int *offsets,
                           const bf16 *x, bf16 *s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * C)
    return;
  int b = i / C;
  if (!offsets)
    s[i] = x[(size_t(b) * T + T - 1) * C + i % C];
  else if (offsets[b + 1] > offsets[b])
    s[i] = x[(size_t(offsets[b + 1]) - 1) * C + i % C];
}
__global__ void kk(int H, const bf16 *k, const bf16 *kw, const bf16 *a0,
                   const bf16 *a12, const bf16 *ka, bf16 *nk, bf16 *neg,
                   bf16 *kka, bf16 *alpha_out) {
  __shared__ float s[32];
  int j = threadIdx.x, c = (blockIdx.x % H) * 64 + j;
  size_t i = size_t(blockIdx.x) * 64 + j;
  float v = to_float(k[i]), u = v * to_float(kw[c]);
  float inv = 1 / fmaxf(sqrtf(sum64(u * u, s)), 1e-12f);
  float a = 1 / (1 + expf(-(to_float(a0[c]) + to_float(a12[i]))));
  if (alpha_out)
    alpha_out[i] = to_bf16(a);
  float b = to_float(ka[c]);
  nk[i] = to_bf16(v * (1 - b + a * b));
  neg[i] = to_bf16(-u * inv);
  kka[i] = to_bf16(u * inv * a);
}
__global__ void post(int H, const bf16 *x, const bf16 *r, const bf16 *k,
                     const bf16 *v, const bf16 *rk, const bf16 *w,
                     const bf16 *b, const bf16 *g, bf16 *y, bf16 *gn_out) {
  __shared__ float s[64];
  int j = threadIdx.x, c = (blockIdx.x % H) * 64 + j;
  size_t i = size_t(blockIdx.x) * 64 + j;
  float a = to_float(x[i]);
  float q_piece = to_float(r[i]) * to_float(k[i]) * to_float(rk[c]);
  if (j >= 32) {
    s[j - 32] = a;
    s[j] = q_piece;
  }
  __syncthreads();
  if (j < 32) {
    float sa = warp_sum(a + s[j]);
    float sq = warp_sum(q_piece + s[j + 32]);
    if (j == 0) {
      s[0] = sa;
      s[1] = sq;
    }
  }
  __syncthreads();
  float mean = s[0] * (1.0f / 64.0f);
  float q = s[1];
  float d = a - mean;
  float inv = rsqrtf(sum64(d * d, s) * (1.0f / 64.0f) + 64e-5f);
  float gn = d * inv * to_float(w[c]) + to_float(b[c]);
  if (gn_out)
    gn_out[i] = to_bf16(gn);
  y[i] = to_bf16((gn + q * to_float(v[i])) * to_float(g[i]));
}
thread_local bool consistent_forward = false;
thread_local bool tensor_forward = false;
thread_local int pin_rows = 64;
void gemm_raw(cudaStream_t st, int M, int K, int N, const bf16 *x,
              const bf16 *w, bf16 *y, bool orig, bool accumulate = false) {
  float a = 1, b = accumulate ? 1.0f : 0.0f;
#ifdef RWKV_USE_HIP
  struct HipLtPlan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    bool valid = false;
  };
  struct Handle {
    hipblasHandle_t blas = nullptr;
    hipblasLtHandle_t lt = nullptr;
    DeviceBuffer<uint8_t> ws;
    std::map<std::tuple<int, int, int, bool>, HipLtPlan> cache;
    cudaStream_t last_stream = reinterpret_cast<cudaStream_t>(~uintptr_t(0));
    Handle() {
      if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipblasCreate");
      if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipblasLtCreate");
      ws.resize(64 * 1024 * 1024, "BF16 hipBLASLt workspace");
    }
    ~Handle() {
      for (auto &kv : cache) {
        if (kv.second.lc) hipblasLtMatrixLayoutDestroy(kv.second.lc);
        if (kv.second.lb) hipblasLtMatrixLayoutDestroy(kv.second.lb);
        if (kv.second.la) hipblasLtMatrixLayoutDestroy(kv.second.la);
        if (kv.second.desc) hipblasLtMatmulDescDestroy(kv.second.desc);
      }
      if (lt) hipblasLtDestroy(lt);
      if (blas) hipblasDestroy(blas);
    }
  };
  static thread_local Handle h;
  // Use cached hipBLASLt for large MFMA-friendly shapes (up to +25% over rocBLAS on MI250).
  if (M >= 64 && (K % 8 == 0) && (N % 8 == 0)) {
    auto key = std::make_tuple(M, K, N, orig);
    auto [it, fresh] = h.cache.try_emplace(key);
    HipLtPlan &plan = it->second;
    if (fresh) {
      hipblasLtMatmulPreference_t pref = nullptr;
      const hipblasOperation_t ta = orig ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const hipblasOperation_t tb = HIPBLAS_OP_N;
      const size_t ws_bytes = h.ws.n;
      hipblasLtMatmulHeuristicResult_t res[16]{};
      int found = 0;
      const bool do_tune = !accumulate && (K >= 32768 || N >= 32768) && M >= 1024 && (M % 256 == 0);
      const int req = do_tune ? 16 : 1;
      if (hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatrixLayoutCreate(&plan.la, HIP_R_16BF, orig ? K : N, orig ? N : K, orig ? K : N) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatrixLayoutCreate(&plan.lb, HIP_R_16BF, K, M, K) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatrixLayoutCreate(&plan.lc, HIP_R_16BF, N, M, N) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulAlgoGetHeuristic(h.lt, plan.desc, plan.la, plan.lb, plan.lc, plan.lc, pref, req, res, &found) == HIPBLAS_STATUS_SUCCESS &&
          found > 0) {
        int best = 0;
        if (do_tune && found > 1) {
          float best_ms = 1e30f;
          hipEvent_t e0 = nullptr, e1 = nullptr;
          if (hipEventCreate(&e0) == hipSuccess && hipEventCreate(&e1) == hipSuccess) {
            for (int cand = 0; cand < found; ++cand) {
              if (hipblasLtMatmul(h.lt, plan.desc, &a, w, plan.la, x, plan.lb, &b,
                                  y, plan.lc, y, plan.lc, &res[cand].algo, h.ws.p, h.ws.n, st) != HIPBLAS_STATUS_SUCCESS)
                continue;
              (void)hipStreamSynchronize(st);
              (void)hipEventRecord(e0, st);
              for (int it_t = 0; it_t < 2; ++it_t)
                (void)hipblasLtMatmul(h.lt, plan.desc, &a, w, plan.la, x, plan.lb, &b,
                                      y, plan.lc, y, plan.lc, &res[cand].algo, h.ws.p, h.ws.n, st);
              (void)hipEventRecord(e1, st);
              (void)hipEventSynchronize(e1);
              float ms = 1e30f;
              (void)hipEventElapsedTime(&ms, e0, e1);
              if (ms < best_ms) { best_ms = ms; best = cand; }
            }
          }
          if (e0) (void)hipEventDestroy(e0);
          if (e1) (void)hipEventDestroy(e1);
        }
        plan.algo = res[best].algo;
        plan.valid = true;
      }
      if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    }
    if (plan.valid) {
      auto e = hipblasLtMatmul(h.lt, plan.desc, &a, w, plan.la, x, plan.lb, &b,
                               y, plan.lc, y, plan.lc, &plan.algo, h.ws.p, h.ws.n, st);
      if (e == HIPBLAS_STATUS_SUCCESS) return;
    }
  }
  if (h.last_stream != st) {
    hipblasSetStream(h.blas, st);
    h.last_stream = st;
  }
  auto e = hipblasGemmEx(h.blas, orig ? HIPBLAS_OP_T : HIPBLAS_OP_N, HIPBLAS_OP_N,
                         N, M, K, &a, w, HIPBLAS_R_16B, orig ? K : N, x,
                         HIPBLAS_R_16B, K, &b, y, HIPBLAS_R_16B, N,
                         HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  if (e != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BF16 hipblasGemmEx");
#else
  struct Handle {
    cublasHandle_t h = nullptr;
    DeviceBuffer<uint8_t> ws;
    cudaStream_t last_stream = reinterpret_cast<cudaStream_t>(~uintptr_t(0));
    bool has_last_mode = false;
    bool last_tensor_forward = false;
    Handle() {
      if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasCreate");
      ws.resize(32 * 1024 * 1024, "BF16 cuBLAS workspace");
    }
    ~Handle() {
      if (h)
        cublasDestroy(h);
    }
  };
  static thread_local Handle h;
  if (h.last_stream != st) {
    if (cublasSetStream(h.h, st) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(h.h, h.ws.p, h.ws.n) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("BF16 cublasSetStream/Workspace");
    h.last_stream = st;
  }
  // Keep FP32 reductions while allowing BF16 Tensor Cores in online mode.
  if (!h.has_last_mode || h.last_tensor_forward != tensor_forward) {
    if (cublasSetMathMode(
            h.h, tensor_forward
                     ? CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
                     : CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("BF16 cublasSetMathMode");
    h.has_last_mode = true;
    h.last_tensor_forward = tensor_forward;
  }
  auto e = cublasGemmEx(
      h.h, orig ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &a, w,
      CUDA_R_16BF, orig ? K : N, x, CUDA_R_16BF, K, &b, y, CUDA_R_16BF, N,
      consistent_forward && !tensor_forward ? CUBLAS_COMPUTE_32F_PEDANTIC
                                             : CUBLAS_COMPUTE_32F,
      consistent_forward && !tensor_forward ? CUBLAS_GEMM_DEFAULT
                                             : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("BF16 cublasGemmEx");
#endif
}
#ifndef RWKV_USE_HIP
// Consistent policies pin one cuBLASLt algorithm per (K, N, layout, compute):
// chosen once at pin_rows rows, without workspace, split-K or any reduction
// scheme. Each output row is then computed by the same kernel with the same
// K order whatever M is, so prefill, decode, teacher forcing and packed lanes
// agree bit for bit, and a GEMM reads its weights once instead of per tile.
// An algorithm that cannot run is an error, never a silent fallback.
struct LtCheck {
  void operator()(cublasStatus_t status, const char *what) const {
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(std::string("cuBLASLt ") + what + ": " +
                               std::to_string(int(status)));
  }
} lt_check;
struct LtLayouts {
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  // Row-major y[M,N] = x[M,K] W as column-major y^T = W^T x^T.
  LtLayouts(int M, int K, int N, bool orig) {
    lt_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, orig ? K : N,
                                        orig ? N : K, orig ? K : N),
             "weight layout");
    lt_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, K, M, K),
             "input layout");
    lt_check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, M, N),
             "output layout");
  }
  ~LtLayouts() {
    for (auto layout : {a, b, c})
      if (layout)
        cublasLtMatrixLayoutDestroy(layout);
  }
};
struct LtGemm {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatmulAlgo_t algo{};
};
void gemm_pinned(cudaStream_t st, int M, int K, int N, const bf16 *x,
                 const bf16 *w, bf16 *y, bool orig, bool accumulate = false) {
  // Pinned algorithms assume 16-byte aligned rows (every model width is a
  // multiple of 8 BF16 values).
  if (K % 8 || N % 8 ||
      (reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(w) |
       reinterpret_cast<std::uintptr_t>(y)) % 16)
    throw std::invalid_argument("consistent GEMM needs 16-byte aligned rows");
  struct Handle {
    cublasLtHandle_t h = nullptr;
    std::map<std::tuple<int, int, bool, bool, int>, LtGemm> pinned;
    Handle() { lt_check(cublasLtCreate(&h), "create"); }
    ~Handle() {
      for (auto &item : pinned)
        cublasLtMatmulDescDestroy(item.second.desc);
      cublasLtDestroy(h);
    }
  };
  static thread_local Handle h;
  auto [it, fresh] =
      h.pinned.try_emplace(std::make_tuple(K, N, orig, tensor_forward, pin_rows));
  LtGemm &g = it->second;
  if (fresh) {
    try {
      lt_check(cublasLtMatmulDescCreate(&g.desc,
                                        tensor_forward
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_32F_PEDANTIC,
                                        CUDA_R_32F),
               "matmul descriptor");
      const cublasOperation_t ta = orig ? CUBLAS_OP_T : CUBLAS_OP_N,
                              tb = CUBLAS_OP_N;
      lt_check(cublasLtMatmulDescSetAttribute(
                   g.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta),
               "transpose A");
      lt_check(cublasLtMatmulDescSetAttribute(
                   g.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb),
               "transpose B");
      cublasLtMatmulPreference_t pref = nullptr;
      lt_check(cublasLtMatmulPreferenceCreate(&pref), "preference");
      const size_t workspace = 0;
      const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE,
                     alignment = 16;
      cublasLtMatmulHeuristicResult_t result{};
      int found = 0;
      cublasStatus_t status =
          cublasLtMatmulPreferenceSetAttribute(
              pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
              sizeof workspace);
      for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        if (status == CUBLAS_STATUS_SUCCESS)
          status = cublasLtMatmulPreferenceSetAttribute(
              pref, attribute, &alignment, sizeof alignment);
      if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
            sizeof reduction);
      if (status == CUBLAS_STATUS_SUCCESS) {
        LtLayouts l(pin_rows, K, N, orig);
        status = cublasLtMatmulAlgoGetHeuristic(h.h, g.desc, l.a, l.b, l.c,
                                                l.c, pref, 1, &result, &found);
      }
      cublasLtMatmulPreferenceDestroy(pref);
      lt_check(status, "algorithm heuristic");
      if (!found)
        throw std::runtime_error("cuBLASLt found no pinned BF16 algorithm");
      g.algo = result.algo;
    } catch (...) {
      if (g.desc)
        cublasLtMatmulDescDestroy(g.desc);
      h.pinned.erase(it);
      throw;
    }
  }
  const float one = 1, beta = accumulate ? 1.0f : 0.0f;
  LtLayouts l(M, K, N, orig);
  lt_check(cublasLtMatmul(h.h, g.desc, &one, w, l.a, x, l.b, &beta, y, l.c, y,
                          l.c, &g.algo, nullptr, 0, st),
           "pinned BF16 matmul");
}
#endif
#ifdef RWKV_USE_HIP
// Pinned zero-workspace hipBLASLt path for consistent BF16 policy on AMD GPUs:
// selects one split-K-free algorithm per (K, N, orig, pin_rows) at workspace=0
// so every row executes the same K reduction independent of M, matching CUDA's
// gemm_pinned at full MFMA speed. Returns false if unsupported so the caller
// can fall back to fixed 32-row tiling.
struct HipLtPinnedLayouts {
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  bool ok = false;
  HipLtPinnedLayouts(int M, int K, int N, bool orig) {
    ok = hipblasLtMatrixLayoutCreate(&a, HIP_R_16BF, orig ? K : N,
                                     orig ? N : K, orig ? K : N) ==
             HIPBLAS_STATUS_SUCCESS &&
         hipblasLtMatrixLayoutCreate(&b, HIP_R_16BF, K, M, K) ==
             HIPBLAS_STATUS_SUCCESS &&
         hipblasLtMatrixLayoutCreate(&c, HIP_R_16BF, N, M, N) ==
             HIPBLAS_STATUS_SUCCESS;
  }
  ~HipLtPinnedLayouts() {
    if (c) hipblasLtMatrixLayoutDestroy(c);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (a) hipblasLtMatrixLayoutDestroy(a);
  }
};
// One-time bitwise row-invariance self-test for a zero-workspace hipBLASLt
// candidate algorithm: verifies M=1, M=255 (odd), and M=510 (even) all succeed
// and produce bit-identical outputs for every row. Automatically rejects
// Stream-K / GSUAMB split-K kernels whose K-reduction order varies across M.
static bool verify_hiplt_row_invariant(hipblasLtHandle_t lt,
                                       hipblasLtMatmulDesc_t desc,
                                       const hipblasLtMatmulAlgo_t &algo,
                                       int K, int N, bool orig,
                                       cudaStream_t st) {
  constexpr int kHalf = 255;
  constexpr int kFull = 510;
  const size_t x_elems = size_t(kFull) * size_t(K);
  const size_t w_elems = size_t(K) * size_t(N);
  const size_t y_elems = size_t(kFull) * size_t(N);
  bf16 *dx = nullptr, *dw = nullptr, *dy_split = nullptr, *dy_full = nullptr,
       *dy_m1 = nullptr;
  auto cleanup = [&]() {
    if (dy_m1) (void)hipFree(dy_m1);
    if (dy_full) (void)hipFree(dy_full);
    if (dy_split) (void)hipFree(dy_split);
    if (dw) (void)hipFree(dw);
    if (dx) (void)hipFree(dx);
  };
  if (hipMalloc(&dx, x_elems * 2) != hipSuccess ||
      hipMalloc(&dw, w_elems * 2) != hipSuccess ||
      hipMalloc(&dy_split, y_elems * 2) != hipSuccess ||
      hipMalloc(&dy_full, y_elems * 2) != hipSuccess ||
      hipMalloc(&dy_m1, size_t(N) * 2) != hipSuccess) {
    cleanup();
    return false;
  }
  std::vector<bf16> hx(x_elems), hw(w_elems);
  uint32_t rng = 0x9e3779b9u ^ uint32_t(K * 131 + N * 17 + (orig ? 1 : 0));
  auto next_bf16 = [&]() -> bf16 {
    rng = rng * 1664525u + 1013904223u;
    const float v = (float(int(rng >> 16) - 32768) / 32768.0f) * 0.5f;
    return to_bf16(v);
  };
  for (size_t i = 0; i < x_elems; ++i) hx[i] = next_bf16();
  for (size_t i = 0; i < w_elems; ++i) hw[i] = next_bf16();
  if (hipMemcpyAsync(dx, hx.data(), x_elems * 2, hipMemcpyHostToDevice, st) !=
          hipSuccess ||
      hipMemcpyAsync(dw, hw.data(), w_elems * 2, hipMemcpyHostToDevice, st) !=
          hipSuccess) {
    cleanup();
    return false;
  }
  const float one = 1.0f, zero = 0.0f;
  HipLtPinnedLayouts l1(1, K, N, orig);
  HipLtPinnedLayouts lh(kHalf, K, N, orig);
  HipLtPinnedLayouts lf(kFull, K, N, orig);
  if (!l1.ok || !lh.ok || !lf.ok) {
    cleanup();
    return false;
  }
  const bool ran_ok =
      hipblasLtMatmul(lt, desc, &one, dw, lh.a, dx, lh.b, &zero, dy_split,
                      lh.c, dy_split, lh.c, &algo, nullptr, 0, st) ==
          HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmul(lt, desc, &one, dw, lh.a, dx + size_t(kHalf) * K, lh.b,
                      &zero, dy_split + size_t(kHalf) * N, lh.c,
                      dy_split + size_t(kHalf) * N, lh.c, &algo, nullptr, 0,
                      st) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmul(lt, desc, &one, dw, lf.a, dx, lf.b, &zero, dy_full, lf.c,
                      dy_full, lf.c, &algo, nullptr, 0, st) ==
          HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmul(lt, desc, &one, dw, l1.a, dx, l1.b, &zero, dy_m1, l1.c,
                      dy_m1, l1.c, &algo, nullptr, 0, st) ==
          HIPBLAS_STATUS_SUCCESS;
  if (!ran_ok || hipStreamSynchronize(st) != hipSuccess) {
    cleanup();
    return false;
  }
  std::vector<uint16_t> h_split(y_elems), h_full(y_elems), h_m1(static_cast<size_t>(N));
  if (hipMemcpy(h_split.data(), dy_split, y_elems * 2, hipMemcpyDeviceToHost) !=
          hipSuccess ||
      hipMemcpy(h_full.data(), dy_full, y_elems * 2, hipMemcpyDeviceToHost) !=
          hipSuccess ||
      hipMemcpy(h_m1.data(), dy_m1, size_t(N) * 2, hipMemcpyDeviceToHost) !=
          hipSuccess) {
    cleanup();
    return false;
  }
  cleanup();
  return std::equal(h_split.begin(), h_split.end(), h_full.begin()) &&
         std::equal(h_m1.begin(), h_m1.end(), h_full.begin());
}

bool gemm_pinned_hip(cudaStream_t st, int M, int K, int N, const bf16 *x,
                     const bf16 *w, bf16 *y, bool orig, bool accumulate) {
  if (K < 64 || N < 64 || (K & 7) != 0 || (N & 7) != 0 ||
      !aligned16(x) || !aligned16(w) || !aligned16(y))
    return false;
  struct PinnedPlan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    bool valid = false;
  };
  struct PinnedHandle {
    hipblasLtHandle_t lt = nullptr;
    std::map<std::tuple<int, int, bool, int>, PinnedPlan> cache;
    PinnedHandle() {
      if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hipblasLtCreate (pinned)");
    }
    ~PinnedHandle() {
      for (auto &kv : cache)
        if (kv.second.desc)
          hipblasLtMatmulDescDestroy(kv.second.desc);
      if (lt)
        hipblasLtDestroy(lt);
    }
  };
  static thread_local PinnedHandle h;
  auto [it, fresh] =
      h.cache.try_emplace(std::make_tuple(K, N, orig, pin_rows));
  PinnedPlan &plan = it->second;
  if (fresh) {
    hipblasLtMatmulPreference_t pref = nullptr;
    const hipblasOperation_t ta = orig ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipblasOperation_t tb = HIPBLAS_OP_N;
    const uint64_t zero_ws = 0;
    const int query_m = std::max(pin_rows, 2048);
    hipblasLtMatmulHeuristicResult_t res[32]{};
    int found = 0;
    HipLtPinnedLayouts lpin(query_m, K, N, orig);
    if (lpin.ok &&
        hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F,
                                  HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &ta, sizeof(ta)) ==
            HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &tb, sizeof(tb)) ==
            HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &zero_ws,
            sizeof(zero_ws)) == HIPBLAS_STATUS_SUCCESS &&
        hipblasLtMatmulAlgoGetHeuristic(h.lt, plan.desc, lpin.a, lpin.b, lpin.c,
                                        lpin.c, pref, 32, res, &found) ==
            HIPBLAS_STATUS_SUCCESS &&
        found > 0) {
      // Only probe up to 8 zero-workspace candidates so startup stays instant.
      int tested = 0;
      for (int c = 0; c < found && tested < 8; ++c) {
        if (res[c].state != HIPBLAS_STATUS_SUCCESS ||
            res[c].workspaceSize != 0)
          continue;
        ++tested;
        if (verify_hiplt_row_invariant(h.lt, plan.desc, res[c].algo, K, N, orig,
                                       st)) {
          plan.algo = res[c].algo;
          plan.valid = true;
          break;
        }
      }
    }
    if (pref)
      hipblasLtMatmulPreferenceDestroy(pref);
    if (!plan.valid && plan.desc) {
      hipblasLtMatmulDescDestroy(plan.desc);
      plan.desc = nullptr;
    }
  }
  if (!plan.valid)
    return false;
  HipLtPinnedLayouts l(M, K, N, orig);
  if (!l.ok)
    return false;
  const float one = 1.0f, beta = accumulate ? 1.0f : 0.0f;
  return hipblasLtMatmul(h.lt, plan.desc, &one, w, l.a, x, l.b, &beta, y, l.c,
                         y, l.c, &plan.algo, nullptr, 0, st) ==
         HIPBLAS_STATUS_SUCCESS;
}
#endif
// Online BF16 and strict policies make each row independent of the GEMM row
// count (CUDA: pinned cuBLASLt algorithm; HIP: pinned zero-workspace hipBLASLt
// for bf16 mode with fixed 32-row tiles for strict mode and fallback) to keep
// prefill, decode and teacher forcing consistent. WKV always sees the true
// sequence length. The default policy lets cuBLAS choose per call.
void gemm(cudaStream_t st, int M, int K, int N, const bf16 *x, const bf16 *w,
          bf16 *y, bool orig, bool accumulate = false) {
  if (!consistent_forward) {
    gemm_raw(st, M, K, N, x, w, y, orig, accumulate);
    return;
  }
#ifndef RWKV_USE_HIP
  gemm_pinned(st, M, K, N, x, w, y, orig, accumulate);
#else
  if (tensor_forward &&
      gemm_pinned_hip(st, M, K, N, x, w, y, orig, accumulate)) {
    return;
  }
  constexpr int tile = 32;
  thread_local DeviceBuffer<bf16> padded_x, padded_y;
  for (int row = 0; row < M; row += tile) {
    int count = std::min(tile, M - row);
    // Reuse staging buffers sized for the largest linear width.
    if (padded_x.n < size_t(tile) * K)
      padded_x.resize(size_t(tile) * K, "BF16 fixed-tile input");
    if (padded_y.n < size_t(tile) * N)
      padded_y.resize(size_t(tile) * N, "BF16 fixed-tile output");
    gpu_check(cudaMemcpyAsync(padded_x.p, x + size_t(row) * K,
                              size_t(count) * K * 2, cudaMemcpyDeviceToDevice,
                              st),
              "pad GEMM input");
    if (count < tile)
      gpu_check(cudaMemsetAsync(padded_x.p + size_t(count) * K, 0,
                                size_t(tile - count) * K * 2, st),
                "zero GEMM padding");
    if (accumulate)
      gpu_check(cudaMemcpyAsync(padded_y.p, y + size_t(row) * N,
                                size_t(count) * N * 2, cudaMemcpyDeviceToDevice,
                                st),
                "load GEMM accum");
    gemm_raw(st, tile, K, N, padded_x.p, w, padded_y.p, orig, accumulate);
    gpu_check(cudaMemcpyAsync(y + size_t(row) * N, padded_y.p,
                              size_t(count) * N * 2, cudaMemcpyDeviceToDevice,
                              st),
              "copy GEMM output");
  }
#endif
}
} // namespace
void set_consistent_forward(bool enabled, bool tensor_cores, int rows) {
  if (rows < 1)
    throw std::invalid_argument("pinned GEMM rows must be positive");
  consistent_forward = enabled;
  tensor_forward = tensor_cores;
  pin_rows = rows;
}
bool consistent_forward_enabled() { return consistent_forward; }
void rwkv7_v3a_linear_bf16_launch(cudaStream_t s, int M, int K, int N,
                                  const bf16 *x, const bf16 *w, bf16 *y) {
  gemm(s, M, K, N, x, w, y, false, false);
}
void rwkv7_v3a_linear_bf16_accumulate_launch(cudaStream_t s, int M, int K,
                                             int N, const bf16 *x,
                                             const bf16 *w, bf16 *y) {
  gemm(s, M, K, N, x, w, y, false, true);
}
void rwkv7_v3a_linear_bf16_orig_launch(cudaStream_t s, int M, int K, int N,
                                       const bf16 *x, const bf16 *w, bf16 *y) {
  gemm(s, M, K, N, x, w, y, true, false);
}
void rwkv7_v3a_linear_bf16_orig_accumulate_launch(cudaStream_t s, int M, int K,
                                                  int N, const bf16 *x,
                                                  const bf16 *w, bf16 *y) {
  gemm(s, M, K, N, x, w, y, true, true);
}
void rwkv7_v3a_layer_norm_bf16_launch(cudaStream_t s, int rows, int C,
                                      const bf16 *x, const bf16 *w,
                                      const bf16 *b, bf16 *y, float eps) {
#ifdef RWKV_USE_HIP
  if (C <= 8192 && (C & 7) == 0 && aligned16(x) && aligned16(w) &&
      aligned16(b) && aligned16(y)) {
    norm_cached_v8<<<rows, 256, 0, s>>>(C, x, nullptr, nullptr, w, b, y, eps);
    return;
  }
#endif
  if (C <= 8192)
    norm_cached<<<rows, 256, 0, s>>>(C, x, nullptr, nullptr, w, b, y, eps);
  else
    norm<<<rows, 256, 0, s>>>(C, x, w, b, y, eps);
}
void rwkv7_add_layer_norm_bf16_launch(cudaStream_t s, int rows, int C,
                                      const bf16 *a, const bf16 *b_in,
                                      bf16 *sum_out, const bf16 *w,
                                      const bf16 *bias, bf16 *y, float eps) {
#ifdef RWKV_USE_HIP
  if (C <= 8192 && (C & 7) == 0 && aligned16(a) && aligned16(b_in) &&
      (!sum_out || aligned16(sum_out)) && aligned16(w) && aligned16(bias) &&
      aligned16(y)) {
    norm_cached_v8<<<rows, 256, 0, s>>>(C, a, b_in, sum_out, w, bias, y, eps);
    return;
  }
#endif
  if (C <= 8192) {
    norm_cached<<<rows, 256, 0, s>>>(C, a, b_in, sum_out, w, bias, y, eps);
  } else {
    rwkv7_v3a_add_bf16_launch(s, a, b_in, sum_out, static_cast<long long>(rows) * C);
    norm<<<rows, 256, 0, s>>>(C, sum_out, w, bias, y, eps);
  }
}
void rwkv7_v3a_add_bf16_launch(cudaStream_t s, const bf16 *x, const bf16 *y,
                               bf16 *z, long long n) {
#ifdef RWKV_USE_HIP
  if ((n & 7) == 0 && aligned16(x) && aligned16(y) && aligned16(z)) {
    const long long n8 = n >> 3;
    element_v8<<<(n8 + 255) / 256, 256, 0, s>>>(x, y, z, n8, 0, 0);
    return;
  }
#endif
  element<<<(n + 255) / 256, 256, 0, s>>>(x, y, z, n, 0, 0);
}
void rwkv7_act_tanh_launch(cudaStream_t s, const bf16 *x, bf16 *y,
                           long long n) {
#ifdef RWKV_USE_HIP
  if ((n & 7) == 0 && aligned16(x) && aligned16(y)) {
    const long long n8 = n >> 3;
    element_v8<<<(n8 + 255) / 256, 256, 0, s>>>(x, nullptr, y, n8, 0, 1);
    return;
  }
#endif
  element<<<(n + 255) / 256, 256, 0, s>>>(x, nullptr, y, n, 0, 1);
}
void rwkv7_act_sigmoid_launch(cudaStream_t s, const bf16 *x, bf16 *y,
                              long long n) {
#ifdef RWKV_USE_HIP
  if ((n & 7) == 0 && aligned16(x) && aligned16(y)) {
    const long long n8 = n >> 3;
    element_v8<<<(n8 + 255) / 256, 256, 0, s>>>(x, nullptr, y, n8, 0, 2);
    return;
  }
#endif
  element<<<(n + 255) / 256, 256, 0, s>>>(x, nullptr, y, n, 0, 2);
}
void rwkv7_relu_square_launch(cudaStream_t s, const bf16 *x, bf16 *y,
                              long long n) {
#ifdef RWKV_USE_HIP
  if ((n & 7) == 0 && aligned16(x) && aligned16(y)) {
    const long long n8 = n >> 3;
    element_v8<<<(n8 + 255) / 256, 256, 0, s>>>(x, nullptr, y, n8, 0, 3);
    return;
  }
#endif
  element<<<(n + 255) / 256, 256, 0, s>>>(x, nullptr, y, n, 0, 3);
}
void rwkv7_add_vec_launch(cudaStream_t s, int C, const bf16 *x, const bf16 *v,
                          bf16 *y, long long n) {
#ifdef RWKV_USE_HIP
  if ((n & 7) == 0 && (C & 7) == 0 && aligned16(x) && aligned16(v) && aligned16(y)) {
    const long long n8 = n >> 3;
    element_v8<<<(n8 + 255) / 256, 256, 0, s>>>(x, v, y, n8, C >> 3, 4);
    return;
  }
#endif
  element<<<(n + 255) / 256, 256, 0, s>>>(x, v, y, n, C, 4);
}
void rwkv7_cmix_mix_launch(cudaStream_t s, int B, int T, int C, const bf16 *x,
                           bf16 *sh, const bf16 *w, bf16 *y,
                           const int *offsets, const std::uint8_t *doc_start) {
  const size_t n = size_t(offsets ? 1 : B) * T * C;
#ifdef RWKV_USE_HIP
  if ((C & 7) == 0 && aligned16(x) && aligned16(sh) && aligned16(w) && aligned16(y)) {
    const size_t n8 = n >> 3;
    mix_v8<<<(n8 + 255) / 256, 256, 0, s>>>(B, T, C, offsets, doc_start, x, sh, w, y);
  } else
#endif
  mix<<<(n + 255) / 256, 256, 0, s>>>(B, T, C, offsets, doc_start, x, sh, w, y);
  shift_last<<<(B * C + 255) / 256, 256, 0, s>>>(B, T, C, offsets, x, sh);
}
void rwkv7_tmix_mix6_launch(cudaStream_t s, int B, int T, int C, const bf16 *x,
                            bf16 *sh, const bf16 *wr, const bf16 *ww,
                            const bf16 *wk, const bf16 *wv, const bf16 *wa,
                            const bf16 *wg, bf16 *r, bf16 *w, bf16 *k, bf16 *v,
                            bf16 *a, bf16 *g, const int *offsets,
                            const std::uint8_t *doc_start) {
  const size_t n = size_t(offsets ? 1 : B) * T * C;
#ifdef RWKV_USE_HIP
  if ((C & 7) == 0 && aligned16(x) && aligned16(sh) && aligned16(wr) &&
      aligned16(ww) && aligned16(wk) && aligned16(wv) && aligned16(wa) &&
      aligned16(wg) && aligned16(r) && aligned16(w) && aligned16(k) &&
      aligned16(v) && aligned16(a) && aligned16(g)) {
    const size_t n8 = n >> 3;
    mix6_v8<<<(n8 + 255) / 256, 256, 0, s>>>(B, T, C, offsets, doc_start, x, sh, wr,
                                             ww, wk, wv, wa, wg, r, w, k, v, a, g);
  } else
#endif
  mix6<<<(n + 255) / 256, 256, 0, s>>>(B, T, C, offsets, doc_start, x, sh, wr,
                                       ww, wk, wv, wa, wg, r, w, k, v, a, g);
  shift_last<<<(B * C + 255) / 256, 256, 0, s>>>(B, T, C, offsets, x, sh);
}
void rwkv7_tmix_kk_a_gate_launch(cudaStream_t s, int B, int T, int C, int H,
                                 const bf16 *k, const bf16 *kw, const bf16 *a0,
                                 const bf16 *a12, const bf16 *ka, bf16 *nk,
                                 bf16 *neg, bf16 *kka, bf16 *alpha_out) {
#ifdef RWKV_USE_HIP
  if (is_wave64_device()) {
    const int total_heads = B * T * H;
    kk_w64<<<(total_heads + 3) / 4, 256, 0, s>>>(H, total_heads, k, kw, a0, a12, ka, nk, neg, kka, alpha_out);
    return;
  }
#endif
  kk<<<B * T * H, 64, 0, s>>>(H, k, kw, a0, a12, ka, nk, neg, kka, alpha_out);
}
void rwkv7_tmix_lnx_rkvres_xg_launch(cudaStream_t s, int B, int T, int C, int H,
                                     const bf16 *x, const bf16 *r,
                                     const bf16 *k, const bf16 *v,
                                     const bf16 *rk, const bf16 *w,
                                     const bf16 *b, const bf16 *g, bf16 *y,
                                     bf16 *gn_out) {
#ifdef RWKV_USE_HIP
  if (is_wave64_device()) {
    const int total_heads = B * T * H;
    post_w64<<<(total_heads + 3) / 4, 256, 0, s>>>(H, total_heads, x, r, k, v, rk, w, b, g, y, gn_out);
    return;
  }
#endif
  post<<<B * T * H, 64, 0, s>>>(H, x, r, k, v, rk, w, b, g, y, gn_out);
}
} // namespace rwkv_bf16_training
namespace rwkv_bf16_training {
__global__ void transpose_kernel(int rows, int cols, const bf16 *x, bf16 *y) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < size_t(rows) * cols)
    y[(i % cols) * rows + i / cols] = x[i];
}
__global__ void gather_rows_kernel(int active_rows, int C, const bf16 *src,
                                   const int *row_indices, bf16 *dst) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(active_rows) * C)
    return;
  int r = int(i / C), c = int(i % C);
  dst[i] = src[size_t(row_indices[r]) * C + c];
}
__global__ void scatter_rows_kernel(int active_rows, int C, const bf16 *src,
                                    const int *row_indices, bf16 *dst) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= size_t(active_rows) * C)
    return;
  int r = int(i / C), c = int(i % C);
  dst[size_t(row_indices[r]) * C + c] = src[i];
}
void rwkv7_gather_rows_bf16_launch(cudaStream_t stream, int active_rows, int C,
                                   const bf16 *src, const int *row_indices,
                                   bf16 *dst) {
  size_t n = size_t(active_rows) * C;
  if (n > 0)
    gather_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        active_rows, C, src, row_indices, dst);
}
void rwkv7_scatter_rows_bf16_launch(cudaStream_t stream, int total_rows,
                                    int active_rows, int C, const bf16 *src,
                                    const int *row_indices, bf16 *dst) {
  gpu_check(cudaMemsetAsync(dst, 0, size_t(total_rows) * C * sizeof(bf16),
                            stream),
            "zero scattered rows");
  size_t n = size_t(active_rows) * C;
  if (n > 0)
    scatter_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        active_rows, C, src, row_indices, dst);
}
void transpose_bf16(int rows, int cols, const bf16 *x, bf16 *y) {
  transpose_kernel<<<(size_t(rows) * cols + 255) / 256, 256>>>(rows, cols, x,
                                                               y);
  gpu_check(cudaDeviceSynchronize(), "transpose BF16 weight");
}
} // namespace rwkv_bf16_training
