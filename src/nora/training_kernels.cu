#ifdef RWKV_USE_HIP
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#endif
#include "rwkv/bf16_training/nora.hpp"
#include "rwkv/bf16_training/ops.hpp"
#include "rwkv/runtime/nora_gemm.cuh"
#include "training_gemm.cuh"
#include <cmath>
#include <stdexcept>
#include <type_traits>
#ifndef RWKV_USE_HIP
#include <cublas_v2.h>
#endif
namespace rwkv_bf16_nora {
namespace {
void validate(int rows, const rwkv_bf16_miss::Linear &w) {
  if (!w.a || !w.d || rows < 1 || rows > 65535 || w.in < 1 || w.out < 1 ||
      w.rank < 1 || w.rank > 1024 || !std::isfinite(w.scale))
    throw std::invalid_argument("invalid NoRA shape/weights");
}
void checked() {
  rwkv_bf16_training::gpu_check(cudaGetLastError(), "NoRA kernel");
}
// Tensor-core / MFMA Matrix-core path, used when the thread's GEMM policy is not consistent
// (--gemm-mode auto): cuBLAS BF16 products with FP32 accumulation instead of
// the FP32 SIMT kernels. An FP32 latent (or its adjoint) enters a product as
// the exact BF16 pair hi + lo, about 16 mantissa bits, so results stay within
// ~1e-5 of the FP32 kernels. Where the latent is the reduced (rank) operand,
// [hi|lo] is one 2*rank-wide operand against a weight repeated twice, giving
// one product and one BF16 rounding of y or dx. Consistent policies keep the
// FP32 kernels, whose rows do not depend on the packed row count.
__global__ void split_latent(size_t n, int rank, const float *x, bf16 *y) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const size_t row = i / rank, j = i % rank;
  const bf16 hi = to_bf16(x[i]);
  y[row * 2 * rank + j] = hi;
  y[row * 2 * rank + rank + j] = to_bf16(x[i] - to_float(hi));
}
__global__ void repeat_weight_kernel(size_t total, int r, int c, bool side,
                                     const bf16 *__restrict__ w,
                                     bf16 *__restrict__ y) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  if (side) {
    const size_t row = i / (2 * size_t(c));
    const size_t col = i % size_t(c);
    y[i] = w[row * size_t(c) + col];
  } else {
    y[i] = w[i % (size_t(r) * c)];
  }
}
__global__ void accum_grad_b_pair(int out, int rank, const float *__restrict__ gp,
                                  float *__restrict__ dst) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = size_t(out) * rank;
  if (i >= total)
    return;
  const size_t row = i / rank, j = i % rank;
  dst[i] += gp[row * 2 * rank + j] + gp[row * 2 * rank + rank + j];
}
__global__ void accum_grad_a_pair(int rank, int in, const float *__restrict__ gp,
                                  float *__restrict__ dst) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = size_t(rank) * in;
  if (i >= total)
    return;
  dst[i] += gp[i] + gp[total + i];
}
#ifdef RWKV_USE_HIP
struct NoraHipLtPlan {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  bool valid = false;
};
#endif
struct Blas {
#ifdef RWKV_USE_HIP
  hipblasHandle_t h = nullptr;
  hipblasLtHandle_t lt = nullptr;
  rwkv_bf16_training::DeviceBuffer<uint8_t> ws;
  std::map<std::tuple<int, int, int, int, int, bool, bool, bool>, NoraHipLtPlan> lt_cache;
#else
  cublasHandle_t h = nullptr;
  rwkv_bf16_training::DeviceBuffer<uint8_t> ws;
#endif
  rwkv_bf16_training::DeviceBuffer<float> latent, grad_pair;
  rwkv_bf16_training::DeviceBuffer<bf16> pair, twice;
  Blas() {
#ifdef RWKV_USE_HIP
    if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("NoRA hipblasCreate");
    if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("NoRA hipblasLtCreate");
    ws.resize(32 * 1024 * 1024, "NoRA hipBLASLt workspace");
#else
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("NoRA cublasCreate");
    ws.resize(32 * 1024 * 1024, "NoRA cuBLAS workspace");
    if (cublasSetWorkspace(h, ws.p, ws.n) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("NoRA cublasSetWorkspace");
#endif
  }
  ~Blas() {
#ifdef RWKV_USE_HIP
    for (auto &kv : lt_cache) {
      if (kv.second.lc) hipblasLtMatrixLayoutDestroy(kv.second.lc);
      if (kv.second.lb) hipblasLtMatrixLayoutDestroy(kv.second.lb);
      if (kv.second.la) hipblasLtMatrixLayoutDestroy(kv.second.la);
      if (kv.second.desc) hipblasLtMatmulDescDestroy(kv.second.desc);
    }
    if (lt) hipblasLtDestroy(lt);
    if (h) hipblasDestroy(h);
#else
    if (h) cublasDestroy(h);
#endif
  }
  cudaStream_t last_stream = reinterpret_cast<cudaStream_t>(~uintptr_t(0));
  template <class T>
  T *grow(rwkv_bf16_training::DeviceBuffer<T> &b, size_t n) {
    if (b.n < n)
      b.resize(n, "NoRA tensor-core scratch");
    return b.p;
  }
};
Blas &blas(cudaStream_t stream) {
  static thread_local Blas b;
  if (b.last_stream != stream) {
#ifdef RWKV_USE_HIP
    if (hipblasSetStream(b.h, stream) != HIPBLAS_STATUS_SUCCESS)
      throw std::runtime_error("NoRA hipblasSetStream");
#else
    if (cublasSetStream(b.h, stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(b.h, b.ws.p, b.ws.n) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("NoRA cublasSetStream/Workspace");
#endif
    b.last_stream = stream;
  }
  return b;
}
bool tensor_cores(const rwkv_bf16_miss::Linear &w) {
  return !rwkv_bf16_training::consistent_forward_enabled() && w.in >= 64 &&
         (w.in & 63) == 0 && w.out >= 64 && (w.out & 63) == 0 &&
         (w.rank & 7) == 0;
}
// Row-major C[M,N] = alpha op(A)[M,K] op(B)[K,N] + beta C (leading dimensions
// lda, ldb in elements), on column-major BLAS as C^T = op(B)^T op(A)^T.
template <class C>
void product(cudaStream_t stream, Blas &bl,
             bool ta, bool tb, int M, int N, int K,
             float alpha, const bf16 *a, int lda, const bf16 *b, int ldb,
             float beta, C *c) {
#ifdef RWKV_USE_HIP
  constexpr bool out_f32 = std::is_same_v<C, float>;
  if (M >= 64 && N >= 64 && K >= 64 && (M & 7) == 0 && (N & 7) == 0 &&
      (K & 7) == 0 && ((reinterpret_cast<uintptr_t>(a) |
                        reinterpret_cast<uintptr_t>(b) |
                        reinterpret_cast<uintptr_t>(c)) & 15u) == 0) {
    auto key = std::make_tuple(M, N, K, lda, ldb, ta, tb, out_f32);
    auto [it, fresh] = bl.lt_cache.try_emplace(key);
    NoraHipLtPlan &plan = it->second;
    if (fresh) {
      hipblasLtMatmulPreference_t pref = nullptr;
      const hipblasOperation_t op_b = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const hipblasOperation_t op_a = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N;
      const hipDataType c_type = out_f32 ? HIP_R_32F : HIP_R_16BF;
      const size_t ws_bytes = bl.ws.n;
      hipblasLtMatmulHeuristicResult_t res{};
      int found = 0;
      const uint64_t b_rows = tb ? K : N, b_cols = tb ? N : K;
      const uint64_t a_rows = ta ? M : K, a_cols = ta ? K : M;
      if (hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_b, sizeof(op_b)) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_a, sizeof(op_a)) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatrixLayoutCreate(&plan.la, HIP_R_16BF, b_rows, b_cols, ldb) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatrixLayoutCreate(&plan.lb, HIP_R_16BF, a_rows, a_cols, lda) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatrixLayoutCreate(&plan.lc, c_type, N, M, N) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)) == HIPBLAS_STATUS_SUCCESS &&
          hipblasLtMatmulAlgoGetHeuristic(bl.lt, plan.desc, plan.la, plan.lb, plan.lc, plan.lc, pref, 1, &res, &found) == HIPBLAS_STATUS_SUCCESS &&
          found > 0) {
        plan.algo = res.algo;
        plan.valid = true;
      }
      if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    }
    if (plan.valid &&
        hipblasLtMatmul(bl.lt, plan.desc, &alpha, b, plan.la, a, plan.lb, &beta,
                        c, plan.lc, c, plan.lc, &plan.algo, bl.ws.p, bl.ws.n,
                        stream) == HIPBLAS_STATUS_SUCCESS) {
      return;
    }
  }
  const auto type = out_f32 ? HIPBLAS_R_32F : HIPBLAS_R_16B;
  if (hipblasGemmEx(bl.h, tb ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                    ta ? HIPBLAS_OP_T : HIPBLAS_OP_N, N, M, K, &alpha, b,
                    HIPBLAS_R_16B, ldb, a, HIPBLAS_R_16B, lda, &beta, c, type, N,
                    HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("NoRA hipblasGemmEx");
#else
  (void)stream;
  const auto type = std::is_same_v<C, float> ? CUDA_R_32F : CUDA_R_16BF;
  if (cublasGemmEx(bl.h, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
                   ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, b,
                   CUDA_R_16BF, ldb, a, CUDA_R_16BF, lda, &beta, c, type, N,
                   CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("NoRA cublasGemmEx");
#endif
}
// [hi|lo] rows x 2*rank.
const bf16 *pair(cudaStream_t stream, Blas &b, const float *x, int rows,
                 int rank) {
  const size_t n = size_t(rows) * rank;
  bf16 *y = b.grow(b.pair, 2 * n);
  split_latent<<<(n + 255) / 256, 256, 0, stream>>>(n, rank, x, y);
  return y;
}
// Row-major w[r, c] copied as [w|w] (r x 2c) when side, else [w;w] (2r x c).
const bf16 *twice(cudaStream_t stream, Blas &b, const bf16 *w, int r, int c,
                  bool side) {
  const size_t total = 2 * size_t(r) * c;
  bf16 *y = b.grow(b.twice, total);
  repeat_weight_kernel<<<(total + 255) / 256, 256, 0, stream>>>(total, r, c,
                                                                side, w, y);
  return y;
}
// s = x A^T (FP32; the tape when s is given); y += scale s B^T.
void forward_tensor(cudaStream_t stream, int rows,
                    const rwkv_bf16_miss::Linear &w, const bf16 *x, bf16 *y,
                    float *s) {
  auto &b = blas(stream);
  const int r = w.rank;
  if (!s)
    s = b.grow(b.latent, size_t(rows) * r);
  product(stream, b, false, true, rows, r, w.in, 1.f, x, w.in, w.a, w.in, 0.f, s);
  if (!y)
    return;
  const bf16 *d_rep =
      w.d_twice ? w.d_twice : twice(stream, b, w.d, w.out, r, true);
  product(stream, b, false, true, rows, w.out, 2 * r, w.scale,
          pair(stream, b, s, rows, r), 2 * r, d_rep, 2 * r, 1.f, y);
}
// dB += scale g^T s; ds = scale g B; dA += ds^T x; dx += ds A.
void backward_tensor(cudaStream_t stream, int rows,
                     const rwkv_bf16_miss::Linear &w, const bf16 *x,
                     const float *s, const bf16 *g, bf16 *dx, float *ds) {
  auto &b = blas(stream);
  const int r = w.rank;
  if (w.gradient) {
    const bf16 *sp = pair(stream, b, s, rows, r);
    float *gp = b.grow(b.grad_pair, size_t(w.out) * 2 * r);
    product(stream, b, true, false, w.out, 2 * r, rows, w.scale, g, w.out, sp, 2 * r,
            0.f, gp);
    const size_t total_b = size_t(w.out) * r;
    accum_grad_b_pair<<<(total_b + 255) / 256, 256, 0, stream>>>(
        w.out, r, gp, w.gradient);
  }
  product(stream, b, false, false, rows, r, w.out, w.scale, g, w.out, w.d, r, 0.f,
          ds);
  const bf16 *dp = pair(stream, b, ds, rows, r);
  if (w.gradient_a) {
    float *gp = b.grow(b.grad_pair, 2 * size_t(r) * w.in);
    product(stream, b, true, false, 2 * r, w.in, rows, 1.f, dp, 2 * r, x, w.in, 0.f,
            gp);
    const size_t total_a = size_t(r) * w.in;
    accum_grad_a_pair<<<(total_a + 255) / 256, 256, 0, stream>>>(
        r, w.in, gp, w.gradient_a);
  }
  if (dx) {
    const bf16 *a_rep =
        w.a_twice ? w.a_twice : twice(stream, b, w.a, r, w.in, false);
    product(stream, b, false, false, rows, w.in, 2 * r, 1.f, dp, 2 * r, a_rep, w.in,
            1.f, dx);
  }
}
__global__ void replay_mix(int lanes, int time, int C, int rows,
                           const int *offsets, const std::uint8_t *doc_start,
                           const bf16 *x, const bf16 *shift, const bf16 *mix,
                           bf16 *y) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * C)
    return;
  int c = i % C;
  long long row = static_cast<long long>(i / C);
  auto at = rwkv_bf16_training::lane_row(offsets, lanes, time, row);
  float v = to_float(x[i]),
        p = (doc_start && doc_start[row])
                ? 0.0f
                : to_float(!at.first ? x[i - C] : shift[at.lane * C + c]);
  y[i] = to_bf16(v + (p - v) * to_float(mix[c]));
}
} // namespace
void forward(cudaStream_t stream, int rows, const rwkv_bf16_miss::Linear &w,
             const bf16 *x, bf16 *y, float *s) {
  validate(rows, w);
  if (!x || (!y && !s))
    throw std::invalid_argument("null NoRA forward input");
  using namespace rwkv_nora_detail;
  if (tensor_cores(w)) {
    forward_tensor(stream, rows, w, x, y, s);
    checked();
    return;
  }
  if (s) {
    detail::projection<true>(stream, rows, w.rank, w.in, x, w.a, s);
    if (y)
      detail::expansion<true>(stream, rows, w.out, w.rank, s, w.d, y, w.scale);
  } else {
    fused<<<dim3((w.out + 255) / 256, rows), 256, w.rank * sizeof(float),
            stream>>>(w.in, w.out, w.rank, w.scale, w.a, w.d, x, y);
  }
  checked();
}
void backward(cudaStream_t stream, int rows, const rwkv_bf16_miss::Linear &w,
              const bf16 *x, const float *s, const bf16 *g, bf16 *dx,
              float *ds) {
  validate(rows, w);
  if (!g || !ds || (w.gradient && !s) || (w.gradient_a && !x))
    throw std::invalid_argument("null NoRA backward input/tape");
  using namespace rwkv_nora_detail;
  if (tensor_cores(w)) {
    backward_tensor(stream, rows, w, x, s, g, dx, ds);
    checked();
    return;
  }
  if (w.gradient)
    detail::grad_weight(stream, w.out, w.rank, rows, g, s, w.gradient, w.scale);
  detail::projection<false>(stream, rows, w.rank, w.out, g, w.d, ds, w.scale);
  if (w.gradient_a)
    detail::grad_weight(stream, w.rank, w.in, rows, ds, x, w.gradient_a);
  if (dx)
    detail::expansion<false>(stream, rows, w.in, w.rank, ds, w.a, dx);
  checked();
}
void refresh_weight_twice(cudaStream_t stream, int rows, int cols, bool side,
                          const bf16 *src, bf16 *dst) {
  if (!src || !dst || rows < 1 || cols < 1)
    throw std::invalid_argument("invalid NoRA repeated weight buffer");
  const size_t total = 2 * size_t(rows) * cols;
  repeat_weight_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      total, rows, cols, side, src, dst);
  checked();
}
void backward_target(cudaStream_t stream,
                     const rwkv_bf16_training::FrozenBlockWeights &w,
                     const rwkv_bf16_training::BlockTapeView &t, int target,
                     const bf16 *g, bf16 *dx) {
  const auto &a = w.miss[target];
  if (!a.a) {
    rwkv_bf16_miss::backward(stream, rwkv_bf16_training::tape_rows(t), a,
                             t.miss_reduced[target], g, dx);
    return;
  }
  int rows = rwkv_bf16_training::tape_rows(t), C = w.channels;
  const bf16 *cached_in = target == 0   ? t.xr
                          : target == 1 ? t.xk
                          : target == 2 ? t.xv
                          : target == 3 ? t.att_post
                          : target == 4 ? t.ffn_mixed
                                        : nullptr;
  const bf16 *x_in = cached_in;
  if (!t.nora_ds || (a.gradient_a && !x_in && !t.nora_input))
    throw std::invalid_argument("missing NoRA replay scratch");
  if (a.gradient_a && !x_in) {
    bf16 *x = t.nora_input;
    using namespace rwkv_bf16_training;
    if (target == 5)
      rwkv7_relu_square_launch(stream, t.ffn_hid, x, size_t(rows) * w.ffn);
    else if (target == 3)
      rwkv7_tmix_lnx_rkvres_xg_launch(
          stream, 1, rows, C, w.heads, t.wkv_y, t.r, t.wkv_k, t.v,
          w.r_k, w.att_group_norm_weight, w.att_group_norm_bias, t.att_gate, x);
    else {
      const bf16 *mix = target == 0   ? w.mix_r
                        : target == 1 ? w.mix_k
                        : target == 2 ? w.mix_v
                                      : w.ffn_mix;
      const bf16 *shift = target == 4 ? t.nora_ffn_shift : t.nora_att_shift;
      if (!shift)
        throw std::invalid_argument("missing NoRA incoming shift");
      replay_mix<<<(size_t(rows) * C + 255) / 256, 256, 0, stream>>>(
          t.batch, t.time, C, rows, t.offsets, t.doc_start,
          target == 4 ? t.ln2 : t.ln1, shift, mix, x);
    }
    x_in = x;
  }
  backward(stream, rows, a, x_in, t.miss_reduced[target], g, dx, t.nora_ds);
}
} // namespace rwkv_bf16_nora
