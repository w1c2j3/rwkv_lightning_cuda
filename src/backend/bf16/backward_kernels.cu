// Copied from src/state_tuning; independent BF16 training implementation.
#include "rwkv/bf16_training/training.hpp"

#include "rwkv/bf16_training/runtime.hpp"

#include <algorithm>
#include <stdexcept>

#include "rwkv/bf16_training/ops.hpp"

namespace rwkv_bf16_training {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_all_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
#ifdef RWKV_USE_HIP
    value += __shfl_xor(value, mask, 32);
#else
    value += __shfl_xor_sync(0xffffffffu, value, mask);
#endif
  }
  return value;
}

// One 32-thread warp computes exact GroupNorm backward for one 64-element head
// group (2 elements in registers per thread, 0 shared memory, 0 barriers).
__global__ void group_norm_64_backward_kernel(int group_count,
                                              std::size_t total_groups,
                                              const bf16 *__restrict__ x,
                                              const bf16 *__restrict__ weight,
                                              const bf16 *__restrict__ dy,
                                              bf16 *__restrict__ dx,
                                              float eps) {
  const std::size_t group =
      (static_cast<std::size_t>(blockIdx.x) * (kThreads >> 5)) +
      (threadIdx.x >> 5);
  if (group >= total_groups)
    return;
  const int lane = threadIdx.x & 31;
  const std::size_t offset = group * 64;
  const int w_offset = static_cast<int>(group % group_count) * 64;
  const float x0 = to_float(x[offset + lane]);
  const float x1 = to_float(x[offset + lane + 32]);
  const float sdy0 =
      to_float(dy[offset + lane]) * to_float(weight[w_offset + lane]);
  const float sdy1 =
      to_float(dy[offset + lane + 32]) * to_float(weight[w_offset + lane + 32]);
  constexpr float kInvN = 1.0f / 64.0f;
  const float mean = warp_all_sum(x0 + x1) * kInvN;
  const float d0 = x0 - mean;
  const float d1 = x1 - mean;
  const float rstd = rsqrtf(warp_all_sum(d0 * d0 + d1 * d1) * kInvN + eps);
  const float xhat0 = d0 * rstd;
  const float xhat1 = d1 * rstd;
  const float mean_d = warp_all_sum(sdy0 + sdy1) * kInvN;
  const float mean_dxh = warp_all_sum(sdy0 * xhat0 + sdy1 * xhat1) * kInvN;
  dx[offset + lane] = to_bf16(rstd * (sdy0 - mean_d - xhat0 * mean_dxh));
  dx[offset + lane + 32] = to_bf16(rstd * (sdy1 - mean_d - xhat1 * mean_dxh));
}

#ifdef RWKV_USE_HIP
__device__ __forceinline__ float wave64_all_sum_bwd(float value) {
#if defined(__GFX8__) || defined(__GFX9__)
#pragma unroll
  for (int mask = 32; mask > 0; mask >>= 1) {
    value += __shfl_xor(value, mask, 64);
  }
#endif
  return value;
}

// One 64-thread CDNA wavefront computes exact GroupNorm backward for one
// 64-element head group (1 element/thread, 0 LDS, 0 barriers, coalesced 128B loads).
__global__ __launch_bounds__(kThreads) void group_norm_64_backward_kernel_w64(
    int group_count, std::size_t total_groups, const bf16 *__restrict__ x,
    const bf16 *__restrict__ weight, const bf16 *__restrict__ dy,
    bf16 *__restrict__ dx, float eps) {
#if defined(__GFX8__) || defined(__GFX9__)
  const std::size_t group =
      (static_cast<std::size_t>(blockIdx.x) * (kThreads >> 6)) +
      (threadIdx.x >> 6);
  if (group >= total_groups)
    return;
  const int lane = threadIdx.x & 63;
  const std::size_t offset = group * 64 + lane;
  const int w_offset = static_cast<int>(group % group_count) * 64 + lane;
  const float xv = to_float(x[offset]);
  const float sdy = to_float(dy[offset]) * to_float(weight[w_offset]);
  constexpr float kInvN = 1.0f / 64.0f;
  const float mean = wave64_all_sum_bwd(xv) * kInvN;
  const float d = xv - mean;
  const float rstd = rsqrtf(wave64_all_sum_bwd(d * d) * kInvN + eps);
  const float xhat = d * rstd;
  const float mean_d = wave64_all_sum_bwd(sdy) * kInvN;
  const float mean_dxh = wave64_all_sum_bwd(sdy * xhat) * kInvN;
  dx[offset] = to_bf16(rstd * (sdy - mean_d - xhat * mean_dxh));
#endif
}
#endif

__device__ __forceinline__ float block_sum(float value, float *scratch) {
  __syncthreads(); // Previous reduction readers must finish before reuse.
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_all_sum(value);
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < (blockDim.x >> 5) ? scratch[lane] : 0.0f;
  if (warp == 0)
    value = warp_all_sum(value);
  if (threadIdx.x == 0)
    scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

__global__ void norm_backward_kernel(int group_size, int group_count,
                                     const bf16 *__restrict__ x,
                                     const bf16 *__restrict__ weight,
                                     const bf16 *__restrict__ dy,
                                     bf16 *__restrict__ dx, float eps,
                                     const bf16 *__restrict__ residual_grad) {
  const int group = blockIdx.x;
  const int offset = group * group_size;
  const int weight_offset = (group % group_count) * group_size;
  float local_x = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float value = to_float(x[offset + i]);
    local_x += value;
  }
  __shared__ float scratch[8];
  const float sum_x = block_sum(local_x, scratch);
  const float mean = sum_x / group_size;
  float local_x2 = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float centered = to_float(x[offset + i]) - mean;
    local_x2 += centered * centered;
  }
  const float variance = block_sum(local_x2, scratch) / group_size;
  const float rstd = rsqrtf(variance + eps);

  float local_d = 0.0f;
  float local_dxhat_xhat = 0.0f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float xhat = (to_float(x[offset + i]) - mean) * rstd;
    const float scaled_dy =
        to_float(dy[offset + i]) * to_float(weight[weight_offset + i]);
    local_d += scaled_dy;
    local_dxhat_xhat += scaled_dy * xhat;
  }
  const float sum_d = block_sum(local_d, scratch);
  const float sum_d_xhat = block_sum(local_dxhat_xhat, scratch);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float xhat = (to_float(x[offset + i]) - mean) * rstd;
    const float scaled_dy =
        to_float(dy[offset + i]) * to_float(weight[weight_offset + i]);
    const float value = rstd * (scaled_dy - sum_d / group_size -
                                xhat * sum_d_xhat / group_size);
    const bf16 d_norm = to_bf16(value);
    dx[offset + i] = residual_grad
                         ? to_bf16(to_float(residual_grad[offset + i]) +
                                   to_float(d_norm))
                         : d_norm;
  }
}

__global__ void norm_backward_cached_kernel(
    int C, const bf16 *__restrict__ x, const bf16 *__restrict__ weight,
    const bf16 *__restrict__ dy, bf16 *__restrict__ dx, float eps,
    const bf16 *__restrict__ residual_grad) {
  const size_t offset = static_cast<size_t>(blockIdx.x) * C;
  float reg_x[32];
  float reg_sdy[32];
  float local_x = 0.0f;
  int n = 0;
  for (int i = threadIdx.x; i < C; i += kThreads, ++n) {
    const float xv = to_float(x[offset + i]);
    reg_x[n] = xv;
    reg_sdy[n] = to_float(dy[offset + i]) * to_float(weight[i]);
    local_x += xv;
  }
  __shared__ float scratch[8];
  const float mean = block_sum(local_x, scratch) / C;
  float local_x2 = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float centered = reg_x[k] - mean;
    local_x2 += centered * centered;
  }
  const float rstd = rsqrtf(block_sum(local_x2, scratch) / C + eps);
  float local_d = 0.0f;
  float local_dxhat_xhat = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float xhat = (reg_x[k] - mean) * rstd;
    const float sdy = reg_sdy[k];
    local_d += sdy;
    local_dxhat_xhat += sdy * xhat;
  }
  const float sum_d = block_sum(local_d, scratch);
  const float sum_d_xhat = block_sum(local_dxhat_xhat, scratch);
  n = 0;
  for (int i = threadIdx.x; i < C; i += kThreads, ++n) {
    const float xhat = (reg_x[n] - mean) * rstd;
    const float value =
        rstd * (reg_sdy[n] - sum_d / C - xhat * sum_d_xhat / C);
    const bf16 d_norm = to_bf16(value);
    dx[offset + i] = residual_grad
                         ? to_bf16(to_float(residual_grad[offset + i]) +
                                   to_float(d_norm))
                         : d_norm;
  }
}

struct SixPointers {
  const bf16 *p0;
  const bf16 *p1;
  const bf16 *p2;
  const bf16 *p3;
  const bf16 *p4;
  const bf16 *p5;
};

__device__ __forceinline__ const bf16 *select_ptr(SixPointers p, int i) {
  switch (i) {
  case 0:
    return p.p0;
  case 1:
    return p.p1;
  case 2:
    return p.p2;
  case 3:
    return p.p3;
  case 4:
    return p.p4;
  default:
    return p.p5;
  }
}

#ifdef RWKV_USE_HIP
struct Bf16x8Bwd { bf16 v[8]; };
__device__ __forceinline__ Bf16x8Bwd ld8_b(const bf16 *p) {
  union { uint4 u; Bf16x8Bwd b; } r;
  r.u = *reinterpret_cast<const uint4 *>(p);
  return r.b;
}
__device__ __forceinline__ void st8_b(bf16 *p, Bf16x8Bwd b) {
  union { uint4 u; Bf16x8Bwd b; } r;
  r.b = b;
  *reinterpret_cast<uint4 *>(p) = r.u;
}
static inline bool aligned16_b(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}
__global__ void time_mix_backward_kernel_v8(
    int B, int T, int C, const int *__restrict__ offsets,
    const std::uint8_t *__restrict__ doc_start, int branches,
    SixPointers grads, SixPointers mixes, bf16 *__restrict__ dx) {
  const int C8 = C >> 3;
  const long long idx8 =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long elements8 = static_cast<long long>(offsets ? 1 : B) * T * C8;
  if (idx8 >= elements8) return;
  const int c8 = static_cast<int>(idx8 % C8);
  const long long row = idx8 / C8;
  const bool last = lane_row(offsets, B, T, row).last;
  const bool next_resets = doc_start && !last && doc_start[row + 1];
  const bool use_next = (!last && !next_resets);
  float val[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int branch = 0; branch < branches; ++branch) {
    const bf16 *grad = select_ptr(grads, branch);
    const Bf16x8Bwd m = ld8_b(select_ptr(mixes, branch) + c8 * 8);
    const Bf16x8Bwd g0 = ld8_b(grad + idx8 * 8);
    Bf16x8Bwd g1;
    if (use_next) g1 = ld8_b(grad + (idx8 + C8) * 8);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float mv = to_float(m.v[k]);
      val[k] += to_float(g0.v[k]) * (1.0f - mv);
      if (use_next) val[k] += to_float(g1.v[k]) * mv;
    }
  }
  Bf16x8Bwd out;
#pragma unroll
  for (int k = 0; k < 8; ++k) out.v[k] = to_bf16(val[k]);
  st8_b(dx + idx8 * 8, out);
}
#endif
__global__ void time_mix_backward_kernel(int B, int T, int C,
                                         const int *__restrict__ offsets,
                                         const std::uint8_t *__restrict__ doc_start,
                                         int branches, SixPointers grads,
                                         SixPointers mixes,
                                         bf16 *__restrict__ dx) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long elements = static_cast<long long>(offsets ? 1 : B) * T * C;
  if (idx >= elements)
    return;
  const int c = static_cast<int>(idx % C);
  const long long row = idx / C;
  const bool last = lane_row(offsets, B, T, row).last;
  const bool next_resets = doc_start && !last && doc_start[row + 1];
  float value = 0.0f;
  for (int branch = 0; branch < branches; ++branch) {
    const bf16 *grad = select_ptr(grads, branch);
    const float mix = to_float(select_ptr(mixes, branch)[c]);
    value += to_float(grad[idx]) * (1.0f - mix);
    if (!last && !next_resets)
      value += to_float(grad[idx + C]) * mix;
  }
  dx[idx] = to_bf16(value);
}

// An empty packed lane gets zero: callers only leave lanes empty while their
// terminal shift adjoint is still zero.
__global__ void shift_backward_kernel(int B, int T, int C,
                                      const int *__restrict__ offsets,
                                      const std::uint8_t *__restrict__ doc_start,
                                      int branches, SixPointers grads,
                                      SixPointers mixes,
                                      bf16 *__restrict__ d_shift) {
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long elements = static_cast<long long>(B) * C;
  if (idx >= elements)
    return;
  const int c = static_cast<int>(idx % C);
  const int batch = static_cast<int>(idx / C);
  const long long first_row =
      offsets ? offsets[batch] : static_cast<long long>(batch) * T;
  if ((offsets && offsets[batch + 1] == offsets[batch]) ||
      (doc_start && doc_start[first_row])) {
    d_shift[idx] = to_bf16(0.0f);
    return;
  }
  const long long first = first_row * C + c;
  float value = 0.0f;
  for (int branch = 0; branch < branches; ++branch) {
    value += to_float(select_ptr(grads, branch)[first]) *
             to_float(select_ptr(mixes, branch)[c]);
  }
  d_shift[idx] = to_bf16(value);
}

enum class ElementOp { ReluSquare, Sigmoid, Tanh, Add, AddInplace };

__global__ void last_shift_kernel(int T, int C, int count, const int *offsets,
                                  bf16 *dx, const bf16 *shift) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = i / C;
  if (i < count && (!offsets || offsets[b + 1] > offsets[b])) {
    const long long end =
        offsets ? offsets[b + 1] : static_cast<long long>(b) * T + T;
    const long long j = (end - 1) * C + i % C;
    dx[j] = to_bf16(to_float(dx[j]) + to_float(shift[i]));
  }
}

#ifdef RWKV_USE_HIP
template <ElementOp Op>
__global__ void element_kernel_v8(const bf16 *__restrict__ x,
                                  const bf16 *__restrict__ y,
                                  bf16 *__restrict__ out, std::size_t n8) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  const Bf16x8Bwd xv = ld8_b(x + i * 8);
  Bf16x8Bwd yv;
  if (y) yv = ld8_b(y + i * 8);
  Bf16x8Bwd res;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float a = to_float(xv.v[k]);
    const float b = y ? to_float(yv.v[k]) : 0.0f;
    float r;
    if constexpr (Op == ElementOp::ReluSquare) r = a > 0.0f ? 2.0f * a * b : 0.0f;
    else if constexpr (Op == ElementOp::Sigmoid) r = a * (1.0f - a) * b;
    else if constexpr (Op == ElementOp::Tanh) r = (1.0f - a * a) * b;
    else r = a + b;
    res.v[k] = to_bf16(r);
  }
  st8_b(out + i * 8, res);
}
#endif
template <ElementOp Op>
__global__ void element_kernel(const bf16 *__restrict__ x,
                               const bf16 *__restrict__ y,
                               bf16 *__restrict__ out, std::size_t elements) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements)
    return;
  const float xv = to_float(x[i]);
  const float yv = y ? to_float(y[i]) : 0.0f;
  float result;
  if constexpr (Op == ElementOp::ReluSquare) {
    result = xv > 0.0f ? 2.0f * xv * yv : 0.0f;
  } else if constexpr (Op == ElementOp::Sigmoid) {
    result = xv * (1.0f - xv) * yv;
  } else if constexpr (Op == ElementOp::Tanh) {
    result = (1.0f - xv * xv) * yv;
  } else {
    result = xv + yv;
  }
  out[i] = to_bf16(result);
}

void require_dims(int rows, int in_features, int out_features) {
  if (rows <= 0 || in_features <= 0 || out_features <= 0) {
    throw std::invalid_argument(
        "state-tuning linear dimensions must be positive");
  }
}

SixPointers pack(const bf16 *const *pointers, int count) {
  SixPointers result{};
  const bf16 **fields[] = {&result.p0, &result.p1, &result.p2,
                           &result.p3, &result.p4, &result.p5};
  for (int i = 0; i < count; ++i) {
    if (!pointers[i])
      throw std::invalid_argument("null time-mix branch pointer");
    *fields[i] = pointers[i];
  }
  return result;
}

int blocks_for(std::size_t elements) {
  return static_cast<int>((elements + kThreads - 1) / kThreads);
}
#ifdef RWKV_USE_HIP
__device__ __forceinline__ float block_sum_hip_bwd(float v, float *s) {
#if defined(__GFX8__) || defined(__GFX9__)
  v += __shfl_xor(v, 32, 64);
  v += __shfl_xor(v, 16, 64);
  v += __shfl_xor(v, 8, 64);
  v += __shfl_xor(v, 4, 64);
  v += __shfl_xor(v, 2, 64);
  v += __shfl_xor(v, 1, 64);
  const int lane = threadIdx.x & 63;
  const int wid = threadIdx.x >> 6;
  if (lane == 0) s[wid] = v;
  __syncthreads();
  const float tot = s[0] + s[1] + s[2] + s[3];
  __syncthreads();
  return tot;
#else
  return block_sum(v, s);
#endif
}

__global__ void norm_backward_cached_kernel_v8(
    int C, const bf16 *__restrict__ x, const bf16 *__restrict__ weight,
    const bf16 *__restrict__ dy, bf16 *__restrict__ dx, float eps,
    const bf16 *__restrict__ residual_grad) {
  const int C8 = C >> 3;
  const size_t off8 = static_cast<size_t>(blockIdx.x) * C8;
  float reg_x[32];
  float reg_sdy[32];
  float local_x = 0.0f;
  int n = 0;
  for (int i8 = threadIdx.x; i8 < C8; i8 += kThreads) {
    const Bf16x8Bwd xv = ld8_b(x + (off8 + i8) * 8);
    const Bf16x8Bwd dyv = ld8_b(dy + (off8 + i8) * 8);
    const Bf16x8Bwd wv = ld8_b(weight + i8 * 8);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float xk = to_float(xv.v[k]);
      reg_x[n] = xk;
      reg_sdy[n] = to_float(dyv.v[k]) * to_float(wv.v[k]);
      local_x += xk;
      ++n;
    }
  }
  __shared__ float scratch[8];
  const float mean = block_sum_hip_bwd(local_x, scratch) / C;
  float local_x2 = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float centered = reg_x[k] - mean;
    local_x2 += centered * centered;
  }
  const float rstd = rsqrtf(block_sum_hip_bwd(local_x2, scratch) / C + eps);
  float local_d = 0.0f;
  float local_dxhat_xhat = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float xhat = (reg_x[k] - mean) * rstd;
    const float sdy = reg_sdy[k];
    local_d += sdy;
    local_dxhat_xhat += sdy * xhat;
  }
  const float mean_d = block_sum_hip_bwd(local_d, scratch) / C;
  const float mean_dxh = block_sum_hip_bwd(local_dxhat_xhat, scratch) / C;
  n = 0;
  for (int i8 = threadIdx.x; i8 < C8; i8 += kThreads) {
    Bf16x8Bwd outv;
    if (residual_grad) {
      const Bf16x8Bwd rgv = ld8_b(residual_grad + (off8 + i8) * 8);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float xhat = (reg_x[n] - mean) * rstd;
        const bf16 d_norm = to_bf16(rstd * (reg_sdy[n] - mean_d - xhat * mean_dxh));
        outv.v[k] = to_bf16(to_float(rgv.v[k]) + to_float(d_norm));
        ++n;
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float xhat = (reg_x[n] - mean) * rstd;
        outv.v[k] = to_bf16(rstd * (reg_sdy[n] - mean_d - xhat * mean_dxh));
        ++n;
      }
    }
    st8_b(dx + (off8 + i8) * 8, outv);
  }
}

template <ElementOp Op>
static void launch_elem_bwd(cudaStream_t stream, const bf16 *x, const bf16 *y,
                            bf16 *out, std::size_t elements) {
  if ((elements & 7) == 0 && aligned16_b(x) && (!y || aligned16_b(y)) && aligned16_b(out)) {
    element_kernel_v8<Op><<<blocks_for(elements >> 3), kThreads, 0, stream>>>(
        x, y, out, elements >> 3);
    return;
  }
  element_kernel<Op><<<blocks_for(elements), kThreads, 0, stream>>>(x, y, out, elements);
}
#endif

} // namespace

void linear_orig_backward_input_bf16(cudaStream_t stream, int rows,
                                     int in_features, int out_features,
                                     const bf16 *dy,
                                     const bf16 *frozen_weight_orig, bf16 *dx) {
  require_dims(rows, in_features, out_features);
  if (!dy || !frozen_weight_orig || !dx)
    throw std::invalid_argument("null linear pointer");
  rwkv7_v3a_linear_bf16_launch(stream, rows, out_features, in_features, dy,
                               frozen_weight_orig, dx);
}

void add_last_shift_gradient(cudaStream_t stream, int batch, int time,
                             int channels, bf16 *dx, const bf16 *dshift,
                             const int *offsets) {
  if (dshift)
    last_shift_kernel<<<blocks_for(batch * channels), kThreads, 0, stream>>>(
        time, channels, batch * channels, offsets, dx, dshift);
}

void linear_runtime_backward_input_bf16(cudaStream_t stream, int rows,
                                        int in_features, int out_features,
                                        const bf16 *dy,
                                        const bf16 *frozen_weight_runtime,
                                        bf16 *dx) {
  require_dims(rows, in_features, out_features);
  if (!dy || !frozen_weight_runtime || !dx)
    throw std::invalid_argument("null linear pointer");
  rwkv7_v3a_linear_bf16_orig_launch(stream, rows, out_features, in_features, dy,
                                    frozen_weight_runtime, dx);
}

void linear_runtime_backward_input_accumulate_bf16(
    cudaStream_t stream, int rows, int in_features, int out_features,
    const bf16 *dy, const bf16 *frozen_weight_runtime, bf16 *dx) {
  require_dims(rows, in_features, out_features);
  if (!dy || !frozen_weight_runtime || !dx)
    throw std::invalid_argument("null linear pointer");
  rwkv7_v3a_linear_bf16_orig_accumulate_launch(
      stream, rows, out_features, in_features, dy, frozen_weight_runtime, dx);
}

void layer_norm_backward_input_bf16(cudaStream_t stream, int rows, int channels,
                                    const bf16 *x, const bf16 *frozen_weight,
                                    const bf16 *dy, bf16 *dx, float eps,
                                    const bf16 *residual_grad) {
  if (rows <= 0 || channels <= 0 || !x || !frozen_weight || !dy || !dx ||
      eps <= 0.0f) {
    throw std::invalid_argument("invalid layer-norm backward arguments");
  }
#ifdef RWKV_USE_HIP
  if (channels <= 8192 && (channels & 7) == 0 && aligned16_b(x) &&
      aligned16_b(frozen_weight) && aligned16_b(dy) && aligned16_b(dx) &&
      (!residual_grad || aligned16_b(residual_grad))) {
    norm_backward_cached_kernel_v8<<<rows, kThreads, 0, stream>>>(
        channels, x, frozen_weight, dy, dx, eps, residual_grad);
    return;
  }
#endif
  if (channels <= 8192) {
    norm_backward_cached_kernel<<<rows, kThreads, 0, stream>>>(
        channels, x, frozen_weight, dy, dx, eps, residual_grad);
  } else {
    norm_backward_kernel<<<rows, kThreads, 0, stream>>>(
        channels, 1, x, frozen_weight, dy, dx, eps, residual_grad);
  }
}

void group_norm_backward_input_bf16(cudaStream_t stream, int rows, int channels,
                                    int groups, const bf16 *x,
                                    const bf16 *frozen_weight, const bf16 *dy,
                                    bf16 *dx, float eps) {
  if (rows <= 0 || channels <= 0 || groups <= 0 || channels % groups != 0 ||
      !x || !frozen_weight || !dy || !dx || eps <= 0.0f) {
    throw std::invalid_argument("invalid group-norm backward arguments");
  }
  const int group_size = channels / groups;
  const std::size_t total_groups = static_cast<std::size_t>(rows) * groups;
  if (group_size == 64) {
#ifdef RWKV_USE_HIP
    if (is_wave64_device()) {
      constexpr std::size_t kWavesPerBlock = kThreads >> 6;
      const int grid = static_cast<int>(
          (total_groups + kWavesPerBlock - 1) / kWavesPerBlock);
      group_norm_64_backward_kernel_w64<<<grid, kThreads, 0, stream>>>(
          groups, total_groups, x, frozen_weight, dy, dx, eps);
      return;
    }
#endif
    constexpr std::size_t kWarpsPerBlock = kThreads >> 5;
    const int grid =
        static_cast<int>((total_groups + kWarpsPerBlock - 1) / kWarpsPerBlock);
    group_norm_64_backward_kernel<<<grid, kThreads, 0, stream>>>(
        groups, total_groups, x, frozen_weight, dy, dx, eps);
  } else {
    norm_backward_kernel<<<rows * groups, kThreads, 0, stream>>>(
        group_size, groups, x, frozen_weight, dy, dx, eps, nullptr);
  }
}

void time_mix_backward_input_bf16(cudaStream_t stream, int batch, int time,
                                  int channels, int branches,
                                  const bf16 *const *branch_grads,
                                  const bf16 *const *frozen_mixes, bf16 *dx,
                                  bf16 *d_shift, const int *offsets,
                                  const std::uint8_t *doc_start) {
  if (batch <= 0 || time <= 0 || channels <= 0 || branches <= 0 ||
      branches > 6 || !branch_grads || !frozen_mixes || !dx) {
    throw std::invalid_argument("invalid time-mix backward arguments");
  }
  const SixPointers grads = pack(branch_grads, branches);
  const SixPointers mixes = pack(frozen_mixes, branches);
  const std::size_t elements =
      static_cast<std::size_t>(offsets ? 1 : batch) * time * channels;
#ifdef RWKV_USE_HIP
  bool ok8 = ((channels & 7) == 0) && aligned16_b(dx);
  for (int b = 0; b < branches && ok8; ++b)
    ok8 = aligned16_b(branch_grads[b]) && aligned16_b(frozen_mixes[b]);
  if (ok8) {
    time_mix_backward_kernel_v8<<<blocks_for(elements >> 3), kThreads, 0, stream>>>(
        batch, time, channels, offsets, doc_start, branches, grads, mixes, dx);
  } else
#endif
  time_mix_backward_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      batch, time, channels, offsets, doc_start, branches, grads, mixes, dx);
  if (d_shift) {
    const std::size_t shift_elements =
        static_cast<std::size_t>(batch) * channels;
    shift_backward_kernel<<<blocks_for(shift_elements), kThreads, 0, stream>>>(
        batch, time, channels, offsets, doc_start, branches, grads, mixes,
        d_shift);
  }
}

void relu_square_backward_input_bf16(cudaStream_t stream, const bf16 *x,
                                     const bf16 *dy, bf16 *dx,
                                     std::size_t elements) {
#ifdef RWKV_USE_HIP
  launch_elem_bwd<ElementOp::ReluSquare>(stream, x, dy, dx, elements);
#else
  element_kernel<ElementOp::ReluSquare>
      <<<blocks_for(elements), kThreads, 0, stream>>>(x, dy, dx, elements);
#endif
}

void sigmoid_backward_input_bf16(cudaStream_t stream, const bf16 *sigmoid_y,
                                 const bf16 *dy, bf16 *dx,
                                 std::size_t elements) {
#ifdef RWKV_USE_HIP
  launch_elem_bwd<ElementOp::Sigmoid>(stream, sigmoid_y, dy, dx, elements);
#else
  element_kernel<ElementOp::Sigmoid>
      <<<blocks_for(elements), kThreads, 0, stream>>>(sigmoid_y, dy, dx,
                                                      elements);
#endif
}

void tanh_backward_input_bf16(cudaStream_t stream, const bf16 *tanh_y,
                              const bf16 *dy, bf16 *dx, std::size_t elements) {
#ifdef RWKV_USE_HIP
  launch_elem_bwd<ElementOp::Tanh>(stream, tanh_y, dy, dx, elements);
#else
  element_kernel<ElementOp::Tanh>
      <<<blocks_for(elements), kThreads, 0, stream>>>(tanh_y, dy, dx, elements);
#endif
}

void add_bf16(cudaStream_t stream, const bf16 *a, const bf16 *b, bf16 *out,
              std::size_t elements) {
#ifdef RWKV_USE_HIP
  launch_elem_bwd<ElementOp::Add>(stream, a, b, out, elements);
#else
  element_kernel<ElementOp::Add>
      <<<blocks_for(elements), kThreads, 0, stream>>>(a, b, out, elements);
#endif
}

void add_inplace_bf16(cudaStream_t stream, bf16 *destination,
                      const bf16 *source, std::size_t elements) {
#ifdef RWKV_USE_HIP
  launch_elem_bwd<ElementOp::AddInplace>(stream, destination, source,
                                         destination, elements);
#else
  element_kernel<ElementOp::AddInplace>
      <<<blocks_for(elements), kThreads, 0, stream>>>(destination, source,
                                                      destination, elements);
#endif
}

} // namespace rwkv_bf16_training
