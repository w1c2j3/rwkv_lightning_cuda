#include "rwkv/runtime/rwkv_training_wkv.hpp"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <stdexcept>

namespace rwkv7_state_tuning {
namespace {

constexpr int kN = 64;
constexpr int kPad = kN + 1;
constexpr int kHalf = 32;
constexpr int kHalfPad = kHalf + 1; // 33: bank-conflict-free on 32-bank LDS
constexpr int kCheckpointInterval = kWkvCheckpointInterval;
constexpr float kWScale = -0.6065306597126334f; // -exp(-0.5)

template <typename T> __device__ __forceinline__ float as_float(T value);
template <> __device__ __forceinline__ float as_float(half value) {
  return __half2float(value);
}
template <> __device__ __forceinline__ float as_float(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T from_float(float value);
template <> __device__ __forceinline__ half from_float(float value) {
  return __float2half_rn(value);
}
template <> __device__ __forceinline__ __hip_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float effective_w(float raw) {
  const float sigmoid = 1.0f / (1.0f + expf(-raw));
  return expf(kWScale * sigmoid);
}

__device__ __forceinline__ float next_state(float s, float w, float sa,
                                            float b, float k, float v) {
  return s * w + sa * b + k * v;
}

__device__ __forceinline__ long long checkpoint_slot(int t, int batch) {
  return t / kCheckpointInterval + batch;
}

// 4-Wave64 (256-thread) cooperative WKV7 forward kernel for MI250 (gfx90a).
// Eliminates the 164-byte register spill and 1-wave/CU bottleneck of the
// original 64-thread kernel: each wave owns 16 rows of state[64][64].
template <typename IO>
__global__ __launch_bounds__(256, 2) void exact_forward_kernel_w4(
    int T, int H, const int *__restrict__ offsets,
    const std::uint8_t *__restrict__ doc_start,
    const float *__restrict__ doc_initial_state,
    const float *__restrict__ initial_state,
    const IO *__restrict__ r_ptr, const IO *__restrict__ w_ptr,
    const IO *__restrict__ k_ptr, const IO *__restrict__ v_ptr,
    const IO *__restrict__ a_ptr, const IO *__restrict__ b_ptr,
    IO *__restrict__ y_ptr, float *__restrict__ final_state,
    float *__restrict__ checkpoints) {
  constexpr int kWaves = 4;
  constexpr int kSlice = kN / kWaves; // 16
  const int batch = blockIdx.y;
  const int head = blockIdx.x;
  const int value = threadIdx.x; // 0..63
  const int wy = threadIdx.y;    // 0..3
  const int k0 = wy * kSlice;
  const long long bh = static_cast<long long>(batch) * H + head;
  const long long state_base = bh * kN * kN;
  const long long doc_base = static_cast<long long>(head) * kN * kN;

  const int lane_off = k0 * kN + value;
  const float *init_ptr = initial_state + state_base + lane_off;
  const float *doc_ptr =
      doc_initial_state ? doc_initial_state + doc_base + lane_off : nullptr;

  float state[kSlice];
#pragma unroll
  for (int j = 0; j < kSlice; ++j) {
    state[j] = init_ptr[j * kN];
  }

  __shared__ float r[kN], w[kN], k[kN], a[kN], b[kN];
  __shared__ float red[2][kWaves][kN];

  const int begin = offsets ? offsets[batch] : batch * T;
  const int end = offsets ? offsets[batch + 1] : begin + T;

  for (int t = begin; t < end; ++t) {
    const long long token = (static_cast<long long>(t) * H + head) * kN;
    if (doc_start && doc_start[t]) {
#pragma unroll
      for (int j = 0; j < kSlice; ++j) {
        state[j] = doc_ptr ? doc_ptr[j * kN] : 0.0f;
      }
    }
    if (checkpoints && (t - begin) % kCheckpointInterval == 0) {
      float *chk_ptr =
          checkpoints + (checkpoint_slot(t, batch) * H + head) * kN * kN + lane_off;
#pragma unroll
      for (int j = 0; j < kSlice; ++j)
        chk_ptr[j * kN] = state[j];
    }
    __syncthreads();
    if (wy == 0) {
      r[value] = as_float(r_ptr[token + value]);
      w[value] = effective_w(as_float(w_ptr[token + value]));
    } else if (wy == 1) {
      k[value] = as_float(k_ptr[token + value]);
      a[value] = as_float(a_ptr[token + value]);
    } else if (wy == 2) {
      b[value] = as_float(b_ptr[token + value]);
    }
    const float vv = as_float(v_ptr[token + value]);
    __syncthreads();

    float sa0 = 0.0f, sa1 = 0.0f;
#pragma unroll
    for (int j = 0; j < kSlice; j += 2) {
      sa0 += state[j] * a[k0 + j];
      sa1 += state[j + 1] * a[k0 + j + 1];
    }
    red[0][wy][value] = sa0 + sa1;
    __syncthreads();

    const float sa = red[0][0][value] + red[0][1][value] +
                     red[0][2][value] + red[0][3][value];
    float yy0 = 0.0f, yy1 = 0.0f;
#pragma unroll
    for (int j = 0; j < kSlice; j += 2) {
      const int key0 = k0 + j;
      const int key1 = k0 + j + 1;
      state[j] = next_state(state[j], w[key0], sa, b[key0], k[key0], vv);
      state[j + 1] = next_state(state[j + 1], w[key1], sa, b[key1], k[key1], vv);
      yy0 += state[j] * r[key0];
      yy1 += state[j + 1] * r[key1];
    }
    red[1][wy][value] = yy0 + yy1;
    __syncthreads();
    if (wy == 0) {
      const float yy = red[1][0][value] + red[1][1][value] +
                       red[1][2][value] + red[1][3][value];
      y_ptr[token + value] = from_float<IO>(yy);
    }
  }
  float *final_ptr = final_state + state_base + lane_off;
#pragma unroll
  for (int j = 0; j < kSlice; ++j) {
    final_ptr[j * kN] = state[j];
  }
}

// 4-Wave64 (256-thread) software-pipelined WKV7 backward kernel for MI250 (gfx90a).
// Prefetches before[t-1] from HBM while computing token t so HBM latency overlaps
// with the transpose FMA pass; parallelizes scalar loads across all 4 waves.
template <typename IO, bool HasDocAccum>
__global__ __launch_bounds__(256, 1) void exact_backward_kernel_w4(
    int T, int H, const int *__restrict__ offsets,
    const std::uint8_t *__restrict__ doc_start,
    const float *__restrict__ doc_initial_state,
    const IO *__restrict__ r_ptr, const IO *__restrict__ w_ptr,
    const IO *__restrict__ k_ptr, const IO *__restrict__ v_ptr,
    const IO *__restrict__ a_ptr, const IO *__restrict__ b_ptr,
    const IO *__restrict__ dy_ptr, const float *__restrict__ ds_final,
    const float *__restrict__ checkpoints, float *__restrict__ scratch,
    float *__restrict__ ds_initial, float *__restrict__ ds_doc_accum,
    IO *__restrict__ dr_ptr, IO *__restrict__ dw_ptr, IO *__restrict__ dk_ptr,
    IO *__restrict__ dv_ptr, IO *__restrict__ da_ptr, IO *__restrict__ db_ptr) {
  constexpr int kWaves = 4;
  constexpr int kSlice = kN / kWaves; // 16
  const int batch = blockIdx.y;
  const int head = blockIdx.x;
  const int value = threadIdx.x; // 0..63
  const int wy = threadIdx.y;    // 0..3
  const int k0 = wy * kSlice;
  const long long bh = static_cast<long long>(batch) * H + head;
  const long long state_base = bh * kN * kN;
  const long long doc_base = static_cast<long long>(head) * kN * kN;

  const int lane_off = k0 * kN + value;
  const float *ds_final_ptr =
      ds_final ? ds_final + state_base + lane_off : nullptr;
  const float *doc_ptr =
      doc_initial_state ? doc_initial_state + doc_base + lane_off : nullptr;

  float dstate[kSlice];
  float doc_accum[HasDocAccum ? kSlice : 1] = {};
#pragma unroll
  for (int j = 0; j < kSlice; ++j) {
    dstate[j] = ds_final_ptr ? ds_final_ptr[j * kN] : 0.0f;
  }

  __shared__ float r[kN], w[kN], w_sig[kN], k[kN], v[kN], a[kN], b[kN], dy[kN];
  __shared__ float dsa[kN];
  __shared__ float sa_history[kCheckpointInterval * kN];
  __shared__ float dnew[kN * kPad];
  __shared__ float previous[kN * kPad];
  __shared__ float red[5][kWaves][kN];
  float *history = scratch + bh * kCheckpointInterval * kN * kN + lane_off;

  const int begin = offsets ? offsets[batch] : batch * T;
  const int end = offsets ? offsets[batch + 1] : begin + T;
  const int last = end > begin
                       ? begin + (end - begin - 1) / kCheckpointInterval *
                                     kCheckpointInterval
                       : begin - kCheckpointInterval;

  for (int first = last; first >= begin; first -= kCheckpointInterval) {
    const int stop = min(first + kCheckpointInterval, end);
    const float *chk_ptr =
        checkpoints + (checkpoint_slot(first, batch) * H + head) * kN * kN +
        lane_off;
    float pref[kSlice];
    if (stop > first) {
      float state[kSlice];
#pragma unroll
      for (int j = 0; j < kSlice; ++j)
        state[j] = chk_ptr[j * kN];
      for (int t = first; t < stop; ++t) {
        if (t > first && doc_start && doc_start[t]) {
#pragma unroll
          for (int j = 0; j < kSlice; ++j) {
            state[j] = doc_ptr ? doc_ptr[j * kN] : 0.0f;
          }
        }
        // Skip writing t==first (already stored in checkpoints[slot]) and
        // t==stop-1 (handed directly to backward via pref[] in registers).
        if (t > first && t + 1 < stop) {
          float *before = history + (t - first) * kN * kN;
#pragma unroll
          for (int j = 0; j < kSlice; ++j)
            before[j * kN] = state[j];
        }
        const long long token = (static_cast<long long>(t) * H + head) * kN;
        __syncthreads();
        if (wy == 0) {
          w[value] = effective_w(as_float(w_ptr[token + value]));
          k[value] = as_float(k_ptr[token + value]);
        } else if (wy == 1) {
          a[value] = as_float(a_ptr[token + value]);
          b[value] = as_float(b_ptr[token + value]);
        }
        const float vv = as_float(v_ptr[token + value]);
        __syncthreads();
        float sa0 = 0.0f, sa1 = 0.0f;
#pragma unroll
        for (int j = 0; j < kSlice; j += 2) {
          sa0 += state[j] * a[k0 + j];
          sa1 += state[j + 1] * a[k0 + j + 1];
        }
        red[0][wy][value] = sa0 + sa1;
        __syncthreads();
        const float state_sa = red[0][0][value] + red[0][1][value] +
                               red[0][2][value] + red[0][3][value];
        if (wy == 0)
          sa_history[(t - first) * kN + value] = state_sa;
        if (t + 1 == stop) {
#pragma unroll
          for (int j = 0; j < kSlice; ++j)
            pref[j] = state[j];
          break;
        }
#pragma unroll
        for (int j = 0; j < kSlice; ++j)
          state[j] = next_state(state[j], w[k0 + j], state_sa, b[k0 + j],
                                k[k0 + j], vv);
      }
    }

    for (int t = stop - 1; t >= first; --t) {
      const long long token = (static_cast<long long>(t) * H + head) * kN;
      const float *sa = sa_history + (t - first) * kN;
      __syncthreads();
#pragma unroll
      for (int j = 0; j < kSlice; ++j)
        previous[(k0 + j) * kPad + value] = pref[j];
      if (wy == 0) {
        r[value] = as_float(r_ptr[token + value]);
        const float raw_w = as_float(w_ptr[token + value]);
        w_sig[value] = 1.0f / (1.0f + expf(-raw_w));
        w[value] = expf(kWScale * w_sig[value]);
      } else if (wy == 1) {
        k[value] = as_float(k_ptr[token + value]);
        v[value] = as_float(v_ptr[token + value]);
      } else if (wy == 2) {
        a[value] = as_float(a_ptr[token + value]);
        b[value] = as_float(b_ptr[token + value]);
      } else {
        dy[value] = as_float(dy_ptr[token + value]);
      }
      __syncthreads();

      const float dy_val = dy[value];
      float local_dsa0 = 0.0f, local_dsa1 = 0.0f;
      float local_dv0 = 0.0f, local_dv1 = 0.0f;
#pragma unroll
      for (int j = 0; j < kSlice; j += 2) {
        const int key0 = k0 + j;
        const int key1 = k0 + j + 1;
        const float d0 = dstate[j] + r[key0] * dy_val;
        const float d1 = dstate[j + 1] + r[key1] * dy_val;
        dnew[key0 * kPad + value] = d0;
        dnew[key1 * kPad + value] = d1;
        local_dsa0 += d0 * b[key0];
        local_dsa1 += d1 * b[key1];
        local_dv0 += d0 * k[key0];
        local_dv1 += d1 * k[key1];
      }
      red[0][wy][value] = local_dsa0 + local_dsa1;
      red[1][wy][value] = local_dv0 + local_dv1;
      // Prefetch t-1 now so L2/HBM latency overlaps with reduction + transpose.
      if (t > first) {
        const float *before_prev =
            (t - 1 == first) ? chk_ptr : (history + (t - 1 - first) * kN * kN);
#pragma unroll
        for (int j = 0; j < kSlice; ++j)
          pref[j] = before_prev[j * kN];
      }
      __syncthreads();
      if (wy == 0) {
        dsa[value] = red[0][0][value] + red[0][1][value] +
                     red[0][2][value] + red[0][3][value];
        const float dv_tot = red[1][0][value] + red[1][1][value] +
                             red[1][2][value] + red[1][3][value];
        dv_ptr[token + value] = from_float<IO>(dv_tot);
      }
      __syncthreads();

      const float wv = w[value], bv = b[value], kv = k[value];
      float dr0 = 0.0f, dr1 = 0.0f, dw0 = 0.0f, dw1 = 0.0f;
      float dk0 = 0.0f, dk1 = 0.0f, db0 = 0.0f, db1 = 0.0f;
      float da0 = 0.0f, da1 = 0.0f;
      const float *prev_row = previous + value * kPad;
      const float *dnew_row = dnew + value * kPad;
#pragma unroll
      for (int j = 0; j < kSlice; j += 2) {
        const int q0 = k0 + j;
        const int q1 = k0 + j + 1;
        const float pr0 = prev_row[q0];
        const float pr1 = prev_row[q1];
        const float ur0 = next_state(pr0, wv, sa[q0], bv, kv, v[q0]);
        const float ur1 = next_state(pr1, wv, sa[q1], bv, kv, v[q1]);
        const float rg0 = dnew_row[q0];
        const float rg1 = dnew_row[q1];
        dr0 += ur0 * dy[q0];
        dr1 += ur1 * dy[q1];
        dw0 += rg0 * pr0;
        dw1 += rg1 * pr1;
        dk0 += rg0 * v[q0];
        dk1 += rg1 * v[q1];
        db0 += rg0 * sa[q0];
        db1 += rg1 * sa[q1];
        da0 += pr0 * dsa[q0];
        da1 += pr1 * dsa[q1];
      }
      red[0][wy][value] = dr0 + dr1;
      red[1][wy][value] = dw0 + dw1;
      red[2][wy][value] = dk0 + dk1;
      red[3][wy][value] = db0 + db1;
      red[4][wy][value] = da0 + da1;
      __syncthreads();

      if (wy == 0) {
        const float dr = red[0][0][value] + red[0][1][value] + red[0][2][value] + red[0][3][value];
        const float dw_eff = red[1][0][value] + red[1][1][value] + red[1][2][value] + red[1][3][value];
        const float dk = red[2][0][value] + red[2][1][value] + red[2][2][value] + red[2][3][value];
        const float db = red[3][0][value] + red[3][1][value] + red[3][2][value] + red[3][3][value];
        const float da = red[4][0][value] + red[4][1][value] + red[4][2][value] + red[4][3][value];
        const float sig = w_sig[value];
        dr_ptr[token + value] = from_float<IO>(dr);
        dw_ptr[token + value] =
            from_float<IO>(kWScale * dw_eff * wv * sig * (1.0f - sig));
        dk_ptr[token + value] = from_float<IO>(dk);
        da_ptr[token + value] = from_float<IO>(da);
        db_ptr[token + value] = from_float<IO>(db);
      }

      const float dsa_v = dsa[value];
#pragma unroll
      for (int j = 0; j < kSlice; ++j) {
        const int key = k0 + j;
        dstate[j] = dnew[key * kPad + value] * w[key] + a[key] * dsa_v;
      }
      if (doc_start && doc_start[t]) {
        if constexpr (HasDocAccum) {
          if (ds_doc_accum) {
#pragma unroll
            for (int j = 0; j < kSlice; ++j)
              doc_accum[j] += dstate[j];
          }
        }
#pragma unroll
        for (int j = 0; j < kSlice; ++j)
          dstate[j] = 0.0f;
      }
    }
  }
  float *ds_init_ptr = ds_initial + state_base + lane_off;
  float *ds_doc_ptr =
      (HasDocAccum && ds_doc_accum) ? ds_doc_accum + state_base + lane_off : nullptr;
#pragma unroll
  for (int j = 0; j < kSlice; ++j) {
    ds_init_ptr[j * kN] = dstate[j];
    if constexpr (HasDocAccum) {
      if (ds_doc_ptr)
        ds_doc_ptr[j * kN] += doc_accum[j];
    }
  }
}

template <typename IO>
void launch_forward(hipStream_t stream, const WkvShape &s, const float *s0,
                    const void *r, const void *w, const void *k, const void *v,
                    const void *a, const void *b, void *y, float *sT,
                    WkvTapeView tape) {
  exact_forward_kernel_w4<IO><<<dim3(s.heads, s.batch), dim3(kN, 4), 0, stream>>>(
      s.time, s.heads, s.offsets, s.doc_start, s.doc_initial_state, s0,
      static_cast<const IO *>(r),
      static_cast<const IO *>(w), static_cast<const IO *>(k),
      static_cast<const IO *>(v), static_cast<const IO *>(a),
      static_cast<const IO *>(b), static_cast<IO *>(y), sT, tape.data);
}

template <typename IO>
void launch_backward(hipStream_t stream, const WkvShape &s, const void *r,
                     const void *w, const void *k, const void *v, const void *a,
                     const void *b, const void *dy, const float *dsT,
                     WkvTapeView tape, float *ds0, void *dr, void *dw, void *dk,
                     void *dv, void *da, void *db, float *ds_doc_accum) {
  if (ds_doc_accum) {
    exact_backward_kernel_w4<IO, true><<<dim3(s.heads, s.batch), dim3(kN, 4), 0, stream>>>(
        s.time, s.heads, s.offsets, s.doc_start, s.doc_initial_state,
        static_cast<const IO *>(r), static_cast<const IO *>(w),
        static_cast<const IO *>(k), static_cast<const IO *>(v),
        static_cast<const IO *>(a), static_cast<const IO *>(b),
        static_cast<const IO *>(dy), dsT, tape.data,
        tape.data + wkv_checkpoint_elements(s), ds0, ds_doc_accum,
        static_cast<IO *>(dr), static_cast<IO *>(dw), static_cast<IO *>(dk),
        static_cast<IO *>(dv), static_cast<IO *>(da), static_cast<IO *>(db));
  } else {
    exact_backward_kernel_w4<IO, false><<<dim3(s.heads, s.batch), dim3(kN, 4), 0, stream>>>(
        s.time, s.heads, s.offsets, s.doc_start, s.doc_initial_state,
        static_cast<const IO *>(r), static_cast<const IO *>(w),
        static_cast<const IO *>(k), static_cast<const IO *>(v),
        static_cast<const IO *>(a), static_cast<const IO *>(b),
        static_cast<const IO *>(dy), dsT, tape.data,
        tape.data + wkv_checkpoint_elements(s), ds0, ds_doc_accum,
        static_cast<IO *>(dr), static_cast<IO *>(dw), static_cast<IO *>(dk),
        static_cast<IO *>(dv), static_cast<IO *>(da), static_cast<IO *>(db));
  }
}

void validate(const WkvShape &shape, WkvTapeView tape) {
  const std::size_t required = wkv_tape_elements(shape);
  if (!tape.data || tape.elements < required) {
    throw std::invalid_argument("state-tuning WKV tape is null or too small");
  }
}

} // namespace

void wkv_forward(hipStream_t stream, IoType io_type, const WkvShape &shape,
                 const float *initial_state, const void *r, const void *w,
                 const void *k, const void *v, const void *a, const void *b,
                 void *y, float *final_state, WkvTapeView tape) {
  if (tape.data || tape.elements)
    validate(shape, tape);
  else
    (void)wkv_tape_elements(shape);
  if (!initial_state || !r || !w || !k || !v || !a || !b || !y ||
      !final_state) {
    throw std::invalid_argument("null state-tuning WKV forward pointer");
  }
  if (io_type == IoType::F16) {
    launch_forward<half>(stream, shape, initial_state, r, w, k, v, a, b, y,
                         final_state, tape);
  } else {
    launch_forward<__hip_bfloat16>(stream, shape, initial_state, r, w, k, v, a,
                                  b, y, final_state, tape);
  }
}

void wkv_backward_input(hipStream_t stream, IoType io_type,
                        const WkvShape &shape, const void *r, const void *w,
                        const void *k, const void *v, const void *a,
                        const void *b, const void *dy, const float *ds_final,
                        WkvTapeView tape, float *ds_initial, void *dr, void *dw,
                        void *dk, void *dv, void *da, void *db,
                        float *ds_doc_accum) {
  validate(shape, tape);
  if (!r || !w || !k || !v || !a || !b || !dy || !ds_initial || !dr || !dw ||
      !dk || !dv || !da || !db) {
    throw std::invalid_argument("null state-tuning WKV backward pointer");
  }
  if (io_type == IoType::F16) {
    launch_backward<half>(stream, shape, r, w, k, v, a, b, dy, ds_final, tape,
                          ds_initial, dr, dw, dk, dv, da, db, ds_doc_accum);
  } else {
    launch_backward<__hip_bfloat16>(stream, shape, r, w, k, v, a, b, dy,
                                   ds_final, tape, ds_initial, dr, dw, dk, dv,
                                   da, db, ds_doc_accum);
  }
}

} // namespace rwkv7_state_tuning
