#include "apply_rotary_pos_emb_plugin.h"

#include <cassert>
#include <cstdint>

#include "cuda_bf16.h"
#include "cuda_fp16.h"

namespace {

template <typename T>
__device__ inline T negateValue(T value);
template <>
__device__ inline float negateValue(float value) {
    return -value;
}
template <>
__device__ inline half negateValue(half value) {
    return __hneg(value);
}
template <>
__device__ inline __nv_bfloat16 negateValue(__nv_bfloat16 value) {
    return __hneg(value);
}

template <typename T>
__device__ inline T addValue(T a, T b);
template <>
__device__ inline float addValue(float a, float b) {
    return a + b;
}
template <>
__device__ inline half addValue(half a, half b) {
    return __hadd(a, b);
}
template <>
__device__ inline __nv_bfloat16 addValue(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

template <typename T>
__device__ inline T mulValue(T a, T b);
template <>
__device__ inline float mulValue(float a, float b) {
    return a * b;
}
template <>
__device__ inline half mulValue(half a, half b) {
    return __hmul(a, b);
}
template <>
__device__ inline __nv_bfloat16 mulValue(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
}

template <typename T>
__global__ void applyRotaryKernel(const T* input, const T* cos, const T* sin, T* output, int batch, int heads, int seq,
                                  int headDim) {
    const int64_t total = static_cast<int64_t>(batch) * heads * seq * headDim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t rem = idx;
    const int d = static_cast<int>(rem % headDim);
    rem /= headDim;
    const int s = static_cast<int>(rem % seq);
    rem /= seq;
    const int h = static_cast<int>(rem % heads);
    rem /= heads;
    const int b = static_cast<int>(rem);

    const int halfHeadDim = headDim / 2;
    const int rotatedD = d < halfHeadDim ? d + halfHeadDim : d - halfHeadDim;
    const bool negateRotated = d < halfHeadDim;

    const int64_t inputBase = (((static_cast<int64_t>(b) * heads + h) * seq + s) * headDim);
    const int64_t ropeBase = ((static_cast<int64_t>(b) * seq + s) * headDim);

    const T x = input[inputBase + d];
    T rotated = input[inputBase + rotatedD];
    if (negateRotated) {
        rotated = negateValue(rotated);
    }

    output[idx] = addValue(mulValue(x, cos[ropeBase + d]), mulValue(rotated, sin[ropeBase + d]));
}

template <typename T>
void launchTypedApplyRotaryPosEmb(const void* q, const void* k, const void* cos, const void* sin, void* qEmbed,
                                  void* kEmbed, int batch, int qHeads, int kHeads, int seq, int headDim,
                                  cudaStream_t stream) {
    constexpr int kThreads = 256;
    const int64_t qTotal = static_cast<int64_t>(batch) * qHeads * seq * headDim;
    const int qBlocks = static_cast<int>((qTotal + kThreads - 1) / kThreads);
    applyRotaryKernel<<<qBlocks, kThreads, 0, stream>>>(static_cast<const T*>(q), static_cast<const T*>(cos),
                                                        static_cast<const T*>(sin), static_cast<T*>(qEmbed), batch,
                                                        qHeads, seq, headDim);

    const int64_t kTotal = static_cast<int64_t>(batch) * kHeads * seq * headDim;
    const int kBlocks = static_cast<int>((kTotal + kThreads - 1) / kThreads);
    applyRotaryKernel<<<kBlocks, kThreads, 0, stream>>>(static_cast<const T*>(k), static_cast<const T*>(cos),
                                                        static_cast<const T*>(sin), static_cast<T*>(kEmbed), batch,
                                                        kHeads, seq, headDim);
}

}  // namespace

void launchApplyRotaryPosEmb(const void* q, const void* k, const void* cos, const void* sin, void* qEmbed, void* kEmbed,
                             int batch, int qHeads, int kHeads, int seq, int headDim, nvinfer1::DataType type,
                             cudaStream_t stream) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            launchTypedApplyRotaryPosEmb<float>(q, k, cos, sin, qEmbed, kEmbed, batch, qHeads, kHeads, seq, headDim,
                                                stream);
            break;
        case nvinfer1::DataType::kHALF:
            launchTypedApplyRotaryPosEmb<half>(q, k, cos, sin, qEmbed, kEmbed, batch, qHeads, kHeads, seq, headDim,
                                               stream);
            break;
        case nvinfer1::DataType::kBF16:
            launchTypedApplyRotaryPosEmb<__nv_bfloat16>(q, k, cos, sin, qEmbed, kEmbed, batch, qHeads, kHeads, seq,
                                                        headDim, stream);
            break;
        default:
            assert(false && "ApplyRotaryPosEmbPlugin only supports FLOAT/HALF/BF16");
            break;
    }
}
