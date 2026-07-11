#include "repeat_kv_plugin.h"

#include <cassert>
#include <cstdint>

#include "cuda_bf16.h"
#include "cuda_fp16.h"

namespace {

template <typename T>
__global__ void repeatKVKernel(const T* input, T* output, int batch, int heads, int seq, int headDim, int groups) {
    const int outHeads = heads * groups;
    const int64_t total = static_cast<int64_t>(batch) * outHeads * seq * headDim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t rem = idx;
    const int d = static_cast<int>(rem % headDim);
    rem /= headDim;
    const int s = static_cast<int>(rem % seq);
    rem /= seq;
    const int outHead = static_cast<int>(rem % outHeads);
    rem /= outHeads;
    const int b = static_cast<int>(rem);

    const int inHead = outHead / groups;
    const int64_t inIdx = (((static_cast<int64_t>(b) * heads + inHead) * seq + s) * headDim + d);
    output[idx] = input[inIdx];
}

template <typename T>
void launchTypedRepeatKV(const void* input, void* output, int batch, int heads, int seq, int headDim, int groups,
                         cudaStream_t stream) {
    const int outHeads = heads * groups;
    const int64_t total = static_cast<int64_t>(batch) * outHeads * seq * headDim;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    repeatKVKernel<<<blocks, kThreads, 0, stream>>>(static_cast<const T*>(input), static_cast<T*>(output), batch, heads,
                                                    seq, headDim, groups);
}

}  // namespace

void launchRepeatKV(const void* input, void* output, int batch, int heads, int seq, int headDim, int groups,
                    nvinfer1::DataType type, cudaStream_t stream) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            launchTypedRepeatKV<float>(input, output, batch, heads, seq, headDim, groups, stream);
            break;
        case nvinfer1::DataType::kHALF:
            launchTypedRepeatKV<half>(input, output, batch, heads, seq, headDim, groups, stream);
            break;
        case nvinfer1::DataType::kBF16:
            launchTypedRepeatKV<__nv_bfloat16>(input, output, batch, heads, seq, headDim, groups, stream);
            break;
        default:
            assert(false && "RepeatKVPlugin only supports FLOAT/HALF/BF16");
            break;
    }
}
