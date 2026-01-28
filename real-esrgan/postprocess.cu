#include "cublas_v2.h"
#include "cuda_utils.h"

using namespace std;

// postprocess (NCHW->NHWC, RGB->BGR, *255, ROUND, uint8)
template <typename T>
__global__ void postprocess_kernel(uint8_t* output, const T* input, const int batchSize, const int height,
                                   const int width, const int channel, const int thread_count) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= thread_count)
        return;

    const int c_idx = index % channel;
    int idx = index / channel;
    const int w_idx = idx % width;
    idx /= width;
    const int h_idx = idx % height;
    const int b_idx = idx / height;

    int g_idx = b_idx * height * width * channel + (2 - c_idx) * height * width + h_idx * width + w_idx;
    float val = (float)input[g_idx];
    float tt = val * 255.f;
    if (tt > 255)
        tt = 255;
    if (tt < 0)
        tt = 0;
    output[index] = (uint8_t)tt;
}

template __global__ void postprocess_kernel<float>(uint8_t* output, const float* input, const int batchSize,
                                                   const int height, const int width, const int channel,
                                                   const int thread_count);
template __global__ void postprocess_kernel<half>(uint8_t* output, const half* input, const int batchSize,
                                                  const int height, const int width, const int channel,
                                                  const int thread_count);

template <typename T>
void postprocess(uint8_t* output, const T* input, int batchSize, int height, int width, int channel,
                 cudaStream_t stream) {
    int thread_count = batchSize * height * width * channel;
    int block = 512;
    int grid = (thread_count - 1) / block + 1;

    postprocess_kernel<T><<<grid, block, 0, stream>>>(output, input, batchSize, height, width, channel, thread_count);
}

template void postprocess<float>(uint8_t* output, const float* input, int batchSize, int height, int width, int channel,
                                 cudaStream_t stream);
template void postprocess<half>(uint8_t* output, const half* input, int batchSize, int height, int width, int channel,
                                cudaStream_t stream);

#include "postprocess.hpp"

namespace nvinfer1 {
int PostprocessPluginV2::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
    uint8_t* output = (uint8_t*)outputs[0];

    const int H = mPostprocess.H;
    const int W = mPostprocess.W;
    const int C = mPostprocess.C;

    if (mDataType == DataType::kFLOAT) {
        const float* input = (const float*)inputs[0];
        postprocess<float>(output, input, batchSize, H, W, C, stream);
    } else if (mDataType == DataType::kHALF) {
        const half* input = (const half*)inputs[0];
        postprocess<half>(output, input, batchSize, H, W, C, stream);
    }

    return 0;
}
}  // namespace nvinfer1
