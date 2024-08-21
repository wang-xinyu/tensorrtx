#include "cuda_utils.h"

using namespace std;

// preprocess (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
__global__ void preprocess_kernel(float* output, uint8_t* input,
    const int batchSize, const int height, const int width, const int channel,
    const int thread_count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= thread_count) return;

    const int w_idx = index % width;
    int idx = index / width;
    const int h_idx = idx % height;
    idx /= height;
    const int c_idx = idx % channel;
    const int b_idx = idx / channel;

    int g_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + 2 - c_idx;

    output[index] = input[g_idx] / 255.f;
}

void preprocess(float* output, uint8_t*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
    int thread_count = batchSize * height * width * channel;
    int block = 512;
    int grid = (thread_count - 1) / block + 1;

    preprocess_kernel << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, thread_count);
}

#include "preprocess.hpp"

namespace nvinfer1
{
    int PreprocessPluginV2::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        uint8_t* input = (uint8_t*)inputs[0];
        float* output = (float*)outputs[0];

        const int H = mPreprocess.H;
        const int W = mPreprocess.W;
        const int C = mPreprocess.C;

        preprocess(output, input, batchSize, H, W, C, stream);

        return 0;
    }
}