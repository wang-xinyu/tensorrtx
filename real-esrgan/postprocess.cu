#include "cuda_utils.h"

using namespace std;

// postprocess (NCHW->NHWC, RGB->BGR, *255, ROUND, uint8)
__global__ void postprocess_kernel(uint8_t* output, float* input,
    const int batchSize, const int height, const int width, const int channel,
    const int thread_count)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= thread_count) return;

    const int c_idx = index % channel;
    int idx = index / channel;
    const int w_idx = idx % width;
    idx /= width;
    const int h_idx = idx % height;
    const int b_idx = idx / height;

    int g_idx = b_idx * height * width * channel + (2 - c_idx)* height * width + h_idx * width + w_idx;
    float tt = input[g_idx] * 255.f;
    if (tt > 255)
        tt = 255;
    output[index] = tt;
}

void postprocess(uint8_t* output, float*input, int batchSize, int height, int width, int channel, cudaStream_t stream)
{
    int thread_count = batchSize * height * width * channel;
    int block = 512;
    int grid = (thread_count - 1) / block + 1;

    postprocess_kernel << <grid, block, 0, stream >> > (output, input, batchSize, height, width, channel, thread_count);
}


#include "postprocess.hpp"

namespace nvinfer1
{
    int PostprocessPluginV2::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        float* input = (float*)inputs[0];
        uint8_t* output = (uint8_t*)outputs[0];

        const int H = mPostprocess.H;
        const int W = mPostprocess.W;
        const int C = mPostprocess.C;

        postprocess(output, input, batchSize, H, W, C, stream);

        return 0;
    }
}