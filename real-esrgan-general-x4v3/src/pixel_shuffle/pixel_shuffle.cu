#include <cuda_runtime.h>
#include <string>
#include "pixel_shuffle/pixel_shuffle.hpp"

// CUDA kernel for PixelShuffle
__global__ void PixelShuffleKernel(const float* input, float* output, int batchSize, int channels, int height,
                                   int width, int upscaleFactor) {
    int outHeight = height * upscaleFactor;
    int outWidth = width * upscaleFactor;
    int outChannels = channels / (upscaleFactor * upscaleFactor);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * outChannels * outHeight * outWidth)
        return;

    int out_w = idx % outWidth;
    int out_h = (idx / outWidth) % outHeight;
    int out_c = (idx / outWidth / outHeight) % outChannels;
    int b = idx / (outWidth * outHeight * outChannels);

    int in_c =
            out_c * upscaleFactor * upscaleFactor + (out_h % upscaleFactor) * upscaleFactor + (out_w % upscaleFactor);
    int in_h = out_h / upscaleFactor;
    int in_w = out_w / upscaleFactor;

    output[idx] = input[((b * channels + in_c) * height + in_h) * width + in_w];
}

int32_t PixelShufflePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);

    int batchSize = inputDesc[0].dims.d[0];
    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];
    int upscaleFactor = mUpscaleFactor;

    int outChannels = channels / (upscaleFactor * upscaleFactor);
    int outHeight = height * upscaleFactor;
    int outWidth = width * upscaleFactor;

    int numElements = batchSize * outChannels * outHeight * outWidth;

    PixelShuffleKernel<<<(numElements + 255) / 256, 256>>>(input, output, batchSize, channels, height, width,
                                                           upscaleFactor);
    return cudaGetLastError() != cudaSuccess;
}
