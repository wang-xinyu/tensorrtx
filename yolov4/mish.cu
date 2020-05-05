#include <cmath>
#include <stdio.h>
#include <cassert>
#include "mish.h"

namespace nvinfer1
{
    MishPlugin::MishPlugin(const int cudaThread) : thread_count_(cudaThread)
    {
    }
    
    MishPlugin::~MishPlugin()
    {
    }
    
    // create the plugin at runtime from a byte stream
    MishPlugin::MishPlugin(const void* data, size_t length)
    {
        assert(length == sizeof(input_size_));
        input_size_ = *reinterpret_cast<const int*>(data);
    }

    void MishPlugin::serialize(void* buffer)
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
    }
    
    size_t MishPlugin::getSerializationSize()
    {  
        return sizeof(input_size_);
    }

    int MishPlugin::initialize()
    { 
        return 0;
    }
    
    Dims MishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    __device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}

    __device__ float softplus_kernel(float x, float threshold = 20) {
        if (x > threshold) return x;                // too large
        else if (x < -threshold) return expf(x);    // too small
        return logf(expf(x) + 1);
    }

    __global__ void mish_kernel(const float *input, float *output, int num_elem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        //float t = exp(input[idx]);
        //if (input[idx] > 20.0) {
        //    t *= t;
        //    output[idx] = (t - 1.0) / (t + 1.0);
        //} else {
        //    float tt = t * t;
        //    output[idx] = (tt + 2.0 * t) / (tt + 2.0 * t + 2.0);
        //}
        //output[idx] *= input[idx];
        output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
    }

    void MishPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        mish_kernel<<<grid_size, block_size>>>(inputs[0], output, input_size_ * batchSize);
    }


    int MishPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }
}

