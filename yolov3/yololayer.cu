#include <cmath>
#include <cuda_runtime.h>
#include "yololayer.h"
#include <stdio.h>

using namespace nvinfer1;

YoloLayerPlugin::YoloLayerPlugin(int class_num, int yolo_grid, int input_dim, int cuda_block, float anchors[6]) {
    class_num_ = class_num;
    yolo_grid_ = yolo_grid;
    input_dim_ = input_dim;
    cuda_block_ = cuda_block;
    memcpy(anchors_, anchors, 6 * sizeof(float));
}

YoloLayerPlugin::~YoloLayerPlugin() {}

// create the plugin at runtime from a byte stream
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
{
    char *d = (char*)(data), *a = d;

    class_num_ = *reinterpret_cast<int*>(d);
    d += sizeof(int);
    yolo_grid_ = *reinterpret_cast<int*>(d);
    d += sizeof(int);
    input_dim_ = *reinterpret_cast<int*>(d);
    d += sizeof(int);
    cuda_block_ = *reinterpret_cast<int*>(d);
    d += sizeof(int);
    memcpy(anchors_, d, 6 * sizeof(float));
    d += 6 * sizeof(float);

    if (d != a + length) {
        fprintf(stderr, "deserialize yololayer plugin failed! \n");
    }
}

void YoloLayerPlugin::serialize(void* buffer)
{
    char* d = static_cast<char*>(buffer), *a = d;

    *reinterpret_cast<int*>(d) = class_num_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = yolo_grid_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = input_dim_;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = cuda_block_;
    d += sizeof(int);
    memcpy(d, anchors_, 6 * sizeof(float));
    d += 6 * sizeof(float);

    if (d != a + getSerializationSize()) {
        fprintf(stderr, "serialize yololayer plugin failed! \n");
    }
}

size_t YoloLayerPlugin::getSerializationSize()
{  
    return sizeof(int) * 4 + sizeof(float) * 6;
}

int YoloLayerPlugin::initialize()
{ 
    return 0;
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    return Dims3(3 * yolo_grid_ * yolo_grid_, 1, 7);
}

__device__ float Logist(float data){ return 1./(1. + exp(-data)); };

__global__ void CalDetection(const float *input, float *output, int noElements,
        int yolo_grid, const float anchors[6], int classes, int input_dim) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= noElements) return;

    int total_grid = yolo_grid * yolo_grid;
    int info_len_i = 5 + classes;
    int info_len_o = 7;
    int input_col = idx;
    int out_row = input_col;

    for (int k = 0; k < 3; ++k) {
        int class_id = 0;
        float max_prob = 0.0;
        for (int i = 5; i < info_len_i; ++i) {
            float p = Logist(input[input_col + k * info_len_i * total_grid + i * total_grid]);
            if (p > max_prob) {
                max_prob = p;
                class_id = i - 5;
            }
        }

        int row = idx / yolo_grid;
        int col = idx % yolo_grid;

        //Location
        output[out_row * info_len_o * 3 + info_len_o * k + 0] = (col + Logist(input[input_col + k * info_len_i * total_grid + 0 * total_grid])) * input_dim / yolo_grid;
        output[out_row * info_len_o * 3 + info_len_o * k + 1] = (row + Logist(input[input_col + k * info_len_i * total_grid + 1 * total_grid])) * input_dim / yolo_grid;
        output[out_row * info_len_o * 3 + info_len_o * k + 2] = exp(input[input_col + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
        output[out_row * info_len_o * 3 + info_len_o * k + 3] = exp(input[input_col + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
        output[out_row * info_len_o * 3 + info_len_o * k + 4] =  Logist(input[input_col + k * info_len_i * total_grid + 4 * total_grid]);
        output[out_row * info_len_o * 3 + info_len_o * k + 5] =  class_id;
        output[out_row * info_len_o * 3 + info_len_o * k + 6] =  max_prob;
    }
}

void YoloLayerPlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize) {
    void* anchors_gpu;
    cudaMalloc(&anchors_gpu, 6 * sizeof(float));
    cudaMemcpy(anchors_gpu, anchors_, 6 * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = cuda_block_;
    int grid_size = (yolo_grid_ * yolo_grid_ * batchSize + block_size - 1) / block_size;
    CalDetection<<<grid_size, block_size>>>
        (inputs[0], output, yolo_grid_ * yolo_grid_ * batchSize, yolo_grid_, (float *)anchors_gpu, class_num_, input_dim_);

    cudaFree(anchors_gpu);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch caldetect kernel (error code %s)!\n", cudaGetErrorString(err));
    }
}

int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) {
    //assert(batchSize == 1);
    //GPU
    //CUDA_CHECK(cudaStreamSynchronize(stream));
    forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
    return 0;
}
