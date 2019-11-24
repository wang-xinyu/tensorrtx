#include <cuda_runtime.h>
#include <stdio.h>
#include "leaky.cuh"


__global__ void _leakyReluKer(float const *in, float *out, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= size)
        return ;

    if (in[index] < 0)
        out[index] = in[index] * 0.1;
    else
        out[index] = in[index];
}

extern "C" void culeaky(float const *in, float *out, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    _leakyReluKer<<<grid_size, block_size>>>(in, out, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch _leakyReluKer kernel (error code %s)!\n", cudaGetErrorString(err));
    }
}
