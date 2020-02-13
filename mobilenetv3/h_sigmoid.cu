#include <cuda_runtime.h>
#include <stdio.h>
#include "h_sigmoid.cuh"


__global__ void _hSigmoidKer(float const *in, float *out, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= size)
        return ;

    if (in[index] > 3 )
        out[index] = 1;
    else if (in[index] < -3)
        out[index] = 0;
    else
        out[index] = (in[index] + 3)/6;
}

extern "C" void cuh_sigmoid(float const *in, float *out, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    _hSigmoidKer<<<grid_size, block_size>>>(in, out, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch _leakyReluKer kernel (error code %s)!\n", cudaGetErrorString(err));
    }
}
