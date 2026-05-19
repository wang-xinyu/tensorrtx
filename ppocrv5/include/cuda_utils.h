#pragma once

#include <cuda_runtime_api.h>
#include <iostream>

#define CUDA_CHECK(call)                                                                                         \
    do {                                                                                                         \
        cudaError_t status = call;                                                                               \
        if (status != cudaSuccess) {                                                                             \
            std::cerr << "CUDA failure: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                              \
            abort();                                                                                             \
        }                                                                                                        \
    } while (0)
