#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cstdint>

#define CUDA_ALIGN 256

template <typename T>
inline size_t get_size_aligned(size_t num_elem) {
    size_t size = num_elem * sizeof(T);
    size_t extra_align = 0;
    if (size % CUDA_ALIGN != 0) {
        extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
    }
    return size + extra_align;
}

template <typename T>
inline T *get_next_ptr(size_t num_elem, void *&workspace, size_t &workspace_size) {
    size_t size = get_size_aligned<T>(num_elem);
    if (size > workspace_size) {
        throw std::runtime_error("Workspace is too small!");
    }
    workspace_size -= size;
    T *ptr = reinterpret_cast<T *>(workspace);
    workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
    return ptr;
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK
