#pragma once

#include <NvInfer.h>
#include <cuda.h>

#if CUDA_VERSION >=11000
#define CUDA_11
#endif

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif
