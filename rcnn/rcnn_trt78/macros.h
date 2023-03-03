#pragma once

#include <NvInfer.h>

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#define CUDA_11 // CUDA 11 TensorRT 8
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif
