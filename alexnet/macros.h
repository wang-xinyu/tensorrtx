#pragma once

#include <NvInfer.h>

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif  // API_EXPORTS

#define TRT_VERSION \
    ((NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + (NV_TENSORRT_PATCH * 10) + NV_TENSORRT_BUILD)

#if TRT_VERSION >= 8000
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif
