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

#define TRT_VERSION_ENCODE(major, minor, patch, build) \
    (((major) * 1000000) + ((minor) * 10000) + ((patch) * 100) + (build))
#define TRT_VERSION TRT_VERSION_ENCODE(NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD)
#define TRT_VERSION_GE(major, minor, patch) (TRT_VERSION >= TRT_VERSION_ENCODE((major), (minor), (patch), 0))
#define TRT_VERSION_LT(major, minor, patch) (!TRT_VERSION_GE((major), (minor), (patch)))

#if TRT_VERSION_LT(7, 2, 2)
#error "TensorRT >= 7.2.2 is required for this demo."
#endif

#if TRT_VERSION_GE(8, 0, 0)
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif
