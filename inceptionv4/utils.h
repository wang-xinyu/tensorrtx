# ifndef TRTX_UTILS_H
# define TRTX_UTILS_H

#include <map>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "assert.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace nvinfer1;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)


std::map<std::string, Weights> loadWeights(const std::string input);

#endif // TRTX_UTILS_H