# ifndef TRTX_UTILS_H
# define TRTX_UTILS_H

#include <map>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "assert.h"
#include <fstream>
#include <iostream>
#include <memory>

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

using namespace nvinfer1;

std::map<std::string, Weights> loadWeights(const std::string input);

#endif // TRTX_UTILS_H