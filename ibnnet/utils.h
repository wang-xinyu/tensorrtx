#pragma once

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

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::map<std::string, Weights> loadWeights(const std::string file);

