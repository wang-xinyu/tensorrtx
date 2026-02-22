#pragma once
#include <cuda_runtime_api.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include "macros.h"

using namespace nvinfer1;

constexpr const std::size_t WORKSPACE_SIZE = 16 << 20;

#define CHECK(status)                                     \
    do {                                                  \
        auto ret = (status);                              \
        if (ret != cudaSuccess) {                         \
            std::cerr << "Cuda failure: " << ret << "\n"; \
            std::abort();                                 \
        }                                                 \
    } while (0)

static void checkTrtEnv(int device = 0) {
#if TRT_VERSION < 8000
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, device));
    const int sm = prop.major * 10 + prop.minor;
    if (sm > 86) {
        std::cerr << "TensorRT < 8 does not support SM > 86 on this GPU.";
        std::abort();
    }
#endif
}

/**
 * @brief TensorRT weight files have a simple space delimited format:
 * [type] [size] <data x size in hex>
 * 
 * @param file input weight file path
 * @return std::map<std::string, nvinfer1::Weights> 
 */
static auto loadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << "\n";
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> wt.count;

        // Load blob
        auto* val = new uint32_t[wt.count];
        input >> std::hex;
        for (auto x = 0ll; x < wt.count; ++x) {
            input >> val[x];
        }
        wt.values = val;
        weightMap[name] = wt;
    }

    return weightMap;
}

static size_t getSize(DataType dt) {
    switch (dt) {
#if TRT_VERSION >= 8510
        case DataType::kUINT8:
#endif
        case DataType::kINT8:
            return sizeof(int8_t);
        case DataType::kFLOAT:
            return sizeof(float);
        case DataType::kHALF:
            return sizeof(int16_t);
        case DataType::kINT32:
            return sizeof(int32_t);
        default: {
            std::cerr << "Unsupported data type\n";
            std::abort();
        }
    }
}
