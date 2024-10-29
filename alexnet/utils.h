#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "macros.h"

#define WORKSPACE_SIZE (16 << 20)

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != cudaSuccess) {                              \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << std::endl;
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
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * wt.count));
        for (uint32_t x = 0; x < wt.count; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        weightMap[name] = wt;
    }

    return weightMap;
}
