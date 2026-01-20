#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

#define WORKSPACE_SIZE (16 << 20)

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != cudaSuccess) {                              \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

static void checkTrtEnv(int device = 0) {
#if TRT_VERSION < 7220
#error "TensorRT >= 7.2.2 is required for this demo."
#endif
#if TRT_VERSION < 8000
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, device));
    const int sm = prop.major * 10 + prop.minor;
    if (sm > 86) {
        throw std::runtime_error("TensorRT < 8 does not support SM > 86 on this GPU.");
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
static std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file) {
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

static std::vector<std::pair<int, float>> topk(const std::vector<float>& v, int k) {
    if (k <= 0)
        return {};
    k = std::min<int>(k, v.size());

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });

    std::vector<std::pair<int, float>> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i)
        out.emplace_back(idx[i], v[idx[i]]);
    return out;
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
        default:
            throw std::runtime_error("Unsupported data type");
    }
}
