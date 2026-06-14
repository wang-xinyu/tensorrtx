#pragma once
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
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
#if TRT_VERSION_LT(8, 0, 0)
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
#if TRT_VERSION_GE(8, 5, 1)
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

static constexpr int32_t kBenchmarkRuns = 200;
static constexpr std::size_t kMaxFirstOutputs = 10;

inline auto percentile(const std::vector<double>& sorted, double percent) -> double {
    assert(!sorted.empty());
    const double rank = percent / 100.0 * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(rank);
    const auto upper = std::min<std::size_t>(lower + 1, sorted.size() - 1);
    if (lower == upper) {
        return sorted[lower];
    }
    const double weight = rank - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

inline void printBenchmark(const std::string& tag, const std::vector<double>& latenciesMs, int64_t batchSize = 1) {
    assert(!latenciesMs.empty());
    auto sorted = latenciesMs;
    std::sort(sorted.begin(), sorted.end());
    const double avg =
            std::accumulate(latenciesMs.begin(), latenciesMs.end(), 0.0) / static_cast<double>(latenciesMs.size());
    std::cout << "[" << tag << "] benchmark_runs=" << latenciesMs.size() << " batch=" << batchSize << " AVG=" << avg
              << "ms P50=" << percentile(sorted, 50.0) << "ms P90=" << percentile(sorted, 90.0)
              << "ms P95=" << percentile(sorted, 95.0) << "ms P99=" << percentile(sorted, 99.0) << "ms\n";
}

inline void printFirstOutputs(const std::string& tag, const float* values, std::size_t count) {
    const auto limit = std::min(count, kMaxFirstOutputs);
    std::cout << "[" << tag << "] first_outputs=";
    for (std::size_t i = 0; i < limit; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::setprecision(4) << values[i];
    }
    std::cout << '\n';
}
