#pragma once
#include <cuda_fp16.h>
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
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "macros.h"

constexpr const std::size_t WORKSPACE_SIZE = 16 << 20;
namespace {
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
        nvinfer1::Weights wt{.type = nvinfer1::DataType::kFLOAT, .values = nullptr, .count = 0};

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

/**
 * @brief a preprocess function aligning with ImageNet preprocess in torchvision, only support 3-channel image
 * 
 * @param img opencv image with BGR layout
 * @param bgr2rgb whether to convert BGR to RGB
  * @param mean_std subtract mean, then divide std
  * @param n batch size
  * @param h resize height
  * @param w resize width
  * @return std::vector<half> contiguous flatten image data in fp16 type (CHW)
  */
static auto preprocess_img(cv::Mat& img, bool bgr2rgb, const std::array<const float, 3>& mean,
                           const std::array<const float, 3>& std, int64_t n, int32_t h, int32_t w) {
    const auto c = img.channels();
    const auto size = c * h * w;
    if (c != 3) {
        std::cerr << "this demo only supports 3 channel input image.\n";
        std::abort();
    }
    if (bgr2rgb) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    cv::resize(img, img, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    // Keep preprocessing in fp32 on CPU for correctness, then pack to fp16 CHW for TensorRT input.
    img.convertTo(img, CV_32FC3, 1.f / 255.f);
    img = (img - cv::Scalar(mean[0], mean[1], mean[2])) / cv::Scalar(std[0], std[1], std[2]);
    std::vector<half> chw(static_cast<std::size_t>(n) * c * h * w);

    // fill all batch with the same input image
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const cv::Vec3f v = img.at<cv::Vec3f>(y, x);
                chw[i * size + 0 * h * w + y * w + x] = __float2half(v[0]);
                chw[i * size + 1 * h * w + y * w + x] = __float2half(v[1]);
                chw[i * size + 2 * h * w + y * w + x] = __float2half(v[2]);
            }
        }
    }
    return chw;
}

static auto topk(const std::vector<float>& v, int k) -> std::vector<std::pair<int, float>> {
    if (k <= 0)
        return {};
    auto stride = std::min<std::ptrdiff_t>(k, static_cast<std::ptrdiff_t>(v.size()));

    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + stride, idx.end(), [&](int a, int b) { return v[a] > v[b]; });

    std::vector<std::pair<int, float>> out;
    out.reserve(stride);
    for (int i = 0; i < stride; ++i)
        out.emplace_back(idx[i], v[idx[i]]);
    return out;
}

static auto loadImagenetLabelMap(const std::string& path) {
    std::map<int, std::string> labels;
    std::ifstream in(path);
    if (!in.is_open()) {
        return labels;
    }
    std::string line;
    while (std::getline(in, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        auto first_quote = line.find('\'', colon);
        if (first_quote == std::string::npos) {
            continue;
        }
        auto second_quote = line.find('\'', first_quote + 1);
        if (second_quote == std::string::npos) {
            continue;
        }
        int idx = std::stoi(line.substr(0, colon));
        labels[idx] = line.substr(first_quote + 1, second_quote - first_quote - 1);
    }
    return labels;
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

}  // namespace
