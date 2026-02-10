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

/**
 * @brief a preprocess function aligning with ImageNet preprocess in torchvision, only support 3-channel image
 * 
 * @param img opencv image with BGR layout
 * @param bgr2rgb whether to convert BGR to RGB
 * @param mean subtract mean
 * @param std divide std
 * @param n batch size
 * @param h resize height
 * @param w resize width
 * @return std::vector<float> contiguous flatten image data in float32 type
 */
static std::vector<float> preprocess_img(cv::Mat& img, bool bgr2rgb, const float mean[3], const float std[3], int n,
                                         int h, int w) {
    const int c = img.channels();
    const std::size_t size = c * h * w;
    if (c != 3) {
        throw std::runtime_error("this demo only supports 3 channel input image.");
    }
    if (bgr2rgb) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }
    cv::resize(img, img, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    img.convertTo(img, CV_32FC3, 1.f / 255);
    img = (img - cv::Scalar(mean[0], mean[1], mean[2])) / cv::Scalar(std[0], std[1], std[2]);
    std::vector<float> chw(n * c * h * w, 0.f);

    // fill all batch with the same input image
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const cv::Vec3f v = img.at<cv::Vec3f>(y, x);
                chw[i * size + 0 * h * w + y * w + x] = v[0];
                chw[i * size + 1 * h * w + y * h + x] = v[1];
                chw[i * size + 2 * h * w + y * h + x] = v[2];
            }
        }
    }
    return chw;
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

static std::map<int, std::string> loadImagenetLabelMap(const std::string& path) {
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

static ILayer* addTransformLayer(INetworkDefinition* network, ITensor& input, bool bgr2rgb, const float mean[3],
                                 const float std[3]) {
    struct ScaleParams {
        std::array<float, 3> shift;
        std::array<float, 3> scale;
    };
    static std::vector<std::unique_ptr<ScaleParams>> gScaleParams;
    auto params = std::make_unique<ScaleParams>();
    params->shift = {-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]};
    params->scale = {1.f / (std[0] * 255.f), 1.f / (std[1] * 255.f), 1.f / (std[2] * 255.f)};

    static const Weights empty{DataType::kFLOAT, nullptr, 0ll};
    const Weights shift{DataType::kFLOAT, params->shift.data(), 3ll};
    const Weights scale{DataType::kFLOAT, params->scale.data(), 3ll};

    gScaleParams.emplace_back(std::move(params));

    ITensor* in = &input;
    if (input.getType() != DataType::kFLOAT) {
#if TRT_VERSION >= 8000
        auto* cast = network->addCast(input, DataType::kFLOAT);
        assert(cast);
        cast->setName("Cast to FP32");
        in = cast->getOutput(0);
#else
        auto* identity = network->addIdentity(input);
        assert(identity);
        identity->setName("Convert to FP32");
        identity->setOutputType(0, DataType::kFLOAT);
        in = identity->getOutput(0);
#endif
    }

    // Convert from NHWC to NCHW
    auto* perm = network->addShuffle(*in);
    assert(perm);
    perm->setName("NHWC -> NCHW");
    perm->setFirstTranspose(Permutation{0, 3, 1, 2});

    // Convert from BGR to RGB (optional)
    ITensor* data{nullptr};
    if (bgr2rgb) {
        auto add_slice = [&](int c, const char* name) -> ITensor* {
            auto dims = perm->getOutput(0)->getDimensions();
            Dims4 start = {0, c, 0, 0}, stride = {1, 1, 1, 1};
            Dims4 size = {dims.d[0], 1, dims.d[2], dims.d[3]};
            auto* _slice = network->addSlice(*perm->getOutput(0), start, size, stride);
            _slice->setName(name);
            assert(_slice && _slice->getNbOutputs() == 1);
            auto d = _slice->getOutput(0)->getDimensions();
            return _slice->getOutput(0);
        };
        ITensor* channels[] = {add_slice(2, "R"), add_slice(1, "G"), add_slice(0, "B")};
        auto* cat = network->addConcatenation(channels, 3);
        assert(cat);
        cat->setName("RGB");
        cat->setAxis(1);
        data = cat->getOutput(0);
    } else {
        data = perm->getOutput(0);
    }

    // Normalize
    auto* trans = network->addScale(*data, ScaleMode::kCHANNEL, shift, scale, empty);
    assert(trans);
    trans->setName("mean & std");
#if TRT_VERSION >= 8000
    trans->setChannelAxis(1);
#endif
    return trans;
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
