#pragma once

#include <cmath>
#include <string>
#include <vector>

#define USE_BF16
// #define USE_FP32
// #define USE_FP16

static const char* kInputIdsName = "input_ids";
static const char* kPosInputIdsName = "pos_input_ids";
static const char* kInputCosName = "input_cos";
static const char* kInputSinName = "input_sin";
static const char* kKeyCacheInputName = "key_cache_input";
static const char* kValueCacheInputName = "value_cache_input";
static const char* kInputMaskName = "input_mask";
static const char* kInputLayerName = "input_layer";
static const char* kOutputTensorName = "output";
static const char* kKeyCacheOutputTensorName = "key_cache_output";
static const char* kValueCacheOutputTensorName = "value_cache_output";
static const int kBatchSize = 1;
static const int kMaxSeqLen = 1024;
static const int kVocabSize = 151936;
static const int kHiddenSize = 1024;
static const int kIntermediateSize = 3072;
static const int kHeadDim = 128;
static const int kNumAttentionHeads = 16;
static const int kNumHiddenLayers = 28;
static const int kNumKeyValueHeads = 8;
static const int kBosTokenId = 151643;
static const int kPadTokenId = 151643;
static const bool kDoSample = true;
inline const std::vector<int32_t> kEosTokenIds = {151645, 151643};
static const float kTemperature = 0.6f;
static const int kTopK = 20;
static const float kTopP = 0.95f;
static const float kRopeTheta = 1000000.0f;
static const float kPartialRotaryFactor = 1.0f;
static const float kRopeAttentionScaling = 1.0f;
static const float kRmsNormEps = 1e-6f;
static const float kQKScale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
static const int kGpuId = 0;
static const char* kInputQuantizationFolder = "./coco_calib";

inline std::string getKeyCacheInputName(int layer_idx) {
    return std::string(kKeyCacheInputName) + "_" + std::to_string(layer_idx);
}

inline std::string getValueCacheInputName(int layer_idx) {
    return std::string(kValueCacheInputName) + "_" + std::to_string(layer_idx);
}

inline std::string getKeyCacheOutputName(int layer_idx) {
    return std::string(kKeyCacheOutputTensorName) + "_" + std::to_string(layer_idx);
}

inline std::string getValueCacheOutputName(int layer_idx) {
    return std::string(kValueCacheOutputTensorName) + "_" + std::to_string(layer_idx);
}

inline std::string getLayerWeightFileName(int layer_idx) {
    return "qwen3-0.6_layer_" + std::to_string(layer_idx) + ".wts";
}
