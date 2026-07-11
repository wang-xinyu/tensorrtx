#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "block.h"
#include "config.h"
#include "model.h"

namespace fs = std::filesystem;

namespace {
nvinfer1::IIdentityLayer* castOutputToFloat(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                            const std::string& name) {
    auto* cast = network->addIdentity(input);
    assert(cast != nullptr);
    cast->setOutputType(0, nvinfer1::DataType::kFLOAT);
    cast->getOutput(0)->setName(name.c_str());
    return cast;
}
}  // namespace

nvinfer1::IHostMemory* buildEngineQwen3_0_6B(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_dir) {
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    nvinfer1::ITensor* input_ids =
            network->addInput(kInputIdsName, nvinfer1::DataType::kINT32, nvinfer1::Dims2{kBatchSize, -1});
    assert(input_ids != nullptr);
    nvinfer1::ITensor* cos_input =
            network->addInput(kInputCosName, nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{kBatchSize, -1, kHeadDim});
    nvinfer1::ITensor* sin_input =
            network->addInput(kInputSinName, nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{kBatchSize, -1, kHeadDim});
    nvinfer1::ITensor* mask_input =
            network->addInput(kInputMaskName, nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{kBatchSize, 1, -1, -1});
    fs::path embedding_path = fs::path(wts_dir) / "qwen3-0.6_embedding.wts";
    std::map<std::string, nvinfer1::Weights> embeddingWeightMap = loadWeights(embedding_path.string());
    auto* embedding_layer = Qwen3Embedding(network, embeddingWeightMap, *input_ids);
    assert(embedding_layer != nullptr);
    nvinfer1::ITensor* hidden_states = embedding_layer->getOutput(0);
    std::vector<std::map<std::string, nvinfer1::Weights>> layerWeightMaps;
    layerWeightMaps.reserve(kNumHiddenLayers);

    for (int layer_idx = 0; layer_idx < kNumHiddenLayers; ++layer_idx) {
        const std::string key_cache_input_name = getKeyCacheInputName(layer_idx);
        const std::string value_cache_input_name = getValueCacheInputName(layer_idx);
        nvinfer1::ITensor* k_cache_input =
                network->addInput(key_cache_input_name.c_str(), nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, -1, kHeadDim});
        nvinfer1::ITensor* v_cache_input =
                network->addInput(value_cache_input_name.c_str(), nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, -1, kHeadDim});
        assert(k_cache_input != nullptr);
        assert(v_cache_input != nullptr);

        fs::path full_path = fs::path(wts_dir) / getLayerWeightFileName(layer_idx);
        layerWeightMaps.push_back(loadWeights(full_path.string()));
        auto& layerWeightMap = layerWeightMaps.back();

        auto decoder = Qwen3DecoderLayer(network, layerWeightMap, *hidden_states, *cos_input, *sin_input,
                                         *k_cache_input, *v_cache_input, *mask_input, layer_idx);
        hidden_states = std::get<0>(decoder)->getOutput(0);

        const std::string key_cache_output_name = getKeyCacheOutputName(layer_idx);
        auto* k_cache_output = castOutputToFloat(network, *std::get<1>(decoder)->getOutput(0), key_cache_output_name);
        network->markOutput(*k_cache_output->getOutput(0));

        const std::string value_cache_output_name = getValueCacheOutputName(layer_idx);
        auto* v_cache_output = castOutputToFloat(network, *std::get<2>(decoder)->getOutput(0), value_cache_output_name);
        network->markOutput(*v_cache_output->getOutput(0));
    }

    fs::path norm_path = fs::path(wts_dir) / "qwen3-0.6_norm.wts";
    std::map<std::string, nvinfer1::Weights> normWeightMap = loadWeights(norm_path.string());
    auto* final_hidden_states = Qwen3RMSNorm(network, normWeightMap, *hidden_states, 3, "");
    assert(final_hidden_states != nullptr);
    auto* last_hidden_state = GatherLastToken(network, *final_hidden_states->getOutput(0));
    assert(last_hidden_state != nullptr);

    fs::path lm_head_path = fs::path(wts_dir) / "qwen3-0.6_lm_head.wts";
    std::map<std::string, nvinfer1::Weights> lmHeadWeightMap = loadWeights(lm_head_path.string());
    auto* logits = addLinear(network, lmHeadWeightMap, *last_hidden_state->getOutput(0), kVocabSize, "");
    assert(logits != nullptr);

    auto* logits_output = castOutputToFloat(network, *logits->getOutput(0), kOutputTensorName);
    network->markOutput(*logits_output->getOutput(0));

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(kInputIdsName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{kBatchSize, 1});
    profile->setDimensions(kInputIdsName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{kBatchSize, 32});
    profile->setDimensions(kInputIdsName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{kBatchSize, kMaxSeqLen});
    profile->setDimensions(kInputCosName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{kBatchSize, 1, kHeadDim});
    profile->setDimensions(kInputCosName, nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims3{kBatchSize, 32, kHeadDim});
    profile->setDimensions(kInputCosName, nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims3{kBatchSize, kMaxSeqLen, kHeadDim});
    profile->setDimensions(kInputSinName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{kBatchSize, 1, kHeadDim});
    profile->setDimensions(kInputSinName, nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims3{kBatchSize, 32, kHeadDim});
    profile->setDimensions(kInputSinName, nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims3{kBatchSize, kMaxSeqLen, kHeadDim});
    for (int layer_idx = 0; layer_idx < kNumHiddenLayers; ++layer_idx) {
        const std::string key_cache_input_name = getKeyCacheInputName(layer_idx);
        const std::string value_cache_input_name = getValueCacheInputName(layer_idx);
        profile->setDimensions(key_cache_input_name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, 0, kHeadDim});
        profile->setDimensions(key_cache_input_name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, 32, kHeadDim});
        profile->setDimensions(key_cache_input_name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, kMaxSeqLen, kHeadDim});
        profile->setDimensions(value_cache_input_name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                               nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, 0, kHeadDim});
        profile->setDimensions(value_cache_input_name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, 32, kHeadDim});
        profile->setDimensions(value_cache_input_name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims4{kBatchSize, kNumKeyValueHeads, kMaxSeqLen, kHeadDim});
    }
    profile->setDimensions(kInputMaskName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{kBatchSize, 1, 1, 1});
    profile->setDimensions(kInputMaskName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{kBatchSize, 1, 32, 64});
    profile->setDimensions(kInputMaskName, nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4{kBatchSize, 1, kMaxSeqLen, kMaxSeqLen * 2});
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 * (1 << 30));  // 1G workspace

#if defined(USE_BF16)
    config->setFlag(nvinfer1::BuilderFlag::kBF16);
#elif defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform supports int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kClsInputW, kClsInputH, kInputQuantizationFolder,
                                                  "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    // Begin building the engine; this may take a while
    std::cout << "[DBG] buildEngineQwen3_0_6B: before buildSerializedNetwork" << std::endl;
    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "[DBG] buildEngineQwen3_0_6B: after buildSerializedNetwork" << std::endl;
    if (!serialized_model) {
        std::cerr << "Failed to build engine!" << std::endl;
    } else {
        std::cout << "Build engine successfully!" << std::endl;
    }

    // Cleanup the network definition and allocated weights
    delete network;

    freeWeights(embeddingWeightMap);
    for (auto& layerWeightMap : layerWeightMaps) {
        freeWeights(layerWeightMap);
    }
    freeWeights(normWeightMap);
    freeWeights(lmHeadWeightMap);

    return serialized_model;
}
