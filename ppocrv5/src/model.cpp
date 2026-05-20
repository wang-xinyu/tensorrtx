#include "model.h"

#include "block.h"
#include "config.h"
#include "ppocrv5_db_layer.h"
#include "ppocrv5_rtdetr_layer.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

bool isMobileWts(const std::string& wtsPath) {
    return wtsPath.find("mobile") != std::string::npos || wtsPath.find("ppocrv5_det") != std::string::npos ||
           wtsPath.find("ppocrv5_rec") != std::string::npos;
}

bool contains(const std::string& text, const std::string& pattern) {
    return text.find(pattern) != std::string::npos;
}

bool startsWith(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool matchesFormulaDebugStage(const std::string& stage) {
    const char* debugStage = std::getenv("PPOCRV5_DEBUG_FORMULA_STAGE");
    return debugStage && stage == debugStage;
}

bool markFormulaDebugStage(INetworkDefinition* network, const std::string& stage, ITensor& tensor) {
    if (!matchesFormulaDebugStage(stage)) {
        return false;
    }
    tensor.setName("output");
    network->markOutput(tensor);
    return true;
}

bool hasFormulaDebugPrefix(const std::string& prefix) {
    const char* debugStage = std::getenv("PPOCRV5_DEBUG_FORMULA_STAGE");
    return debugStage && startsWith(debugStage, prefix);
}

int readEnvInt(const char* name, int defaultValue, int minValue, int maxValue) {
    const char* text = std::getenv(name);
    if (!text || !text[0]) {
        return defaultValue;
    }

    int value = std::atoi(text);
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

void setCommonBuilderConfig(IBuilder* builder, IBuilderConfig* config, DataType dt) {
    int workspaceMiB = readEnvInt("PPOCRV5_TRT_WORKSPACE_MB", static_cast<int>(kMaxBuilderWorkspaceMiB), 1, 8192);
    int builderOptLevel = readEnvInt("PPOCRV5_BUILDER_OPT_LEVEL", kDefaultBuilderOptLevel, 0, kMaxBuilderOptLevel);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, static_cast<size_t>(workspaceMiB) << 20);
    config->setBuilderOptimizationLevel(builderOptLevel);
    if (dt == DataType::kHALF && builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
    }
}

IHostMemory* buildSerializedNetwork(IBuilder* builder, IBuilderConfig* config, INetworkDefinition* network,
                                    std::map<std::string, Weights>& weightMap) {
    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    delete network;
    freeWeights(weightMap);
    if (!serializedModel) {
        throw std::runtime_error("failed to build TensorRT engine");
    }
    std::cout << "Build engine successfully!" << std::endl;
    return serializedModel;
}

void addDetOptimizationProfile(IBuilder* builder, IBuilderConfig* config) {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(kDetInputTensorName, OptProfileSelector::kMIN, Dims4{1, 3, 64, 64});
    profile->setDimensions(kDetInputTensorName, OptProfileSelector::kOPT, Dims4{1, 3, kDetResizeLong, kDetResizeLong});
    profile->setDimensions(kDetInputTensorName, OptProfileSelector::kMAX, Dims4{1, 3, 1536, 1536});
    config->addOptimizationProfile(profile);
}

void addRecOptimizationProfile(IBuilder* builder, IBuilderConfig* config) {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(kRecInputTensorName, OptProfileSelector::kMIN, Dims4{1, 3, kRecInputH, kRecMinW});
    profile->setDimensions(kRecInputTensorName, OptProfileSelector::kOPT, Dims4{1, 3, kRecInputH, kRecOptW});
    profile->setDimensions(kRecInputTensorName, OptProfileSelector::kMAX, Dims4{1, 3, kRecInputH, kRecMaxW});
    config->addOptimizationProfile(profile);
}

IHostMemory* buildPPLCNetX1_0Model(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    bool textline = contains(wtsPath, "textline");
    int inputH = textline ? 80 : 224;
    int inputW = textline ? 160 : 224;
    int classCount = contains(wtsPath, "doc_ori") ? 4 : 2;
    DimsHW downStride = textline ? DimsHW{2, 1} : DimsHW{2, 2};

    ITensor* data = network->addInput("x", dt, Dims4{1, 3, inputH, inputW});
    if (!data) {
        throw std::runtime_error("failed to add PP-LCNet input");
    }
    const char* debugStage = std::getenv("PPOCRV5_DEBUG_LCNET_STAGE");
    auto markDebugStage = [&](const char* stage, ITensor& tensor) -> bool {
        if (!debugStage || std::strcmp(debugStage, stage) != 0) {
            return false;
        }
        tensor.setName("output");
        network->markOutput(tensor);
        return true;
    };

    IElementWiseLayer* stem =
            convBnHSwishByPrefix(network, weightMap, *data, 16, 3, 2, 1, 1, "conv2d_0", "batch_norm2d_0");
    if (markDebugStage("stem", *stem->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b0 =
            ppLcNetBlock(network, weightMap, *stem->getOutput(0), 16, 32, 1, 1, 2, 2, 3, DimsHW{1, 1}, false);
    if (markDebugStage("b0", *b0->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b1 =
            ppLcNetBlock(network, weightMap, *b0->getOutput(0), 32, 64, 3, 3, 4, 4, 3, downStride, false);
    if (markDebugStage("b1", *b1->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b2 =
            ppLcNetBlock(network, weightMap, *b1->getOutput(0), 64, 64, 5, 5, 6, 6, 3, DimsHW{1, 1}, false);
    if (markDebugStage("b2", *b2->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b3 =
            ppLcNetBlock(network, weightMap, *b2->getOutput(0), 64, 128, 7, 7, 8, 8, 3, downStride, false);
    if (markDebugStage("b3", *b3->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b4 =
            ppLcNetBlock(network, weightMap, *b3->getOutput(0), 128, 128, 9, 9, 10, 10, 3, DimsHW{1, 1}, false);
    if (markDebugStage("b4", *b4->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b5 =
            ppLcNetBlock(network, weightMap, *b4->getOutput(0), 128, 256, 11, 11, 12, 12, 3, downStride, false);
    if (markDebugStage("b5", *b5->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b6 =
            ppLcNetBlock(network, weightMap, *b5->getOutput(0), 256, 256, 13, 13, 14, 14, 5, DimsHW{1, 1}, false);
    if (markDebugStage("b6", *b6->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b7 =
            ppLcNetBlock(network, weightMap, *b6->getOutput(0), 256, 256, 15, 15, 16, 16, 5, DimsHW{1, 1}, false);
    if (markDebugStage("b7", *b7->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b8 =
            ppLcNetBlock(network, weightMap, *b7->getOutput(0), 256, 256, 17, 17, 18, 18, 5, DimsHW{1, 1}, false);
    if (markDebugStage("b8", *b8->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b9 =
            ppLcNetBlock(network, weightMap, *b8->getOutput(0), 256, 256, 19, 19, 20, 20, 5, DimsHW{1, 1}, false);
    if (markDebugStage("b9", *b9->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b10 =
            ppLcNetBlock(network, weightMap, *b9->getOutput(0), 256, 256, 21, 21, 22, 22, 5, DimsHW{1, 1}, false);
    if (markDebugStage("b10", *b10->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b11 =
            ppLcNetBlock(network, weightMap, *b10->getOutput(0), 256, 512, 23, 23, 26, 24, 5, downStride, true);
    if (markDebugStage("b11", *b11->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* b12 =
            ppLcNetBlock(network, weightMap, *b11->getOutput(0), 512, 512, 27, 25, 30, 26, 5, DimsHW{1, 1}, true);
    if (markDebugStage("b12", *b12->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }

    IReduceLayer* pool = addGlobalAvgPool2d(network, *b12->getOutput(0), "pool2d_0");
    if (markDebugStage("pool", *pool->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv =
            addConv2d(network, *pool->getOutput(0), getWeightsByPrefix(weightMap, "conv2d_31.w_0"), emptywts, 1280,
                      DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, DimsHW{1, 1}, 1, "conv2d_31");
    IElementWiseLayer* act = addHardSwish(network, *conv->getOutput(0));
    act->setName("conv2d_31_hardswish");
    if (markDebugStage("head", *act->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* dropout = addScalarMul(network, weightMap, *act->getOutput(0), 0.8f, "dropout_0");
    IShuffleLayer* flatten = addShuffle(network, *dropout->getOutput(0), Dims2{0, 1280}, "flatten_0");
    IElementWiseLayer* logits =
            addLinear2dByPrefix(network, weightMap, *flatten->getOutput(0), 1280, classCount, "linear_0");
    if (markDebugStage("logits", *logits->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ISoftMaxLayer* prob = addSoftmax(network, *logits->getOutput(0), 1 << 1, "softmax_0");
    prob->getOutput(0)->setName("output");
    network->markOutput(*prob->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

Weights addOwnedFloatWeights(std::map<std::string, Weights>& weightMap, const std::string& name,
                             const std::vector<float>& values) {
    auto* data = reinterpret_cast<float*>(malloc(sizeof(float) * values.size()));
    std::memcpy(data, values.data(), sizeof(float) * values.size());
    Weights wt{DataType::kFLOAT, data, static_cast<int64_t>(values.size())};
    weightMap[name] = wt;
    return wt;
}

Dims makeDims1(int d0) {
    Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = d0;
    return dims;
}

Dims makeDimsN(const std::vector<int>& values) {
    Dims dims{};
    dims.nbDims = static_cast<int32_t>(values.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = values[i];
    }
    return dims;
}

Weights addOwnedIntWeights(std::map<std::string, Weights>& weightMap, const std::string& name,
                           const std::vector<int32_t>& values) {
    auto* data = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * values.size()));
    std::memcpy(data, values.data(), sizeof(int32_t) * values.size());
    Weights wt{DataType::kINT32, data, static_cast<int64_t>(values.size())};
    weightMap[name] = wt;
    return wt;
}

Weights addOwnedBoolWeights(std::map<std::string, Weights>& weightMap, const std::string& name,
                            const std::vector<bool>& values) {
    auto* data = reinterpret_cast<bool*>(malloc(sizeof(bool) * values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    Weights wt{DataType::kBOOL, data, static_cast<int64_t>(values.size())};
    weightMap[name] = wt;
    return wt;
}

Weights addOwnedInt64Weights(std::map<std::string, Weights>& weightMap, const std::string& name,
                             const std::vector<int64_t>& values) {
#if NV_TENSORRT_MAJOR >= 10
    auto* data = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t) * values.size()));
    std::memcpy(data, values.data(), sizeof(int64_t) * values.size());
    Weights wt{DataType::kINT64, data, static_cast<int64_t>(values.size())};
#else
    auto* data = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = static_cast<int32_t>(values[i]);
    }
    Weights wt{DataType::kINT32, data, static_cast<int64_t>(values.size())};
#endif
    weightMap[name] = wt;
    return wt;
}

DataType formulaIndexDataType() {
#if NV_TENSORRT_MAJOR >= 10
    return DataType::kINT64;
#else
    return DataType::kINT32;
#endif
}

ITensor* addFloatConstantTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                const std::string& name, Dims dims, const std::vector<float>& values) {
    IConstantLayer* constant = network->addConstant(dims, addOwnedFloatWeights(weightMap, name, values));
    assert(constant);
    constant->setName(name.c_str());
    return constant->getOutput(0);
}

ITensor* addIntConstantTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                              const std::string& name, Dims dims, const std::vector<int32_t>& values) {
    IConstantLayer* constant = network->addConstant(dims, addOwnedIntWeights(weightMap, name, values));
    assert(constant);
    constant->setName(name.c_str());
    return constant->getOutput(0);
}

ITensor* addInt64ConstantTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                const std::string& name, Dims dims, const std::vector<int64_t>& values) {
    IConstantLayer* constant = network->addConstant(dims, addOwnedInt64Weights(weightMap, name, values));
    assert(constant);
    constant->setName(name.c_str());
    return constant->getOutput(0);
}

ITensor* addBoolConstantTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                               const std::string& name, Dims dims, const std::vector<bool>& values) {
    IConstantLayer* constant = network->addConstant(dims, addOwnedBoolWeights(weightMap, name, values));
    assert(constant);
    constant->setName(name.c_str());
    return constant->getOutput(0);
}

ITensor* addLinear2dTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                           int inChannels, int outChannels, const std::string& linearName, const std::string& lname) {
    IConstantLayer* weight =
            network->addConstant(Dims2{inChannels, outChannels}, getWeights(weightMap, linearName + ".w_0"));
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    IMatrixMultiplyLayer* matmul =
            network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight->getOutput(0), MatrixOperation::kNONE);
    assert(matmul);
    matmul->setName((lname + "_matmul").c_str());
    IConstantLayer* bias = network->addConstant(Dims2{1, outChannels}, getWeights(weightMap, linearName + ".b_0"));
    assert(bias);
    bias->setName((lname + "_bias").c_str());
    IElementWiseLayer* add =
            network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add->getOutput(0);
}

ITensor* addLinearNoBiasTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                               int inChannels, int outChannels, const std::string& linearName,
                               const std::string& lname) {
    IConstantLayer* weight =
            network->addConstant(Dims3{1, inChannels, outChannels}, getWeights(weightMap, linearName + ".w_0"));
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    IMatrixMultiplyLayer* matmul =
            network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight->getOutput(0), MatrixOperation::kNONE);
    assert(matmul);
    matmul->setName(lname.c_str());
    return matmul->getOutput(0);
}

ITensor* addLinearTransposeTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                  ITensor& input, int rows, int cols, int outChannels, const std::string& linearName,
                                  const std::string& lname) {
    IConstantLayer* weight = network->addConstant(Dims2{rows, cols}, getWeights(weightMap, linearName + ".w_0"));
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight->getOutput(0),
                                                              MatrixOperation::kTRANSPOSE);
    assert(matmul);
    matmul->setName((lname + "_matmul").c_str());
    IConstantLayer* bias = network->addConstant(Dims2{1, outChannels}, getWeights(weightMap, linearName + ".b_0"));
    assert(bias);
    bias->setName((lname + "_bias").c_str());
    IElementWiseLayer* add =
            network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add->getOutput(0);
}

ITensor* addLinearTransposeTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                  ITensor& input, int rows, int cols, int outChannels, const std::string& weightName,
                                  const std::string& biasName, const std::string& lname) {
    IConstantLayer* weight = network->addConstant(Dims2{rows, cols}, getWeights(weightMap, weightName));
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight->getOutput(0),
                                                              MatrixOperation::kTRANSPOSE);
    assert(matmul);
    matmul->setName((lname + "_matmul").c_str());
    IConstantLayer* bias = network->addConstant(Dims2{1, outChannels}, getWeights(weightMap, biasName));
    assert(bias);
    bias->setName((lname + "_bias").c_str());
    IElementWiseLayer* add =
            network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add->getOutput(0);
}

ITensor* addSlice2dTensor(INetworkDefinition* network, ITensor& input, int offset, int width,
                          const std::string& lname) {
    ISliceLayer* slice = network->addSlice(input, Dims2{0, offset}, Dims2{1, width}, Dims2{1, 1});
    assert(slice);
    slice->setName(lname.c_str());
    return slice->getOutput(0);
}

ITensor* addOneHotTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& index,
                         int depth, int axis, const std::string& lname) {
    ITensor* values = addFloatConstantTensor(network, weightMap, lname + "_values", makeDims1(2), {0.0f, 1.0f});
    ITensor* depthTensor = addIntConstantTensor(network, weightMap, lname + "_depth", Dims{}, {depth});
    IOneHotLayer* oneHot = network->addOneHot(index, *values, *depthTensor, axis);
    assert(oneHot);
    oneHot->setName(lname.c_str());
    return oneHot->getOutput(0);
}

ITensor* addTensorAtIndex(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& base,
                          ITensor& update, ITensor& index, int depth, int channels, const std::string& lname) {
    ITensor* values = addBoolConstantTensor(network, weightMap, lname + "_mask_values", makeDims1(2), {false, true});
    ITensor* depthTensor = addIntConstantTensor(network, weightMap, lname + "_depth", Dims{}, {depth});
    IOneHotLayer* oneHot = network->addOneHot(index, *values, *depthTensor, 0);
    assert(oneHot);
    oneHot->setName((lname + "_one_hot").c_str());

    IShuffleLayer* mask = network->addShuffle(*oneHot->getOutput(0));
    assert(mask);
    mask->setName((lname + "_mask").c_str());
    if (channels > 1) {
        mask->setReshapeDimensions(Dims3{1, depth, 1});
    } else {
        mask->setReshapeDimensions(Dims2{1, depth});
    }

    IShuffleLayer* body = network->addShuffle(update);
    assert(body);
    body->setName((lname + "_update").c_str());
    if (channels > 1) {
        body->setReshapeDimensions(Dims3{1, 1, channels});
    } else {
        body->setReshapeDimensions(Dims2{1, 1});
    }

    ISelectLayer* select = network->addSelect(*mask->getOutput(0), *body->getOutput(0), base);
    assert(select);
    select->setName(lname.c_str());
    return select->getOutput(0);
}

ITensor* addAnyTokenTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& ids,
                           int token, const std::string& lname) {
    ITensor* eos = addIntConstantTensor(network, weightMap, lname + "_eos", Dims2{1, 1}, {token});
    IElementWiseLayer* equal = network->addElementWise(ids, *eos, ElementWiseOperation::kEQUAL);
    assert(equal);
    equal->setName((lname + "_equal").c_str());
    ICastLayer* cast = network->addCast(*equal->getOutput(0), DataType::kINT32);
    assert(cast);
    cast->setName((lname + "_cast").c_str());
    IReduceLayer* any = network->addReduce(*cast->getOutput(0), ReduceOperation::kMAX, (1U << 0) | (1U << 1), false);
    assert(any);
    any->setName((lname + "_any").c_str());
    ITensor* zero = addIntConstantTensor(network, weightMap, lname + "_zero", Dims{}, {0});
    IElementWiseLayer* greater = network->addElementWise(*any->getOutput(0), *zero, ElementWiseOperation::kGREATER);
    assert(greater);
    greater->setName(lname.c_str());
    return greater->getOutput(0);
}

ITensor* addScalarIntSum(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int value, const std::string& lname) {
    ITensor* constant = addIntConstantTensor(network, weightMap, lname + "_value", Dims{}, {value});
    IElementWiseLayer* sum = network->addElementWise(input, *constant, ElementWiseOperation::kSUM);
    assert(sum);
    sum->setName(lname.c_str());
    return sum->getOutput(0);
}

ITensor* addScalarLessThan(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                           int value, const std::string& lname) {
    ITensor* constant = addIntConstantTensor(network, weightMap, lname + "_value", Dims{}, {value});
    IElementWiseLayer* less = network->addElementWise(input, *constant, ElementWiseOperation::kLESS);
    assert(less);
    less->setName(lname.c_str());
    return less->getOutput(0);
}

ITensor* addScalarInt64Sum(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                           int64_t value, const std::string& lname) {
    ITensor* constant = addInt64ConstantTensor(network, weightMap, lname + "_value", makeDims1(1), {value});
    IElementWiseLayer* sum = network->addElementWise(input, *constant, ElementWiseOperation::kSUM);
    assert(sum);
    sum->setName(lname.c_str());
    return sum->getOutput(0);
}

ITensor* addLogicalNotTensor(INetworkDefinition* network, ITensor& input, const std::string& lname) {
    IUnaryLayer* logicalNot = network->addUnary(input, UnaryOperation::kNOT);
    assert(logicalNot);
    logicalNot->setName(lname.c_str());
    return logicalNot->getOutput(0);
}

ITensor* addLogicalAndTensor(INetworkDefinition* network, ITensor& a, ITensor& b, const std::string& lname) {
    IElementWiseLayer* logicalAnd = network->addElementWise(a, b, ElementWiseOperation::kAND);
    assert(logicalAnd);
    logicalAnd->setName(lname.c_str());
    return logicalAnd->getOutput(0);
}

ITensor* addDynamicSliceEndShape(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                 ITensor& length, int channels, const std::string& lname) {
    IShuffleLayer* lengthVector = network->addShuffle(length);
    assert(lengthVector);
    lengthVector->setName((lname + "_length").c_str());
    lengthVector->setReshapeDimensions(makeDims1(1));
    std::vector<ITensor*> pieces{
            addIntConstantTensor(network, weightMap, lname + "_batch", makeDims1(1), {1}), lengthVector->getOutput(0),
            addIntConstantTensor(network, weightMap, lname + "_channels", makeDims1(1), {channels})};
    IConcatenationLayer* shape = addConcat(network, pieces, 0, lname);
    return shape->getOutput(0);
}

ITensor* addSLANetCspBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                           int leftConv, int leftBn, int rightConv0, int rightBn0, int rightConv1, int rightBn1,
                           int rightDw, int rightDwBn, int rightConv2, int rightBn2, int outConv, int outBn,
                           const std::string& lname) {
    IElementWiseLayer* left = convBnHSwish(network, weightMap, input, 48, 1, 1, 0, "conv2d_" + std::to_string(leftConv),
                                           "batch_norm2d_" + std::to_string(leftBn));
    IElementWiseLayer* right0 =
            convBnHSwish(network, weightMap, input, 48, 1, 1, 0, "conv2d_" + std::to_string(rightConv0),
                         "batch_norm2d_" + std::to_string(rightBn0));
    IElementWiseLayer* right1 =
            convBnHSwish(network, weightMap, *right0->getOutput(0), 48, 1, 1, 0, "conv2d_" + std::to_string(rightConv1),
                         "batch_norm2d_" + std::to_string(rightBn1));
    IElementWiseLayer* rightDwLayer =
            convBnHSwish(network, weightMap, *right1->getOutput(0), 48, 5, 1, 2, 48,
                         "conv2d_" + std::to_string(rightDw), "batch_norm2d_" + std::to_string(rightDwBn));
    IElementWiseLayer* right2 =
            convBnHSwish(network, weightMap, *rightDwLayer->getOutput(0), 48, 1, 1, 0,
                         "conv2d_" + std::to_string(rightConv2), "batch_norm2d_" + std::to_string(rightBn2));
    IConcatenationLayer* concat =
            addConcat(network, {right2->getOutput(0), left->getOutput(0)}, 1, lname + "_inner_cat");
    IElementWiseLayer* out = convBnHSwish(network, weightMap, *concat->getOutput(0), 96, 1, 1, 0,
                                          "conv2d_" + std::to_string(outConv), "batch_norm2d_" + std::to_string(outBn));
    return out->getOutput(0);
}

ITensor* addSLANetResizeTo(INetworkDefinition* network, ITensor& input, int height, int width,
                           const std::string& lname) {
    IResizeLayer* resize = network->addResize(input);
    assert(resize);
    resize->setName(lname.c_str());
    resize->setResizeMode(InterpolationMode::kNEAREST);
    resize->setOutputDimensions(Dims4{1, 96, height, width});
    return resize->getOutput(0);
}

IElementWiseLayer* addLayerNormLastDim(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                       ITensor& input, int channels, const std::string& layerNormName,
                                       const std::string& lname, float epsValue) {
    Dims inputDims = input.getDimensions();
    uint32_t axis = 1U << (inputDims.nbDims - 1);
    IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, axis, true);
    assert(mean);
    mean->setName((lname + "_mean").c_str());
    IElementWiseLayer* centered = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(centered);
    centered->setName((lname + "_centered").c_str());
    IElementWiseLayer* square =
            network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), ElementWiseOperation::kPROD);
    assert(square);
    square->setName((lname + "_square").c_str());
    IReduceLayer* var = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, axis, true);
    assert(var);
    var->setName((lname + "_var").c_str());

    Dims scalarDims{};
    scalarDims.nbDims = inputDims.nbDims;
    for (int i = 0; i < scalarDims.nbDims; ++i) {
        scalarDims.d[i] = 1;
    }
    ITensor* eps = addFloatConstantTensor(network, weightMap, lname + "_eps", scalarDims, {epsValue});
    IElementWiseLayer* varEps = network->addElementWise(*var->getOutput(0), *eps, ElementWiseOperation::kSUM);
    assert(varEps);
    varEps->setName((lname + "_var_eps").c_str());
    IUnaryLayer* stddev = network->addUnary(*varEps->getOutput(0), UnaryOperation::kSQRT);
    assert(stddev);
    stddev->setName((lname + "_std").c_str());
    IElementWiseLayer* norm =
            network->addElementWise(*centered->getOutput(0), *stddev->getOutput(0), ElementWiseOperation::kDIV);
    assert(norm);
    norm->setName((lname + "_norm").c_str());

    Dims affineDims{};
    affineDims.nbDims = inputDims.nbDims;
    for (int i = 0; i < affineDims.nbDims - 1; ++i) {
        affineDims.d[i] = 1;
    }
    affineDims.d[affineDims.nbDims - 1] = channels;
    IConstantLayer* gamma = network->addConstant(affineDims, getWeights(weightMap, layerNormName + ".w_0"));
    assert(gamma);
    gamma->setName((lname + "_gamma").c_str());
    IElementWiseLayer* scaled =
            network->addElementWise(*norm->getOutput(0), *gamma->getOutput(0), ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((lname + "_scaled").c_str());
    IConstantLayer* beta = network->addConstant(affineDims, getWeights(weightMap, layerNormName + ".b_0"));
    assert(beta);
    beta->setName((lname + "_beta").c_str());
    IElementWiseLayer* shifted =
            network->addElementWise(*scaled->getOutput(0), *beta->getOutput(0), ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName(lname.c_str());
    return shifted;
}

IElementWiseLayer* addLayerNormLastDim(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                       ITensor& input, int channels, const std::string& gammaName,
                                       const std::string& betaName, const std::string& lname, float epsValue) {
    Dims inputDims = input.getDimensions();
    uint32_t axis = 1U << (inputDims.nbDims - 1);
    IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, axis, true);
    assert(mean);
    mean->setName((lname + "_mean").c_str());
    IElementWiseLayer* centered = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(centered);
    centered->setName((lname + "_centered").c_str());
    IElementWiseLayer* square =
            network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), ElementWiseOperation::kPROD);
    assert(square);
    square->setName((lname + "_square").c_str());
    IReduceLayer* var = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, axis, true);
    assert(var);
    var->setName((lname + "_var").c_str());

    Dims scalarDims{};
    scalarDims.nbDims = inputDims.nbDims;
    for (int i = 0; i < scalarDims.nbDims; ++i) {
        scalarDims.d[i] = 1;
    }
    ITensor* eps = addFloatConstantTensor(network, weightMap, lname + "_eps", scalarDims, {epsValue});
    IElementWiseLayer* varEps = network->addElementWise(*var->getOutput(0), *eps, ElementWiseOperation::kSUM);
    assert(varEps);
    varEps->setName((lname + "_var_eps").c_str());
    IUnaryLayer* stddev = network->addUnary(*varEps->getOutput(0), UnaryOperation::kSQRT);
    assert(stddev);
    stddev->setName((lname + "_std").c_str());
    IElementWiseLayer* norm =
            network->addElementWise(*centered->getOutput(0), *stddev->getOutput(0), ElementWiseOperation::kDIV);
    assert(norm);
    norm->setName((lname + "_norm").c_str());

    Dims affineDims{};
    affineDims.nbDims = inputDims.nbDims;
    for (int i = 0; i < affineDims.nbDims - 1; ++i) {
        affineDims.d[i] = 1;
    }
    affineDims.d[affineDims.nbDims - 1] = channels;
    IConstantLayer* gamma = network->addConstant(affineDims, getWeights(weightMap, gammaName));
    assert(gamma);
    gamma->setName((lname + "_gamma").c_str());
    IElementWiseLayer* scaled =
            network->addElementWise(*norm->getOutput(0), *gamma->getOutput(0), ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((lname + "_scaled").c_str());
    IConstantLayer* beta = network->addConstant(affineDims, getWeights(weightMap, betaName));
    assert(beta);
    beta->setName((lname + "_beta").c_str());
    IElementWiseLayer* shifted =
            network->addElementWise(*scaled->getOutput(0), *beta->getOutput(0), ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName(lname.c_str());
    return shifted;
}

ITensor* addSLANeXtLayerNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                               int channels, const std::string& gammaName, const std::string& betaName,
                               const std::string& lname) {
    IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, 1U << 1, true);
    assert(mean);
    mean->setName((lname + "_mean").c_str());
    IElementWiseLayer* centered = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(centered);
    centered->setName((lname + "_centered").c_str());
    IElementWiseLayer* square =
            network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), ElementWiseOperation::kPROD);
    assert(square);
    square->setName((lname + "_square").c_str());
    IReduceLayer* var = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, 1U << 1, true);
    assert(var);
    var->setName((lname + "_var").c_str());
    ITensor* eps = addFloatConstantTensor(network, weightMap, lname + "_eps", Dims4{1, 1, 1, 1}, {1e-6f});
    IElementWiseLayer* varEps = network->addElementWise(*var->getOutput(0), *eps, ElementWiseOperation::kSUM);
    assert(varEps);
    varEps->setName((lname + "_var_eps").c_str());
    IUnaryLayer* stddev = network->addUnary(*varEps->getOutput(0), UnaryOperation::kSQRT);
    assert(stddev);
    stddev->setName((lname + "_std").c_str());
    IElementWiseLayer* norm =
            network->addElementWise(*centered->getOutput(0), *stddev->getOutput(0), ElementWiseOperation::kDIV);
    assert(norm);
    norm->setName((lname + "_norm").c_str());

    IConstantLayer* gamma = network->addConstant(Dims4{1, channels, 1, 1}, getWeights(weightMap, gammaName));
    assert(gamma);
    gamma->setName((lname + "_gamma").c_str());
    IElementWiseLayer* scaled =
            network->addElementWise(*norm->getOutput(0), *gamma->getOutput(0), ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((lname + "_scaled").c_str());
    IConstantLayer* beta = network->addConstant(Dims4{1, channels, 1, 1}, getWeights(weightMap, betaName));
    assert(beta);
    beta->setName((lname + "_beta").c_str());
    IElementWiseLayer* shifted =
            network->addElementWise(*scaled->getOutput(0), *beta->getOutput(0), ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName(lname.c_str());
    return shifted->getOutput(0);
}

Weights makeSLANeXtRelativeWeights(std::map<std::string, Weights>& weightMap, const std::string& tableName, int size,
                                   bool heightAxis, const std::string& name) {
    Weights table = getWeights(weightMap, tableName + ".w_0");
    int headDim = 64;
    int rows = size * 2 - 1;
    if (table.count != rows * headDim) {
        throw std::runtime_error("unexpected SLANeXt relative position shape for " + tableName);
    }
    int count = size * size * headDim;
    auto* values = reinterpret_cast<float*>(malloc(sizeof(float) * count));
    const auto* src = reinterpret_cast<const float*>(table.values);
    int index = 0;
    for (int q = 0; q < size; ++q) {
        for (int k = 0; k < size; ++k) {
            int rel = q - k + size - 1;
            for (int c = 0; c < headDim; ++c) {
                values[index++] = src[rel * headDim + c];
            }
        }
    }
    Weights rel{DataType::kFLOAT, values, count};
    weightMap[name + (heightAxis ? ".h" : ".w")] = rel;
    return rel;
}

ITensor* addSLANeXtRelativeAttention(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& q,
                                     ITensor& qk, int unitHeads, int size, const std::string& relHName,
                                     const std::string& relWName, const std::string& lname) {
    IShuffleLayer* qSpatial = addShuffle(network, q, makeDimsN({unitHeads, size, size, 64}), lname + "_q_spatial");
    IShuffleLayer* qH =
            addShuffle(network, *qSpatial->getOutput(0), makeDimsN({unitHeads, size, size, 1, 64}), lname + "_q_h");
    IConstantLayer* relH =
            network->addConstant(makeDimsN({1, size, 1, size, 64}),
                                 makeSLANeXtRelativeWeights(weightMap, relHName, size, true, lname + "_rel_h"));
    assert(relH);
    relH->setName((lname + "_rel_h").c_str());
    IElementWiseLayer* hProd =
            network->addElementWise(*qH->getOutput(0), *relH->getOutput(0), ElementWiseOperation::kPROD);
    assert(hProd);
    hProd->setName((lname + "_h_prod").c_str());
    IReduceLayer* hBias = network->addReduce(*hProd->getOutput(0), ReduceOperation::kSUM, 1U << 4, false);
    assert(hBias);
    hBias->setName((lname + "_h_bias").c_str());
    IShuffleLayer* hBiasShape = addShuffle(network, *hBias->getOutput(0), makeDimsN({unitHeads, size, size, size, 1}),
                                           lname + "_h_bias_shape");

    IShuffleLayer* qW =
            addShuffle(network, *qSpatial->getOutput(0), makeDimsN({unitHeads, size, size, 1, 64}), lname + "_q_w");
    IConstantLayer* relW =
            network->addConstant(makeDimsN({1, 1, size, size, 64}),
                                 makeSLANeXtRelativeWeights(weightMap, relWName, size, false, lname + "_rel_w"));
    assert(relW);
    relW->setName((lname + "_rel_w").c_str());
    IElementWiseLayer* wProd =
            network->addElementWise(*qW->getOutput(0), *relW->getOutput(0), ElementWiseOperation::kPROD);
    assert(wProd);
    wProd->setName((lname + "_w_prod").c_str());
    IReduceLayer* wBias = network->addReduce(*wProd->getOutput(0), ReduceOperation::kSUM, 1U << 4, false);
    assert(wBias);
    wBias->setName((lname + "_w_bias").c_str());
    IShuffleLayer* wBiasShape = addShuffle(network, *wBias->getOutput(0), makeDimsN({unitHeads, size, size, 1, size}),
                                           lname + "_w_bias_shape");

    IShuffleLayer* qkSpatial =
            addShuffle(network, qk, makeDimsN({unitHeads, size, size, size, size}), lname + "_qk_spatial");
    IElementWiseLayer* qkH = addSum(network, *qkSpatial->getOutput(0), *hBiasShape->getOutput(0), lname + "_qk_h");
    IElementWiseLayer* qkRel = addSum(network, *qkH->getOutput(0), *wBiasShape->getOutput(0), lname + "_qk_rel");
    IShuffleLayer* out =
            addShuffle(network, *qkRel->getOutput(0), Dims3{unitHeads, size * size, size * size}, lname + "_out");
    return out->getOutput(0);
}

ITensor* addSLANeXtAttention(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int units, int size, int linearBase, int relBase, const std::string& lname) {
    int tokens = size * size;
    int unitHeads = units * 12;
    IShuffleLayer* sequence = addShuffle(network, input, Dims3{units, tokens, 768}, lname + "_sequence");
    std::string qkvName = "linear_" + std::to_string(linearBase);
    IElementWiseLayer* q =
            addLinearPart(network, weightMap, *sequence->getOutput(0), 768, 768, qkvName, 0, lname + "_q");
    IElementWiseLayer* k =
            addLinearPart(network, weightMap, *sequence->getOutput(0), 768, 768, qkvName, 1, lname + "_k");
    IElementWiseLayer* v =
            addLinearPart(network, weightMap, *sequence->getOutput(0), 768, 768, qkvName, 2, lname + "_v");

    IShuffleLayer* qReshape = addShuffle(network, *q->getOutput(0), Dims4{units, tokens, 12, 64}, lname + "_q_reshape");
    IShuffleLayer* kReshape = addShuffle(network, *k->getOutput(0), Dims4{units, tokens, 12, 64}, lname + "_k_reshape");
    IShuffleLayer* vReshape = addShuffle(network, *v->getOutput(0), Dims4{units, tokens, 12, 64}, lname + "_v_reshape");
    IShuffleLayer* qPermute =
            addPermute(network, *qReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_q_permute");
    IShuffleLayer* kPermute =
            addPermute(network, *kReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_k_permute");
    IShuffleLayer* vPermute =
            addPermute(network, *vReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_v_permute");
    IShuffleLayer* qFlat =
            addShuffle(network, *qPermute->getOutput(0), Dims3{unitHeads, tokens, 64}, lname + "_q_flat");
    if (markFormulaDebugStage(network, lname + "_q_flat", *qFlat->getOutput(0))) {
        return qFlat->getOutput(0);
    }
    IShuffleLayer* kFlat =
            addShuffle(network, *kPermute->getOutput(0), Dims3{unitHeads, tokens, 64}, lname + "_k_flat");
    if (markFormulaDebugStage(network, lname + "_k_flat", *kFlat->getOutput(0))) {
        return kFlat->getOutput(0);
    }
    IShuffleLayer* vFlat =
            addShuffle(network, *vPermute->getOutput(0), Dims3{unitHeads, tokens, 64}, lname + "_v_flat");
    if (markFormulaDebugStage(network, lname + "_v_flat", *vFlat->getOutput(0))) {
        return vFlat->getOutput(0);
    }

    IElementWiseLayer* qScale =
            addScalarMul(network, weightMap, *qFlat->getOutput(0), 1.0f / std::sqrt(64.0f), lname + "_q_scale");
    if (markFormulaDebugStage(network, lname + "_q_scale", *qScale->getOutput(0))) {
        return qScale->getOutput(0);
    }
    IMatrixMultiplyLayer* qk = network->addMatrixMultiply(*qScale->getOutput(0), MatrixOperation::kNONE,
                                                          *kFlat->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(qk);
    qk->setName((lname + "_qk").c_str());
    if (markFormulaDebugStage(network, lname + "_qk", *qk->getOutput(0))) {
        return qk->getOutput(0);
    }
    ITensor* qkRel = addSLANeXtRelativeAttention(network, weightMap, *qFlat->getOutput(0), *qk->getOutput(0), unitHeads,
                                                 size, "create_parameter_" + std::to_string(relBase),
                                                 "create_parameter_" + std::to_string(relBase + 1), lname + "_rel");
    if (markFormulaDebugStage(network, lname + "_qk_rel", *qkRel)) {
        return qkRel;
    }
    ISoftMaxLayer* attn = addSoftmax(network, *qkRel, 1 << 2, lname + "_softmax");
    if (markFormulaDebugStage(network, lname + "_softmax", *attn->getOutput(0))) {
        return attn->getOutput(0);
    }
    IMatrixMultiplyLayer* context = network->addMatrixMultiply(*attn->getOutput(0), MatrixOperation::kNONE,
                                                               *vFlat->getOutput(0), MatrixOperation::kNONE);
    assert(context);
    context->setName((lname + "_context").c_str());
    if (markFormulaDebugStage(network, lname + "_context", *context->getOutput(0))) {
        return context->getOutput(0);
    }
    IShuffleLayer* contextReshape =
            addShuffle(network, *context->getOutput(0), Dims4{units, 12, tokens, 64}, lname + "_context_reshape");
    IShuffleLayer* contextPermute =
            addPermute(network, *contextReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_context_permute");
    IShuffleLayer* contextSequence =
            addShuffle(network, *contextPermute->getOutput(0), Dims3{units, tokens, 768}, lname + "_context_sequence");
    if (markFormulaDebugStage(network, lname + "_context_sequence", *contextSequence->getOutput(0))) {
        return contextSequence->getOutput(0);
    }
    IElementWiseLayer* proj = addLinear(network, weightMap, *contextSequence->getOutput(0), 768, 768,
                                        "linear_" + std::to_string(linearBase + 1));
    if (markFormulaDebugStage(network, lname + "_proj", *proj->getOutput(0))) {
        return proj->getOutput(0);
    }
    IShuffleLayer* spatial =
            addShuffle(network, *proj->getOutput(0), Dims4{units, size, size, 768}, lname + "_spatial");
    return spatial->getOutput(0);
}

ITensor* addSLANeXtWindowPartition(INetworkDefinition* network, ITensor& input, int spatial, int paddedSpatial,
                                   int windowSize, const std::string& lname) {
    int windows = paddedSpatial / windowSize;
    ISliceLayer* pad =
            network->addSlice(input, Dims4{0, 0, 0, 0}, Dims4{1, paddedSpatial, paddedSpatial, 768}, Dims4{1, 1, 1, 1});
    assert(pad);
    pad->setMode(SampleMode::kFILL);
    pad->setName((lname + "_pad").c_str());
    IShuffleLayer* blocks =
            addShuffle(network, *pad->getOutput(0), makeDimsN({1, windows, windowSize, windows, windowSize, 768}),
                       lname + "_blocks");
    IShuffleLayer* permute =
            addPermute(network, *blocks->getOutput(0), Permutation{0, 1, 3, 2, 4, 5}, lname + "_permute");
    IShuffleLayer* windowTensor = addShuffle(network, *permute->getOutput(0),
                                             Dims4{windows * windows, windowSize, windowSize, 768}, lname + "_windows");
    (void)spatial;
    return windowTensor->getOutput(0);
}

ITensor* addSLANeXtWindowUnpartition(INetworkDefinition* network, ITensor& input, int spatial, int paddedSpatial,
                                     int windowSize, const std::string& lname) {
    int windows = paddedSpatial / windowSize;
    IShuffleLayer* blocks = addShuffle(network, input, makeDimsN({1, windows, windows, windowSize, windowSize, 768}),
                                       lname + "_blocks");
    IShuffleLayer* permute =
            addPermute(network, *blocks->getOutput(0), Permutation{0, 1, 3, 2, 4, 5}, lname + "_permute");
    IShuffleLayer* padded =
            addShuffle(network, *permute->getOutput(0), Dims4{1, paddedSpatial, paddedSpatial, 768}, lname + "_padded");
    ISliceLayer* slice = network->addSlice(*padded->getOutput(0), Dims4{0, 0, 0, 0}, Dims4{1, spatial, spatial, 768},
                                           Dims4{1, 1, 1, 1});
    assert(slice);
    slice->setName((lname + "_slice").c_str());
    return slice->getOutput(0);
}

ITensor* addSLANeXtLinearSpatial(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 int units, int height, int width, int inChannels, int outChannels,
                                 const std::string& linearName, const std::string& lname) {
    IShuffleLayer* sequence = addShuffle(network, input, Dims3{units, height * width, inChannels}, lname + "_sequence");
    IElementWiseLayer* linear =
            addLinear(network, weightMap, *sequence->getOutput(0), inChannels, outChannels, linearName);
    IShuffleLayer* spatial =
            addShuffle(network, *linear->getOutput(0), Dims4{units, height, width, outChannels}, lname + "_spatial");
    return spatial->getOutput(0);
}

ITensor* addSLANeXtBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int blockIndex, int spatial, int windowSize, int paddedSpatial, bool globalAttention,
                         const std::string& lname) {
    int linearBase = blockIndex * 4;
    int layerNormBase = blockIndex * 2;
    int relBase = blockIndex * 2 + 1;

    IElementWiseLayer* ln0 = addLayerNormLastDim(network, weightMap, input, 768,
                                                 "layer_norm_" + std::to_string(layerNormBase), lname + "_ln0", 1e-6f);
    if (markFormulaDebugStage(network, lname + "_ln0", *ln0->getOutput(0))) {
        return ln0->getOutput(0);
    }
    ITensor* attnInput = ln0->getOutput(0);
    ITensor* attnOut{nullptr};
    if (globalAttention) {
        attnOut = addSLANeXtAttention(network, weightMap, *attnInput, 1, spatial, linearBase, relBase,
                                      lname + "_global_attn");
        if (markFormulaDebugStage(network, lname + "_global_attn", *attnOut)) {
            return attnOut;
        }
        if (hasFormulaDebugPrefix(lname + "_global_attn")) {
            return attnOut;
        }
    } else {
        int windows = (paddedSpatial / windowSize) * (paddedSpatial / windowSize);
        ITensor* windowInput = addSLANeXtWindowPartition(network, *attnInput, spatial, paddedSpatial, windowSize,
                                                         lname + "_partition");
        if (markFormulaDebugStage(network, lname + "_partition", *windowInput)) {
            return windowInput;
        }
        ITensor* windowAttn = addSLANeXtAttention(network, weightMap, *windowInput, windows, windowSize, linearBase,
                                                  relBase, lname + "_window_attn");
        if (markFormulaDebugStage(network, lname + "_window_attn", *windowAttn)) {
            return windowAttn;
        }
        if (hasFormulaDebugPrefix(lname + "_window_attn")) {
            return windowAttn;
        }
        attnOut = addSLANeXtWindowUnpartition(network, *windowAttn, spatial, paddedSpatial, windowSize,
                                              lname + "_unpartition");
        if (markFormulaDebugStage(network, lname + "_unpartition", *attnOut)) {
            return attnOut;
        }
    }
    IElementWiseLayer* attnSum = addSum(network, input, *attnOut, lname + "_attn_sum");
    if (markFormulaDebugStage(network, lname + "_attn_sum", *attnSum->getOutput(0))) {
        return attnSum->getOutput(0);
    }

    IElementWiseLayer* ln1 =
            addLayerNormLastDim(network, weightMap, *attnSum->getOutput(0), 768,
                                "layer_norm_" + std::to_string(layerNormBase + 1), lname + "_ln1", 1e-6f);
    if (markFormulaDebugStage(network, lname + "_ln1", *ln1->getOutput(0))) {
        return ln1->getOutput(0);
    }
    ITensor* mlp0 = addSLANeXtLinearSpatial(network, weightMap, *ln1->getOutput(0), 1, spatial, spatial, 768, 3072,
                                            "linear_" + std::to_string(linearBase + 2), lname + "_mlp0");
    if (markFormulaDebugStage(network, lname + "_mlp0", *mlp0)) {
        return mlp0;
    }
    ITensor* gelu = addGeluTensor(network, *mlp0, lname + "_gelu");
    if (markFormulaDebugStage(network, lname + "_gelu", *gelu)) {
        return gelu;
    }
    ITensor* mlp1 = addSLANeXtLinearSpatial(network, weightMap, *gelu, 1, spatial, spatial, 3072, 768,
                                            "linear_" + std::to_string(linearBase + 3), lname + "_mlp1");
    if (markFormulaDebugStage(network, lname + "_mlp1", *mlp1)) {
        return mlp1;
    }
    IElementWiseLayer* out = addSum(network, *attnSum->getOutput(0), *mlp1, lname + "_mlp_sum");
    if (markFormulaDebugStage(network, lname + "_out", *out->getOutput(0))) {
        return out->getOutput(0);
    }
    return out->getOutput(0);
}

ITensor* addSLANeXtBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int blockIndex, bool globalAttention, const std::string& lname) {
    return addSLANeXtBlock(network, weightMap, input, blockIndex, 32, 14, 42, globalAttention, lname);
}

ITensor* addConvBnSiluTensorByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                     ITensor& input, int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding,
                                     int groups, const std::string& convName, const std::string& bnName) {
    ITensor* bn = addConvBnTensorByPrefix(network, weightMap, input, outChannels, ksize, stride, padding, groups,
                                          convName, bnName);
    return addSiluTensor(network, *bn, convName + "_silu");
}

ITensor* addConvBiasSiluTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                               int outChannels, const std::string& convName) {
    ITensor* conv = addConvBiasTensor(network, weightMap, input, outChannels, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1},
                                      convName);
    return addSiluTensor(network, *conv, convName + "_silu");
}

struct RtDetrBackboneFeatures {
    ITensor* c3{nullptr};
    ITensor* c4{nullptr};
    ITensor* c5{nullptr};
};

struct RtDetrNeckFeatures {
    ITensor* p3{nullptr};
    ITensor* p4{nullptr};
    ITensor* p5{nullptr};
    ITensor* memory{nullptr};
};

RtDetrBackboneFeatures addRtDetrHgNetBackbone(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                              ITensor& data) {
    ITensor* stem0 = addConvBnReluTensorByPrefix(network, weightMap, data, 32, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1},
                                                 1, "conv2d_0", "batch_norm2d_80");
    ITensor* stem1 = addSameConvBnReluTensorByPrefix(network, weightMap, *stem0, 16, 2, "conv2d_1", "batch_norm2d_81");
    ITensor* stem2 = addSameConvBnReluTensorByPrefix(network, weightMap, *stem1, 32, 2, "conv2d_2", "batch_norm2d_82");
    IPaddingLayer* stemPoolPad = network->addPaddingNd(*stem0, DimsHW{0, 0}, DimsHW{1, 1});
    assert(stemPoolPad);
    stemPoolPad->setName("rtdetr_stem_pool_same_pad");
    IPoolingLayer* stemPool = addPool2d(network, *stemPoolPad->getOutput(0), PoolingType::kMAX, DimsHW{2, 2},
                                        DimsHW{1, 1}, DimsHW{0, 0}, "rtdetr_stem_pool");
    IConcatenationLayer* stemConcat =
            addConcat(network, std::vector<ITensor*>{stemPool->getOutput(0), stem2}, 1, "rtdetr_stem_concat");

    ITensor* stage1Prep0 = addConvBnReluTensorByPrefix(network, weightMap, *stemConcat->getOutput(0), 32, DimsHW{3, 3},
                                                       DimsHW{2, 2}, DimsHW{1, 1}, 1, "conv2d_3", "batch_norm2d_83");
    ITensor* stage1Prep1 = addConvBnReluTensorByPrefix(network, weightMap, *stage1Prep0, 48, DimsHW{1, 1}, DimsHW{1, 1},
                                                       DimsHW{0, 0}, 1, "conv2d_4", "batch_norm2d_84");
    ITensor* c3 = addHgConvBlockByPrefix(network, weightMap, *stage1Prep1, 48, 6, 5, 85, 11, 91, 64, 12, 92, 128);
    ITensor* c4Low = addHgStandardBlockByPrefix(network, weightMap, *c3, 96, 6, 13, 93, 128, DimsHW{2, 2}, 14, 94, 20,
                                                100, 256, 21, 101, 512);

    ITensor* stage3Down = addConvBnTensorByPrefix(network, weightMap, *c4Low, 512, DimsHW{3, 3}, DimsHW{2, 2},
                                                  DimsHW{1, 1}, 512, "conv2d_22", "batch_norm2d_102");
    ITensor* stage3A = addHgLightBlockByPrefix(network, weightMap, *stage3Down, 192, 6, 23, 103, 35, 115, 512, 36, 116,
                                               1024, false);
    ITensor* stage3B =
            addHgLightBlockByPrefix(network, weightMap, *stage3A, 192, 6, 37, 117, 49, 129, 512, 50, 130, 1024, true);
    ITensor* c4 =
            addHgLightBlockByPrefix(network, weightMap, *stage3B, 192, 6, 51, 131, 63, 143, 512, 64, 144, 1024, true);

    ITensor* stage4Down = addConvBnTensorByPrefix(network, weightMap, *c4, 1024, DimsHW{3, 3}, DimsHW{2, 2},
                                                  DimsHW{1, 1}, 1024, "conv2d_65", "batch_norm2d_145");
    ITensor* c5 = addHgLightBlockByPrefix(network, weightMap, *stage4Down, 384, 6, 66, 146, 78, 158, 1024, 79, 159,
                                          2048, false);
    RtDetrBackboneFeatures features;
    features.c3 = c4Low;
    features.c4 = c4;
    features.c5 = c5;
    return features;
}

ITensor* addRtDetrEncoderAttention(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                   ITensor& sequence, int spatial, int height, int width) {
    IConstantLayer* pos = network->addConstant(Dims3{1, spatial, 256}, getWeightsByPrefix(weightMap, "eager_tmp_0"));
    assert(pos);
    pos->setName("rtdetr_encoder_pos");
    IElementWiseLayer* withPos = addSum(network, sequence, *pos->getOutput(0), "rtdetr_encoder_pos_sum");

    IElementWiseLayer* q = addLinearPartByPrefix(network, weightMap, *withPos->getOutput(0), 256, 256,
                                                 "multi_head_attention_0", 0, "rtdetr_encoder_q");
    IElementWiseLayer* k = addLinearPartByPrefix(network, weightMap, *withPos->getOutput(0), 256, 256,
                                                 "multi_head_attention_0", 1, "rtdetr_encoder_k");
    IElementWiseLayer* v = addLinearPartByPrefix(network, weightMap, sequence, 256, 256, "multi_head_attention_0", 2,
                                                 "rtdetr_encoder_v");
    IShuffleLayer* qReshape =
            addShuffle(network, *q->getOutput(0), Dims4{1, spatial, 8, 32}, "rtdetr_encoder_q_reshape");
    IShuffleLayer* kReshape =
            addShuffle(network, *k->getOutput(0), Dims4{1, spatial, 8, 32}, "rtdetr_encoder_k_reshape");
    IShuffleLayer* vReshape =
            addShuffle(network, *v->getOutput(0), Dims4{1, spatial, 8, 32}, "rtdetr_encoder_v_reshape");
    IShuffleLayer* qPermute =
            addPermute(network, *qReshape->getOutput(0), Permutation{0, 2, 1, 3}, "rtdetr_encoder_q_permute");
    IShuffleLayer* kPermute =
            addPermute(network, *kReshape->getOutput(0), Permutation{0, 2, 1, 3}, "rtdetr_encoder_k_permute");
    IShuffleLayer* vPermute =
            addPermute(network, *vReshape->getOutput(0), Permutation{0, 2, 1, 3}, "rtdetr_encoder_v_permute");
    IMatrixMultiplyLayer* qk = network->addMatrixMultiply(*qPermute->getOutput(0), MatrixOperation::kNONE,
                                                          *kPermute->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(qk);
    qk->setName("rtdetr_encoder_qk");
    IElementWiseLayer* qkScale =
            addScalarMul(network, weightMap, *qk->getOutput(0), 1.0f / std::sqrt(32.0f), "rtdetr_encoder_qk_scale");
    ISoftMaxLayer* attn = addSoftmax(network, *qkScale->getOutput(0), 1 << 3, "rtdetr_encoder_softmax");
    IMatrixMultiplyLayer* context = network->addMatrixMultiply(*attn->getOutput(0), MatrixOperation::kNONE,
                                                               *vPermute->getOutput(0), MatrixOperation::kNONE);
    assert(context);
    context->setName("rtdetr_encoder_context");
    IShuffleLayer* contextPermute =
            addPermute(network, *context->getOutput(0), Permutation{0, 2, 1, 3}, "rtdetr_encoder_context_permute");
    IShuffleLayer* contextReshape = addShuffle(network, *contextPermute->getOutput(0), Dims3{1, spatial, 256},
                                               "rtdetr_encoder_context_reshape");
    IElementWiseLayer* proj = addLinearByPrefix(network, weightMap, *contextReshape->getOutput(0), 256, 256, "linear_0",
                                                "rtdetr_encoder_attn_proj");
    IElementWiseLayer* attnSum = addSum(network, sequence, *proj->getOutput(0), "rtdetr_encoder_attn_sum");
    IElementWiseLayer* norm0 = addLayerNormByPrefix(network, weightMap, *attnSum->getOutput(0), 256, "layer_norm_0",
                                                    "rtdetr_encoder_norm0");
    IElementWiseLayer* mlp0 =
            addLinearByPrefix(network, weightMap, *norm0->getOutput(0), 256, 1024, "linear_1", "rtdetr_encoder_mlp0");
    ITensor* gelu = addGeluTensor(network, *mlp0->getOutput(0), "rtdetr_encoder_gelu");
    IElementWiseLayer* mlp1 =
            addLinearByPrefix(network, weightMap, *gelu, 1024, 256, "linear_2", "rtdetr_encoder_mlp1");
    IElementWiseLayer* mlpSum = addSum(network, *norm0->getOutput(0), *mlp1->getOutput(0), "rtdetr_encoder_mlp_sum");
    IElementWiseLayer* norm1 = addLayerNormByPrefix(network, weightMap, *mlpSum->getOutput(0), 256, "layer_norm_1",
                                                    "rtdetr_encoder_norm1");
    IShuffleLayer* trans = addPermute(network, *norm1->getOutput(0), Permutation{0, 2, 1}, "rtdetr_encoder_ncw");
    IShuffleLayer* out = addShuffle(network, *trans->getOutput(0), Dims4{1, 256, height, width}, "rtdetr_encoder_map");
    return out->getOutput(0);
}

ITensor* addRtDetrCspRepLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int baseConv, int baseBn, int rep0, int rep1, int rep2, int shortcutConv, int shortcutBn,
                              const std::string& lname) {
    ITensor* left =
            addConvBnSiluTensorByPrefix(network, weightMap, input, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
                                        "conv2d_" + std::to_string(baseConv), "batch_norm2d_" + std::to_string(baseBn));
    ITensor* rep = addConvBiasSiluTensor(network, weightMap, *left, 256, "conv2d_" + std::to_string(rep0));
    rep = addConvBiasSiluTensor(network, weightMap, *rep, 256, "conv2d_" + std::to_string(rep1));
    rep = addConvBiasSiluTensor(network, weightMap, *rep, 256, "conv2d_" + std::to_string(rep2));
    ITensor* shortcut = addConvBnSiluTensorByPrefix(network, weightMap, input, 256, DimsHW{1, 1}, DimsHW{1, 1},
                                                    DimsHW{0, 0}, 1, "conv2d_" + std::to_string(shortcutConv),
                                                    "batch_norm2d_" + std::to_string(shortcutBn));
    IElementWiseLayer* sum = addSum(network, *rep, *shortcut, lname);
    return sum->getOutput(0);
}

RtDetrNeckFeatures addRtDetrHybridEncoder(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                          const RtDetrBackboneFeatures& backbone, int inputSize) {
    int p3Size = inputSize / 8;
    int p4Size = inputSize / 16;
    int p5Size = inputSize / 32;
    ITensor* c3Proj = addConvBnTensorByPrefix(network, weightMap, *backbone.c3, 256, DimsHW{1, 1}, DimsHW{1, 1},
                                              DimsHW{0, 0}, 1, "conv2d_80", "batch_norm2d_160");
    ITensor* c4Proj = addConvBnTensorByPrefix(network, weightMap, *backbone.c4, 256, DimsHW{1, 1}, DimsHW{1, 1},
                                              DimsHW{0, 0}, 1, "conv2d_81", "batch_norm2d_161");
    ITensor* c5Proj = addConvBnTensorByPrefix(network, weightMap, *backbone.c5, 256, DimsHW{1, 1}, DimsHW{1, 1},
                                              DimsHW{0, 0}, 1, "conv2d_82", "batch_norm2d_162");

    IShuffleLayer* c5Flat = addShuffle(network, *c5Proj, Dims3{1, 256, p5Size * p5Size}, "rtdetr_c5_flatten");
    IShuffleLayer* c5Seq = addPermute(network, *c5Flat->getOutput(0), Permutation{0, 2, 1}, "rtdetr_c5_sequence");
    ITensor* encodedC5 =
            addRtDetrEncoderAttention(network, weightMap, *c5Seq->getOutput(0), p5Size * p5Size, p5Size, p5Size);

    ITensor* fpn5 = addConvBnSiluTensorByPrefix(network, weightMap, *encodedC5, 256, DimsHW{1, 1}, DimsHW{1, 1},
                                                DimsHW{0, 0}, 1, "conv2d_83", "batch_norm2d_163");
    IResizeLayer* fpn5Up =
            addNearestResize(network, *fpn5, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, "rtdetr_fpn5_up");
    IConcatenationLayer* td4Cat = addConcat(network, {fpn5Up->getOutput(0), c4Proj}, 1, "rtdetr_td4_cat");
    ITensor* td4 = addRtDetrCspRepLayer(network, weightMap, *td4Cat->getOutput(0), 84, 164, 122, 123, 124, 85, 165,
                                        "rtdetr_td4_csp");
    ITensor* fpn4 = addConvBnSiluTensorByPrefix(network, weightMap, *td4, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0},
                                                1, "conv2d_92", "batch_norm2d_172");
    IResizeLayer* fpn4Up =
            addNearestResize(network, *fpn4, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, "rtdetr_fpn4_up");
    IConcatenationLayer* td3Cat = addConcat(network, {fpn4Up->getOutput(0), c3Proj}, 1, "rtdetr_td3_cat");
    ITensor* p3 = addRtDetrCspRepLayer(network, weightMap, *td3Cat->getOutput(0), 93, 173, 125, 126, 127, 94, 174,
                                       "rtdetr_td3_csp");

    ITensor* p3Down = addConvBnSiluTensorByPrefix(network, weightMap, *p3, 256, DimsHW{3, 3}, DimsHW{2, 2},
                                                  DimsHW{1, 1}, 1, "conv2d_101", "batch_norm2d_181");
    IConcatenationLayer* bu4Cat = addConcat(network, {p3Down, fpn4}, 1, "rtdetr_bu4_cat");
    ITensor* p4 = addRtDetrCspRepLayer(network, weightMap, *bu4Cat->getOutput(0), 102, 182, 128, 129, 130, 103, 183,
                                       "rtdetr_bu4_csp");
    ITensor* p4Down = addConvBnSiluTensorByPrefix(network, weightMap, *p4, 256, DimsHW{3, 3}, DimsHW{2, 2},
                                                  DimsHW{1, 1}, 1, "conv2d_110", "batch_norm2d_190");
    IConcatenationLayer* bu5Cat = addConcat(network, {p4Down, fpn5}, 1, "rtdetr_bu5_cat");
    ITensor* p5 = addRtDetrCspRepLayer(network, weightMap, *bu5Cat->getOutput(0), 111, 191, 131, 132, 133, 112, 192,
                                       "rtdetr_bu5_csp");

    ITensor* m3 = addConvBnTensorByPrefix(network, weightMap, *p3, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
                                          "conv2d_119", "batch_norm2d_199");
    ITensor* m4 = addConvBnTensorByPrefix(network, weightMap, *p4, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
                                          "conv2d_120", "batch_norm2d_200");
    ITensor* m5 = addConvBnTensorByPrefix(network, weightMap, *p5, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
                                          "conv2d_121", "batch_norm2d_201");
    IShuffleLayer* m3Flat = addShuffle(network, *m3, Dims3{1, 256, p3Size * p3Size}, "rtdetr_m3_flatten");
    IShuffleLayer* m3Seq = addPermute(network, *m3Flat->getOutput(0), Permutation{0, 2, 1}, "rtdetr_m3_seq");
    IShuffleLayer* m4Flat = addShuffle(network, *m4, Dims3{1, 256, p4Size * p4Size}, "rtdetr_m4_flatten");
    IShuffleLayer* m4Seq = addPermute(network, *m4Flat->getOutput(0), Permutation{0, 2, 1}, "rtdetr_m4_seq");
    IShuffleLayer* m5Flat = addShuffle(network, *m5, Dims3{1, 256, p5Size * p5Size}, "rtdetr_m5_flatten");
    IShuffleLayer* m5Seq = addPermute(network, *m5Flat->getOutput(0), Permutation{0, 2, 1}, "rtdetr_m5_seq");
    IConcatenationLayer* memory =
            addConcat(network, {m3Seq->getOutput(0), m4Seq->getOutput(0), m5Seq->getOutput(0)}, 1, "rtdetr_memory");
    RtDetrNeckFeatures features;
    features.p3 = m3;
    features.p4 = m4;
    features.p5 = m5;
    features.memory = memory->getOutput(0);
    return features;
}

ITensor* addScalarConstantLike(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                               float value, const std::string& lname) {
    Dims dims{};
    dims.nbDims = input.getDimensions().nbDims;
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = 1;
    }
    return addFloatConstantTensor(network, weightMap, lname, dims, {value});
}

ITensor* addScalarAddTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                            float value, const std::string& lname) {
    ITensor* scalar = addScalarConstantLike(network, weightMap, input, value, lname + "_scalar");
    IElementWiseLayer* add = network->addElementWise(input, *scalar, ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add->getOutput(0);
}

ITensor* addScalarSubFromTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, float value,
                                ITensor& input, const std::string& lname) {
    ITensor* scalar = addScalarConstantLike(network, weightMap, input, value, lname + "_scalar");
    IElementWiseLayer* sub = network->addElementWise(*scalar, input, ElementWiseOperation::kSUB);
    assert(sub);
    sub->setName(lname.c_str());
    return sub->getOutput(0);
}

ITensor* addScalarClipTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             float minValue, float maxValue, const std::string& lname) {
    ITensor* minTensor = addScalarConstantLike(network, weightMap, input, minValue, lname + "_min");
    IElementWiseLayer* maxLayer = network->addElementWise(input, *minTensor, ElementWiseOperation::kMAX);
    assert(maxLayer);
    maxLayer->setName((lname + "_max").c_str());
    ITensor* maxTensor = addScalarConstantLike(network, weightMap, input, maxValue, lname + "_max_value");
    IElementWiseLayer* minLayer =
            network->addElementWise(*maxLayer->getOutput(0), *maxTensor, ElementWiseOperation::kMIN);
    assert(minLayer);
    minLayer->setName(lname.c_str());
    return minLayer->getOutput(0);
}

ITensor* addInverseSigmoidTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 const std::string& lname) {
    ITensor* clipped = addScalarClipTensor(network, weightMap, input, 1e-5f, 1.0f - 1e-5f, lname + "_clip");
    ITensor* denom = addScalarSubFromTensor(network, weightMap, 1.0f, *clipped, lname + "_one_minus");
    IElementWiseLayer* ratio = network->addElementWise(*clipped, *denom, ElementWiseOperation::kDIV);
    assert(ratio);
    ratio->setName((lname + "_ratio").c_str());
    IUnaryLayer* logit = network->addUnary(*ratio->getOutput(0), UnaryOperation::kLOG);
    assert(logit);
    logit->setName(lname.c_str());
    return logit->getOutput(0);
}

ITensor* addRtDetrSelfAttentionLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                     ITensor& target, ITensor& targetWithPos, int order, const std::string& lname) {
    IElementWiseLayer* q = addLinearPartByPrefixOrder(network, weightMap, targetWithPos, 256, 256,
                                                      "multi_head_attention_1", order, 0, lname + "_q");
    IElementWiseLayer* k = addLinearPartByPrefixOrder(network, weightMap, targetWithPos, 256, 256,
                                                      "multi_head_attention_1", order, 1, lname + "_k");
    IElementWiseLayer* v = addLinearPartByPrefixOrder(network, weightMap, target, 256, 256, "multi_head_attention_1",
                                                      order, 2, lname + "_v");
    IShuffleLayer* qReshape = addShuffle(network, *q->getOutput(0), Dims4{1, 300, 8, 32}, lname + "_q_reshape");
    IShuffleLayer* kReshape = addShuffle(network, *k->getOutput(0), Dims4{1, 300, 8, 32}, lname + "_k_reshape");
    IShuffleLayer* vReshape = addShuffle(network, *v->getOutput(0), Dims4{1, 300, 8, 32}, lname + "_v_reshape");
    IShuffleLayer* qPermute =
            addPermute(network, *qReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_q_permute");
    IShuffleLayer* kPermute =
            addPermute(network, *kReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_k_permute");
    IShuffleLayer* vPermute =
            addPermute(network, *vReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_v_permute");
    IMatrixMultiplyLayer* qk = network->addMatrixMultiply(*qPermute->getOutput(0), MatrixOperation::kNONE,
                                                          *kPermute->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(qk);
    qk->setName((lname + "_qk").c_str());
    IElementWiseLayer* qkScale =
            addScalarMul(network, weightMap, *qk->getOutput(0), 1.0f / std::sqrt(32.0f), lname + "_qk_scale");
    ISoftMaxLayer* attn = addSoftmax(network, *qkScale->getOutput(0), 1 << 3, lname + "_softmax");
    IMatrixMultiplyLayer* context = network->addMatrixMultiply(*attn->getOutput(0), MatrixOperation::kNONE,
                                                               *vPermute->getOutput(0), MatrixOperation::kNONE);
    assert(context);
    context->setName((lname + "_context").c_str());
    IShuffleLayer* contextPermute =
            addPermute(network, *context->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_context_permute");
    IShuffleLayer* contextReshape =
            addShuffle(network, *contextPermute->getOutput(0), Dims3{1, 300, 256}, lname + "_context_reshape");
    IElementWiseLayer* proj = addLinearByPrefixOrder(network, weightMap, *contextReshape->getOutput(0), 256, 256,
                                                     "linear_3", order, lname + "_proj");
    return proj->getOutput(0);
}

ITensor* addRtDetrDeformableAttentionLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                           ITensor& memory, ITensor& target, ITensor& reference, int memoryLength,
                                           int order, const std::string& lname) {
    IElementWiseLayer* value =
            addLinearByPrefixOrder(network, weightMap, memory, 256, 256, "linear_6", order, lname + "_value");
    IShuffleLayer* valueReshape =
            addShuffle(network, *value->getOutput(0), Dims4{1, memoryLength, 8, 32}, lname + "_value_reshape");
    IElementWiseLayer* offsets =
            addLinearByPrefixOrder(network, weightMap, target, 256, 192, "linear_4", order, lname + "_offsets");
    IShuffleLayer* offsetsReshape =
            addShuffle(network, *offsets->getOutput(0), makeDimsN({1, 300, 8, 3, 4, 2}), lname + "_offsets_reshape");
    IElementWiseLayer* attn =
            addLinearByPrefixOrder(network, weightMap, target, 256, 96, "linear_5", order, lname + "_attn");
    IShuffleLayer* attnReshape =
            addShuffle(network, *attn->getOutput(0), Dims4{1, 300, 8, 12}, lname + "_attn_reshape");
    ISoftMaxLayer* attnSoftmax = addSoftmax(network, *attnReshape->getOutput(0), 1 << 3, lname + "_attn_softmax");
    IShuffleLayer* attnFinal =
            addShuffle(network, *attnSoftmax->getOutput(0), makeDimsN({1, 300, 8, 3, 4}), lname + "_attn_final");
    ITensor* pluginInputs[] = {valueReshape->getOutput(0), &reference, offsetsReshape->getOutput(0),
                               attnFinal->getOutput(0)};
    Ppocrv5RtDetrPlugin* plugin = new Ppocrv5RtDetrPlugin();
    IPluginV2Layer* deform = network->addPluginV2(pluginInputs, 4, *plugin);
    assert(deform);
    deform->setName(lname.c_str());
    return deform->getOutput(0);
}

ITensor* addRtDetrDecoderLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& memory,
                               ITensor& target, ITensor*& reference, int memoryLength, int order,
                               const std::string& lname) {
    IElementWiseLayer* queryPos0 =
            addLinearByPrefix(network, weightMap, *reference, 4, 512, "linear_10", lname + "_query_pos0");
    IActivationLayer* queryPosAct = addRelu(network, *queryPos0->getOutput(0), lname + "_query_pos_relu");
    IElementWiseLayer* queryPos1 = addLinearByPrefix(network, weightMap, *queryPosAct->getOutput(0), 512, 256,
                                                     "linear_11", lname + "_query_pos1");
    IElementWiseLayer* targetWithPos = addSum(network, target, *queryPos1->getOutput(0), lname + "_target_pos");
    ITensor* selfAttn = addRtDetrSelfAttentionLayer(network, weightMap, target, *targetWithPos->getOutput(0), order,
                                                    lname + "_self_attn");
    IElementWiseLayer* selfSum = addSum(network, target, *selfAttn, lname + "_self_sum");
    IElementWiseLayer* selfNorm = addLayerNormByPrefixOrder(network, weightMap, *selfSum->getOutput(0), 256,
                                                            "layer_norm_2", order, lname + "_self_norm");

    ITensor* cross = addRtDetrDeformableAttentionLayer(network, weightMap, memory, *selfNorm->getOutput(0), *reference,
                                                       memoryLength, order, lname + "_deform_attn");
    IElementWiseLayer* crossProj =
            addLinearByPrefixOrder(network, weightMap, *cross, 256, 256, "linear_7", order, lname + "_cross_proj");
    IElementWiseLayer* crossSum =
            addSum(network, *selfNorm->getOutput(0), *crossProj->getOutput(0), lname + "_cross_sum");
    IElementWiseLayer* crossNorm = addLayerNormByPrefixOrder(network, weightMap, *crossSum->getOutput(0), 256,
                                                             "layer_norm_3", order, lname + "_cross_norm");
    IElementWiseLayer* ffn0 = addLinearByPrefixOrder(network, weightMap, *crossNorm->getOutput(0), 256, 1024,
                                                     "linear_8", order, lname + "_ffn0");
    IActivationLayer* ffnRelu = addRelu(network, *ffn0->getOutput(0), lname + "_ffn_relu");
    IElementWiseLayer* ffn1 = addLinearByPrefixOrder(network, weightMap, *ffnRelu->getOutput(0), 1024, 256, "linear_9",
                                                     order, lname + "_ffn1");
    IElementWiseLayer* ffnSum = addSum(network, *crossNorm->getOutput(0), *ffn1->getOutput(0), lname + "_ffn_sum");
    IElementWiseLayer* ffnNorm = addLayerNormByPrefixOrder(network, weightMap, *ffnSum->getOutput(0), 256,
                                                           "layer_norm_4", order, lname + "_ffn_norm");

    int boxHead = 23 + order * 3;
    IElementWiseLayer* box0 = addLinearByPrefix(network, weightMap, *ffnNorm->getOutput(0), 256, 256,
                                                "linear_" + std::to_string(boxHead), lname + "_box0");
    IActivationLayer* boxRelu0 = addRelu(network, *box0->getOutput(0), lname + "_box_relu0");
    IElementWiseLayer* box1 = addLinearByPrefix(network, weightMap, *boxRelu0->getOutput(0), 256, 256,
                                                "linear_" + std::to_string(boxHead + 1), lname + "_box1");
    IActivationLayer* boxRelu1 = addRelu(network, *box1->getOutput(0), lname + "_box_relu1");
    IElementWiseLayer* box2 = addLinearByPrefix(network, weightMap, *boxRelu1->getOutput(0), 256, 4,
                                                "linear_" + std::to_string(boxHead + 2), lname + "_box2");
    ITensor* referenceLogit = addInverseSigmoidTensor(network, weightMap, *reference, lname + "_reference_logit");
    IElementWiseLayer* boxUpdate = addSum(network, *box2->getOutput(0), *referenceLogit, lname + "_box_update");
    IActivationLayer* newReference = addSigmoid(network, *boxUpdate->getOutput(0), lname + "_reference");
    reference = newReference->getOutput(0);
    return ffnNorm->getOutput(0);
}

}  // namespace

IHostMemory* buildPPOCRv5MobileDet(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput(kDetInputTensorName, dt, Dims4{1, 3, -1, -1});
    if (!data) {
        throw std::runtime_error("failed to add PP-OCRv5 mobile det input");
    }
    addDetOptimizationProfile(builder, config);

    IScaleLayer* stem = convBn(network, weightMap, *data, 16, 3, 2, 1, "conv2d_0", "batch_norm2d_0");
    IScaleLayer* stage0Dw =
            learnableRepLayer(network, weightMap, *stem->getOutput(0), 16, 3, 1, 1, 16, "conv2d_161", 0, true);
    IScaleLayer* stage0Pw =
            learnableRepLayer(network, weightMap, *stage0Dw->getOutput(0), 32, 1, 1, 0, 1, "conv2d_162", 2, true);
    IScaleLayer* stage1Dw =
            learnableRepLayer(network, weightMap, *stage0Pw->getOutput(0), 32, 3, 2, 1, 32, "conv2d_163", 4, false);
    IScaleLayer* stage1Pw =
            learnableRepLayer(network, weightMap, *stage1Dw->getOutput(0), 48, 1, 1, 0, 1, "conv2d_164", 6, true);
    IScaleLayer* stage1Block1Dw =
            learnableRepLayer(network, weightMap, *stage1Pw->getOutput(0), 48, 3, 1, 1, 48, "conv2d_165", 8, true);
    IScaleLayer* stage1Block1Pw = learnableRepLayer(network, weightMap, *stage1Block1Dw->getOutput(0), 48, 1, 1, 0, 1,
                                                    "conv2d_166", 10, true);

    IScaleLayer* stage2Dw = learnableRepLayer(network, weightMap, *stage1Block1Pw->getOutput(0), 48, 3, 2, 1, 48,
                                              "conv2d_167", 12, false);
    IScaleLayer* stage2Pw =
            learnableRepLayer(network, weightMap, *stage2Dw->getOutput(0), 96, 1, 1, 0, 1, "conv2d_168", 14, true);
    IScaleLayer* stage2Block1Dw =
            learnableRepLayer(network, weightMap, *stage2Pw->getOutput(0), 96, 3, 1, 1, 96, "conv2d_169", 16, true);
    IScaleLayer* stage2Block1Pw = learnableRepLayer(network, weightMap, *stage2Block1Dw->getOutput(0), 96, 1, 1, 0, 1,
                                                    "conv2d_170", 18, true);

    IScaleLayer* stage3Dw = learnableRepLayer(network, weightMap, *stage2Block1Pw->getOutput(0), 96, 3, 2, 1, 96,
                                              "conv2d_171", 20, false);
    IScaleLayer* stage3Pw =
            learnableRepLayer(network, weightMap, *stage3Dw->getOutput(0), 192, 1, 1, 0, 1, "conv2d_172", 22, true);
    IScaleLayer* stage3Block1Dw =
            learnableRepLayer(network, weightMap, *stage3Pw->getOutput(0), 192, 5, 1, 2, 192, "conv2d_173", 24, true);
    IScaleLayer* stage3Block1Pw = learnableRepLayer(network, weightMap, *stage3Block1Dw->getOutput(0), 192, 1, 1, 0, 1,
                                                    "conv2d_174", 26, true);
    IScaleLayer* stage3Block2Dw = learnableRepLayer(network, weightMap, *stage3Block1Pw->getOutput(0), 192, 5, 1, 2,
                                                    192, "conv2d_175", 28, true);
    IScaleLayer* stage3Block2Pw = learnableRepLayer(network, weightMap, *stage3Block2Dw->getOutput(0), 192, 1, 1, 0, 1,
                                                    "conv2d_176", 30, true);
    IScaleLayer* stage3Block3Dw = learnableRepLayer(network, weightMap, *stage3Block2Pw->getOutput(0), 192, 5, 1, 2,
                                                    192, "conv2d_177", 32, true);
    IScaleLayer* stage3Block3Pw = learnableRepLayer(network, weightMap, *stage3Block3Dw->getOutput(0), 192, 1, 1, 0, 1,
                                                    "conv2d_178", 34, true);
    IScaleLayer* stage3Block4Dw = learnableRepLayer(network, weightMap, *stage3Block3Pw->getOutput(0), 192, 5, 1, 2,
                                                    192, "conv2d_179", 36, true);
    IScaleLayer* stage3Block4Pw = learnableRepLayer(network, weightMap, *stage3Block4Dw->getOutput(0), 192, 1, 1, 0, 1,
                                                    "conv2d_180", 38, true);

    IScaleLayer* stage4Dw = learnableRepLayer(network, weightMap, *stage3Block4Pw->getOutput(0), 192, 5, 2, 2, 192,
                                              "conv2d_181", 40, false);
    IElementWiseLayer* stage4Se =
            seLayer(network, weightMap, *stage4Dw->getOutput(0), 48, 192, "conv2d_96", "conv2d_97");
    IScaleLayer* stage4Pw =
            learnableRepLayer(network, weightMap, *stage4Se->getOutput(0), 384, 1, 1, 0, 1, "conv2d_182", 42, true);
    IScaleLayer* stage4Block1Dw =
            learnableRepLayer(network, weightMap, *stage4Pw->getOutput(0), 384, 5, 1, 2, 384, "conv2d_183", 44, true);
    IElementWiseLayer* stage4Block1Se =
            seLayer(network, weightMap, *stage4Block1Dw->getOutput(0), 96, 384, "conv2d_107", "conv2d_108");
    IScaleLayer* stage4Block1Pw = learnableRepLayer(network, weightMap, *stage4Block1Se->getOutput(0), 384, 1, 1, 0, 1,
                                                    "conv2d_184", 46, true);
    IScaleLayer* stage4Block2Dw = learnableRepLayer(network, weightMap, *stage4Block1Pw->getOutput(0), 384, 5, 1, 2,
                                                    384, "conv2d_185", 48, true);
    IScaleLayer* stage4Block2Pw = learnableRepLayer(network, weightMap, *stage4Block2Dw->getOutput(0), 384, 1, 1, 0, 1,
                                                    "conv2d_186", 50, true);
    IScaleLayer* stage4Block3Dw = learnableRepLayer(network, weightMap, *stage4Block2Pw->getOutput(0), 384, 5, 1, 2,
                                                    384, "conv2d_187", 52, true);
    IScaleLayer* stage4Block3Pw = learnableRepLayer(network, weightMap, *stage4Block3Dw->getOutput(0), 384, 1, 1, 0, 1,
                                                    "conv2d_188", 54, true);

    IConvolutionLayer* lateral0 =
            convBias(network, weightMap, *stage1Block1Pw->getOutput(0), 12, 1, 1, 0, "conv2d_131");
    IConvolutionLayer* lateral1 =
            convBias(network, weightMap, *stage2Block1Pw->getOutput(0), 18, 1, 1, 0, "conv2d_132");
    IConvolutionLayer* lateral2 =
            convBias(network, weightMap, *stage3Block4Pw->getOutput(0), 42, 1, 1, 0, "conv2d_133");
    IConvolutionLayer* lateral3 =
            convBias(network, weightMap, *stage4Block3Pw->getOutput(0), 360, 1, 1, 0, "conv2d_134");

    IElementWiseLayer* rse3 = rseLayer(network, weightMap, *lateral3->getOutput(0), 96, 24, 1, 0, "conv2d_153",
                                       "conv2d_154", "conv2d_155");
    IElementWiseLayer* rse2 = rseLayer(network, weightMap, *lateral2->getOutput(0), 96, 24, 1, 0, "conv2d_147",
                                       "conv2d_148", "conv2d_149");
    IElementWiseLayer* rse1 = rseLayer(network, weightMap, *lateral1->getOutput(0), 96, 24, 1, 0, "conv2d_141",
                                       "conv2d_142", "conv2d_143");
    IElementWiseLayer* rse0 = rseLayer(network, weightMap, *lateral0->getOutput(0), 96, 24, 1, 0, "conv2d_135",
                                       "conv2d_136", "conv2d_137");

    IResizeLayer* up3To2 = addNearestResize(network, *rse3->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                                            "nearest_interp_579");
    IElementWiseLayer* fuse2 = addSum(network, *rse2->getOutput(0), *up3To2->getOutput(0), "add_580");
    IResizeLayer* up2To1 = addNearestResize(network, *fuse2->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                                            "nearest_interp_581");
    IElementWiseLayer* fuse1 = addSum(network, *rse1->getOutput(0), *up2To1->getOutput(0), "add_582");
    IResizeLayer* up1To0 = addNearestResize(network, *fuse1->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                                            "nearest_interp_583");
    IElementWiseLayer* fuse0 = addSum(network, *rse0->getOutput(0), *up1To0->getOutput(0), "add_584");

    IElementWiseLayer* out3 =
            rseLayer(network, weightMap, *rse3->getOutput(0), 24, 6, 3, 1, "conv2d_156", "conv2d_157", "conv2d_158");
    IElementWiseLayer* out2 =
            rseLayer(network, weightMap, *fuse2->getOutput(0), 24, 6, 3, 1, "conv2d_150", "conv2d_151", "conv2d_152");
    IElementWiseLayer* out1 =
            rseLayer(network, weightMap, *fuse1->getOutput(0), 24, 6, 3, 1, "conv2d_144", "conv2d_145", "conv2d_146");
    IElementWiseLayer* out0 =
            rseLayer(network, weightMap, *fuse0->getOutput(0), 24, 6, 3, 1, "conv2d_138", "conv2d_139", "conv2d_140");

    IResizeLayer* out3Up = addNearestResize(network, *out3->getOutput(0), std::vector<float>{1.0f, 1.0f, 8.0f, 8.0f},
                                            "nearest_interp_645");
    IResizeLayer* out2Up = addNearestResize(network, *out2->getOutput(0), std::vector<float>{1.0f, 1.0f, 4.0f, 4.0f},
                                            "nearest_interp_646");
    IResizeLayer* out1Up = addNearestResize(network, *out1->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                                            "nearest_interp_647");
    IConcatenationLayer* neckOut = addConcat(
            network,
            std::vector<ITensor*>{out3Up->getOutput(0), out2Up->getOutput(0), out1Up->getOutput(0), out0->getOutput(0)},
            1, "concat_650");

    IScaleLayer* headConv =
            convBn(network, weightMap, *neckOut->getOutput(0), 24, 3, 1, 1, "conv2d_159", "batch_norm_0");
    IActivationLayer* headRelu0 = addRelu(network, *headConv->getOutput(0), "relu_653");
    IDeconvolutionLayer* headDeconv0 =
            deconvBias(network, weightMap, *headRelu0->getOutput(0), 24, 2, 2, 0, "conv2d_transpose_0");
    IScaleLayer* headBn1 = addBatchNorm2d(network, weightMap, *headDeconv0->getOutput(0), "batch_norm_1", 1e-5f);
    headBn1->setName("batch_norm_1");
    IActivationLayer* headRelu1 = addRelu(network, *headBn1->getOutput(0), "relu_660");
    IDeconvolutionLayer* headDeconv1 =
            deconvBias(network, weightMap, *headRelu1->getOutput(0), 1, 2, 2, 0, "conv2d_transpose_1");
    Ppocrv5DbPlugin* dbPlugin = new Ppocrv5DbPlugin();
    ITensor* dbInputs[] = {headDeconv1->getOutput(0)};
    IPluginV2Layer* pred = network->addPluginV2(dbInputs, 1, *dbPlugin);
    if (!pred) {
        throw std::runtime_error("failed to add PP-OCRv5 DB plugin");
    }
    pred->setName("ppocrv5_db_plugin");
    pred->getOutput(0)->setName(kDetOutputTensorName);
    network->markOutput(*pred->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildPPOCRv5ServerDet(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput(kDetInputTensorName, dt, Dims4{1, 3, -1, -1});
    if (!data) {
        throw std::runtime_error("failed to add PP-OCRv5 server det input");
    }
    addDetOptimizationProfile(builder, config);

    ITensor* stem0 = addConvBnReluTensor(network, weightMap, *data, 32, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1,
                                         "conv2d_0", "batch_norm2d_0");
    ITensor* stem1 = addSameConvBnReluTensor(network, weightMap, *stem0, 16, 2, "conv2d_1", "batch_norm2d_1");
    ITensor* stem2 = addSameConvBnReluTensor(network, weightMap, *stem1, 32, 2, "conv2d_2", "batch_norm2d_2");
    IPaddingLayer* stemPoolPad = network->addPaddingNd(*stem0, DimsHW{0, 0}, DimsHW{1, 1});
    assert(stemPoolPad);
    stemPoolPad->setName("server_det_max_pool2d_0_same_pad");
    IPoolingLayer* stemPool = addPool2d(network, *stemPoolPad->getOutput(0), PoolingType::kMAX, DimsHW{2, 2},
                                        DimsHW{1, 1}, DimsHW{0, 0}, "server_det_max_pool2d_0");
    IConcatenationLayer* stemConcat =
            addConcat(network, std::vector<ITensor*>{stemPool->getOutput(0), stem2}, 1, "server_det_stem_concat");

    ITensor* stage1Prep0 = addConvBnReluTensor(network, weightMap, *stemConcat->getOutput(0), 32, DimsHW{3, 3},
                                               DimsHW{2, 2}, DimsHW{1, 1}, 1, "conv2d_3", "batch_norm2d_3");
    ITensor* stage1Prep1 = addConvBnReluTensor(network, weightMap, *stage1Prep0, 48, DimsHW{1, 1}, DimsHW{1, 1},
                                               DimsHW{0, 0}, 1, "conv2d_4", "batch_norm2d_4");
    ITensor* c2 = addHgConvBlock(network, weightMap, *stage1Prep1, 48, 6, 5, 5, 11, 11, 64, 12, 12, 128);
    ITensor* c3 = addHgStandardBlock(network, weightMap, *c2, 96, 6, 13, 13, 128, DimsHW{2, 2}, 14, 14, 20, 20, 256, 21,
                                     21, 512);

    ITensor* stage3Down = addConvBnTensor(network, weightMap, *c3, 512, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 512,
                                          "conv2d_22", "batch_norm2d_22");
    ITensor* stage3A =
            addHgLightBlock(network, weightMap, *stage3Down, 192, 6, 23, 23, 35, 35, 512, 36, 36, 1024, false);
    ITensor* stage3B = addHgLightBlock(network, weightMap, *stage3A, 192, 6, 37, 37, 49, 49, 512, 50, 50, 1024, true);
    ITensor* c4 = addHgLightBlock(network, weightMap, *stage3B, 192, 6, 51, 51, 63, 63, 512, 64, 64, 1024, true);

    ITensor* stage4Down = addConvBnTensor(network, weightMap, *c4, 1024, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1024,
                                          "conv2d_65", "batch_norm2d_65");
    ITensor* c5 = addHgLightBlock(network, weightMap, *stage4Down, 384, 6, 66, 66, 78, 78, 1024, 79, 79, 2048, false);

    ITensor* l5 =
            addConvNoBiasTensor(network, weightMap, *c5, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_92");
    ITensor* l4 =
            addConvNoBiasTensor(network, weightMap, *c4, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_88");
    ITensor* l3 =
            addConvNoBiasTensor(network, weightMap, *c3, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_84");
    ITensor* l2 =
            addConvNoBiasTensor(network, weightMap, *c2, 256, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_81");

    IResizeLayer* up5 =
            addNearestResize(network, *l5, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, "server_det_up5_to_4");
    IElementWiseLayer* p4 = addSum(network, *l4, *up5->getOutput(0), "server_det_p4_sum");
    IResizeLayer* up4 = addNearestResize(network, *p4->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                                         "server_det_up4_to_3");
    IElementWiseLayer* p3 = addSum(network, *l3, *up4->getOutput(0), "server_det_p3_sum");
    IResizeLayer* up3 = addNearestResize(network, *p3->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                                         "server_det_up3_to_2");
    IElementWiseLayer* p2 = addSum(network, *l2, *up3->getOutput(0), "server_det_p2_sum");

    ITensor* p5Conv =
            addConvNoBiasTensor(network, weightMap, *l5, 64, DimsHW{9, 9}, DimsHW{1, 1}, DimsHW{4, 4}, 1, "conv2d_93");
    ITensor* p4Conv = addConvNoBiasTensor(network, weightMap, *p4->getOutput(0), 64, DimsHW{9, 9}, DimsHW{1, 1},
                                          DimsHW{4, 4}, 1, "conv2d_89");
    ITensor* p3Conv = addConvNoBiasTensor(network, weightMap, *p3->getOutput(0), 64, DimsHW{9, 9}, DimsHW{1, 1},
                                          DimsHW{4, 4}, 1, "conv2d_85");
    ITensor* p2Conv = addConvNoBiasTensor(network, weightMap, *p2->getOutput(0), 64, DimsHW{9, 9}, DimsHW{1, 1},
                                          DimsHW{4, 4}, 1, "conv2d_82");

    ITensor* p2Down = addConvNoBiasTensor(network, weightMap, *p2Conv, 64, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1,
                                          "conv2d_86");
    IElementWiseLayer* n3 = addSum(network, *p3Conv, *p2Down, "server_det_n3_sum");
    ITensor* n3Down = addConvNoBiasTensor(network, weightMap, *n3->getOutput(0), 64, DimsHW{3, 3}, DimsHW{2, 2},
                                          DimsHW{1, 1}, 1, "conv2d_90");
    IElementWiseLayer* n4 = addSum(network, *p4Conv, *n3Down, "server_det_n4_sum");
    ITensor* n4Down = addConvNoBiasTensor(network, weightMap, *n4->getOutput(0), 64, DimsHW{3, 3}, DimsHW{2, 2},
                                          DimsHW{1, 1}, 1, "conv2d_94");
    IElementWiseLayer* n5 = addSum(network, *p5Conv, *n4Down, "server_det_n5_sum");

    ITensor* o2Pre = addConvNoBiasTensor(network, weightMap, *p2Conv, 64, DimsHW{9, 9}, DimsHW{1, 1}, DimsHW{4, 4}, 1,
                                         "conv2d_83");
    ITensor* o3Pre = addConvNoBiasTensor(network, weightMap, *n3->getOutput(0), 64, DimsHW{9, 9}, DimsHW{1, 1},
                                         DimsHW{4, 4}, 1, "conv2d_87");
    ITensor* o4Pre = addConvNoBiasTensor(network, weightMap, *n4->getOutput(0), 64, DimsHW{9, 9}, DimsHW{1, 1},
                                         DimsHW{4, 4}, 1, "conv2d_91");
    ITensor* o5Pre = addConvNoBiasTensor(network, weightMap, *n5->getOutput(0), 64, DimsHW{9, 9}, DimsHW{1, 1},
                                         DimsHW{4, 4}, 1, "conv2d_95");

    ITensor* o5 = addLargeKernelBlock(network, weightMap, *o5Pre, 129, 137, 131, 134, 138, 132, 135, 139, 133, 136, 130,
                                      "batch_norm2d_83");
    ITensor* o4 = addLargeKernelBlock(network, weightMap, *o4Pre, 118, 126, 120, 123, 127, 121, 124, 128, 122, 125, 119,
                                      "batch_norm2d_82");
    ITensor* o3 = addLargeKernelBlock(network, weightMap, *o3Pre, 107, 115, 109, 112, 116, 110, 113, 117, 111, 114, 108,
                                      "batch_norm2d_81");
    ITensor* o2 = addLargeKernelBlock(network, weightMap, *o2Pre, 96, 104, 98, 101, 105, 99, 102, 106, 100, 103, 97,
                                      "batch_norm2d_80");

    IResizeLayer* o5Up = addNearestResize(network, *o5, std::vector<float>{1.0f, 1.0f, 8.0f, 8.0f}, "server_det_o5_up");
    IResizeLayer* o4Up = addNearestResize(network, *o4, std::vector<float>{1.0f, 1.0f, 4.0f, 4.0f}, "server_det_o4_up");
    IResizeLayer* o3Up = addNearestResize(network, *o3, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, "server_det_o3_up");
    IConcatenationLayer* neckOut =
            addConcat(network, std::vector<ITensor*>{o5Up->getOutput(0), o4Up->getOutput(0), o3Up->getOutput(0), o2}, 1,
                      "server_det_neck_concat");

    IScaleLayer* headConv =
            convBn(network, weightMap, *neckOut->getOutput(0), 64, 3, 1, 1, "conv2d_140", "batch_norm_0");
    IActivationLayer* headRelu0 = addRelu(network, *headConv->getOutput(0), "server_det_head_relu0");
    IDeconvolutionLayer* headDeconv0 =
            deconvBias(network, weightMap, *headRelu0->getOutput(0), 64, 2, 2, 0, "conv2d_transpose_0");
    IScaleLayer* headBn1 = addBatchNorm2d(network, weightMap, *headDeconv0->getOutput(0), "batch_norm_1", 1e-5f);
    headBn1->setName("batch_norm_1");
    IActivationLayer* headRelu1 = addRelu(network, *headBn1->getOutput(0), "server_det_head_relu1");
    IDeconvolutionLayer* shrinkLogit =
            deconvBias(network, weightMap, *headRelu1->getOutput(0), 1, 2, 2, 0, "conv2d_transpose_1");
    IActivationLayer* shrink = addSigmoid(network, *shrinkLogit->getOutput(0), "server_det_shrink_sigmoid");
    IResizeLayer* headFeatureUp =
            addNearestResize(network, *headRelu1->getOutput(0), std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f},
                             "server_det_head_feature_up");
    IConcatenationLayer* threshInput =
            addConcat(network, std::vector<ITensor*>{shrink->getOutput(0), headFeatureUp->getOutput(0)}, 1,
                      "server_det_thresh_concat");
    IScaleLayer* threshConv =
            convBn(network, weightMap, *threshInput->getOutput(0), 64, 3, 1, 1, "conv2d_142", "batch_norm_4");
    IActivationLayer* threshRelu = addRelu(network, *threshConv->getOutput(0), "server_det_thresh_relu");
    ITensor* threshLogit = addConvBiasTensor(network, weightMap, *threshRelu->getOutput(0), 1, DimsHW{1, 1},
                                             DimsHW{1, 1}, DimsHW{0, 0}, "conv2d_143");
    IActivationLayer* thresh = addSigmoid(network, *threshLogit, "server_det_thresh_sigmoid");
    IElementWiseLayer* fuse = addSum(network, *shrink->getOutput(0), *thresh->getOutput(0), "server_det_fuse");
    IElementWiseLayer* output = addScalarMul(network, weightMap, *fuse->getOutput(0), 0.5f, "server_det_scale");
    output->getOutput(0)->setName(kDetOutputTensorName);
    network->markOutput(*output->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildPPOCRv5MobileRec(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput(kRecInputTensorName, dt, Dims4{1, 3, kRecInputH, -1});
    if (!data) {
        throw std::runtime_error("failed to add PP-OCRv5 mobile rec input");
    }
    addRecOptimizationProfile(builder, config);

    IScaleLayer* stem = convBn(network, weightMap, *data, 16, 3, 2, 1, "conv2d_0", "batch_norm2d_0");
    IScaleLayer* stage0Dw =
            learnableRepLayer(network, weightMap, *stem->getOutput(0), 16, 3, 1, 1, 16, "conv2d_136", 0, true);
    IScaleLayer* stage0Pw =
            learnableRepLayer(network, weightMap, *stage0Dw->getOutput(0), 32, 1, 1, 0, 1, "conv2d_137", 2, true);

    IScaleLayer* stage1Dw =
            learnableRepLayer(network, weightMap, *stage0Pw->getOutput(0), 32, 3, 1, 1, 32, "conv2d_138", 4, true);
    IScaleLayer* stage1Pw =
            learnableRepLayer(network, weightMap, *stage1Dw->getOutput(0), 64, 1, 1, 0, 1, "conv2d_139", 6, true);
    IScaleLayer* stage1Block1Dw =
            learnableRepLayer(network, weightMap, *stage1Pw->getOutput(0), 64, 3, 1, 1, 64, "conv2d_140", 8, true);
    IScaleLayer* stage1Block1Pw = learnableRepLayer(network, weightMap, *stage1Block1Dw->getOutput(0), 64, 1, 1, 0, 1,
                                                    "conv2d_141", 10, true);

    IScaleLayer* stage2Dw = learnableRepLayer(network, weightMap, *stage1Block1Pw->getOutput(0), 64, DimsHW{3, 3},
                                              DimsHW{2, 1}, DimsHW{1, 1}, 64, "conv2d_142", 12, true);
    IScaleLayer* stage2Pw =
            learnableRepLayer(network, weightMap, *stage2Dw->getOutput(0), 128, 1, 1, 0, 1, "conv2d_143", 14, true);
    IScaleLayer* stage2Block1Dw =
            learnableRepLayer(network, weightMap, *stage2Pw->getOutput(0), 128, 3, 1, 1, 128, "conv2d_144", 16, true);
    IScaleLayer* stage2Block1Pw = learnableRepLayer(network, weightMap, *stage2Block1Dw->getOutput(0), 128, 1, 1, 0, 1,
                                                    "conv2d_145", 18, true);
    IScaleLayer* stage2Block2Dw =
            learnableRepLayer(network, weightMap, *stage2Block1Pw->getOutput(0), 128, DimsHW{3, 3}, DimsHW{1, 2},
                              DimsHW{1, 1}, 128, "conv2d_146", 20, true);
    IScaleLayer* stage2Block2Pw = learnableRepLayer(network, weightMap, *stage2Block2Dw->getOutput(0), 240, 1, 1, 0, 1,
                                                    "conv2d_147", 22, true);

    IScaleLayer* stage3Dw = learnableRepLayer(network, weightMap, *stage2Block2Pw->getOutput(0), 240, 5, 1, 2, 240,
                                              "conv2d_148", 24, true);
    IScaleLayer* stage3Pw =
            learnableRepLayer(network, weightMap, *stage3Dw->getOutput(0), 240, 1, 1, 0, 1, "conv2d_149", 26, true);
    IScaleLayer* stage3Block1Dw =
            learnableRepLayer(network, weightMap, *stage3Pw->getOutput(0), 240, 5, 1, 2, 240, "conv2d_150", 28, true);
    IScaleLayer* stage3Block1Pw = learnableRepLayer(network, weightMap, *stage3Block1Dw->getOutput(0), 240, 1, 1, 0, 1,
                                                    "conv2d_151", 30, true);
    IScaleLayer* stage3Block2Dw = learnableRepLayer(network, weightMap, *stage3Block1Pw->getOutput(0), 240, 5, 1, 2,
                                                    240, "conv2d_152", 32, true);
    IScaleLayer* stage3Block2Pw = learnableRepLayer(network, weightMap, *stage3Block2Dw->getOutput(0), 240, 1, 1, 0, 1,
                                                    "conv2d_153", 34, true);
    IScaleLayer* stage3Block3Dw = learnableRepLayer(network, weightMap, *stage3Block2Pw->getOutput(0), 240, 5, 1, 2,
                                                    240, "conv2d_154", 36, true);
    IScaleLayer* stage3Block3Pw = learnableRepLayer(network, weightMap, *stage3Block3Dw->getOutput(0), 240, 1, 1, 0, 1,
                                                    "conv2d_155", 38, true);

    IScaleLayer* stage4Dw = learnableRepLayer(network, weightMap, *stage3Block3Pw->getOutput(0), 240, DimsHW{5, 5},
                                              DimsHW{2, 1}, DimsHW{2, 2}, 240, "conv2d_156", 40, true);
    IElementWiseLayer* stage4Se =
            seLayer(network, weightMap, *stage4Dw->getOutput(0), 60, 240, "conv2d_96", "conv2d_97");
    IScaleLayer* stage4Pw =
            learnableRepLayer(network, weightMap, *stage4Se->getOutput(0), 480, 1, 1, 0, 1, "conv2d_157", 42, true);
    IScaleLayer* stage4Block1Dw =
            learnableRepLayer(network, weightMap, *stage4Pw->getOutput(0), 480, 5, 1, 2, 480, "conv2d_158", 44, true);
    IElementWiseLayer* stage4Block1Se =
            seLayer(network, weightMap, *stage4Block1Dw->getOutput(0), 120, 480, "conv2d_107", "conv2d_108");
    IScaleLayer* stage4Block1Pw = learnableRepLayer(network, weightMap, *stage4Block1Se->getOutput(0), 480, 1, 1, 0, 1,
                                                    "conv2d_159", 46, true);
    IScaleLayer* stage4Block2Dw =
            learnableRepLayer(network, weightMap, *stage4Block1Pw->getOutput(0), 480, DimsHW{5, 5}, DimsHW{2, 1},
                              DimsHW{2, 2}, 480, "conv2d_160", 48, true);
    IScaleLayer* stage4Block2Pw = learnableRepLayer(network, weightMap, *stage4Block2Dw->getOutput(0), 480, 1, 1, 0, 1,
                                                    "conv2d_161", 50, true);
    IScaleLayer* stage4Block3Dw = learnableRepLayer(network, weightMap, *stage4Block2Pw->getOutput(0), 480, 5, 1, 2,
                                                    480, "conv2d_162", 52, true);
    IScaleLayer* stage4Block3Pw = learnableRepLayer(network, weightMap, *stage4Block3Dw->getOutput(0), 480, 1, 1, 0, 1,
                                                    "conv2d_163", 54, true);

    IPoolingLayer* backboneOut = addPool2d(network, *stage4Block3Pw->getOutput(0), PoolingType::kAVERAGE, DimsHW{3, 2},
                                           DimsHW{3, 2}, DimsHW{0, 0}, "rec_backbone_avgpool");

    IElementWiseLayer* encConv0 = convBnSwish(network, weightMap, *backboneOut->getOutput(0), 60, DimsHW{1, 3},
                                              DimsHW{1, 1}, DimsHW{0, 1}, 1, "conv2d_131", "batch_norm2d_146");
    IElementWiseLayer* encConv1 = convBnSwish(network, weightMap, *encConv0->getOutput(0), 120, DimsHW{1, 1},
                                              DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_132", "batch_norm2d_147");

    IShuffleLayer* flatten = addShuffle(network, *encConv1->getOutput(0), Dims3{0, 120, -1}, "rec_flatten");
    IShuffleLayer* seq = addPermute(network, *flatten->getOutput(0), Permutation{0, 2, 1}, "rec_im2seq");

    ITensor* svtr0 = addSvtrBlock(network, weightMap, *seq->getOutput(0), "layer_norm_0", "linear_0", "linear_1",
                                  "layer_norm_1", "linear_2", "linear_3", "svtr_block_0");
    ITensor* svtr1 = addSvtrBlock(network, weightMap, *svtr0, "layer_norm_2", "linear_4", "linear_5", "layer_norm_3",
                                  "linear_6", "linear_7", "svtr_block_1");
    IElementWiseLayer* svtrNorm = addLayerNorm(network, weightMap, *svtr1, 120, "layer_norm_4", 1e-6f);

    IShuffleLayer* svtrReshape = addShuffle(network, *svtrNorm->getOutput(0), Dims4{0, 1, -1, 120}, "svtr_to_nhwc");
    IShuffleLayer* svtrNchw = addPermute(network, *svtrReshape->getOutput(0), Permutation{0, 3, 1, 2}, "svtr_to_nchw");
    IElementWiseLayer* encConv2 = convBnSwish(network, weightMap, *svtrNchw->getOutput(0), 480, DimsHW{1, 1},
                                              DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_133", "batch_norm2d_148");

    IConcatenationLayer* encConcat = addConcat(
            network, std::vector<ITensor*>{backboneOut->getOutput(0), encConv2->getOutput(0)}, 1, "rec_svtr_concat");
    IElementWiseLayer* encConv3 = convBnSwish(network, weightMap, *encConcat->getOutput(0), 60, DimsHW{1, 3},
                                              DimsHW{1, 1}, DimsHW{0, 1}, 1, "conv2d_134", "batch_norm2d_149");
    IElementWiseLayer* encConv4 = convBnSwish(network, weightMap, *encConv3->getOutput(0), 120, DimsHW{1, 1},
                                              DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_135", "batch_norm2d_150");

    IShuffleLayer* squeeze = addShuffle(network, *encConv4->getOutput(0), Dims3{0, 120, -1}, "rec_squeeze_h");
    IShuffleLayer* ctcInput = addPermute(network, *squeeze->getOutput(0), Permutation{0, 2, 1}, "rec_ctc_input");
    IElementWiseLayer* logits = addLinear(network, weightMap, *ctcInput->getOutput(0), 120, kRecClassCount, "linear_8");
    ISoftMaxLayer* prob = addSoftmax(network, *logits->getOutput(0), 1 << 2, "rec_ctc_softmax");
    prob->getOutput(0)->setName(kRecOutputTensorName);
    network->markOutput(*prob->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildPPOCRv5ServerRec(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput(kRecInputTensorName, dt, Dims4{1, 3, kRecInputH, -1});
    if (!data) {
        throw std::runtime_error("failed to add PP-OCRv5 server rec input");
    }
    addRecOptimizationProfile(builder, config);
    const char* debugStage = std::getenv("PPOCRV5_DEBUG_SERVER_REC_STAGE");
    auto markDebugStage = [&](const char* stage, ITensor& tensor) -> bool {
        if (!debugStage || std::strcmp(debugStage, stage) != 0) {
            return false;
        }
        tensor.setName(kRecOutputTensorName);
        network->markOutput(tensor);
        return true;
    };

    ITensor* stem0 = addConvBnReluTensor(network, weightMap, *data, 32, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1,
                                         "conv2d_0", "batch_norm2d_0");
    if (markDebugStage("stem0", *stem0)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* stem1 = addSameConvBnReluTensor(network, weightMap, *stem0, 16, 2, "conv2d_1", "batch_norm2d_1");
    if (markDebugStage("stem1", *stem1)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* stem2 = addSameConvBnReluTensor(network, weightMap, *stem1, 32, 2, "conv2d_2", "batch_norm2d_2");
    if (markDebugStage("stem2", *stem2)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IPaddingLayer* stemPoolPad = network->addPaddingNd(*stem0, DimsHW{0, 0}, DimsHW{1, 1});
    assert(stemPoolPad);
    stemPoolPad->setName("max_pool2d_0_same_pad");
    IPoolingLayer* stemPool = addPool2d(network, *stemPoolPad->getOutput(0), PoolingType::kMAX, DimsHW{2, 2},
                                        DimsHW{1, 1}, DimsHW{0, 0}, "max_pool2d_0");
    IConcatenationLayer* stemConcat =
            addConcat(network, std::vector<ITensor*>{stemPool->getOutput(0), stem2}, 1, "server_rec_stem_concat");
    if (markDebugStage("stemcat", *stemConcat->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }

    ITensor* stage1Prep0 = addConvBnReluTensor(network, weightMap, *stemConcat->getOutput(0), 32, DimsHW{3, 3},
                                               DimsHW{1, 1}, DimsHW{1, 1}, 1, "conv2d_3", "batch_norm2d_3");
    if (markDebugStage("prep0", *stage1Prep0)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* stage1Prep1 = addConvBnReluTensor(network, weightMap, *stage1Prep0, 48, DimsHW{1, 1}, DimsHW{1, 1},
                                               DimsHW{0, 0}, 1, "conv2d_4", "batch_norm2d_4");
    if (markDebugStage("prep1", *stage1Prep1)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* stage1 = addHgStandardBlock(network, weightMap, *stage1Prep1, 48, 6, 5, 5, 48, DimsHW{2, 1}, 6, 6, 12, 12,
                                         64, 13, 13, 128);
    if (markDebugStage("c2", *stage1)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* stage2 = addHgStandardBlock(network, weightMap, *stage1, 96, 6, 14, 14, 128, DimsHW{1, 2}, 15, 15, 21, 21,
                                         256, 22, 22, 512);
    if (markDebugStage("c3", *stage2)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }

    ITensor* stage3Down = addConvBnTensor(network, weightMap, *stage2, 512, DimsHW{3, 3}, DimsHW{2, 1}, DimsHW{1, 1},
                                          512, "conv2d_23", "batch_norm2d_23");
    ITensor* stage3A =
            addHgLightBlock(network, weightMap, *stage3Down, 192, 6, 24, 24, 36, 36, 512, 37, 37, 1024, false);
    ITensor* stage3B = addHgLightBlock(network, weightMap, *stage3A, 192, 6, 38, 38, 50, 50, 512, 51, 51, 1024, true);
    ITensor* stage3C = addHgLightBlock(network, weightMap, *stage3B, 192, 6, 52, 52, 64, 64, 512, 65, 65, 1024, true);
    if (markDebugStage("c4", *stage3C)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }

    ITensor* stage4Down = addConvBnTensor(network, weightMap, *stage3C, 1024, DimsHW{3, 3}, DimsHW{2, 1}, DimsHW{1, 1},
                                          1024, "conv2d_66", "batch_norm2d_66");
    ITensor* stage4 =
            addHgLightBlock(network, weightMap, *stage4Down, 384, 6, 67, 67, 79, 79, 1024, 80, 80, 2048, false);
    if (markDebugStage("c5", *stage4)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }

    IPoolingLayer* backboneOut = addPool2d(network, *stage4, PoolingType::kAVERAGE, DimsHW{3, 2}, DimsHW{3, 2},
                                           DimsHW{0, 0}, "server_rec_backbone_avgpool");
    if (markDebugStage("backbone", *backboneOut->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* encConv0 = convBnSwish(network, weightMap, *backboneOut->getOutput(0), 256, DimsHW{1, 3},
                                              DimsHW{1, 1}, DimsHW{0, 1}, 1, "conv2d_82", "batch_norm2d_81");
    IElementWiseLayer* encConv1 = convBnSwish(network, weightMap, *encConv0->getOutput(0), 120, DimsHW{1, 1},
                                              DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_83", "batch_norm2d_82");

    IShuffleLayer* flatten = addShuffle(network, *encConv1->getOutput(0), Dims3{0, 120, -1}, "server_rec_flatten");
    IShuffleLayer* seq = addPermute(network, *flatten->getOutput(0), Permutation{0, 2, 1}, "server_rec_im2seq");

    ITensor* svtr0 = addSvtrBlock(network, weightMap, *seq->getOutput(0), "layer_norm_0", "linear_1", "linear_2",
                                  "layer_norm_1", "linear_3", "linear_4", "server_svtr_block_0");
    ITensor* svtr1 = addSvtrBlock(network, weightMap, *svtr0, "layer_norm_2", "linear_5", "linear_6", "layer_norm_3",
                                  "linear_7", "linear_8", "server_svtr_block_1");
    IElementWiseLayer* svtrNorm = addLayerNorm(network, weightMap, *svtr1, 120, "layer_norm_4", 1e-6f);

    IShuffleLayer* svtrReshape =
            addShuffle(network, *svtrNorm->getOutput(0), Dims4{0, 1, -1, 120}, "server_svtr_to_nhwc");
    IShuffleLayer* svtrNchw =
            addPermute(network, *svtrReshape->getOutput(0), Permutation{0, 3, 1, 2}, "server_svtr_to_nchw");
    IElementWiseLayer* encConv2 = convBnSwish(network, weightMap, *svtrNchw->getOutput(0), 2048, DimsHW{1, 1},
                                              DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_84", "batch_norm2d_83");

    IConcatenationLayer* encConcat =
            addConcat(network, std::vector<ITensor*>{backboneOut->getOutput(0), encConv2->getOutput(0)}, 1,
                      "server_rec_svtr_concat");
    IElementWiseLayer* encConv3 = convBnSwish(network, weightMap, *encConcat->getOutput(0), 256, DimsHW{1, 3},
                                              DimsHW{1, 1}, DimsHW{0, 1}, 1, "conv2d_85", "batch_norm2d_84");
    IElementWiseLayer* encConv4 = convBnSwish(network, weightMap, *encConv3->getOutput(0), 120, DimsHW{1, 1},
                                              DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_86", "batch_norm2d_85");

    IShuffleLayer* squeeze = addShuffle(network, *encConv4->getOutput(0), Dims3{0, 120, -1}, "server_rec_squeeze_h");
    IShuffleLayer* ctcInput = addPermute(network, *squeeze->getOutput(0), Permutation{0, 2, 1}, "server_rec_ctc_input");
    IElementWiseLayer* logits = addLinear(network, weightMap, *ctcInput->getOutput(0), 120, kRecClassCount, "linear_9");
    ISoftMaxLayer* prob = addSoftmax(network, *logits->getOutput(0), 1 << 2, "server_rec_ctc_softmax");
    prob->getOutput(0)->setName(kRecOutputTensorName);
    network->markOutput(*prob->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildUVDocModel(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    static const int inputH = 800;
    static const int inputW = 800;
    ITensor* data = network->addInput("image", dt, Dims4{1, 3, inputH, inputW});
    if (!data) {
        throw std::runtime_error("failed to add UVDoc input");
    }

    ITensor* resized = addBilinearResizeTensor(network, *data, 3, 712, 488, "uvdoc_input_resize");
    ITensor* stem0 = addConvBnReluTensor(network, weightMap, *resized, 32, 5, 2, 2, 1, "conv2d_0", "batch_norm2d_0");
    ITensor* stem1 = addConvBnReluTensor(network, weightMap, *stem0, 32, 5, 2, 2, 1, "conv2d_1", "batch_norm2d_1");

    ITensor* c1 = addUvdocResidualBlock(network, weightMap, *stem1, 32, 2, 2, 3, 3, 1);
    c1 = addUvdocResidualBlock(network, weightMap, *c1, 32, 4, 4, 5, 5, 3);
    c1 = addUvdocResidualBlock(network, weightMap, *c1, 32, 6, 6, 7, 7, 3);

    ITensor* c2 = addUvdocDownBlock(network, weightMap, *c1, 64, 8, 8, 9, 9, 10, 10);
    c2 = addUvdocResidualBlock(network, weightMap, *c2, 64, 11, 11, 12, 12, 3);
    c2 = addUvdocResidualBlock(network, weightMap, *c2, 64, 13, 13, 14, 14, 3);
    c2 = addUvdocResidualBlock(network, weightMap, *c2, 64, 15, 15, 16, 16, 3);

    ITensor* c3 = addUvdocDownBlock(network, weightMap, *c2, 128, 17, 17, 18, 18, 19, 19);
    c3 = addUvdocResidualBlock(network, weightMap, *c3, 128, 20, 20, 21, 21, 3);
    c3 = addUvdocResidualBlock(network, weightMap, *c3, 128, 22, 22, 23, 23, 3);
    c3 = addUvdocResidualBlock(network, weightMap, *c3, 128, 24, 24, 25, 25, 3);
    c3 = addUvdocResidualBlock(network, weightMap, *c3, 128, 26, 26, 27, 27, 3);
    c3 = addUvdocResidualBlock(network, weightMap, *c3, 128, 28, 28, 29, 29, 3);

    ITensor* branch0 = addConvBnReluTensor(network, weightMap, *c3, 128, 3, 1, 1, 1, "conv2d_30", "batch_norm2d_30");
    ITensor* branch1 = addConvBnReluTensor(network, weightMap, *c3, 128, 3, 1, 2, 2, "conv2d_31", "batch_norm2d_31");
    ITensor* branch2 = addConvBnReluTensor(network, weightMap, *c3, 128, 3, 1, 5, 5, "conv2d_32", "batch_norm2d_32");
    ITensor* branch3 = addConvBnReluTensor(network, weightMap, *c3, 128, 3, 1, 8, 8, "conv2d_33", "batch_norm2d_33");
    branch3 = addConvBnReluTensor(network, weightMap, *branch3, 128, 3, 1, 3, 3, "conv2d_34", "batch_norm2d_34");
    branch3 = addConvBnReluTensor(network, weightMap, *branch3, 128, 3, 1, 2, 2, "conv2d_35", "batch_norm2d_35");
    ITensor* branch4 = addConvBnReluTensor(network, weightMap, *c3, 128, 3, 1, 12, 12, "conv2d_36", "batch_norm2d_36");
    branch4 = addConvBnReluTensor(network, weightMap, *branch4, 128, 3, 1, 7, 7, "conv2d_37", "batch_norm2d_37");
    branch4 = addConvBnReluTensor(network, weightMap, *branch4, 128, 3, 1, 4, 4, "conv2d_38", "batch_norm2d_38");
    ITensor* branch5 = addConvBnReluTensor(network, weightMap, *c3, 128, 3, 1, 18, 18, "conv2d_39", "batch_norm2d_39");
    branch5 = addConvBnReluTensor(network, weightMap, *branch5, 128, 3, 1, 12, 12, "conv2d_40", "batch_norm2d_40");
    branch5 = addConvBnReluTensor(network, weightMap, *branch5, 128, 3, 1, 6, 6, "conv2d_41", "batch_norm2d_41");

    IConcatenationLayer* context =
            addConcat(network, std::vector<ITensor*>{branch0, branch1, branch2, branch3, branch4, branch5}, 1,
                      "uvdoc_context_concat");
    ITensor* fuse = addConvBnReluTensor(network, weightMap, *context->getOutput(0), 128, 1, 1, 0, 1, "conv2d_42",
                                        "batch_norm2d_42");

    ITensor* headPad0 = addUvdocReflectPad2d(network, *fuse, 1, 128, 45, 31, 2, "uvdoc_head_reflect_pad0");
    ITensor* headConv0 = addConvBnTensor(network, weightMap, *headPad0, 32, DimsHW{5, 5}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
                                         "conv2d_43", "batch_norm2d_43");
    ITensor* headAct0 = addUvdocPRelu(network, weightMap, *headConv0, "p_re_lu_0", "p_re_lu_0");
    ITensor* headPad1 = addUvdocReflectPad2d(network, *headAct0, 1, 32, 45, 31, 2, "uvdoc_head_reflect_pad1");
    ITensor* lowGrid =
            addConvBiasTensor(network, weightMap, *headPad1, 2, DimsHW{5, 5}, DimsHW{1, 1}, DimsHW{0, 0}, "conv2d_44");
    ITensor* grid = addBilinearResizeTensor(network, *lowGrid, 2, inputH, inputW, "uvdoc_grid_resize");
    IShuffleLayer* nhwcGrid = addPermute(network, *grid, Permutation{0, 2, 3, 1}, "uvdoc_grid_nhwc");

    IGridSampleLayer* output = network->addGridSample(*data, *nhwcGrid->getOutput(0));
    assert(output);
    output->setInterpolationMode(InterpolationMode::kLINEAR);
    output->setAlignCorners(true);
    output->setSampleMode(SampleMode::kFILL);
    output->setName("uvdoc_grid_sample");
    output->getOutput(0)->setName("output");
    network->markOutput(*output->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildSLANetPlusModel(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput("x", dt, Dims4{1, 3, 800, 800});
    if (!data) {
        throw std::runtime_error("failed to add SLANet_plus input");
    }

    IElementWiseLayer* stem = convBnHSwish(network, weightMap, *data, 16, 3, 2, 1, "conv2d_0", "batch_norm_0");
    IElementWiseLayer* b0 = slanetLcNetBlock(network, weightMap, *stem->getOutput(0), 16, 32, 1, 1, 2, 2, 3, 1, false);
    IElementWiseLayer* b1 = slanetLcNetBlock(network, weightMap, *b0->getOutput(0), 32, 64, 3, 3, 4, 4, 3, 2, false);
    IElementWiseLayer* b2 = slanetLcNetBlock(network, weightMap, *b1->getOutput(0), 64, 64, 5, 5, 6, 6, 3, 1, false);
    IElementWiseLayer* b3 = slanetLcNetBlock(network, weightMap, *b2->getOutput(0), 64, 128, 7, 7, 8, 8, 3, 2, false);
    IElementWiseLayer* b4 =
            slanetLcNetBlock(network, weightMap, *b3->getOutput(0), 128, 128, 9, 9, 10, 10, 3, 1, false);
    IElementWiseLayer* b5 =
            slanetLcNetBlock(network, weightMap, *b4->getOutput(0), 128, 256, 11, 11, 12, 12, 3, 2, false);
    IElementWiseLayer* b6 =
            slanetLcNetBlock(network, weightMap, *b5->getOutput(0), 256, 256, 13, 13, 14, 14, 5, 1, false);
    IElementWiseLayer* b7 =
            slanetLcNetBlock(network, weightMap, *b6->getOutput(0), 256, 256, 15, 15, 16, 16, 5, 1, false);
    IElementWiseLayer* b8 =
            slanetLcNetBlock(network, weightMap, *b7->getOutput(0), 256, 256, 17, 17, 18, 18, 5, 1, false);
    IElementWiseLayer* b9 =
            slanetLcNetBlock(network, weightMap, *b8->getOutput(0), 256, 256, 19, 19, 20, 20, 5, 1, false);
    IElementWiseLayer* b10 =
            slanetLcNetBlock(network, weightMap, *b9->getOutput(0), 256, 256, 21, 21, 22, 22, 5, 1, false);
    IElementWiseLayer* b11 =
            slanetLcNetBlock(network, weightMap, *b10->getOutput(0), 256, 512, 23, 23, 26, 24, 5, 2, true);
    IElementWiseLayer* b12 =
            slanetLcNetBlock(network, weightMap, *b11->getOutput(0), 512, 512, 27, 25, 30, 26, 5, 1, true);

    IElementWiseLayer* p2 =
            convBnHSwish(network, weightMap, *b2->getOutput(0), 96, 1, 1, 0, "conv2d_31", "batch_norm2d_0");
    IElementWiseLayer* p3 =
            convBnHSwish(network, weightMap, *b4->getOutput(0), 96, 1, 1, 0, "conv2d_32", "batch_norm2d_1");
    IElementWiseLayer* p4 =
            convBnHSwish(network, weightMap, *b10->getOutput(0), 96, 1, 1, 0, "conv2d_33", "batch_norm2d_2");
    IElementWiseLayer* p5 =
            convBnHSwish(network, weightMap, *b12->getOutput(0), 96, 1, 1, 0, "conv2d_34", "batch_norm2d_3");

    ITensor* p5Up = addSLANetResizeTo(network, *p5->getOutput(0), 50, 50, "slanet_p5_up");
    IConcatenationLayer* td4Cat = addConcat(network, {p5Up, p4->getOutput(0)}, 1, "slanet_td4_cat");
    ITensor* td4 = addSLANetCspBlock(network, weightMap, *td4Cat->getOutput(0), 36, 5, 35, 4, 38, 7, 39, 8, 40, 9, 37,
                                     6, "slanet_td4");

    ITensor* td4Up = addSLANetResizeTo(network, *td4, 100, 100, "slanet_td4_up");
    IConcatenationLayer* td3Cat = addConcat(network, {td4Up, p3->getOutput(0)}, 1, "slanet_td3_cat");
    ITensor* td3 = addSLANetCspBlock(network, weightMap, *td3Cat->getOutput(0), 42, 11, 41, 10, 44, 13, 45, 14, 46, 15,
                                     43, 12, "slanet_td3");

    ITensor* td3Up = addSLANetResizeTo(network, *td3, 200, 200, "slanet_td3_up");
    IConcatenationLayer* td2Cat = addConcat(network, {td3Up, p2->getOutput(0)}, 1, "slanet_td2_cat");
    ITensor* td2 = addSLANetCspBlock(network, weightMap, *td2Cat->getOutput(0), 48, 17, 47, 16, 50, 19, 51, 20, 52, 21,
                                     49, 18, "slanet_td2");

    IElementWiseLayer* down3Dw =
            convBnHSwish(network, weightMap, *td2, 96, 5, 2, 2, 96, "conv2d_53", "batch_norm2d_22");
    IElementWiseLayer* down3Pw =
            convBnHSwish(network, weightMap, *down3Dw->getOutput(0), 96, 1, 1, 0, "conv2d_54", "batch_norm2d_23");
    IConcatenationLayer* bu3Cat = addConcat(network, {down3Pw->getOutput(0), td3}, 1, "slanet_bu3_cat");
    ITensor* bu3 = addSLANetCspBlock(network, weightMap, *bu3Cat->getOutput(0), 56, 25, 55, 24, 58, 27, 59, 28, 60, 29,
                                     57, 26, "slanet_bu3");

    IElementWiseLayer* down4Dw =
            convBnHSwish(network, weightMap, *bu3, 96, 5, 2, 2, 96, "conv2d_61", "batch_norm2d_30");
    IElementWiseLayer* down4Pw =
            convBnHSwish(network, weightMap, *down4Dw->getOutput(0), 96, 1, 1, 0, "conv2d_62", "batch_norm2d_31");
    IConcatenationLayer* bu4Cat = addConcat(network, {down4Pw->getOutput(0), td4}, 1, "slanet_bu4_cat");
    ITensor* bu4 = addSLANetCspBlock(network, weightMap, *bu4Cat->getOutput(0), 64, 33, 63, 32, 66, 35, 67, 36, 68, 37,
                                     65, 34, "slanet_bu4");

    IElementWiseLayer* down5Dw =
            convBnHSwish(network, weightMap, *bu4, 96, 5, 2, 2, 96, "conv2d_69", "batch_norm2d_38");
    IElementWiseLayer* down5Pw =
            convBnHSwish(network, weightMap, *down5Dw->getOutput(0), 96, 1, 1, 0, "conv2d_70", "batch_norm2d_39");
    IConcatenationLayer* bu5Cat = addConcat(network, {down5Pw->getOutput(0), p5->getOutput(0)}, 1, "slanet_bu5_cat");
    ITensor* bu5 = addSLANetCspBlock(network, weightMap, *bu5Cat->getOutput(0), 72, 41, 71, 40, 74, 43, 75, 44, 76, 45,
                                     73, 42, "slanet_bu5");

    IShuffleLayer* flatten = addShuffle(network, *bu5, Dims3{1, 96, 625}, "slanet_flatten");
    IShuffleLayer* sequence = addPermute(network, *flatten->getOutput(0), Permutation{0, 2, 1}, "slanet_sequence");
    ITensor* attnFeature =
            addLinearNoBiasTensor(network, weightMap, *sequence->getOutput(0), 96, 256, "linear_0", "linear_0");

    ITensor* initialCond = addBoolConstantTensor(network, weightMap, "slanet_init_cond", Dims{}, {true});
    ITensor* initialEos = addBoolConstantTensor(network, weightMap, "slanet_init_eos", Dims{}, {false});
    ITensor* initialHidden = addFloatConstantTensor(network, weightMap, "slanet_init_hidden", Dims2{1, 256},
                                                    std::vector<float>(256, 0.0f));
    ITensor* initialCounter = addIntConstantTensor(network, weightMap, "slanet_init_counter", Dims{}, {0});
    ITensor* initialLoc = addFloatConstantTensor(network, weightMap, "slanet_init_loc", Dims3{1, 501, 8},
                                                 std::vector<float>(501 * 8, 0.0f));
    ITensor* initialPrevId = addIntConstantTensor(network, weightMap, "slanet_init_prev_id", makeDims1(1), {0});
    ITensor* initialIds =
            addIntConstantTensor(network, weightMap, "slanet_init_ids", Dims2{1, 501}, std::vector<int32_t>(501, 0));
    ITensor* initialChar = addFloatConstantTensor(network, weightMap, "slanet_init_char", Dims3{1, 501, 50},
                                                  std::vector<float>(501 * 50, 0.0f));
    ITensor* initialPrevLoc = addFloatConstantTensor(network, weightMap, "slanet_init_prev_loc", Dims2{1, 8},
                                                     std::vector<float>(8, 0.0f));
    ITensor* initialPrevChar = addFloatConstantTensor(network, weightMap, "slanet_init_prev_char", Dims2{1, 50},
                                                      std::vector<float>(50, 0.0f));

    ILoop* loop = network->addLoop();
    assert(loop);
    loop->setName("slanet_decoder_loop");
    IRecurrenceLayer* condRec = loop->addRecurrence(*initialCond);
    IRecurrenceLayer* eosRec = loop->addRecurrence(*initialEos);
    IRecurrenceLayer* hiddenRec = loop->addRecurrence(*initialHidden);
    IRecurrenceLayer* counterRec = loop->addRecurrence(*initialCounter);
    IRecurrenceLayer* locRec = loop->addRecurrence(*initialLoc);
    IRecurrenceLayer* prevIdRec = loop->addRecurrence(*initialPrevId);
    IRecurrenceLayer* idsRec = loop->addRecurrence(*initialIds);
    IRecurrenceLayer* charRec = loop->addRecurrence(*initialChar);
    IRecurrenceLayer* prevLocRec = loop->addRecurrence(*initialPrevLoc);
    IRecurrenceLayer* prevCharRec = loop->addRecurrence(*initialPrevChar);
    assert(condRec && eosRec && hiddenRec && counterRec && locRec && prevIdRec && idsRec && charRec && prevLocRec &&
           prevCharRec);
    loop->addTripLimit(*condRec->getOutput(0), TripLimit::kWHILE);

    ITensor* prevOneHot = addOneHotTensor(network, weightMap, *prevIdRec->getOutput(0), 50, 1, "slanet_prev_id");
    ITensor* hiddenProj =
            addLinear2dTensor(network, weightMap, *hiddenRec->getOutput(0), 256, 256, "linear_1", "linear_1");
    IShuffleLayer* hiddenProjUnsqueeze =
            addShuffle(network, *hiddenProj, Dims3{1, 1, 256}, "slanet_hidden_proj_unsqueeze");
    IElementWiseLayer* attnSum =
            addSum(network, *attnFeature, *hiddenProjUnsqueeze->getOutput(0), "slanet_attention_sum");
    IActivationLayer* attnTanh = network->addActivation(*attnSum->getOutput(0), ActivationType::kTANH);
    assert(attnTanh);
    attnTanh->setName("slanet_attention_tanh");
    ITensor* attnLogit =
            addLinearNoBiasTensor(network, weightMap, *attnTanh->getOutput(0), 256, 1, "linear_2", "linear_2");
    ISoftMaxLayer* attn = addSoftmax(network, *attnLogit, 1 << 1, "slanet_attention_softmax");
    IShuffleLayer* attnT = addPermute(network, *attn->getOutput(0), Permutation{0, 2, 1}, "slanet_attention_t");
    IMatrixMultiplyLayer* contextMatmul = network->addMatrixMultiply(*attnT->getOutput(0), MatrixOperation::kNONE,
                                                                     *sequence->getOutput(0), MatrixOperation::kNONE);
    assert(contextMatmul);
    contextMatmul->setName("slanet_context_matmul");
    IShuffleLayer* context = addShuffle(network, *contextMatmul->getOutput(0), Dims2{1, 96}, "slanet_context");
    IConcatenationLayer* decoderInput =
            addConcat(network, {context->getOutput(0), prevOneHot}, 1, "slanet_decoder_input");

    ITensor* inputGate = addLinearTransposeTensor(network, weightMap, *decoderInput->getOutput(0), 768, 146, 768,
                                                  "gru_cell_0.w_0", "gru_cell_0.b_0", "gru_cell_0_input");
    ITensor* hiddenGate = addLinearTransposeTensor(network, weightMap, *hiddenRec->getOutput(0), 768, 256, 768,
                                                   "gru_cell_0.w_1", "gru_cell_0.b_1", "gru_cell_0_hidden");
    ITensor* iR = addSlice2dTensor(network, *inputGate, 0, 256, "slanet_gru_ir");
    ITensor* iZ = addSlice2dTensor(network, *inputGate, 256, 256, "slanet_gru_iz");
    ITensor* iN = addSlice2dTensor(network, *inputGate, 512, 256, "slanet_gru_in");
    ITensor* hR = addSlice2dTensor(network, *hiddenGate, 0, 256, "slanet_gru_hr");
    ITensor* hZ = addSlice2dTensor(network, *hiddenGate, 256, 256, "slanet_gru_hz");
    ITensor* hN = addSlice2dTensor(network, *hiddenGate, 512, 256, "slanet_gru_hn");
    IElementWiseLayer* resetSum = addSum(network, *iR, *hR, "slanet_gru_reset_sum");
    IActivationLayer* resetGate = addSigmoid(network, *resetSum->getOutput(0), "slanet_gru_reset");
    IElementWiseLayer* updateSum = addSum(network, *iZ, *hZ, "slanet_gru_update_sum");
    IActivationLayer* updateGate = addSigmoid(network, *updateSum->getOutput(0), "slanet_gru_update");
    IElementWiseLayer* resetHidden =
            network->addElementWise(*resetGate->getOutput(0), *hN, ElementWiseOperation::kPROD);
    assert(resetHidden);
    resetHidden->setName("slanet_gru_reset_hidden");
    IElementWiseLayer* newSum = addSum(network, *iN, *resetHidden->getOutput(0), "slanet_gru_new_sum");
    IActivationLayer* newGate = network->addActivation(*newSum->getOutput(0), ActivationType::kTANH);
    assert(newGate);
    newGate->setName("slanet_gru_new");
    IElementWiseLayer* hiddenDelta =
            network->addElementWise(*hiddenRec->getOutput(0), *newGate->getOutput(0), ElementWiseOperation::kSUB);
    assert(hiddenDelta);
    hiddenDelta->setName("slanet_gru_hidden_delta");
    IElementWiseLayer* updateDelta =
            network->addElementWise(*hiddenDelta->getOutput(0), *updateGate->getOutput(0), ElementWiseOperation::kPROD);
    assert(updateDelta);
    updateDelta->setName("slanet_gru_update_delta");
    IElementWiseLayer* nextHidden =
            addSum(network, *updateDelta->getOutput(0), *newGate->getOutput(0), "slanet_gru_next_hidden");

    ITensor* charHidden =
            addLinear2dTensor(network, weightMap, *nextHidden->getOutput(0), 256, 256, "linear_3", "linear_3");
    ITensor* charLogits = addLinear2dTensor(network, weightMap, *charHidden, 256, 50, "linear_4", "linear_4");
    ITensor* locHidden =
            addLinear2dTensor(network, weightMap, *nextHidden->getOutput(0), 256, 256, "linear_5", "linear_5");
    ITensor* locLogits = addLinear2dTensor(network, weightMap, *locHidden, 256, 8, "linear_6", "linear_6");
    IActivationLayer* loc = addSigmoid(network, *locLogits, "slanet_loc_sigmoid");

    ITopKLayer* topk = network->addTopK(*charLogits, TopKOperation::kMAX, 1, 1 << 1);
    assert(topk);
    topk->setName("slanet_argmax");
    IShuffleLayer* nextId = addShuffle(network, *topk->getOutput(1), makeDims1(1), "slanet_next_id");

    ITensor* nextLocTensor = addTensorAtIndex(network, weightMap, *locRec->getOutput(0), *loc->getOutput(0),
                                              *counterRec->getOutput(0), 501, 8, "slanet_loc_write");
    ITensor* nextCharTensor = addTensorAtIndex(network, weightMap, *charRec->getOutput(0), *charLogits,
                                               *counterRec->getOutput(0), 501, 50, "slanet_char_write");
    ITensor* nextIdsTensor = addTensorAtIndex(network, weightMap, *idsRec->getOutput(0), *nextId->getOutput(0),
                                              *counterRec->getOutput(0), 501, 1, "slanet_id_write");
    ITensor* eosNow = addAnyTokenTensor(network, weightMap, *nextIdsTensor, 49, "slanet_eos_any");
    ITensor* nextCounter = addScalarIntSum(network, weightMap, *counterRec->getOutput(0), 1, "slanet_counter_next");
    ITensor* underMax = addScalarLessThan(network, weightMap, *nextCounter, 501, "slanet_under_max");
    ITensor* notEos = addLogicalNotTensor(network, *eosNow, "slanet_not_eos");
    ITensor* nextCond = addLogicalAndTensor(network, *underMax, *notEos, "slanet_next_cond");

    condRec->setInput(1, *nextCond);
    eosRec->setInput(1, *eosNow);
    hiddenRec->setInput(1, *nextHidden->getOutput(0));
    counterRec->setInput(1, *nextCounter);
    locRec->setInput(1, *nextLocTensor);
    prevIdRec->setInput(1, *nextId->getOutput(0));
    idsRec->setInput(1, *nextIdsTensor);
    charRec->setInput(1, *nextCharTensor);
    prevLocRec->setInput(1, *loc->getOutput(0));
    prevCharRec->setInput(1, *charLogits);

    ILoopOutputLayer* loopCounter = loop->addLoopOutput(*counterRec->getOutput(0), LoopOutput::kLAST_VALUE);
    ILoopOutputLayer* loopLoc = loop->addLoopOutput(*locRec->getOutput(0), LoopOutput::kLAST_VALUE);
    ILoopOutputLayer* loopChar = loop->addLoopOutput(*charRec->getOutput(0), LoopOutput::kLAST_VALUE);
    assert(loopCounter && loopLoc && loopChar);

    ITensor* outputLength = addScalarIntSum(network, weightMap, *loopCounter->getOutput(0), 1, "slanet_output_len");
    ITensor* locSize = addDynamicSliceEndShape(network, weightMap, *outputLength, 8, "slanet_loc_size");
    ISliceLayer* locSlice = network->addSlice(*loopLoc->getOutput(0), Dims3{0, 0, 0}, Dims3{1, 1, 8}, Dims3{1, 1, 1});
    assert(locSlice);
    locSlice->setName("slanet_loc_slice");
    locSlice->setInput(2, *locSize);
    ITensor* charSize = addDynamicSliceEndShape(network, weightMap, *outputLength, 50, "slanet_char_size");
    ISliceLayer* charSlice =
            network->addSlice(*loopChar->getOutput(0), Dims3{0, 0, 0}, Dims3{1, 1, 50}, Dims3{1, 1, 1});
    assert(charSlice);
    charSlice->setName("slanet_char_slice");
    charSlice->setInput(2, *charSize);
    ISoftMaxLayer* charProb = addSoftmax(network, *charSlice->getOutput(0), 1 << 2, "slanet_char_softmax");

    locSlice->getOutput(0)->setName("fetch_name_0");
    charProb->getOutput(0)->setName("fetch_name_1");
    network->markOutput(*locSlice->getOutput(0));
    network->markOutput(*charProb->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildSLANeXtWiredModel(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                    const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput("x", dt, Dims4{1, 3, 512, 512});
    if (!data) {
        throw std::runtime_error("failed to add SLANeXt wired input");
    }

    ITensor* patch =
            addConvBiasTensor(network, weightMap, *data, 768, DimsHW{16, 16}, DimsHW{16, 16}, DimsHW{0, 0}, "conv2d_0");
    IShuffleLayer* patchNhwc = addPermute(network, *patch, Permutation{0, 2, 3, 1}, "slanext_patch_nhwc");
    IConstantLayer* pos = network->addConstant(Dims4{1, 32, 32, 768}, getWeights(weightMap, "create_parameter_0.w_0"));
    assert(pos);
    pos->setName("slanext_pos_embed");
    IElementWiseLayer* body = addSum(network, *patchNhwc->getOutput(0), *pos->getOutput(0), "slanext_pos_add");
    ITensor* feature = body->getOutput(0);

    for (int i = 0; i < 12; ++i) {
        bool globalAttention = (i % 3) == 2;
        feature =
                addSLANeXtBlock(network, weightMap, *feature, i, globalAttention, "slanext_block_" + std::to_string(i));
    }

    IShuffleLayer* featureNchw = addPermute(network, *feature, Permutation{0, 3, 1, 2}, "slanext_feature_nchw");
    ITensor* neck0 = addConvNoBiasTensor(network, weightMap, *featureNchw->getOutput(0), 256, DimsHW{1, 1},
                                         DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_1");
    ITensor* norm0 = addSLANeXtLayerNorm2d(network, weightMap, *neck0, 256, "create_parameter_25.w_0",
                                           "create_parameter_26.w_0", "slanext_neck_norm0");
    ITensor* neck1 = addConvNoBiasTensor(network, weightMap, *norm0, 256, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1}, 1,
                                         "conv2d_2");
    ITensor* norm1 = addSLANeXtLayerNorm2d(network, weightMap, *neck1, 256, "create_parameter_27.w_0",
                                           "create_parameter_28.w_0", "slanext_neck_norm1");
    ITensor* neck2 = addConvNoBiasTensor(network, weightMap, *norm1, 512, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1,
                                         "conv2d_3");

    IShuffleLayer* flatten = addShuffle(network, *neck2, Dims3{1, 512, 256}, "slanext_flatten");
    IShuffleLayer* sequence = addPermute(network, *flatten->getOutput(0), Permutation{0, 2, 1}, "slanext_sequence");

    ITensor* attnFeature = addLinearNoBiasTensor(network, weightMap, *sequence->getOutput(0), 512, 512, "linear_48",
                                                 "slanext_linear_48");
    ITensor* initialCond = addBoolConstantTensor(network, weightMap, "slanext_init_cond", Dims{}, {true});
    ITensor* initialEos = addBoolConstantTensor(network, weightMap, "slanext_init_eos", Dims{}, {false});
    ITensor* initialHidden = addFloatConstantTensor(network, weightMap, "slanext_init_hidden", Dims2{1, 512},
                                                    std::vector<float>(512, 0.0f));
    ITensor* initialCounter = addIntConstantTensor(network, weightMap, "slanext_init_counter", Dims{}, {0});
    ITensor* initialLoc = addFloatConstantTensor(network, weightMap, "slanext_init_loc", Dims3{1, 501, 8},
                                                 std::vector<float>(501 * 8, 0.0f));
    ITensor* initialPrevId = addIntConstantTensor(network, weightMap, "slanext_init_prev_id", makeDims1(1), {0});
    ITensor* initialIds =
            addIntConstantTensor(network, weightMap, "slanext_init_ids", Dims2{1, 501}, std::vector<int32_t>(501, 0));
    ITensor* initialChar = addFloatConstantTensor(network, weightMap, "slanext_init_char", Dims3{1, 501, 50},
                                                  std::vector<float>(501 * 50, 0.0f));
    ITensor* initialPrevLoc = addFloatConstantTensor(network, weightMap, "slanext_init_prev_loc", Dims2{1, 8},
                                                     std::vector<float>(8, 0.0f));
    ITensor* initialPrevChar = addFloatConstantTensor(network, weightMap, "slanext_init_prev_char", Dims2{1, 50},
                                                      std::vector<float>(50, 0.0f));

    ILoop* loop = network->addLoop();
    assert(loop);
    loop->setName("slanext_decoder_loop");
    IRecurrenceLayer* condRec = loop->addRecurrence(*initialCond);
    IRecurrenceLayer* eosRec = loop->addRecurrence(*initialEos);
    IRecurrenceLayer* hiddenRec = loop->addRecurrence(*initialHidden);
    IRecurrenceLayer* counterRec = loop->addRecurrence(*initialCounter);
    IRecurrenceLayer* locRec = loop->addRecurrence(*initialLoc);
    IRecurrenceLayer* prevIdRec = loop->addRecurrence(*initialPrevId);
    IRecurrenceLayer* idsRec = loop->addRecurrence(*initialIds);
    IRecurrenceLayer* charRec = loop->addRecurrence(*initialChar);
    IRecurrenceLayer* prevLocRec = loop->addRecurrence(*initialPrevLoc);
    IRecurrenceLayer* prevCharRec = loop->addRecurrence(*initialPrevChar);
    assert(condRec && eosRec && hiddenRec && counterRec && locRec && prevIdRec && idsRec && charRec && prevLocRec &&
           prevCharRec);
    loop->addTripLimit(*condRec->getOutput(0), TripLimit::kWHILE);

    ITensor* prevOneHot = addOneHotTensor(network, weightMap, *prevIdRec->getOutput(0), 50, 1, "slanext_prev_id");
    ITensor* hiddenProj = addLinear2dTensor(network, weightMap, *hiddenRec->getOutput(0), 512, 512, "linear_49",
                                            "slanext_hidden_proj");
    IShuffleLayer* hiddenProjUnsqueeze =
            addShuffle(network, *hiddenProj, Dims3{1, 1, 512}, "slanext_hidden_proj_unsqueeze");
    IElementWiseLayer* attnSum =
            addSum(network, *attnFeature, *hiddenProjUnsqueeze->getOutput(0), "slanext_attention_sum");
    IActivationLayer* attnTanh = network->addActivation(*attnSum->getOutput(0), ActivationType::kTANH);
    assert(attnTanh);
    attnTanh->setName("slanext_attention_tanh");
    ITensor* attnLogit = addLinearNoBiasTensor(network, weightMap, *attnTanh->getOutput(0), 512, 1, "linear_50",
                                               "slanext_attention_logit");
    ISoftMaxLayer* attn = addSoftmax(network, *attnLogit, 1 << 1, "slanext_attention_softmax");
    IShuffleLayer* attnT = addPermute(network, *attn->getOutput(0), Permutation{0, 2, 1}, "slanext_attention_t");
    IMatrixMultiplyLayer* contextMatmul = network->addMatrixMultiply(*attnT->getOutput(0), MatrixOperation::kNONE,
                                                                     *sequence->getOutput(0), MatrixOperation::kNONE);
    assert(contextMatmul);
    contextMatmul->setName("slanext_context_matmul");
    IShuffleLayer* context = addShuffle(network, *contextMatmul->getOutput(0), Dims2{1, 512}, "slanext_context");
    IConcatenationLayer* decoderInput =
            addConcat(network, {context->getOutput(0), prevOneHot}, 1, "slanext_decoder_input");

    ITensor* inputGate = addLinearTransposeTensor(network, weightMap, *decoderInput->getOutput(0), 1536, 562, 1536,
                                                  "gru_cell_0.w_0", "gru_cell_0.b_0", "slanext_gru_input");
    ITensor* hiddenGate = addLinearTransposeTensor(network, weightMap, *hiddenRec->getOutput(0), 1536, 512, 1536,
                                                   "gru_cell_0.w_1", "gru_cell_0.b_1", "slanext_gru_hidden");
    ITensor* iR = addSlice2dTensor(network, *inputGate, 0, 512, "slanext_gru_ir");
    ITensor* iZ = addSlice2dTensor(network, *inputGate, 512, 512, "slanext_gru_iz");
    ITensor* iN = addSlice2dTensor(network, *inputGate, 1024, 512, "slanext_gru_in");
    ITensor* hR = addSlice2dTensor(network, *hiddenGate, 0, 512, "slanext_gru_hr");
    ITensor* hZ = addSlice2dTensor(network, *hiddenGate, 512, 512, "slanext_gru_hz");
    ITensor* hN = addSlice2dTensor(network, *hiddenGate, 1024, 512, "slanext_gru_hn");
    IElementWiseLayer* resetSum = addSum(network, *iR, *hR, "slanext_gru_reset_sum");
    IActivationLayer* resetGate = addSigmoid(network, *resetSum->getOutput(0), "slanext_gru_reset");
    IElementWiseLayer* updateSum = addSum(network, *iZ, *hZ, "slanext_gru_update_sum");
    IActivationLayer* updateGate = addSigmoid(network, *updateSum->getOutput(0), "slanext_gru_update");
    IElementWiseLayer* resetHidden =
            network->addElementWise(*resetGate->getOutput(0), *hN, ElementWiseOperation::kPROD);
    assert(resetHidden);
    resetHidden->setName("slanext_gru_reset_hidden");
    IElementWiseLayer* newSum = addSum(network, *iN, *resetHidden->getOutput(0), "slanext_gru_new_sum");
    IActivationLayer* newGate = network->addActivation(*newSum->getOutput(0), ActivationType::kTANH);
    assert(newGate);
    newGate->setName("slanext_gru_new");
    IElementWiseLayer* hiddenDelta =
            network->addElementWise(*hiddenRec->getOutput(0), *newGate->getOutput(0), ElementWiseOperation::kSUB);
    assert(hiddenDelta);
    hiddenDelta->setName("slanext_gru_hidden_delta");
    IElementWiseLayer* updateDelta =
            network->addElementWise(*hiddenDelta->getOutput(0), *updateGate->getOutput(0), ElementWiseOperation::kPROD);
    assert(updateDelta);
    updateDelta->setName("slanext_gru_update_delta");
    IElementWiseLayer* nextHidden =
            addSum(network, *updateDelta->getOutput(0), *newGate->getOutput(0), "slanext_gru_next_hidden");

    ITensor* charHidden = addLinear2dTensor(network, weightMap, *nextHidden->getOutput(0), 512, 512, "linear_51",
                                            "slanext_char_hidden");
    ITensor* charLogits =
            addLinear2dTensor(network, weightMap, *charHidden, 512, 50, "linear_52", "slanext_char_logits");
    ITensor* locHidden = addLinear2dTensor(network, weightMap, *nextHidden->getOutput(0), 512, 512, "linear_53",
                                           "slanext_loc_hidden");
    ITensor* locLogits = addLinear2dTensor(network, weightMap, *locHidden, 512, 8, "linear_54", "slanext_loc_logits");
    IActivationLayer* loc = addSigmoid(network, *locLogits, "slanext_loc_sigmoid");

    ITopKLayer* topk = network->addTopK(*charLogits, TopKOperation::kMAX, 1, 1 << 1);
    assert(topk);
    topk->setName("slanext_argmax");
    IShuffleLayer* nextId = addShuffle(network, *topk->getOutput(1), makeDims1(1), "slanext_next_id");

    ITensor* nextLocTensor = addTensorAtIndex(network, weightMap, *locRec->getOutput(0), *loc->getOutput(0),
                                              *counterRec->getOutput(0), 501, 8, "slanext_loc_write");
    ITensor* nextCharTensor = addTensorAtIndex(network, weightMap, *charRec->getOutput(0), *charLogits,
                                               *counterRec->getOutput(0), 501, 50, "slanext_char_write");
    ITensor* nextIdsTensor = addTensorAtIndex(network, weightMap, *idsRec->getOutput(0), *nextId->getOutput(0),
                                              *counterRec->getOutput(0), 501, 1, "slanext_id_write");
    ITensor* eosNow = addAnyTokenTensor(network, weightMap, *nextIdsTensor, 49, "slanext_eos_any");
    ITensor* nextCounter = addScalarIntSum(network, weightMap, *counterRec->getOutput(0), 1, "slanext_counter_next");
    ITensor* underMax = addScalarLessThan(network, weightMap, *nextCounter, 501, "slanext_under_max");
    ITensor* notEos = addLogicalNotTensor(network, *eosNow, "slanext_not_eos");
    ITensor* nextCond = addLogicalAndTensor(network, *underMax, *notEos, "slanext_next_cond");

    condRec->setInput(1, *nextCond);
    eosRec->setInput(1, *eosNow);
    hiddenRec->setInput(1, *nextHidden->getOutput(0));
    counterRec->setInput(1, *nextCounter);
    locRec->setInput(1, *nextLocTensor);
    prevIdRec->setInput(1, *nextId->getOutput(0));
    idsRec->setInput(1, *nextIdsTensor);
    charRec->setInput(1, *nextCharTensor);
    prevLocRec->setInput(1, *loc->getOutput(0));
    prevCharRec->setInput(1, *charLogits);

    ILoopOutputLayer* loopCounter = loop->addLoopOutput(*counterRec->getOutput(0), LoopOutput::kLAST_VALUE);
    ILoopOutputLayer* loopLoc = loop->addLoopOutput(*locRec->getOutput(0), LoopOutput::kLAST_VALUE);
    ILoopOutputLayer* loopChar = loop->addLoopOutput(*charRec->getOutput(0), LoopOutput::kLAST_VALUE);
    assert(loopCounter && loopLoc && loopChar);

    ITensor* outputLength = addScalarIntSum(network, weightMap, *loopCounter->getOutput(0), 1, "slanext_output_len");
    ITensor* locSize = addDynamicSliceEndShape(network, weightMap, *outputLength, 8, "slanext_loc_size");
    ISliceLayer* locSlice = network->addSlice(*loopLoc->getOutput(0), Dims3{0, 0, 0}, Dims3{1, 1, 8}, Dims3{1, 1, 1});
    assert(locSlice);
    locSlice->setName("slanext_loc_slice");
    locSlice->setInput(2, *locSize);
    ITensor* charSize = addDynamicSliceEndShape(network, weightMap, *outputLength, 50, "slanext_char_size");
    ISliceLayer* charSlice =
            network->addSlice(*loopChar->getOutput(0), Dims3{0, 0, 0}, Dims3{1, 1, 50}, Dims3{1, 1, 1});
    assert(charSlice);
    charSlice->setName("slanext_char_slice");
    charSlice->setInput(2, *charSize);
    ISoftMaxLayer* charProb = addSoftmax(network, *charSlice->getOutput(0), 1 << 2, "slanext_char_softmax");

    locSlice->getOutput(0)->setName("fetch_name_0");
    charProb->getOutput(0)->setName("fetch_name_1");
    network->markOutput(*locSlice->getOutput(0));
    network->markOutput(*charProb->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildRtDetrDocumentModel(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                      const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    int p5Spatial = static_cast<int>(getWeightsByPrefix(weightMap, "eager_tmp_0").count / 256);
    int p5Size = static_cast<int>(std::sqrt(static_cast<float>(p5Spatial)) + 0.5f);
    int inputSize = p5Size * 32;
    int p3Size = inputSize / 8;
    int p4Size = inputSize / 16;
    int memoryLength = p3Size * p3Size + p4Size * p4Size + p5Size * p5Size;

    ITensor* data = network->addInput("image", dt, Dims4{1, 3, inputSize, inputSize});
    if (!data) {
        throw std::runtime_error("failed to add RT-DETR document input");
    }

    int classCount = static_cast<int>(getWeightsByPrefix(weightMap, "linear_22.b_0").count);
    RtDetrBackboneFeatures backbone = addRtDetrHgNetBackbone(network, weightMap, *data);
    RtDetrNeckFeatures neck = addRtDetrHybridEncoder(network, weightMap, backbone, inputSize);

    IElementWiseLayer* encoderProj =
            addLinearByPrefix(network, weightMap, *neck.memory, 256, 256, "linear_12", "rtdetr_encoder_proj");
    IElementWiseLayer* encoderNorm = addLayerNormByPrefix(network, weightMap, *encoderProj->getOutput(0), 256,
                                                          "layer_norm_5", "rtdetr_encoder_norm");
    IElementWiseLayer* encoderScores = addLinearByPrefix(network, weightMap, *encoderNorm->getOutput(0), 256,
                                                         classCount, "linear_13", "rtdetr_encoder_scores");
    IElementWiseLayer* box0 = addLinearByPrefix(network, weightMap, *encoderNorm->getOutput(0), 256, 256, "linear_14",
                                                "rtdetr_encoder_box0");
    IActivationLayer* boxRelu0 = addRelu(network, *box0->getOutput(0), "rtdetr_encoder_box_relu0");
    IElementWiseLayer* box1 = addLinearByPrefix(network, weightMap, *boxRelu0->getOutput(0), 256, 256, "linear_15",
                                                "rtdetr_encoder_box1");
    IActivationLayer* boxRelu1 = addRelu(network, *box1->getOutput(0), "rtdetr_encoder_box_relu1");
    IElementWiseLayer* box2 =
            addLinearByPrefix(network, weightMap, *boxRelu1->getOutput(0), 256, 4, "linear_16", "rtdetr_encoder_box2");
    IConstantLayer* anchors =
            network->addConstant(Dims3{1, memoryLength, 4}, getWeightsByPrefix(weightMap, "eager_tmp_1"));
    assert(anchors);
    anchors->setName("rtdetr_anchors");
    IElementWiseLayer* encoderBoxes =
            addSum(network, *box2->getOutput(0), *anchors->getOutput(0), "rtdetr_encoder_boxes");

    IReduceLayer* scoreMax = network->addReduce(*encoderScores->getOutput(0), ReduceOperation::kMAX, 1U << 2, false);
    assert(scoreMax);
    scoreMax->setName("rtdetr_encoder_score_max");
    ITopKLayer* topk = network->addTopK(*scoreMax->getOutput(0), TopKOperation::kMAX, 300, 1 << 1);
    assert(topk);
    topk->setName("rtdetr_encoder_topk");
    IGatherLayer* topBoxes = network->addGather(*encoderBoxes->getOutput(0), *topk->getOutput(1), 1);
    assert(topBoxes);
    topBoxes->setName("rtdetr_gather_boxes");
    topBoxes->setNbElementWiseDims(1);
    IGatherLayer* topTarget = network->addGather(*encoderNorm->getOutput(0), *topk->getOutput(1), 1);
    assert(topTarget);
    topTarget->setName("rtdetr_gather_target");
    topTarget->setNbElementWiseDims(1);

    IActivationLayer* reference0 = addSigmoid(network, *topBoxes->getOutput(0), "rtdetr_reference0");
    ITensor* reference = reference0->getOutput(0);
    ITensor* target = topTarget->getOutput(0);
    for (int i = 0; i < 6; ++i) {
        target = addRtDetrDecoderLayer(network, weightMap, *neck.memory, *target, reference, memoryLength, i,
                                       "rtdetr_decoder_" + std::to_string(i));
    }

    IElementWiseLayer* logits =
            addLinearByPrefix(network, weightMap, *target, 256, classCount, "linear_22", "rtdetr_logits");
    IActivationLayer* scores = addSigmoid(network, *logits->getOutput(0), "rtdetr_scores");
    reference->setName("boxes");
    scores->getOutput(0)->setName("scores");
    network->markOutput(*reference);
    network->markOutput(*scores->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildPPOCRv5Model(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    if (contains(wtsPath, "pp_lcnet_x1_0_doc_ori") || contains(wtsPath, "pp_lcnet_x1_0_table_cls") ||
        contains(wtsPath, "pp_lcnet_x1_0_textline_ori")) {
        return buildPPLCNetX1_0Model(builder, config, dt, wtsPath);
    }
    if (contains(wtsPath, "uvdoc")) {
        return buildUVDocModel(builder, config, dt, wtsPath);
    }
    if (contains(wtsPath, "slanet_plus")) {
        return buildSLANetPlusModel(builder, config, dt, wtsPath);
    }
    if (contains(wtsPath, "slanext_wired")) {
        return buildSLANeXtWiredModel(builder, config, dt, wtsPath);
    }
    if (contains(wtsPath, "pp_docblocklayout") || contains(wtsPath, "pp_doclayout_plus_l") ||
        contains(wtsPath, "rt_detr_l_wired_table_cell_det") || contains(wtsPath, "rt_detr_l_wireless_table_cell_det")) {
        return buildRtDetrDocumentModel(builder, config, dt, wtsPath);
    }
    throw std::runtime_error("direct TensorRT layer implementation is not registered for WTS: " + wtsPath);
}

ITensor* addFormulaEmbedding(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& tokenIds,
                             const std::string& lname) {
    IConstantLayer* table = network->addConstant(Dims2{50000, 512}, getWeights(weightMap, "embedding_3.w_0"));
    assert(table);
    table->setName((lname + "_table").c_str());
    IGatherLayer* gather = network->addGather(*table->getOutput(0), tokenIds, 0);
    assert(gather);
    gather->setName(lname.c_str());
    return gather->getOutput(0);
}

ITensor* addFormulaPositionEmbedding(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                     ITensor& positionState, const std::string& lname) {
    ITensor* offset = addInt64ConstantTensor(network, weightMap, lname + "_offset", makeDims1(1), {1});
    IElementWiseLayer* position = network->addElementWise(positionState, *offset, ElementWiseOperation::kSUM);
    assert(position);
    position->setName((lname + "_ids").c_str());
    IConstantLayer* table =
            network->addConstant(Dims2{2562, 512}, getWeights(weightMap, "m_bart_learned_positional_embedding_3.w_0"));
    assert(table);
    table->setName((lname + "_table").c_str());
    IGatherLayer* gather = network->addGather(*table->getOutput(0), *position->getOutput(0), 0);
    assert(gather);
    gather->setName((lname + "_gather").c_str());
    IShuffleLayer* shaped = addShuffle(network, *gather->getOutput(0), Dims3{1, 1, 512}, lname);
    return shaped->getOutput(0);
}

ITensor* addFormulaProjectHeads(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                int tokens, const std::string& linearName, const std::string& lname) {
    IElementWiseLayer* linear = addLinear(network, weightMap, input, 512, 512, linearName);
    linear->setName(lname.c_str());
    IShuffleLayer* reshape = addShuffle(network, *linear->getOutput(0), Dims4{1, tokens, 16, 32}, lname + "_reshape");
    IShuffleLayer* permute = addPermute(network, *reshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_permute");
    return permute->getOutput(0);
}

ITensor* addFormulaAttention(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& q,
                             ITensor& k, ITensor& v, const std::string& lname) {
    IElementWiseLayer* qScale = addScalarMul(network, weightMap, q, 1.0f / std::sqrt(32.0f), lname + "_q_scale");
    IMatrixMultiplyLayer* qk =
            network->addMatrixMultiply(*qScale->getOutput(0), MatrixOperation::kNONE, k, MatrixOperation::kTRANSPOSE);
    assert(qk);
    qk->setName((lname + "_qk").c_str());
    ISoftMaxLayer* attn = addSoftmax(network, *qk->getOutput(0), 1 << 3, lname + "_softmax");
    IMatrixMultiplyLayer* context =
            network->addMatrixMultiply(*attn->getOutput(0), MatrixOperation::kNONE, v, MatrixOperation::kNONE);
    assert(context);
    context->setName((lname + "_context").c_str());
    IShuffleLayer* permute =
            addPermute(network, *context->getOutput(0), Permutation{0, 2, 1, 3}, lname + "_context_permute");
    IShuffleLayer* output = addShuffle(network, *permute->getOutput(0), Dims3{1, 1, 512}, lname + "_output");
    return output->getOutput(0);
}

struct FormulaDecoderStates {
    std::vector<ITensor*> next;
};

ITensor* addFormulaDecoderLayer(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& hidden,
                                ITensor& memory, ITensor& selfKeyState, ITensor& selfValueState, int layerIndex,
                                FormulaDecoderStates& states) {
    int linearBase = 300 + layerIndex * 10;
    int lnBase = 103 + layerIndex * 3;
    int stateBase = 6 + layerIndex * 4;
    std::string lname = "formula_decoder_" + std::to_string(layerIndex);

    IElementWiseLayer* selfNorm = addLayerNormLastDim(
            network, weightMap, hidden, 512, "layer_norm_" + std::to_string(lnBase), lname + "_self_norm", 1e-5f);
    ITensor* q = addFormulaProjectHeads(network, weightMap, *selfNorm->getOutput(0), 1,
                                        "linear_" + std::to_string(linearBase + 2), lname + "_self_q");
    ITensor* kNew = addFormulaProjectHeads(network, weightMap, *selfNorm->getOutput(0), 1,
                                           "linear_" + std::to_string(linearBase), lname + "_self_k");
    ITensor* vNew = addFormulaProjectHeads(network, weightMap, *selfNorm->getOutput(0), 1,
                                           "linear_" + std::to_string(linearBase + 1), lname + "_self_v");
    IConcatenationLayer* kAll = addConcat(network, {&selfKeyState, kNew}, 2, lname + "_self_k_cache");
    IConcatenationLayer* vAll = addConcat(network, {&selfValueState, vNew}, 2, lname + "_self_v_cache");
    states.next[stateBase] = kAll->getOutput(0);
    states.next[stateBase + 1] = vAll->getOutput(0);
    ITensor* selfContext =
            addFormulaAttention(network, weightMap, *q, *kAll->getOutput(0), *vAll->getOutput(0), lname + "_self");
    IElementWiseLayer* selfProj =
            addLinear(network, weightMap, *selfContext, 512, 512, "linear_" + std::to_string(linearBase + 3));
    selfProj->setName((lname + "_self_proj").c_str());
    IElementWiseLayer* selfSum = addSum(network, hidden, *selfProj->getOutput(0), lname + "_self_sum");

    IElementWiseLayer* crossNorm =
            addLayerNormLastDim(network, weightMap, *selfSum->getOutput(0), 512,
                                "layer_norm_" + std::to_string(lnBase + 1), lname + "_cross_norm", 1e-5f);
    ITensor* crossQ = addFormulaProjectHeads(network, weightMap, *crossNorm->getOutput(0), 1,
                                             "linear_" + std::to_string(linearBase + 6), lname + "_cross_q");
    ITensor* crossK = addFormulaProjectHeads(network, weightMap, memory, 144,
                                             "linear_" + std::to_string(linearBase + 4), lname + "_cross_k");
    ITensor* crossV = addFormulaProjectHeads(network, weightMap, memory, 144,
                                             "linear_" + std::to_string(linearBase + 5), lname + "_cross_v");
    states.next[stateBase + 2] = crossK;
    states.next[stateBase + 3] = crossV;
    ITensor* crossContext = addFormulaAttention(network, weightMap, *crossQ, *crossK, *crossV, lname + "_cross");
    IElementWiseLayer* crossProj =
            addLinear(network, weightMap, *crossContext, 512, 512, "linear_" + std::to_string(linearBase + 7));
    crossProj->setName((lname + "_cross_proj").c_str());
    IElementWiseLayer* crossSum =
            addSum(network, *selfSum->getOutput(0), *crossProj->getOutput(0), lname + "_cross_sum");

    IElementWiseLayer* ffnNorm =
            addLayerNormLastDim(network, weightMap, *crossSum->getOutput(0), 512,
                                "layer_norm_" + std::to_string(lnBase + 2), lname + "_ffn_norm", 1e-5f);
    IElementWiseLayer* ffn0 = addLinear(network, weightMap, *ffnNorm->getOutput(0), 512, 2048,
                                        "linear_" + std::to_string(linearBase + 8));
    ffn0->setName((lname + "_ffn0").c_str());
    ITensor* gelu = addGeluTensor(network, *ffn0->getOutput(0), lname + "_gelu");
    IElementWiseLayer* ffn1 =
            addLinear(network, weightMap, *gelu, 2048, 512, "linear_" + std::to_string(linearBase + 9));
    ffn1->setName((lname + "_ffn1").c_str());
    IElementWiseLayer* out = addSum(network, *crossSum->getOutput(0), *ffn1->getOutput(0), lname + "_ffn_sum");
    return out->getOutput(0);
}

void addFormulaDecoderOptimizationProfile(IBuilder* builder, IBuilderConfig* config) {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("state_3", OptProfileSelector::kMIN, Dims2{1, 1});
    profile->setDimensions("state_3", OptProfileSelector::kOPT, Dims2{1, 1});
    profile->setDimensions("state_3", OptProfileSelector::kMAX, Dims2{1, 1});
    profile->setDimensions("state_5", OptProfileSelector::kMIN, Dims2{1, 1});
    profile->setDimensions("state_5", OptProfileSelector::kOPT, Dims2{1, 16});
    profile->setDimensions("state_5", OptProfileSelector::kMAX, Dims2{1, kFormulaMaxLength + 1});
    for (int i = 6; i <= 37; ++i) {
        std::string name = "state_" + std::to_string(i);
        profile->setDimensions(name.c_str(), OptProfileSelector::kMIN, Dims4{1, 16, 0, 32});
        profile->setDimensions(name.c_str(), OptProfileSelector::kOPT, Dims4{1, 16, 144, 32});
        profile->setDimensions(name.c_str(), OptProfileSelector::kMAX, Dims4{1, 16, kFormulaMaxLength + 1, 32});
    }
    config->addOptimizationProfile(profile);
}

IHostMemory* buildPPFormulaNetEncoderDirect(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                            const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* data = network->addInput("x", dt, Dims4{1, 1, kFormulaInputH, kFormulaInputW});
    if (!data) {
        throw std::runtime_error("failed to add PP-FormulaNet encoder input");
    }
    const char* debugStage = std::getenv("PPOCRV5_DEBUG_FORMULA_STAGE");
    auto markDebugStage = [&](const char* stage, ITensor& tensor) -> bool {
        if (!debugStage || std::strcmp(debugStage, stage) != 0) {
            return false;
        }
        tensor.setName("output");
        network->markOutput(tensor);
        return true;
    };

    IConcatenationLayer* rgb = addConcat(network, {data, data, data}, 1, "formula_gray_to_rgb");
    if (markDebugStage("rgb", *rgb->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* patch = addConvBiasTensor(network, weightMap, *rgb->getOutput(0), 768, DimsHW{16, 16}, DimsHW{16, 16},
                                       DimsHW{0, 0}, "conv2d_0");
    if (markDebugStage("patch", *patch)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IShuffleLayer* patchNhwc = addPermute(network, *patch, Permutation{0, 2, 3, 1}, "formula_patch_nhwc");
    if (markDebugStage("patch_nhwc", *patchNhwc->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IConstantLayer* pos = network->addConstant(Dims4{1, 48, 48, 768}, getWeights(weightMap, "create_parameter_0.w_0"));
    assert(pos);
    pos->setName("formula_pos_embed");
    IElementWiseLayer* body = addSum(network, *patchNhwc->getOutput(0), *pos->getOutput(0), "formula_pos_add");
    if (markDebugStage("pos_add", *body->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* feature = body->getOutput(0);

    for (int i = 0; i < 12; ++i) {
        bool globalAttention = (i % 3) == 2;
        feature = addSLANeXtBlock(network, weightMap, *feature, i, 48, 14, 56, globalAttention,
                                  "formula_block_" + std::to_string(i));
        if (hasFormulaDebugPrefix("formula_block_" + std::to_string(i) + "_")) {
            return buildSerializedNetwork(builder, config, network, weightMap);
        }
        std::string stage = "block" + std::to_string(i);
        if (markDebugStage(stage.c_str(), *feature)) {
            return buildSerializedNetwork(builder, config, network, weightMap);
        }
    }

    IShuffleLayer* featureNchw = addPermute(network, *feature, Permutation{0, 3, 1, 2}, "formula_feature_nchw");
    if (markDebugStage("feature_nchw", *featureNchw->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* neck0 = addConvNoBiasTensor(network, weightMap, *featureNchw->getOutput(0), 256, DimsHW{1, 1},
                                         DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_1");
    if (markDebugStage("neck0", *neck0)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* norm0 = addSLANeXtLayerNorm2d(network, weightMap, *neck0, 256, "create_parameter_25.w_0",
                                           "create_parameter_26.w_0", "formula_neck_norm0");
    if (markDebugStage("neck_norm0", *norm0)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* neck1 = addConvNoBiasTensor(network, weightMap, *norm0, 256, DimsHW{3, 3}, DimsHW{1, 1}, DimsHW{1, 1}, 1,
                                         "conv2d_2");
    if (markDebugStage("neck1", *neck1)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* norm1 = addSLANeXtLayerNorm2d(network, weightMap, *neck1, 256, "create_parameter_27.w_0",
                                           "create_parameter_28.w_0", "formula_neck_norm1");
    if (markDebugStage("neck_norm1", *norm1)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* neck2 = addConvNoBiasTensor(network, weightMap, *norm1, 512, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1,
                                         "conv2d_3");
    if (markDebugStage("neck2", *neck2)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    ITensor* neck3 = addConvNoBiasTensor(network, weightMap, *neck2, 1024, DimsHW{3, 3}, DimsHW{2, 2}, DimsHW{1, 1}, 1,
                                         "conv2d_4");
    if (markDebugStage("neck3", *neck3)) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }

    IShuffleLayer* flatten = addShuffle(network, *neck3, Dims3{1, 1024, 144}, "formula_flatten");
    if (markDebugStage("flatten", *flatten->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IShuffleLayer* sequence = addPermute(network, *flatten->getOutput(0), Permutation{0, 2, 1}, "formula_sequence");
    if (markDebugStage("sequence", *sequence->getOutput(0))) {
        return buildSerializedNetwork(builder, config, network, weightMap);
    }
    IElementWiseLayer* memory = addLinear(network, weightMap, *sequence->getOutput(0), 1024, 1024, "linear_48");
    memory->setName("formula_memory_proj");
    memory->getOutput(0)->setName("formula_memory");
    network->markOutput(*memory->getOutput(0));

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildPPFormulaNetDecoderDirect(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                            const std::string& wtsPath) {
    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    setCommonBuilderConfig(builder, config, dt);

    ITensor* memory1024 = network->addInput("formula_memory", dt, Dims3{1, 144, 1024});
    if (!memory1024) {
        throw std::runtime_error("failed to add PP-FormulaNet decoder memory input");
    }

    std::vector<ITensor*> states(kFormulaStateCount + 1, nullptr);
    states[1] = network->addInput("state_1", DataType::kBOOL, Dims{});
    states[2] = network->addInput("state_2", formulaIndexDataType(), makeDims1(1));
    states[3] = network->addInput("state_3", formulaIndexDataType(), Dims2{1, -1});
    states[4] = network->addInput("state_4", DataType::kFLOAT, Dims{});
    states[5] = network->addInput("state_5", formulaIndexDataType(), Dims2{1, -1});
    for (int i = 6; i <= 37; ++i) {
        std::string name = "state_" + std::to_string(i);
        states[i] = network->addInput(name.c_str(), dt, Dims4{1, 16, -1, 32});
    }
    states[38] = network->addInput("state_38", formulaIndexDataType(), makeDims1(1));
    for (int i = 1; i <= kFormulaStateCount; ++i) {
        if (!states[i]) {
            throw std::runtime_error("failed to add PP-FormulaNet decoder state input");
        }
    }
    addFormulaDecoderOptimizationProfile(builder, config);

    IElementWiseLayer* memory = addLinear(network, weightMap, *memory1024, 1024, 512, "linear_380");
    memory->setName("formula_memory_proj");

    ITensor* tokenEmbedding = addFormulaEmbedding(network, weightMap, *states[3], "formula_token_embedding");
    IElementWiseLayer* tokenScaled =
            addScalarMul(network, weightMap, *tokenEmbedding, std::sqrt(512.0f), "formula_token_scale");
    ITensor* positionEmbedding =
            addFormulaPositionEmbedding(network, weightMap, *states[38], "formula_position_embedding");
    IElementWiseLayer* embeddingSum =
            addSum(network, *tokenScaled->getOutput(0), *positionEmbedding, "formula_embedding_sum");
    IElementWiseLayer* hiddenNorm =
            addLayerNormLastDim(network, weightMap, *embeddingSum->getOutput(0), 512, "create_parameter_43.w_0",
                                "create_parameter_44.w_0", "formula_embedding_norm", 1e-5f);
    ITensor* hidden = hiddenNorm->getOutput(0);

    FormulaDecoderStates nextStates;
    nextStates.next.resize(kFormulaStateCount + 1, nullptr);
    for (int i = 0; i < 8; ++i) {
        hidden = addFormulaDecoderLayer(network, weightMap, *hidden, *memory->getOutput(0), *states[6 + i * 4],
                                        *states[7 + i * 4], i, nextStates);
    }
    IElementWiseLayer* finalNorm =
            addLayerNormLastDim(network, weightMap, *hidden, 512, "layer_norm_127", "formula_final_norm", 1e-5f);
    ITensor* logits = addLinearNoBiasTensor(network, weightMap, *finalNorm->getOutput(0), 512, 50000, "linear_299",
                                            "formula_logits");
    ITopKLayer* topk = network->addTopK(*logits, TopKOperation::kMAX, 1, 1 << 2);
    assert(topk);
    topk->setName("formula_argmax");
    IShuffleLayer* nextTokenI32 = addShuffle(network, *topk->getOutput(1), Dims2{1, 1}, "formula_next_token_i32");
    ICastLayer* nextTokenCast = network->addCast(*nextTokenI32->getOutput(0), formulaIndexDataType());
    assert(nextTokenCast);
    nextTokenCast->setName("formula_next_token");
    ITensor* nextToken = nextTokenCast->getOutput(0);

    IConcatenationLayer* generated = addConcat(network, {states[5], nextToken}, 1, "formula_generated_ids");
    ITensor* nextCounter = addScalarInt64Sum(network, weightMap, *states[2], 1, "formula_counter_next");
    ITensor* nextPosition = addScalarInt64Sum(network, weightMap, *states[38], 1, "formula_position_next");

    ITensor* eos = addInt64ConstantTensor(network, weightMap, "formula_eos", Dims2{1, 1}, {kFormulaEosId});
    IElementWiseLayer* isEos = network->addElementWise(*nextToken, *eos, ElementWiseOperation::kEQUAL);
    assert(isEos);
    isEos->setName("formula_is_eos");
    IUnaryLayer* keepToken = network->addUnary(*isEos->getOutput(0), UnaryOperation::kNOT);
    assert(keepToken);
    keepToken->setName("formula_keep_token");
    IShuffleLayer* condition = addShuffle(network, *keepToken->getOutput(0), Dims{}, "condition");

    ITensor* trueTensor = addBoolConstantTensor(network, weightMap, "formula_state1_true", Dims{}, {true});
    IElementWiseLayer* state1 =
            network->addElementWise(*condition->getOutput(0), *trueTensor, ElementWiseOperation::kAND);
    assert(state1);
    state1->setName("formula_state1");
    ITensor* zeroFloat = addFloatConstantTensor(network, weightMap, "formula_state4_zero", Dims{}, {0.0f});
    IElementWiseLayer* state4 = network->addElementWise(*states[4], *zeroFloat, ElementWiseOperation::kSUM);
    assert(state4);
    state4->setName("formula_state4");

    nextStates.next[1] = state1->getOutput(0);
    nextStates.next[2] = nextCounter;
    nextStates.next[3] = nextToken;
    nextStates.next[4] = state4->getOutput(0);
    nextStates.next[5] = generated->getOutput(0);
    nextStates.next[38] = nextPosition;

    condition->getOutput(0)->setName("condition");
    network->markOutput(*condition->getOutput(0));
    for (int i = 1; i <= kFormulaStateCount; ++i) {
        if (!nextStates.next[i]) {
            throw std::runtime_error("missing PP-FormulaNet decoder next_state_" + std::to_string(i));
        }
        std::string name = "next_state_" + std::to_string(i);
        nextStates.next[i]->setName(name.c_str());
        network->markOutput(*nextStates.next[i]);
    }

    return buildSerializedNetwork(builder, config, network, weightMap);
}

IHostMemory* buildEnginePPOCRv5Det(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::cout << "Building PP-OCRv5 det engine from WTS" << std::endl;
    return isMobileWts(wtsPath) ? buildPPOCRv5MobileDet(builder, config, dt, wtsPath)
                                : buildPPOCRv5ServerDet(builder, config, dt, wtsPath);
}

IHostMemory* buildEnginePPOCRv5Det(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath,
                                   const std::string& variant) {
    std::cout << "Building PP-OCRv5 det engine from WTS, variant=" << variant << std::endl;
    if (variant == "m") {
        return buildPPOCRv5MobileDet(builder, config, dt, wtsPath);
    }
    if (variant == "s") {
        return buildPPOCRv5ServerDet(builder, config, dt, wtsPath);
    }
    throw std::runtime_error("unknown PP-OCRv5 det variant, use m or s: " + variant);
}

IHostMemory* buildEnginePPOCRv5Rec(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath) {
    std::cout << "Building PP-OCRv5 rec engine from WTS" << std::endl;
    return isMobileWts(wtsPath) ? buildPPOCRv5MobileRec(builder, config, dt, wtsPath)
                                : buildPPOCRv5ServerRec(builder, config, dt, wtsPath);
}

IHostMemory* buildEnginePPOCRv5Rec(IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wtsPath,
                                   const std::string& variant) {
    std::cout << "Building PP-OCRv5 rec engine from WTS, variant=" << variant << std::endl;
    if (variant == "m") {
        return buildPPOCRv5MobileRec(builder, config, dt, wtsPath);
    }
    if (variant == "s") {
        return buildPPOCRv5ServerRec(builder, config, dt, wtsPath);
    }
    throw std::runtime_error("unknown PP-OCRv5 rec variant, use m or s: " + variant);
}

IHostMemory* buildEnginePPOCRv5Model(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                     const std::string& wtsPath) {
    std::cout << "Building document model engine from WTS" << std::endl;
    return buildPPOCRv5Model(builder, config, dt, wtsPath);
}

IHostMemory* buildEnginePPFormulaNetEncoder(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                            const std::string& wtsPath) {
    std::cout << "Building PP-FormulaNet encoder engine from WTS" << std::endl;
    return buildPPFormulaNetEncoderDirect(builder, config, dt, wtsPath);
}

IHostMemory* buildEnginePPFormulaNetDecoder(IBuilder* builder, IBuilderConfig* config, DataType dt,
                                            const std::string& wtsPath) {
    std::cout << "Building PP-FormulaNet decoder-step engine from WTS" << std::endl;
    return buildPPFormulaNetDecoderDirect(builder, config, dt, wtsPath);
}
