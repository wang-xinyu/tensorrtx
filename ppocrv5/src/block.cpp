#include "block.h"
#include "calibrator.h"
#include "config.h"

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

using namespace nvinfer1;

int calculateP(int ksize) {
    return ksize / 3;
}

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    std::ifstream input(file);
    if (!input.is_open()) {
        throw std::runtime_error("Unable to load weight file: " + file);
    }

    int32_t count = 0;
    input >> count;
    if (count <= 0) {
        throw std::runtime_error("Invalid weight count in: " + file);
    }

    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size = 0;
        std::string name;
        input >> name >> std::dec >> size;
        auto* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0; x < size; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

void freeWeights(std::map<std::string, nvinfer1::Weights>& weightMap) {
    for (auto& mem : weightMap) {
        free(const_cast<void*>(mem.second.values));
    }
    weightMap.clear();
}

nvinfer1::Weights getWeights(std::map<std::string, nvinfer1::Weights>& weightMap, const std::string& name) {
    auto it = weightMap.find(name);
    if (it == weightMap.end()) {
        throw std::runtime_error("missing weight: " + name);
    }
    return it->second;
}

nvinfer1::Weights getWeightsByPrefix(std::map<std::string, nvinfer1::Weights>& weightMap, const std::string& name) {
    auto it = weightMap.find(name);
    if (it != weightMap.end()) {
        return it->second;
    }
    std::string prefix = name + "_";
    for (auto& item : weightMap) {
        if (item.first.compare(0, prefix.size(), prefix) == 0) {
            return item.second;
        }
    }
    throw std::runtime_error("missing weight prefix: " + name);
}

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      const std::string& lname, float eps) {
    nvinfer1::Weights gamma = getWeights(weightMap, lname + ".w_0");
    nvinfer1::Weights beta = getWeights(weightMap, lname + ".b_0");
    nvinfer1::Weights mean = getWeights(weightMap, lname + ".w_1");
    nvinfer1::Weights var = getWeights(weightMap, lname + ".w_2");
    int len = static_cast<int>(gamma.count);

    auto* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    auto* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    auto* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    const auto* gammaVal = reinterpret_cast<const float*>(gamma.values);
    const auto* betaVal = reinterpret_cast<const float*>(beta.values);
    const auto* meanVal = reinterpret_cast<const float*>(mean.values);
    const auto* varVal = reinterpret_cast<const float*>(var.values);

    for (int i = 0; i < len; ++i) {
        scval[i] = gammaVal[i] / sqrtf(varVal[i] + eps);
        shval[i] = betaVal[i] - meanVal[i] * scval[i];
        pval[i] = 1.0f;
    }

    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    return addChannelScale(network, input, shift, scale, power, lname);
}

nvinfer1::IScaleLayer* addBatchNorm2dByPrefix(nvinfer1::INetworkDefinition* network,
                                              std::map<std::string, nvinfer1::Weights>& weightMap,
                                              nvinfer1::ITensor& input, const std::string& lname, float eps) {
    nvinfer1::Weights gamma = getWeightsByPrefix(weightMap, lname + ".w_0");
    nvinfer1::Weights beta = getWeightsByPrefix(weightMap, lname + ".b_0");
    nvinfer1::Weights mean = getWeightsByPrefix(weightMap, lname + ".w_1");
    nvinfer1::Weights var = getWeightsByPrefix(weightMap, lname + ".w_2");
    int len = static_cast<int>(gamma.count);

    auto* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    auto* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    auto* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    const auto* gammaVal = reinterpret_cast<const float*>(gamma.values);
    const auto* betaVal = reinterpret_cast<const float*>(beta.values);
    const auto* meanVal = reinterpret_cast<const float*>(mean.values);
    const auto* varVal = reinterpret_cast<const float*>(var.values);

    for (int i = 0; i < len; ++i) {
        scval[i] = gammaVal[i] / sqrtf(varVal[i] + eps);
        shval[i] = betaVal[i] - meanVal[i] * scval[i];
        pval[i] = 1.0f;
    }

    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scval, len};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shval, len};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, pval, len};
    weightMap[lname + ".prefix.scale"] = scale;
    weightMap[lname + ".prefix.shift"] = shift;
    weightMap[lname + ".prefix.power"] = power;
    return addChannelScale(network, input, shift, scale, power, lname);
}

nvinfer1::IConvolutionLayer* addConv2d(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       const std::string& wname, int outChannels, nvinfer1::DimsHW ksize,
                                       nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups) {
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto* conv = network->addConvolutionNd(input, outChannels, ksize, getWeights(weightMap, wname), emptywts);
    assert(conv);
    conv->setStrideNd(stride);
    conv->setPaddingNd(padding);
    conv->setNbGroups(groups);
    return conv;
}

nvinfer1::IConvolutionLayer* addConv2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       nvinfer1::Weights kernel, nvinfer1::Weights bias, int outChannels,
                                       nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
                                       nvinfer1::DimsHW dilation, int groups, const std::string& lname) {
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(input, outChannels, ksize, kernel, bias);
    assert(conv);
    conv->setName(lname.c_str());
    conv->setStrideNd(stride);
    conv->setPaddingNd(padding);
    conv->setDilationNd(dilation);
    conv->setNbGroups(groups);
    return conv;
}

nvinfer1::IConvolutionLayer* convBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, int k, int s, int p, const std::string& convName) {
    return convBias(network, weightMap, input, outChannels, k, s, p, 1, convName);
}

nvinfer1::IConvolutionLayer* convBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, int k, int s, int p, int groups, const std::string& convName) {
    return addConv2d(network, input, getWeights(weightMap, convName + ".w_0"), getWeights(weightMap, convName + ".b_0"),
                     outChannels, nvinfer1::DimsHW{k, k}, nvinfer1::DimsHW{s, s}, nvinfer1::DimsHW{p, p},
                     nvinfer1::DimsHW{1, 1}, groups, convName);
}

nvinfer1::IConvolutionLayer* convBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                      nvinfer1::DimsHW padding, int groups, const std::string& convName) {
    return addConv2d(network, input, getWeights(weightMap, convName + ".w_0"), getWeights(weightMap, convName + ".b_0"),
                     outChannels, ksize, stride, padding, nvinfer1::DimsHW{1, 1}, groups, convName);
}

nvinfer1::IScaleLayer* convBn(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                              int outChannels, int k, int s, int p, const std::string& convName,
                              const std::string& bnName) {
    return convBn(network, weightMap, input, outChannels, k, s, p, 1, convName, bnName);
}

nvinfer1::IScaleLayer* convBn(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                              int outChannels, int k, int s, int p, int groups, const std::string& convName,
                              const std::string& bnName) {
    return convBn(network, weightMap, input, outChannels, nvinfer1::DimsHW{k, k}, nvinfer1::DimsHW{s, s},
                  nvinfer1::DimsHW{p, p}, groups, convName, bnName);
}

nvinfer1::IScaleLayer* convBn(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                              int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                              nvinfer1::DimsHW padding, int groups, const std::string& convName,
                              const std::string& bnName) {
    nvinfer1::IConvolutionLayer* conv =
            addConv2d(network, weightMap, input, convName + ".w_0", outChannels, ksize, stride, padding, groups);
    conv->setName(convName.c_str());
    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), bnName, 1e-5f);
    bn->setName(bnName.c_str());
    return bn;
}

nvinfer1::IElementWiseLayer* convBnHSwish(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int outChannels, int k, int s, int p, const std::string& convName,
                                          const std::string& bnName) {
    return convBnHSwish(network, weightMap, input, outChannels, k, s, p, 1, convName, bnName);
}

nvinfer1::IElementWiseLayer* convBnHSwish(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int outChannels, int k, int s, int p, int groups, const std::string& convName,
                                          const std::string& bnName) {
    nvinfer1::IScaleLayer* bn = convBn(network, weightMap, input, outChannels, k, s, p, groups, convName, bnName);
    nvinfer1::IElementWiseLayer* hswish = addHardSwish(network, *bn->getOutput(0));
    hswish->setName((convName + "_hardswish").c_str());
    return hswish;
}

nvinfer1::IElementWiseLayer* convBnHSwishByPrefix(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, int outChannels, int k, int s, int p,
                                                  int groups, const std::string& convName, const std::string& bnName) {
    return convBnHSwishByPrefix(network, weightMap, input, outChannels, nvinfer1::DimsHW{k, k}, nvinfer1::DimsHW{s, s},
                                nvinfer1::DimsHW{p, p}, groups, convName, bnName);
}

nvinfer1::IElementWiseLayer* convBnHSwishByPrefix(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, int outChannels, nvinfer1::DimsHW ksize,
                                                  nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups,
                                                  const std::string& convName, const std::string& bnName) {
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv = network->addConvolutionNd(
            input, outChannels, ksize, getWeightsByPrefix(weightMap, convName + ".w_0"), emptywts);
    assert(conv);
    conv->setStrideNd(stride);
    conv->setPaddingNd(padding);
    conv->setNbGroups(groups);
    conv->setName(convName.c_str());
    nvinfer1::IScaleLayer* bn = addBatchNorm2dByPrefix(network, weightMap, *conv->getOutput(0), bnName, 1e-5f);
    bn->setName(bnName.c_str());
    nvinfer1::IElementWiseLayer* hswish = addHardSwish(network, *bn->getOutput(0));
    hswish->setName((convName + "_hardswish").c_str());
    return hswish;
}

nvinfer1::IElementWiseLayer* convBnSwish(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                         nvinfer1::DimsHW padding, int groups, const std::string& convName,
                                         const std::string& bnName) {
    nvinfer1::IScaleLayer* bn =
            convBn(network, weightMap, input, outChannels, ksize, stride, padding, groups, convName, bnName);
    nvinfer1::IElementWiseLayer* swish = addSwish(network, *bn->getOutput(0), convName + "_swish");
    return swish;
}

nvinfer1::IDeconvolutionLayer* addDeconv2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                           nvinfer1::Weights kernel, nvinfer1::Weights bias, int outChannels,
                                           nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
                                           int groups, const std::string& lname) {
    nvinfer1::IDeconvolutionLayer* deconv = network->addDeconvolutionNd(input, outChannels, ksize, kernel, bias);
    assert(deconv);
    deconv->setName(lname.c_str());
    deconv->setStrideNd(stride);
    deconv->setPaddingNd(padding);
    deconv->setNbGroups(groups);
    return deconv;
}

nvinfer1::IDeconvolutionLayer* deconvBias(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int outChannels, int k, int s, int p, const std::string& weightName) {
    return addDeconv2d(network, input, getWeights(weightMap, weightName + ".w_0"),
                       getWeights(weightMap, weightName + ".b_0"), outChannels, nvinfer1::DimsHW{k, k},
                       nvinfer1::DimsHW{s, s}, nvinfer1::DimsHW{p, p}, 1, weightName);
}

nvinfer1::IScaleLayer* addChannelScale(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power,
                                       const std::string& lname) {
    nvinfer1::IScaleLayer* layer = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(layer);
    layer->setName(lname.c_str());
    return layer;
}

nvinfer1::IScaleLayer* addChannelBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      const std::string& biasName, const std::string& lname) {
    nvinfer1::Weights bias = getWeights(weightMap, biasName);
    int len = static_cast<int>(bias.count);
    auto* scaleVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    auto* powerVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        scaleVal[i] = 1.0f;
        powerVal[i] = 1.0f;
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scaleVal, len};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, powerVal, len};
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".power"] = power;
    return addChannelScale(network, input, bias, scale, power, lname);
}

nvinfer1::IScaleLayer* addUniformAffine(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                        const std::string& scaleName, const std::string& biasName,
                                        const std::string& lname) {
    auto* powerVal = reinterpret_cast<float*>(malloc(sizeof(float)));
    powerVal[0] = 1.0f;
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, powerVal, 1};
    weightMap[lname + ".power"] = power;
    nvinfer1::IScaleLayer* layer =
            network->addScale(input, nvinfer1::ScaleMode::kUNIFORM, getWeights(weightMap, biasName),
                              getWeights(weightMap, scaleName), power);
    assert(layer);
    layer->setName(lname.c_str());
    return layer;
}

nvinfer1::IScaleLayer* learnableRepLayer(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int outChannels, int k, int s, int p, int groups, const std::string& convName,
                                         int affineIndex, bool withAct) {
    return learnableRepLayer(network, weightMap, input, outChannels, nvinfer1::DimsHW{k, k}, nvinfer1::DimsHW{s, s},
                             nvinfer1::DimsHW{p, p}, groups, convName, affineIndex, withAct);
}

nvinfer1::IScaleLayer* learnableRepLayer(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                         nvinfer1::DimsHW padding, int groups, const std::string& convName,
                                         int affineIndex, bool withAct) {
    nvinfer1::IConvolutionLayer* conv = addConv2d(network, input, getWeights(weightMap, convName + ".w_0"),
                                                  getWeights(weightMap, convName + ".b_0"), outChannels, ksize, stride,
                                                  padding, nvinfer1::DimsHW{1, 1}, groups, convName);
    nvinfer1::IScaleLayer* affine0 = addUniformAffine(network, weightMap, *conv->getOutput(0),
                                                      "learnable_affine_block_" + std::to_string(affineIndex) + ".w_0",
                                                      "learnable_affine_block_" + std::to_string(affineIndex) + ".w_1",
                                                      "learnable_affine_block_" + std::to_string(affineIndex));
    if (!withAct) {
        return affine0;
    }
    nvinfer1::IElementWiseLayer* hswish = addHardSwish(network, *affine0->getOutput(0));
    hswish->setName((convName + "_hardswish").c_str());
    nvinfer1::IScaleLayer* affine1 =
            addUniformAffine(network, weightMap, *hswish->getOutput(0),
                             "learnable_affine_block_" + std::to_string(affineIndex + 1) + ".w_0",
                             "learnable_affine_block_" + std::to_string(affineIndex + 1) + ".w_1",
                             "learnable_affine_block_" + std::to_string(affineIndex + 1));
    return affine1;
}

nvinfer1::IReduceLayer* addGlobalAvgPool2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                           const std::string& lname) {
    nvinfer1::IReduceLayer* pool =
            network->addReduce(input, nvinfer1::ReduceOperation::kAVG, (1U << 2) | (1U << 3), true);
    assert(pool);
    pool->setName(lname.c_str());
    return pool;
}

nvinfer1::IElementWiseLayer* seLayer(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                     int squeezeChannels, int outChannels, const std::string& conv0Name,
                                     const std::string& conv1Name) {
    nvinfer1::IReduceLayer* pool = addGlobalAvgPool2d(network, input, conv0Name + "_pool");
    nvinfer1::IConvolutionLayer* conv0 =
            convBias(network, weightMap, *pool->getOutput(0), squeezeChannels, 1, 1, 0, conv0Name);
    nvinfer1::IActivationLayer* relu = addRelu(network, *conv0->getOutput(0), conv0Name + "_relu");
    nvinfer1::IConvolutionLayer* conv1 =
            convBias(network, weightMap, *relu->getOutput(0), outChannels, 1, 1, 0, conv1Name);
    nvinfer1::IActivationLayer* hardSigmoid = addHardSigmoid(network, *conv1->getOutput(0));
    hardSigmoid->setName((conv1Name + "_hardsigmoid").c_str());
    nvinfer1::IElementWiseLayer* scale =
            network->addElementWise(input, *hardSigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(scale);
    scale->setName((conv0Name + "_mul").c_str());
    return scale;
}

nvinfer1::IElementWiseLayer* seLayerByPrefix(nvinfer1::INetworkDefinition* network,
                                             std::map<std::string, nvinfer1::Weights>& weightMap,
                                             nvinfer1::ITensor& input, int squeezeChannels, int outChannels,
                                             const std::string& conv0Name, const std::string& conv1Name) {
    nvinfer1::IReduceLayer* pool = addGlobalAvgPool2d(network, input, conv0Name + "_pool");
    nvinfer1::IConvolutionLayer* conv0 =
            addConv2d(network, *pool->getOutput(0), getWeightsByPrefix(weightMap, conv0Name + ".w_0"),
                      getWeightsByPrefix(weightMap, conv0Name + ".b_0"), squeezeChannels, nvinfer1::DimsHW{1, 1},
                      nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{0, 0}, nvinfer1::DimsHW{1, 1}, 1, conv0Name);
    nvinfer1::IActivationLayer* relu = addRelu(network, *conv0->getOutput(0), conv0Name + "_relu");
    nvinfer1::IConvolutionLayer* conv1 =
            addConv2d(network, *relu->getOutput(0), getWeightsByPrefix(weightMap, conv1Name + ".w_0"),
                      getWeightsByPrefix(weightMap, conv1Name + ".b_0"), outChannels, nvinfer1::DimsHW{1, 1},
                      nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{0, 0}, nvinfer1::DimsHW{1, 1}, 1, conv1Name);
    nvinfer1::IActivationLayer* hardSigmoid = addHardSigmoid(network, *conv1->getOutput(0));
    hardSigmoid->setName((conv1Name + "_hardsigmoid").c_str());
    nvinfer1::IElementWiseLayer* scale =
            network->addElementWise(input, *hardSigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(scale);
    scale->setName((conv0Name + "_mul").c_str());
    return scale;
}

nvinfer1::IElementWiseLayer* rseLayer(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, int squeezeChannels, int k, int p, const std::string& convName,
                                      const std::string& conv0Name, const std::string& conv1Name) {
    nvinfer1::IConvolutionLayer* conv =
            addConv2d(network, weightMap, input, convName + ".w_0", outChannels, nvinfer1::DimsHW{k, k},
                      nvinfer1::DimsHW{1, 1}, nvinfer1::DimsHW{p, p}, 1);
    conv->setName(convName.c_str());
    nvinfer1::IReduceLayer* pool = addGlobalAvgPool2d(network, *conv->getOutput(0), convName + "_pool");
    nvinfer1::IConvolutionLayer* conv0 =
            convBias(network, weightMap, *pool->getOutput(0), squeezeChannels, 1, 1, 0, conv0Name);
    nvinfer1::IActivationLayer* relu = addRelu(network, *conv0->getOutput(0), conv0Name + "_relu");
    nvinfer1::IConvolutionLayer* conv1 =
            convBias(network, weightMap, *relu->getOutput(0), outChannels, 1, 1, 0, conv1Name);
    nvinfer1::IActivationLayer* hardSigmoid = addHardSigmoid(network, *conv1->getOutput(0));
    hardSigmoid->setAlpha(0.2f);
    hardSigmoid->setName((conv1Name + "_hardsigmoid").c_str());
    nvinfer1::IElementWiseLayer* scale = network->addElementWise(*conv->getOutput(0), *hardSigmoid->getOutput(0),
                                                                 nvinfer1::ElementWiseOperation::kPROD);
    assert(scale);
    scale->setName((convName + "_scale").c_str());
    nvinfer1::IElementWiseLayer* sum =
            network->addElementWise(*conv->getOutput(0), *scale->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(sum);
    sum->setName((convName + "_sum").c_str());
    return sum;
}

IElementWiseLayer* ppLcNetBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                int inChannels, int outChannels, int dwName, int dwBnName, int pwName, int pwBnName,
                                int kernel, DimsHW stride, bool useSe) {
    IElementWiseLayer* dw = convBnHSwishByPrefix(
            network, weightMap, input, inChannels, DimsHW{kernel, kernel}, stride, DimsHW{kernel / 2, kernel / 2},
            inChannels, "conv2d_" + std::to_string(dwName), "batch_norm2d_" + std::to_string(dwBnName));
    ITensor* body = dw->getOutput(0);
    if (useSe) {
        int squeezeChannels = inChannels / 4;
        IElementWiseLayer* se =
                seLayerByPrefix(network, weightMap, *body, squeezeChannels, inChannels,
                                "conv2d_" + std::to_string(dwName + 1), "conv2d_" + std::to_string(dwName + 2));
        body = se->getOutput(0);
    }
    return convBnHSwishByPrefix(network, weightMap, *body, outChannels, 1, 1, 0, 1, "conv2d_" + std::to_string(pwName),
                                "batch_norm2d_" + std::to_string(pwBnName));
}

IElementWiseLayer* slanetLcNetBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                    ITensor& input, int inChannels, int outChannels, int dwName, int dwBnName,
                                    int pwName, int pwBnName, int kernel, int stride, bool useSe) {
    IElementWiseLayer* dw = convBnHSwish(network, weightMap, input, inChannels, kernel, stride, kernel / 2, inChannels,
                                         "conv2d_" + std::to_string(dwName), "batch_norm_" + std::to_string(dwBnName));
    ITensor* body = dw->getOutput(0);
    if (useSe) {
        int squeezeChannels = inChannels / 4;
        IElementWiseLayer* se = seLayer(network, weightMap, *body, squeezeChannels, inChannels,
                                        "conv2d_" + std::to_string(dwName + 1), "conv2d_" + std::to_string(dwName + 2));
        body = se->getOutput(0);
    }
    return convBnHSwish(network, weightMap, *body, outChannels, 1, 1, 0, 1, "conv2d_" + std::to_string(pwName),
                        "batch_norm_" + std::to_string(pwBnName));
}

ITensor* addSvtrAttention(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                          const std::string& qkvName, const std::string& projName, const std::string& lname) {
    IElementWiseLayer* q = addLinearPart(network, weightMap, input, 120, 120, qkvName, 0, lname + ".q");
    IElementWiseLayer* k = addLinearPart(network, weightMap, input, 120, 120, qkvName, 1, lname + ".k");
    IElementWiseLayer* v = addLinearPart(network, weightMap, input, 120, 120, qkvName, 2, lname + ".v");

    IShuffleLayer* qReshape = addShuffle(network, *q->getOutput(0), Dims4{0, -1, 8, 15}, lname + ".q.reshape");
    IShuffleLayer* kReshape = addShuffle(network, *k->getOutput(0), Dims4{0, -1, 8, 15}, lname + ".k.reshape");
    IShuffleLayer* vReshape = addShuffle(network, *v->getOutput(0), Dims4{0, -1, 8, 15}, lname + ".v.reshape");

    IShuffleLayer* qPermute =
            addPermute(network, *qReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + ".q.permute");
    IShuffleLayer* kPermute =
            addPermute(network, *kReshape->getOutput(0), Permutation{0, 2, 3, 1}, lname + ".k.permute");
    IShuffleLayer* vPermute =
            addPermute(network, *vReshape->getOutput(0), Permutation{0, 2, 1, 3}, lname + ".v.permute");
    IElementWiseLayer* qScale =
            addScalarMul(network, weightMap, *qPermute->getOutput(0), 1.0f / std::sqrt(15.0f), lname + ".q.scale");

    IMatrixMultiplyLayer* qk = network->addMatrixMultiply(*qScale->getOutput(0), MatrixOperation::kNONE,
                                                          *kPermute->getOutput(0), MatrixOperation::kNONE);
    qk->setName((lname + ".qk").c_str());
    ISoftMaxLayer* attn = addSoftmax(network, *qk->getOutput(0), 1 << 3, lname + ".softmax");
    IMatrixMultiplyLayer* context = network->addMatrixMultiply(*attn->getOutput(0), MatrixOperation::kNONE,
                                                               *vPermute->getOutput(0), MatrixOperation::kNONE);
    context->setName((lname + ".context").c_str());
    IShuffleLayer* contextPermute =
            addPermute(network, *context->getOutput(0), Permutation{0, 2, 1, 3}, lname + ".context.permute");
    IShuffleLayer* contextReshape =
            addShuffle(network, *contextPermute->getOutput(0), Dims3{0, -1, 120}, lname + ".context.reshape");
    IElementWiseLayer* proj = addLinear(network, weightMap, *contextReshape->getOutput(0), 120, 120, projName);
    return proj->getOutput(0);
}

ITensor* addSvtrBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                      const std::string& ln0Name, const std::string& qkvName, const std::string& projName,
                      const std::string& ln1Name, const std::string& mlp0Name, const std::string& mlp1Name,
                      const std::string& lname) {
    IElementWiseLayer* ln0 = addLayerNorm(network, weightMap, input, 120, ln0Name);
    ITensor* attn = addSvtrAttention(network, weightMap, *ln0->getOutput(0), qkvName, projName, lname + ".attn");
    IElementWiseLayer* attnSum = addSum(network, input, *attn, lname + ".attn.sum");

    IElementWiseLayer* ln1 = addLayerNorm(network, weightMap, *attnSum->getOutput(0), 120, ln1Name);
    IElementWiseLayer* mlp0 = addLinear(network, weightMap, *ln1->getOutput(0), 120, 240, mlp0Name);
    IElementWiseLayer* mlpAct = addSwish(network, *mlp0->getOutput(0), lname + ".mlp.swish");
    IElementWiseLayer* mlp1 = addLinear(network, weightMap, *mlpAct->getOutput(0), 240, 120, mlp1Name);
    IElementWiseLayer* mlpSum = addSum(network, *attnSum->getOutput(0), *mlp1->getOutput(0), lname + ".mlp.sum");
    return mlpSum->getOutput(0);
}

ITensor* addHgConvBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                        int bodyChannels, int bodyCount, int bodyStartName, int bodyStartBnName, int squeezeName,
                        int squeezeBnName, int squeezeChannels, int exciteName, int exciteBnName, int exciteChannels) {
    std::vector<ITensor*> features{&input};
    ITensor* current = &input;
    for (int i = 0; i < bodyCount; ++i) {
        current = addConvBnReluTensor(network, weightMap, *current, bodyChannels, DimsHW{3, 3}, DimsHW{1, 1},
                                      DimsHW{1, 1}, 1, "conv2d_" + std::to_string(bodyStartName + i),
                                      "batch_norm2d_" + std::to_string(bodyStartBnName + i));
        features.push_back(current);
    }
    IConcatenationLayer* concat = addConcat(network, features, 1, "hg_conv_concat_" + std::to_string(squeezeName));
    ITensor* squeeze = addConvBnReluTensor(network, weightMap, *concat->getOutput(0), squeezeChannels, DimsHW{1, 1},
                                           DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_" + std::to_string(squeezeName),
                                           "batch_norm2d_" + std::to_string(squeezeBnName));
    return addConvBnReluTensor(network, weightMap, *squeeze, exciteChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0},
                               1, "conv2d_" + std::to_string(exciteName),
                               "batch_norm2d_" + std::to_string(exciteBnName));
}

ITensor* addHgStandardBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                            int bodyChannels, int bodyCount, int firstDwName, int firstDwBnName, int firstDwChannels,
                            DimsHW firstStride, int bodyStartName, int bodyStartBnName, int squeezeName,
                            int squeezeBnName, int squeezeChannels, int exciteName, int exciteBnName,
                            int exciteChannels) {
    ITensor* first = addConvBnTensor(network, weightMap, input, firstDwChannels, DimsHW{3, 3}, firstStride,
                                     DimsHW{1, 1}, firstDwChannels, "conv2d_" + std::to_string(firstDwName),
                                     "batch_norm2d_" + std::to_string(firstDwBnName));
    std::vector<ITensor*> features{first};
    ITensor* current = first;
    for (int i = 0; i < bodyCount; ++i) {
        current = addConvBnReluTensor(network, weightMap, *current, bodyChannels, DimsHW{3, 3}, DimsHW{1, 1},
                                      DimsHW{1, 1}, 1, "conv2d_" + std::to_string(bodyStartName + i),
                                      "batch_norm2d_" + std::to_string(bodyStartBnName + i));
        features.push_back(current);
    }
    IConcatenationLayer* concat = addConcat(network, features, 1, "hg_standard_concat_" + std::to_string(squeezeName));
    ITensor* squeeze = addConvBnReluTensor(network, weightMap, *concat->getOutput(0), squeezeChannels, DimsHW{1, 1},
                                           DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_" + std::to_string(squeezeName),
                                           "batch_norm2d_" + std::to_string(squeezeBnName));
    return addConvBnReluTensor(network, weightMap, *squeeze, exciteChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0},
                               1, "conv2d_" + std::to_string(exciteName),
                               "batch_norm2d_" + std::to_string(exciteBnName));
}

ITensor* addHgLightBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& identity,
                         int bodyChannels, int bodyCount, int bodyStartName, int bodyStartBnName, int squeezeName,
                         int squeezeBnName, int squeezeChannels, int exciteName, int exciteBnName, int exciteChannels,
                         bool residual) {
    std::vector<ITensor*> features{&identity};
    ITensor* current = &identity;
    for (int i = 0; i < bodyCount; ++i) {
        int convName = bodyStartName + i * 2;
        int dwName = convName + 1;
        int convBnName = bodyStartBnName + i * 2;
        int dwBnName = convBnName + 1;
        current =
                addConvBnTensor(network, weightMap, *current, bodyChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
                                "conv2d_" + std::to_string(convName), "batch_norm2d_" + std::to_string(convBnName));
        current = addConvBnReluTensor(network, weightMap, *current, bodyChannels, DimsHW{5, 5}, DimsHW{1, 1},
                                      DimsHW{2, 2}, bodyChannels, "conv2d_" + std::to_string(dwName),
                                      "batch_norm2d_" + std::to_string(dwBnName));
        features.push_back(current);
    }
    IConcatenationLayer* concat = addConcat(network, features, 1, "hg_light_concat_" + std::to_string(squeezeName));
    ITensor* squeeze = addConvBnReluTensor(network, weightMap, *concat->getOutput(0), squeezeChannels, DimsHW{1, 1},
                                           DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_" + std::to_string(squeezeName),
                                           "batch_norm2d_" + std::to_string(squeezeBnName));
    ITensor* excite = addConvBnReluTensor(network, weightMap, *squeeze, exciteChannels, DimsHW{1, 1}, DimsHW{1, 1},
                                          DimsHW{0, 0}, 1, "conv2d_" + std::to_string(exciteName),
                                          "batch_norm2d_" + std::to_string(exciteBnName));
    if (!residual) {
        return excite;
    }
    IElementWiseLayer* sum = addSum(network, identity, *excite, "hg_light_residual_" + std::to_string(exciteName));
    return sum->getOutput(0);
}

ITensor* addHgConvBlockByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                int bodyChannels, int bodyCount, int bodyStartName, int bodyStartBnName,
                                int squeezeName, int squeezeBnName, int squeezeChannels, int exciteName,
                                int exciteBnName, int exciteChannels) {
    std::vector<ITensor*> features{&input};
    ITensor* current = &input;
    for (int i = 0; i < bodyCount; ++i) {
        current = addConvBnReluTensorByPrefix(network, weightMap, *current, bodyChannels, DimsHW{3, 3}, DimsHW{1, 1},
                                              DimsHW{1, 1}, 1, "conv2d_" + std::to_string(bodyStartName + i),
                                              "batch_norm2d_" + std::to_string(bodyStartBnName + i));
        features.push_back(current);
    }
    IConcatenationLayer* concat =
            addConcat(network, features, 1, "hg_conv_prefix_concat_" + std::to_string(squeezeName));
    ITensor* squeeze = addConvBnReluTensorByPrefix(
            network, weightMap, *concat->getOutput(0), squeezeChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
            "conv2d_" + std::to_string(squeezeName), "batch_norm2d_" + std::to_string(squeezeBnName));
    return addConvBnReluTensorByPrefix(network, weightMap, *squeeze, exciteChannels, DimsHW{1, 1}, DimsHW{1, 1},
                                       DimsHW{0, 0}, 1, "conv2d_" + std::to_string(exciteName),
                                       "batch_norm2d_" + std::to_string(exciteBnName));
}

ITensor* addHgStandardBlockByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                    ITensor& input, int bodyChannels, int bodyCount, int firstDwName, int firstDwBnName,
                                    int firstDwChannels, DimsHW firstStride, int bodyStartName, int bodyStartBnName,
                                    int squeezeName, int squeezeBnName, int squeezeChannels, int exciteName,
                                    int exciteBnName, int exciteChannels) {
    ITensor* first = addConvBnTensorByPrefix(network, weightMap, input, firstDwChannels, DimsHW{3, 3}, firstStride,
                                             DimsHW{1, 1}, firstDwChannels, "conv2d_" + std::to_string(firstDwName),
                                             "batch_norm2d_" + std::to_string(firstDwBnName));
    std::vector<ITensor*> features{first};
    ITensor* current = first;
    for (int i = 0; i < bodyCount; ++i) {
        current = addConvBnReluTensorByPrefix(network, weightMap, *current, bodyChannels, DimsHW{3, 3}, DimsHW{1, 1},
                                              DimsHW{1, 1}, 1, "conv2d_" + std::to_string(bodyStartName + i),
                                              "batch_norm2d_" + std::to_string(bodyStartBnName + i));
        features.push_back(current);
    }
    IConcatenationLayer* concat =
            addConcat(network, features, 1, "hg_standard_prefix_concat_" + std::to_string(squeezeName));
    ITensor* squeeze = addConvBnReluTensorByPrefix(
            network, weightMap, *concat->getOutput(0), squeezeChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
            "conv2d_" + std::to_string(squeezeName), "batch_norm2d_" + std::to_string(squeezeBnName));
    return addConvBnReluTensorByPrefix(network, weightMap, *squeeze, exciteChannels, DimsHW{1, 1}, DimsHW{1, 1},
                                       DimsHW{0, 0}, 1, "conv2d_" + std::to_string(exciteName),
                                       "batch_norm2d_" + std::to_string(exciteBnName));
}

ITensor* addHgLightBlockByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                 ITensor& identity, int bodyChannels, int bodyCount, int bodyStartName,
                                 int bodyStartBnName, int squeezeName, int squeezeBnName, int squeezeChannels,
                                 int exciteName, int exciteBnName, int exciteChannels, bool residual) {
    std::vector<ITensor*> features{&identity};
    ITensor* current = &identity;
    for (int i = 0; i < bodyCount; ++i) {
        int convName = bodyStartName + i * 2;
        int dwName = convName + 1;
        int convBnName = bodyStartBnName + i * 2;
        int dwBnName = convBnName + 1;
        current = addConvBnTensorByPrefix(network, weightMap, *current, bodyChannels, DimsHW{1, 1}, DimsHW{1, 1},
                                          DimsHW{0, 0}, 1, "conv2d_" + std::to_string(convName),
                                          "batch_norm2d_" + std::to_string(convBnName));
        current = addConvBnReluTensorByPrefix(network, weightMap, *current, bodyChannels, DimsHW{5, 5}, DimsHW{1, 1},
                                              DimsHW{2, 2}, bodyChannels, "conv2d_" + std::to_string(dwName),
                                              "batch_norm2d_" + std::to_string(dwBnName));
        features.push_back(current);
    }
    IConcatenationLayer* concat =
            addConcat(network, features, 1, "hg_light_prefix_concat_" + std::to_string(squeezeName));
    ITensor* squeeze = addConvBnReluTensorByPrefix(
            network, weightMap, *concat->getOutput(0), squeezeChannels, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0}, 1,
            "conv2d_" + std::to_string(squeezeName), "batch_norm2d_" + std::to_string(squeezeBnName));
    ITensor* excite = addConvBnReluTensorByPrefix(network, weightMap, *squeeze, exciteChannels, DimsHW{1, 1},
                                                  DimsHW{1, 1}, DimsHW{0, 0}, 1, "conv2d_" + std::to_string(exciteName),
                                                  "batch_norm2d_" + std::to_string(exciteBnName));
    if (!residual) {
        return excite;
    }
    IElementWiseLayer* sum =
            addSum(network, identity, *excite, "hg_light_prefix_residual_" + std::to_string(exciteName));
    return sum->getOutput(0);
}

ITensor* addLargeKernelBranch(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int conv0Name, DimsHW conv0Kernel, DimsHW conv0Padding, int conv1Name, DimsHW conv1Kernel,
                              DimsHW conv1Padding, int conv2Name, DimsHW conv2Kernel, DimsHW conv2Padding) {
    ITensor* conv0 = addConvBiasTensor(network, weightMap, input, 32, conv0Kernel, DimsHW{1, 1}, conv0Padding,
                                       "conv2d_" + std::to_string(conv0Name));
    ITensor* conv1 = addConvBiasTensor(network, weightMap, *conv0, 32, conv1Kernel, DimsHW{1, 1}, conv1Padding,
                                       "conv2d_" + std::to_string(conv1Name));
    return addConvBiasTensor(network, weightMap, *conv1, 32, conv2Kernel, DimsHW{1, 1}, conv2Padding,
                             "conv2d_" + std::to_string(conv2Name));
}

ITensor* addLargeKernelBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int reduceName, int branch7Name0, int branch7Name1, int branch7Name2, int branch5Name0,
                             int branch5Name1, int branch5Name2, int branch3Name0, int branch3Name1, int branch3Name2,
                             int expandName, const std::string& bnName) {
    ITensor* reduce = addConvBiasTensor(network, weightMap, input, 32, DimsHW{1, 1}, DimsHW{1, 1}, DimsHW{0, 0},
                                        "conv2d_" + std::to_string(reduceName));
    ITensor* branch7 =
            addLargeKernelBranch(network, weightMap, *reduce, branch7Name0, DimsHW{7, 7}, DimsHW{3, 3}, branch7Name1,
                                 DimsHW{1, 7}, DimsHW{0, 3}, branch7Name2, DimsHW{7, 1}, DimsHW{3, 0});
    ITensor* branch5 =
            addLargeKernelBranch(network, weightMap, *reduce, branch5Name0, DimsHW{5, 5}, DimsHW{2, 2}, branch5Name1,
                                 DimsHW{1, 5}, DimsHW{0, 2}, branch5Name2, DimsHW{5, 1}, DimsHW{2, 0});
    ITensor* branch3 =
            addLargeKernelBranch(network, weightMap, *reduce, branch3Name0, DimsHW{3, 3}, DimsHW{1, 1}, branch3Name1,
                                 DimsHW{1, 3}, DimsHW{0, 1}, branch3Name2, DimsHW{3, 1}, DimsHW{1, 0});
    IElementWiseLayer* sum0 =
            addSum(network, *branch7, *branch5, "large_kernel_sum_" + std::to_string(expandName) + "_0");
    IElementWiseLayer* sum1 =
            addSum(network, *sum0->getOutput(0), *branch3, "large_kernel_sum_" + std::to_string(expandName) + "_1");
    ITensor* expand = addConvBiasTensor(network, weightMap, *sum1->getOutput(0), 64, DimsHW{1, 1}, DimsHW{1, 1},
                                        DimsHW{0, 0}, "conv2d_" + std::to_string(expandName));
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *expand, bnName, 1e-5f);
    bn->setName(bnName.c_str());
    IActivationLayer* relu = addRelu(network, *bn->getOutput(0), bnName + "_relu");
    IElementWiseLayer* residual =
            addSum(network, input, *relu->getOutput(0), "large_kernel_residual_" + std::to_string(expandName));
    return residual->getOutput(0);
}

ITensor* addUvdocResidualBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                               int channels, int conv0Name, int bn0Name, int conv1Name, int bn1Name, int dilation) {
    ITensor* conv0 =
            addConvBiasBnReluTensor(network, weightMap, input, channels, 5, 1, dilation * 2, dilation,
                                    "conv2d_" + std::to_string(conv0Name), "batch_norm2d_" + std::to_string(bn0Name));
    ITensor* conv1 =
            addConvBiasBnTensor(network, weightMap, *conv0, channels, 5, 1, dilation * 2, dilation,
                                "conv2d_" + std::to_string(conv1Name), "batch_norm2d_" + std::to_string(bn1Name));
    IElementWiseLayer* sum = addSum(network, *conv1, input, "uvdoc_residual_" + std::to_string(conv1Name));
    IActivationLayer* relu = addRelu(network, *sum->getOutput(0), "uvdoc_residual_relu_" + std::to_string(conv1Name));
    return relu->getOutput(0);
}

ITensor* addUvdocDownBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                           int channels, int skipConvName, int skipBnName, int conv0Name, int bn0Name, int conv1Name,
                           int bn1Name) {
    ITensor* skip =
            addConvBiasBnTensor(network, weightMap, input, channels, 5, 2, 2, 1,
                                "conv2d_" + std::to_string(skipConvName), "batch_norm2d_" + std::to_string(skipBnName));
    ITensor* conv0 =
            addConvBiasBnReluTensor(network, weightMap, input, channels, 5, 2, 2, 1,
                                    "conv2d_" + std::to_string(conv0Name), "batch_norm2d_" + std::to_string(bn0Name));
    ITensor* conv1 =
            addConvBiasBnTensor(network, weightMap, *conv0, channels, 5, 1, 2, 1, "conv2d_" + std::to_string(conv1Name),
                                "batch_norm2d_" + std::to_string(bn1Name));
    IElementWiseLayer* sum = addSum(network, *conv1, *skip, "uvdoc_down_residual_" + std::to_string(conv1Name));
    IActivationLayer* relu =
            addRelu(network, *sum->getOutput(0), "uvdoc_down_residual_relu_" + std::to_string(conv1Name));
    return relu->getOutput(0);
}

ITensor* addUvdocReflectPad2d(INetworkDefinition* network, ITensor& input, int batch, int channels, int height,
                              int width, int pad, const std::string& lname) {
    ISliceLayer* slice =
            network->addSlice(input, Dims4{0, 0, -pad, -pad}, Dims4{batch, channels, height + pad * 2, width + pad * 2},
                              Dims4{1, 1, 1, 1});
    assert(slice);
    slice->setMode(SampleMode::kREFLECT);
    slice->setName(lname.c_str());
    return slice->getOutput(0);
}

ITensor* addUvdocPRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                       const std::string& preluName, const std::string& lname) {
    IConstantLayer* slope = network->addConstant(Dims4{1, 1, 1, 1}, getWeights(weightMap, preluName + ".w_0"));
    assert(slope);
    slope->setName((lname + "_slope").c_str());
    IParametricReLULayer* prelu = network->addParametricReLU(input, *slope->getOutput(0));
    assert(prelu);
    prelu->setName(lname.c_str());
    return prelu->getOutput(0);
}

nvinfer1::IConcatenationLayer* addConcat(nvinfer1::INetworkDefinition* network,
                                         const std::vector<nvinfer1::ITensor*>& inputs, int axis,
                                         const std::string& lname) {
    nvinfer1::IConcatenationLayer* cat =
            network->addConcatenation(const_cast<nvinfer1::ITensor**>(inputs.data()), static_cast<int>(inputs.size()));
    assert(cat);
    cat->setAxis(axis);
    cat->setName(lname.c_str());
    return cat;
}

nvinfer1::IResizeLayer* addNearestResize(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                         const std::vector<float>& scales, const std::string& lname) {
    nvinfer1::IResizeLayer* resize = network->addResize(input);
    assert(resize);
    resize->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    resize->setScales(scales.data(), static_cast<int>(scales.size()));
    resize->setName(lname.c_str());
    return resize;
}

nvinfer1::IElementWiseLayer* addSum(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input0,
                                    nvinfer1::ITensor& input1, const std::string& lname) {
    nvinfer1::IElementWiseLayer* layer = network->addElementWise(input0, input1, nvinfer1::ElementWiseOperation::kSUM);
    assert(layer);
    layer->setName(lname.c_str());
    return layer;
}

nvinfer1::IPoolingLayer* addPool2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   nvinfer1::PoolingType type, int k, int s, int p, const std::string& lname) {
    return addPool2d(network, input, type, nvinfer1::DimsHW{k, k}, nvinfer1::DimsHW{s, s}, nvinfer1::DimsHW{p, p},
                     lname);
}

nvinfer1::IPoolingLayer* addPool2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   nvinfer1::PoolingType type, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                   nvinfer1::DimsHW padding, const std::string& lname) {
    nvinfer1::IPoolingLayer* pool = network->addPoolingNd(input, type, ksize);
    assert(pool);
    pool->setStrideNd(stride);
    pool->setPaddingNd(padding);
    pool->setName(lname.c_str());
    return pool;
}

nvinfer1::IActivationLayer* addRelu(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                    const std::string& lname) {
    nvinfer1::IActivationLayer* relu = network->addActivation(input, nvinfer1::ActivationType::kRELU);
    assert(relu);
    relu->setName(lname.c_str());
    return relu;
}

nvinfer1::IActivationLayer* addSigmoid(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       const std::string& lname) {
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(input, nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid);
    sigmoid->setName(lname.c_str());
    return sigmoid;
}

nvinfer1::IElementWiseLayer* addSwish(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                      const std::string& lname) {
    nvinfer1::IActivationLayer* sigmoid = addSigmoid(network, input, lname + "_sigmoid");
    nvinfer1::IElementWiseLayer* swish =
            network->addElementWise(input, *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(swish);
    swish->setName(lname.c_str());
    return swish;
}

nvinfer1::IActivationLayer* addHardSigmoid(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input) {
    auto* layer = network->addActivation(input, nvinfer1::ActivationType::kHARD_SIGMOID);
    assert(layer);
    layer->setAlpha(1.0f / 6.0f);
    layer->setBeta(0.5f);
    return layer;
}

nvinfer1::IElementWiseLayer* addHardSwish(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input) {
    auto* hardSigmoid = addHardSigmoid(network, input);
    auto* layer = network->addElementWise(input, *hardSigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(layer);
    return layer;
}

nvinfer1::IShuffleLayer* addShuffle(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                    nvinfer1::Dims reshapeDims, const std::string& lname) {
    nvinfer1::IShuffleLayer* layer = network->addShuffle(input);
    assert(layer);
    layer->setReshapeDimensions(reshapeDims);
    layer->setName(lname.c_str());
    return layer;
}

nvinfer1::IShuffleLayer* addPermute(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                    nvinfer1::Permutation permutation, const std::string& lname) {
    nvinfer1::IShuffleLayer* layer = network->addShuffle(input);
    assert(layer);
    layer->setFirstTranspose(permutation);
    layer->setName(lname.c_str());
    return layer;
}

nvinfer1::IElementWiseLayer* addLinear(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int inChannels, int outChannels, const std::string& linearName) {
    return addLinear(network, input, getWeights(weightMap, linearName + ".w_0"),
                     getWeights(weightMap, linearName + ".b_0"), inChannels, outChannels, linearName);
}

nvinfer1::IElementWiseLayer* addLinear(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       nvinfer1::Weights kernel, nvinfer1::Weights bias, int inChannels,
                                       int outChannels, const std::string& lname) {
    nvinfer1::IConstantLayer* weight = network->addConstant(nvinfer1::Dims3{1, inChannels, outChannels}, kernel);
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    nvinfer1::IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(
            input, nvinfer1::MatrixOperation::kNONE, *weight->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    assert(matmul);
    matmul->setName((lname + "_matmul").c_str());

    nvinfer1::IConstantLayer* biasLayer = network->addConstant(nvinfer1::Dims3{1, 1, outChannels}, bias);
    assert(biasLayer);
    biasLayer->setName((lname + "_bias").c_str());
    nvinfer1::IElementWiseLayer* add = network->addElementWise(*matmul->getOutput(0), *biasLayer->getOutput(0),
                                                               nvinfer1::ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add;
}

nvinfer1::IElementWiseLayer* addLinear2dByPrefix(nvinfer1::INetworkDefinition* network,
                                                 std::map<std::string, nvinfer1::Weights>& weightMap,
                                                 nvinfer1::ITensor& input, int inChannels, int outChannels,
                                                 const std::string& linearName) {
    nvinfer1::IConstantLayer* weight = network->addConstant(nvinfer1::Dims2{inChannels, outChannels},
                                                            getWeightsByPrefix(weightMap, linearName + ".w_0"));
    assert(weight);
    weight->setName((linearName + "_weight").c_str());
    nvinfer1::IMatrixMultiplyLayer* matmul = network->addMatrixMultiply(
            input, nvinfer1::MatrixOperation::kNONE, *weight->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    assert(matmul);
    matmul->setName((linearName + "_matmul").c_str());

    nvinfer1::IConstantLayer* bias =
            network->addConstant(nvinfer1::Dims2{1, outChannels}, getWeightsByPrefix(weightMap, linearName + ".b_0"));
    assert(bias);
    bias->setName((linearName + "_bias").c_str());
    nvinfer1::IElementWiseLayer* add =
            network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(add);
    add->setName(linearName.c_str());
    return add;
}

Weights sliceLinearKernel(std::map<std::string, Weights>& weightMap, const std::string& linearName, int inChannels,
                          int outChannels, int part) {
    Weights src = getWeights(weightMap, linearName + ".w_0");
    const auto* srcVal = reinterpret_cast<const float*>(src.values);
    auto* dstVal = reinterpret_cast<float*>(malloc(sizeof(float) * inChannels * outChannels));
    int srcStride = outChannels * 3;
    for (int i = 0; i < inChannels; ++i) {
        std::memcpy(dstVal + i * outChannels, srcVal + i * srcStride + part * outChannels, sizeof(float) * outChannels);
    }
    std::string name = linearName + ".part" + std::to_string(part) + ".w";
    Weights dst{DataType::kFLOAT, dstVal, inChannels * outChannels};
    weightMap[name] = dst;
    return dst;
}

Weights sliceLinearBias(std::map<std::string, Weights>& weightMap, const std::string& linearName, int outChannels,
                        int part) {
    Weights src = getWeights(weightMap, linearName + ".b_0");
    const auto* srcVal = reinterpret_cast<const float*>(src.values);
    auto* dstVal = reinterpret_cast<float*>(malloc(sizeof(float) * outChannels));
    std::memcpy(dstVal, srcVal + part * outChannels, sizeof(float) * outChannels);
    std::string name = linearName + ".part" + std::to_string(part) + ".b";
    Weights dst{DataType::kFLOAT, dstVal, outChannels};
    weightMap[name] = dst;
    return dst;
}

Weights sliceLinearKernelByPrefix(std::map<std::string, Weights>& weightMap, const std::string& linearName,
                                  int inChannels, int outChannels, int part) {
    Weights src = getWeightsByPrefix(weightMap, linearName + ".w_0");
    const auto* srcVal = reinterpret_cast<const float*>(src.values);
    auto* dstVal = reinterpret_cast<float*>(malloc(sizeof(float) * inChannels * outChannels));
    int srcStride = outChannels * 3;
    for (int i = 0; i < inChannels; ++i) {
        std::memcpy(dstVal + i * outChannels, srcVal + i * srcStride + part * outChannels, sizeof(float) * outChannels);
    }
    std::string name = linearName + ".prefix.part" + std::to_string(part) + ".w";
    Weights dst{DataType::kFLOAT, dstVal, inChannels * outChannels};
    weightMap[name] = dst;
    return dst;
}

Weights sliceLinearBiasByPrefix(std::map<std::string, Weights>& weightMap, const std::string& linearName,
                                int outChannels, int part) {
    Weights src = getWeightsByPrefix(weightMap, linearName + ".b_0");
    const auto* srcVal = reinterpret_cast<const float*>(src.values);
    auto* dstVal = reinterpret_cast<float*>(malloc(sizeof(float) * outChannels));
    std::memcpy(dstVal, srcVal + part * outChannels, sizeof(float) * outChannels);
    std::string name = linearName + ".prefix.part" + std::to_string(part) + ".b";
    Weights dst{DataType::kFLOAT, dstVal, outChannels};
    weightMap[name] = dst;
    return dst;
}

IElementWiseLayer* addLinearPart(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 int inChannels, int outChannels, const std::string& linearName, int part,
                                 const std::string& lname) {
    return addLinear(network, input, sliceLinearKernel(weightMap, linearName, inChannels, outChannels, part),
                     sliceLinearBias(weightMap, linearName, outChannels, part), inChannels, outChannels, lname);
}

IElementWiseLayer* addLinearPartByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                         ITensor& input, int inChannels, int outChannels, const std::string& linearName,
                                         int part, const std::string& lname) {
    return addLinear(network, input, sliceLinearKernelByPrefix(weightMap, linearName, inChannels, outChannels, part),
                     sliceLinearBiasByPrefix(weightMap, linearName, outChannels, part), inChannels, outChannels, lname);
}

IElementWiseLayer* addLinearByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                     ITensor& input, int inChannels, int outChannels, const std::string& linearName,
                                     const std::string& lname) {
    IConstantLayer* weight =
            network->addConstant(Dims3{1, inChannels, outChannels}, getWeightsByPrefix(weightMap, linearName + ".w_0"));
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    IMatrixMultiplyLayer* matmul =
            network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight->getOutput(0), MatrixOperation::kNONE);
    assert(matmul);
    matmul->setName((lname + "_matmul").c_str());
    IConstantLayer* bias =
            network->addConstant(Dims3{1, 1, outChannels}, getWeightsByPrefix(weightMap, linearName + ".b_0"));
    assert(bias);
    bias->setName((lname + "_bias").c_str());
    IElementWiseLayer* add =
            network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add;
}

int deepcopyOrder(const std::string& key, const std::string& prefix) {
    std::string marker = prefix + "_deepcopy_";
    size_t pos = key.find(marker);
    if (pos == std::string::npos) {
        return 0;
    }
    pos += marker.size();
    int value = 0;
    while (pos < key.size() && std::isdigit(static_cast<unsigned char>(key[pos]))) {
        value = value * 10 + (key[pos] - '0');
        ++pos;
    }
    return value;
}

Weights getWeightsByPrefixOrder(std::map<std::string, Weights>& weightMap, const std::string& prefix, int order) {
    std::vector<std::pair<int, std::string>> matched;
    for (const auto& item : weightMap) {
        if (item.first == prefix || item.first.compare(0, prefix.size() + 1, prefix + "_") == 0) {
            matched.push_back(std::make_pair(deepcopyOrder(item.first, prefix), item.first));
        }
    }
    std::sort(matched.begin(), matched.end(),
              [](const std::pair<int, std::string>& a, const std::pair<int, std::string>& b) {
                  if (a.first != b.first) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });
    if (order < 0 || order >= static_cast<int>(matched.size())) {
        throw std::runtime_error("missing ordered weight prefix: " + prefix + " order=" + std::to_string(order));
    }
    return weightMap[matched[order].second];
}

Weights sliceLinearKernelByPrefixOrder(std::map<std::string, Weights>& weightMap, const std::string& linearName,
                                       int order, int inChannels, int outChannels, int part) {
    Weights src = getWeightsByPrefixOrder(weightMap, linearName + ".w_0", order);
    const auto* srcVal = reinterpret_cast<const float*>(src.values);
    auto* dstVal = reinterpret_cast<float*>(malloc(sizeof(float) * inChannels * outChannels));
    int srcStride = outChannels * 3;
    for (int i = 0; i < inChannels; ++i) {
        std::memcpy(dstVal + i * outChannels, srcVal + i * srcStride + part * outChannels, sizeof(float) * outChannels);
    }
    std::string name = linearName + ".ordered." + std::to_string(order) + ".part" + std::to_string(part) + ".w";
    Weights dst{DataType::kFLOAT, dstVal, inChannels * outChannels};
    weightMap[name] = dst;
    return dst;
}

Weights sliceLinearBiasByPrefixOrder(std::map<std::string, Weights>& weightMap, const std::string& linearName,
                                     int order, int outChannels, int part) {
    Weights src = getWeightsByPrefixOrder(weightMap, linearName + ".b_0", order);
    const auto* srcVal = reinterpret_cast<const float*>(src.values);
    auto* dstVal = reinterpret_cast<float*>(malloc(sizeof(float) * outChannels));
    std::memcpy(dstVal, srcVal + part * outChannels, sizeof(float) * outChannels);
    std::string name = linearName + ".ordered." + std::to_string(order) + ".part" + std::to_string(part) + ".b";
    Weights dst{DataType::kFLOAT, dstVal, outChannels};
    weightMap[name] = dst;
    return dst;
}

IElementWiseLayer* addLinearPartByPrefixOrder(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                              ITensor& input, int inChannels, int outChannels,
                                              const std::string& linearName, int order, int part,
                                              const std::string& lname) {
    return addLinear(network, input,
                     sliceLinearKernelByPrefixOrder(weightMap, linearName, order, inChannels, outChannels, part),
                     sliceLinearBiasByPrefixOrder(weightMap, linearName, order, outChannels, part), inChannels,
                     outChannels, lname);
}

IElementWiseLayer* addLinearByPrefixOrder(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                          ITensor& input, int inChannels, int outChannels,
                                          const std::string& linearName, int order, const std::string& lname) {
    IConstantLayer* weight = network->addConstant(Dims3{1, inChannels, outChannels},
                                                  getWeightsByPrefixOrder(weightMap, linearName + ".w_0", order));
    assert(weight);
    weight->setName((lname + "_weight").c_str());
    IMatrixMultiplyLayer* matmul =
            network->addMatrixMultiply(input, MatrixOperation::kNONE, *weight->getOutput(0), MatrixOperation::kNONE);
    assert(matmul);
    matmul->setName((lname + "_matmul").c_str());
    IConstantLayer* bias = network->addConstant(Dims3{1, 1, outChannels},
                                                getWeightsByPrefixOrder(weightMap, linearName + ".b_0", order));
    assert(bias);
    bias->setName((lname + "_bias").c_str());
    IElementWiseLayer* add =
            network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), ElementWiseOperation::kSUM);
    assert(add);
    add->setName(lname.c_str());
    return add;
}

IElementWiseLayer* addLayerNormByPrefixOrder(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                             ITensor& input, int channels, const std::string& layerNormName, int order,
                                             const std::string& lname) {
    IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, 1U << 2, true);
    assert(mean);
    mean->setName((lname + "_mean").c_str());
    IElementWiseLayer* centered = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(centered);
    centered->setName((lname + "_centered").c_str());
    IElementWiseLayer* square =
            network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), ElementWiseOperation::kPROD);
    assert(square);
    square->setName((lname + "_square").c_str());
    IReduceLayer* var = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, 1U << 2, true);
    assert(var);
    var->setName((lname + "_var").c_str());
    auto* epsVal = reinterpret_cast<float*>(malloc(sizeof(float)));
    epsVal[0] = 1e-5f;
    Weights epsWts{DataType::kFLOAT, epsVal, 1};
    weightMap[lname + ".eps"] = epsWts;
    IConstantLayer* eps = network->addConstant(Dims3{1, 1, 1}, epsWts);
    assert(eps);
    eps->setName((lname + "_eps").c_str());
    IElementWiseLayer* varEps =
            network->addElementWise(*var->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
    assert(varEps);
    varEps->setName((lname + "_var_eps").c_str());
    IUnaryLayer* stddev = network->addUnary(*varEps->getOutput(0), UnaryOperation::kSQRT);
    assert(stddev);
    stddev->setName((lname + "_std").c_str());
    IElementWiseLayer* norm =
            network->addElementWise(*centered->getOutput(0), *stddev->getOutput(0), ElementWiseOperation::kDIV);
    assert(norm);
    norm->setName((lname + "_norm").c_str());
    IConstantLayer* gamma = network->addConstant(Dims3{1, 1, channels},
                                                 getWeightsByPrefixOrder(weightMap, layerNormName + ".w_0", order));
    assert(gamma);
    gamma->setName((lname + "_gamma").c_str());
    IElementWiseLayer* scaled =
            network->addElementWise(*norm->getOutput(0), *gamma->getOutput(0), ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((lname + "_scaled").c_str());
    IConstantLayer* beta = network->addConstant(Dims3{1, 1, channels},
                                                getWeightsByPrefixOrder(weightMap, layerNormName + ".b_0", order));
    assert(beta);
    beta->setName((lname + "_beta").c_str());
    IElementWiseLayer* shifted =
            network->addElementWise(*scaled->getOutput(0), *beta->getOutput(0), ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName(lname.c_str());
    return shifted;
}

IElementWiseLayer* addLayerNormByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                        ITensor& input, int channels, const std::string& layerNormName,
                                        const std::string& lname) {
    IReduceLayer* mean = network->addReduce(input, ReduceOperation::kAVG, 1U << 2, true);
    assert(mean);
    mean->setName((lname + "_mean").c_str());
    IElementWiseLayer* centered = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(centered);
    centered->setName((lname + "_centered").c_str());
    IElementWiseLayer* square =
            network->addElementWise(*centered->getOutput(0), *centered->getOutput(0), ElementWiseOperation::kPROD);
    assert(square);
    square->setName((lname + "_square").c_str());
    IReduceLayer* var = network->addReduce(*square->getOutput(0), ReduceOperation::kAVG, 1U << 2, true);
    assert(var);
    var->setName((lname + "_var").c_str());
    auto* epsVal = reinterpret_cast<float*>(malloc(sizeof(float)));
    epsVal[0] = 1e-5f;
    Weights epsWts{DataType::kFLOAT, epsVal, 1};
    weightMap[lname + ".eps"] = epsWts;
    IConstantLayer* eps = network->addConstant(Dims3{1, 1, 1}, epsWts);
    assert(eps);
    eps->setName((lname + "_eps").c_str());
    IElementWiseLayer* varEps =
            network->addElementWise(*var->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
    assert(varEps);
    varEps->setName((lname + "_var_eps").c_str());
    IUnaryLayer* stddev = network->addUnary(*varEps->getOutput(0), UnaryOperation::kSQRT);
    assert(stddev);
    stddev->setName((lname + "_std").c_str());
    IElementWiseLayer* norm =
            network->addElementWise(*centered->getOutput(0), *stddev->getOutput(0), ElementWiseOperation::kDIV);
    assert(norm);
    norm->setName((lname + "_norm").c_str());
    IConstantLayer* gamma =
            network->addConstant(Dims3{1, 1, channels}, getWeightsByPrefix(weightMap, layerNormName + ".w_0"));
    assert(gamma);
    gamma->setName((lname + "_gamma").c_str());
    IElementWiseLayer* scaled =
            network->addElementWise(*norm->getOutput(0), *gamma->getOutput(0), ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((lname + "_scaled").c_str());
    IConstantLayer* beta =
            network->addConstant(Dims3{1, 1, channels}, getWeightsByPrefix(weightMap, layerNormName + ".b_0"));
    assert(beta);
    beta->setName((lname + "_beta").c_str());
    IElementWiseLayer* shifted =
            network->addElementWise(*scaled->getOutput(0), *beta->getOutput(0), ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName(lname.c_str());
    return shifted;
}

IElementWiseLayer* addScalarMul(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                float value, const std::string& lname) {
    auto* scalar = reinterpret_cast<float*>(malloc(sizeof(float)));
    scalar[0] = value;
    Weights scalarWts{DataType::kFLOAT, scalar, 1};
    weightMap[lname + ".scalar"] = scalarWts;

    Dims dims{};
    dims.nbDims = input.getDimensions().nbDims;
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = 1;
    }
    IConstantLayer* constant = network->addConstant(dims, scalarWts);
    IElementWiseLayer* mul = network->addElementWise(input, *constant->getOutput(0), ElementWiseOperation::kPROD);
    mul->setName(lname.c_str());
    return mul;
}

ITensor* addSiluTensor(INetworkDefinition* network, ITensor& input, const std::string& lname) {
    IElementWiseLayer* silu = addSwish(network, input, lname);
    return silu->getOutput(0);
}

ITensor* addGeluTensor(INetworkDefinition* network, ITensor& input, const std::string& lname) {
    Dims dims{};
    dims.nbDims = input.getDimensions().nbDims;
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = 1;
    }

    static const float kInvSqrt2 = 0.7071067811865475f;
    static const float kOne = 1.0f;
    static const float kHalf = 0.5f;

    IConstantLayer* invSqrt2 = network->addConstant(dims, Weights{DataType::kFLOAT, &kInvSqrt2, 1});
    assert(invSqrt2);
    invSqrt2->setName((lname + "_inv_sqrt2").c_str());
    IElementWiseLayer* scaled = network->addElementWise(input, *invSqrt2->getOutput(0), ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((lname + "_scale").c_str());
    IUnaryLayer* erf = network->addUnary(*scaled->getOutput(0), UnaryOperation::kERF);
    assert(erf);
    erf->setName((lname + "_erf").c_str());
    IConstantLayer* one = network->addConstant(dims, Weights{DataType::kFLOAT, &kOne, 1});
    assert(one);
    one->setName((lname + "_one").c_str());
    IElementWiseLayer* shifted =
            network->addElementWise(*erf->getOutput(0), *one->getOutput(0), ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName((lname + "_shift").c_str());
    IElementWiseLayer* product = network->addElementWise(input, *shifted->getOutput(0), ElementWiseOperation::kPROD);
    assert(product);
    product->setName((lname + "_prod").c_str());
    IConstantLayer* half = network->addConstant(dims, Weights{DataType::kFLOAT, &kHalf, 1});
    assert(half);
    half->setName((lname + "_half").c_str());
    IElementWiseLayer* gelu =
            network->addElementWise(*product->getOutput(0), *half->getOutput(0), ElementWiseOperation::kPROD);
    assert(gelu);
    gelu->setName(lname.c_str());
    return gelu->getOutput(0);
}

ITensor* addConvBnTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding, int groups,
                         const std::string& convName, const std::string& bnName) {
    IScaleLayer* bn = convBn(network, weightMap, input, outChannels, ksize, stride, padding, groups, convName, bnName);
    return bn->getOutput(0);
}

ITensor* addConvBnReluTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding, int groups,
                             const std::string& convName, const std::string& bnName) {
    ITensor* bn =
            addConvBnTensor(network, weightMap, input, outChannels, ksize, stride, padding, groups, convName, bnName);
    IActivationLayer* relu = addRelu(network, *bn, convName + "_relu");
    return relu->getOutput(0);
}

ITensor* addConvNoBiasTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding, int groups,
                             const std::string& convName) {
    IConvolutionLayer* conv =
            addConv2d(network, weightMap, input, convName + ".w_0", outChannels, ksize, stride, padding, groups);
    conv->setName(convName.c_str());
    return conv->getOutput(0);
}

ITensor* addConvNoBiasTensorByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                     ITensor& input, int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding,
                                     int groups, const std::string& convName) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = addConv2d(network, input, getWeightsByPrefix(weightMap, convName + ".w_0"), emptywts,
                                        outChannels, ksize, stride, padding, DimsHW{1, 1}, groups, convName);
    return conv->getOutput(0);
}

ITensor* addConvBiasTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                           int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding, const std::string& convName) {
    IConvolutionLayer* conv = convBias(network, weightMap, input, outChannels, ksize, stride, padding, 1, convName);
    return conv->getOutput(0);
}

ITensor* addConvBiasTensorByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                   ITensor& input, int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding,
                                   int groups, const std::string& convName) {
    IConvolutionLayer* conv = addConv2d(network, input, getWeightsByPrefix(weightMap, convName + ".w_0"),
                                        getWeightsByPrefix(weightMap, convName + ".b_0"), outChannels, ksize, stride,
                                        padding, DimsHW{1, 1}, groups, convName);
    return conv->getOutput(0);
}

ITensor* addSameConvBnReluTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 int outChannels, int kernel, const std::string& convName, const std::string& bnName) {
    IPaddingLayer* pad = network->addPaddingNd(input, DimsHW{0, 0}, DimsHW{kernel - 1, kernel - 1});
    assert(pad);
    pad->setName((convName + "_same_pad").c_str());
    return addConvBnReluTensor(network, weightMap, *pad->getOutput(0), outChannels, DimsHW{kernel, kernel},
                               DimsHW{1, 1}, DimsHW{0, 0}, 1, convName, bnName);
}

ITensor* addConvBnTensorByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding, int groups,
                                 const std::string& convName, const std::string& bnName) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = addConv2d(network, input, getWeightsByPrefix(weightMap, convName + ".w_0"), emptywts,
                                        outChannels, ksize, stride, padding, DimsHW{1, 1}, groups, convName);
    IScaleLayer* bn = addBatchNorm2dByPrefix(network, weightMap, *conv->getOutput(0), bnName, 1e-5f);
    bn->setName(bnName.c_str());
    return bn->getOutput(0);
}

ITensor* addConvBnReluTensorByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                     ITensor& input, int outChannels, DimsHW ksize, DimsHW stride, DimsHW padding,
                                     int groups, const std::string& convName, const std::string& bnName) {
    ITensor* bn = addConvBnTensorByPrefix(network, weightMap, input, outChannels, ksize, stride, padding, groups,
                                          convName, bnName);
    IActivationLayer* relu = addRelu(network, *bn, convName + "_relu");
    return relu->getOutput(0);
}

ITensor* addSameConvBnReluTensorByPrefix(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                         ITensor& input, int outChannels, int kernel, const std::string& convName,
                                         const std::string& bnName) {
    IPaddingLayer* pad = network->addPaddingNd(input, DimsHW{0, 0}, DimsHW{kernel - 1, kernel - 1});
    assert(pad);
    pad->setName((convName + "_same_pad").c_str());
    return addConvBnReluTensorByPrefix(network, weightMap, *pad->getOutput(0), outChannels, DimsHW{kernel, kernel},
                                       DimsHW{1, 1}, DimsHW{0, 0}, 1, convName, bnName);
}

ITensor* addBilinearResizeTensor(INetworkDefinition* network, ITensor& input, int channels, int height, int width,
                                 const std::string& lname) {
    IResizeLayer* resize = network->addResize(input);
    assert(resize);
    resize->setResizeMode(InterpolationMode::kLINEAR);
    resize->setCoordinateTransformation(ResizeCoordinateTransformation::kALIGN_CORNERS);
    resize->setOutputDimensions(Dims4{1, channels, height, width});
    resize->setName(lname.c_str());
    return resize->getOutput(0);
}

ITensor* addConvBiasBnTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int outChannels, int kernel, int stride, int padding, int dilation,
                             const std::string& convName, const std::string& bnName) {
    IConvolutionLayer* conv =
            addConv2d(network, input, getWeights(weightMap, convName + ".w_0"),
                      getWeights(weightMap, convName + ".b_0"), outChannels, DimsHW{kernel, kernel},
                      DimsHW{stride, stride}, DimsHW{padding, padding}, DimsHW{dilation, dilation}, 1, convName);
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), bnName, 1e-5f);
    bn->setName(bnName.c_str());
    return bn->getOutput(0);
}

ITensor* addConvBiasBnReluTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 int outChannels, int kernel, int stride, int padding, int dilation,
                                 const std::string& convName, const std::string& bnName) {
    ITensor* bn = addConvBiasBnTensor(network, weightMap, input, outChannels, kernel, stride, padding, dilation,
                                      convName, bnName);
    IActivationLayer* relu = addRelu(network, *bn, convName + "_relu");
    return relu->getOutput(0);
}

ITensor* addConvBnReluTensor(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int outChannels, int kernel, int stride, int padding, int dilation,
                             const std::string& convName, const std::string& bnName) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = addConv2d(network, input, getWeights(weightMap, convName + ".w_0"), emptywts, outChannels,
                                        DimsHW{kernel, kernel}, DimsHW{stride, stride}, DimsHW{padding, padding},
                                        DimsHW{dilation, dilation}, 1, convName);
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), bnName, 1e-5f);
    bn->setName(bnName.c_str());
    IActivationLayer* relu = addRelu(network, *bn->getOutput(0), convName + "_relu");
    return relu->getOutput(0);
}

nvinfer1::IElementWiseLayer* addLayerNorm(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int channels, const std::string& layerNormName) {
    return addLayerNorm(network, weightMap, input, channels, layerNormName, 1e-5f);
}

nvinfer1::IElementWiseLayer* addLayerNorm(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int channels, const std::string& layerNormName, float epsValue) {
    nvinfer1::IReduceLayer* mean = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, 1U << 2, true);
    assert(mean);
    mean->setName((layerNormName + "_mean").c_str());
    nvinfer1::IElementWiseLayer* centered =
            network->addElementWise(input, *mean->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
    assert(centered);
    centered->setName((layerNormName + "_centered").c_str());
    nvinfer1::IElementWiseLayer* square = network->addElementWise(*centered->getOutput(0), *centered->getOutput(0),
                                                                  nvinfer1::ElementWiseOperation::kPROD);
    assert(square);
    square->setName((layerNormName + "_square").c_str());
    nvinfer1::IReduceLayer* var =
            network->addReduce(*square->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 2, true);
    assert(var);
    var->setName((layerNormName + "_var").c_str());

    auto* epsVal = reinterpret_cast<float*>(malloc(sizeof(float)));
    epsVal[0] = epsValue;
    nvinfer1::Weights epsWts{nvinfer1::DataType::kFLOAT, epsVal, 1};
    weightMap[layerNormName + ".eps"] = epsWts;
    nvinfer1::IConstantLayer* eps = network->addConstant(nvinfer1::Dims3{1, 1, 1}, epsWts);
    assert(eps);
    eps->setName((layerNormName + "_eps").c_str());
    nvinfer1::IElementWiseLayer* varEps =
            network->addElementWise(*var->getOutput(0), *eps->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(varEps);
    varEps->setName((layerNormName + "_var_eps").c_str());
    nvinfer1::IUnaryLayer* stddev = network->addUnary(*varEps->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    assert(stddev);
    stddev->setName((layerNormName + "_std").c_str());
    nvinfer1::IElementWiseLayer* norm = network->addElementWise(*centered->getOutput(0), *stddev->getOutput(0),
                                                                nvinfer1::ElementWiseOperation::kDIV);
    assert(norm);
    norm->setName((layerNormName + "_norm").c_str());

    nvinfer1::IConstantLayer* gamma =
            network->addConstant(nvinfer1::Dims3{1, 1, channels}, getWeights(weightMap, layerNormName + ".w_0"));
    assert(gamma);
    gamma->setName((layerNormName + "_gamma").c_str());
    nvinfer1::IElementWiseLayer* scaled =
            network->addElementWise(*norm->getOutput(0), *gamma->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(scaled);
    scaled->setName((layerNormName + "_scaled").c_str());
    nvinfer1::IConstantLayer* beta =
            network->addConstant(nvinfer1::Dims3{1, 1, channels}, getWeights(weightMap, layerNormName + ".b_0"));
    assert(beta);
    beta->setName((layerNormName + "_beta").c_str());
    nvinfer1::IElementWiseLayer* shifted =
            network->addElementWise(*scaled->getOutput(0), *beta->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(shifted);
    shifted->setName(layerNormName.c_str());
    return shifted;
}

nvinfer1::ISoftMaxLayer* addSoftmax(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, int axes,
                                    const std::string& lname) {
    nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(input);
    assert(softmax);
    softmax->setAxes(axes);
    softmax->setName(lname.c_str());
    return softmax;
}
