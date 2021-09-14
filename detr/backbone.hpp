#pragma once
#include <map>
#include "common.hpp"

enum RESNETTYPE {
    R18 = 0,
    R34,
    R50,
    R101,
    R152
};

const std::map<RESNETTYPE, std::vector<int>> num_blocks_per_stage = {
    {R18, {2, 2, 2, 2}},
    {R34, {3, 4, 6, 3}},
    {R50, {3, 4, 6, 3}},
    {R101, {3, 4, 23, 3}},
    {R152, {3, 8, 36, 3}}
};

IScaleLayer* addBatchNorm2d(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
ITensor& input,
const std::string& lname,
float eps = 1e-5
) {
    float *gamma = (float*)(weightMap[lname + ".weight"].values);
    float *beta = (float*)(weightMap[lname + ".bias"].values);
    float *mean = (float*)(weightMap[lname + ".running_mean"].values);
    float *var = (float*)(weightMap[lname + ".running_var"].values);
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* BasicStem(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& input,
int out_channels,
int group_num = 1
) {
    // conv1
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(
        input,
        out_channels,
        DimsHW{ 7, 7 },
        weightMap[lname + ".conv1.weight"],
        emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 3, 3 });
    conv1->setNbGroups(group_num);

    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1");
    assert(bn1);

    auto r1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(r1);

    auto max_pool2d = network->addPoolingNd(*r1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    max_pool2d->setStrideNd(DimsHW{ 2, 2 });
    max_pool2d->setPaddingNd(DimsHW{ 1, 1 });
    auto mp_dim = max_pool2d->getOutput(0)->getDimensions();
    return max_pool2d;
}

ITensor* BasicBlock(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& input,
int in_channels,
int out_channels,
int stride = 1
) {
    // conv1
    IConvolutionLayer* conv1 = network->addConvolutionNd(
        input,
        out_channels,
        DimsHW{ 3, 3 },
        weightMap[lname + ".conv1.weight"],
        weightMap[lname + ".conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 1, 1 });

    auto r1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(r1);

    // conv2
    IConvolutionLayer* conv2 = network->addConvolutionNd(
        *r1->getOutput(0),
        out_channels, DimsHW{ 3, 3 },
        weightMap[lname + ".conv2.weight"],
        weightMap[lname + ".conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    conv2->setPaddingNd(DimsHW{ 1, 1 });

    // shortcut
    ITensor* shortcut_value = nullptr;
    if (in_channels != out_channels) {
        auto shortcut = network->addConvolutionNd(
            input,
            out_channels,
            DimsHW{ 1, 1 },
            weightMap[lname + ".shortcut.weight"],
            weightMap[lname + ".shortcut.bias"]);
        assert(shortcut);
        shortcut->setStrideNd(DimsHW{ stride, stride });
        shortcut_value = shortcut->getOutput(0);
    } else {
        shortcut_value = &input;
    }

    // add
    auto ew = network->addElementWise(*conv2->getOutput(0), *shortcut_value, ElementWiseOperation::kSUM);
    assert(ew);

    auto r3 = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
    assert(r3);

    return r3->getOutput(0);
}

ITensor* BottleneckBlock(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& input,
int in_channels,
int bottleneck_channels,
int out_channels,
int stride = 1,
int dilation = 1,
int group_num = 1
) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // conv1
    IConvolutionLayer* conv1 = network->addConvolutionNd(
        input,
        bottleneck_channels,
        DimsHW{ 1, 1 },
        weightMap[lname + ".conv1.weight"],
        emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setNbGroups(group_num);

    auto bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1");
    assert(bn1);

    auto r1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(r1);

    // conv2
    IConvolutionLayer* conv2 = network->addConvolutionNd(
        *r1->getOutput(0),
        bottleneck_channels,
        DimsHW{ 3, 3 },
        weightMap[lname + ".conv2.weight"],
        emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ stride, stride });
    conv2->setPaddingNd(DimsHW{ 1 * dilation, 1 * dilation });
    conv2->setDilationNd(DimsHW{ dilation, dilation });
    conv2->setNbGroups(group_num);

    auto bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2");
    assert(bn2);

    auto r2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(r2);

    // conv3
    IConvolutionLayer* conv3 = network->addConvolutionNd(
        *r2->getOutput(0),
        out_channels,
        DimsHW{ 1, 1 },
        weightMap[lname + ".conv3.weight"],
        emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{ 1, 1 });
    conv3->setNbGroups(group_num);

    auto bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3");
    assert(bn3);

    // shortcut
    ITensor* shortcut_value = nullptr;
    if (in_channels != out_channels) {
        auto shortcut = network->addConvolutionNd(
            input,
            out_channels,
            DimsHW{ 1, 1 },
            weightMap[lname + ".downsample.0.weight"],
            emptywts);
        assert(shortcut);
        shortcut->setStrideNd(DimsHW{stride, stride});
        shortcut->setNbGroups(group_num);

        auto shortcut_bn = addBatchNorm2d(network, weightMap, *shortcut->getOutput(0), lname + ".downsample.1");
        assert(shortcut_bn);
        shortcut_value = shortcut_bn->getOutput(0);
    } else {
        shortcut_value = &input;
    }

    // add
    auto ew = network->addElementWise(*bn3->getOutput(0), *shortcut_value, ElementWiseOperation::kSUM);
    assert(ew);

    auto r3 = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
    assert(r3);

    return r3->getOutput(0);
}

ITensor* MakeStage(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& input,
int stage,
RESNETTYPE resnet_type,
int in_channels,
int bottleneck_channels,
int out_channels,
int first_stride = 1,
int dilation = 1
) {
    ITensor* out = &input;
    for (int i = 0; i < stage; i++) {
        std::string layerName = lname + "." + std::to_string(i);
        int stride = i == 0 ? first_stride : 1;

        if (resnet_type == R18 || resnet_type == R34)
            out = BasicBlock(network, weightMap, layerName, *out, in_channels, out_channels, stride);
        else
            out = BottleneckBlock(
                network,
                weightMap,
                layerName,
                *out,
                in_channels,
                bottleneck_channels,
                out_channels,
                stride,
                dilation);

        in_channels = out_channels;
    }
    return out;
}

ITensor* BuildResNet(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
ITensor& input,
RESNETTYPE resnet_type,
int stem_out_channels,
int bottleneck_channels,
int res2_out_channels,
int res5_dilation = 1
) {
    assert(res5_dilation == 1 || res5_dilation == 2);  // "res5_dilation must be 1 or 2"
    if (resnet_type == R18 || resnet_type == R34) {
        assert(res2_out_channels == 64);  // "res2_out_channels must be 64 for R18/R34")
        assert(res5_dilation == 1);  // "res5_dilation must be 1 for R18/R34")
    }

    int out_channels = res2_out_channels;
    ITensor* out = nullptr;
    // stem
    auto stem = BasicStem(network, weightMap, "backbone.0.body", input, stem_out_channels);
    out = stem->getOutput(0);

    // res
    for (int i = 0; i < 4; i++) {
        int dilation = (i == 3) ? res5_dilation : 1;
        int first_stride = (i == 0 || (i == 3 && dilation == 2)) ? 1 : 2;
        out = MakeStage(
            network,
            weightMap,
            "backbone.0.body.layer" + std::to_string(i + 1),
            *out,
            num_blocks_per_stage.at(resnet_type)[i],
            resnet_type,
            stem_out_channels,
            bottleneck_channels,
            out_channels,
            first_stride,
            dilation);
        stem_out_channels = out_channels;
        bottleneck_channels *= 2;
        out_channels *= 2;
    }
    return out;
}
