#pragma once
#include <vector>
#include <map>
#include <string>
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

ILayer* BasicStem(INetworkDefinition *network,
std::map<std::string, Weights>& weightMap,
const std::string& lname, ITensor& input,
int out_channels,
int group_num = 1) {
    // conv1
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, out_channels, DimsHW{ 7, 7 },
    weightMap[lname + ".conv1.weight"],
    weightMap[lname + ".conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 3, 3 });
    conv1->setNbGroups(group_num);

    auto r1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(r1);

    auto max_pool2d = network->addPoolingNd(*r1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    max_pool2d->setStrideNd(DimsHW{ 2, 2 });
    max_pool2d->setPaddingNd(DimsHW{ 1, 1 });
    auto mp_dim = max_pool2d->getOutput(0)->getDimensions();
    return max_pool2d;
}

ITensor* BottleneckBlock(INetworkDefinition *network,
std::map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& input,
int in_channels,
int bottleneck_channels,
int out_channels,
int stride = 1,
int dilation = 1,
int group_num = 1) {
    // conv1
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, bottleneck_channels, DimsHW{ 1, 1 },
    weightMap[lname + ".conv1.weight"],
    weightMap[lname + ".conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setNbGroups(group_num);

    auto r1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(r1);

    // conv2
    IConvolutionLayer* conv2 = network->addConvolutionNd(*r1->getOutput(0), bottleneck_channels, DimsHW{ 3, 3 },
    weightMap[lname + ".conv2.weight"],
    weightMap[lname + ".conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    conv2->setPaddingNd(DimsHW{ 1 * dilation, 1 * dilation });
    conv2->setDilationNd(DimsHW{ dilation, dilation });
    conv2->setNbGroups(group_num);

    auto r2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(r2);

    // conv3
    IConvolutionLayer* conv3 = network->addConvolutionNd(*r2->getOutput(0), out_channels, DimsHW{ 1, 1 },
    weightMap[lname + ".conv3.weight"],
    weightMap[lname + ".conv3.bias"]);
    assert(conv3);
    conv3->setStrideNd(DimsHW{ 1, 1 });
    conv3->setNbGroups(group_num);

    // shortcut
    ITensor* shortcut_value = nullptr;
    if (in_channels != out_channels) {
        auto shortcut = network->addConvolutionNd(input, out_channels, DimsHW{ 1, 1 },
        weightMap[lname + ".shortcut.weight"],
        weightMap[lname + ".shortcut.bias"]);
        assert(shortcut);
        shortcut->setStrideNd(DimsHW{stride, stride});
        shortcut->setNbGroups(group_num);
        shortcut_value = shortcut->getOutput(0);
    } else {
        shortcut_value = &input;
    }

    // add
    auto ew = network->addElementWise(*conv3->getOutput(0), *shortcut_value, ElementWiseOperation::kSUM);
    assert(ew);

    auto r3 = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
    assert(r3);

    return r3->getOutput(0);
}

ITensor* MakeStage(INetworkDefinition *network,
std::map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& input,
int stage,
int in_channels,
int bottleneck_channels,
int out_channels,
int first_stride = 1,
int dilation = 1) {
    ITensor* out = &input;
    for (int i = 0; i < stage; i++) {
        if (i == 0)
            out = BottleneckBlock(network, weightMap,
            lname + "." + std::to_string(i), *out, in_channels,
            bottleneck_channels, out_channels, first_stride, dilation);
        else
            out = BottleneckBlock(network, weightMap,
            lname + "." + std::to_string(i), *out, in_channels,
            bottleneck_channels, out_channels, 1, dilation);
        in_channels = out_channels;
    }
    return out;
}

ITensor* BuildResNet(INetworkDefinition *network,
std::map<std::string, Weights>& weightMap,
ITensor& input,
RESNETTYPE resnet_type,
int stem_out_channels,
int bottleneck_channels,
int res2_out_channels,
int res5_dilation = 1) {
    assert(res5_dilation == 1 || res5_dilation == 2);  // "res5_dilation must be 1 or 2"
    if (resnet_type == R18 || resnet_type == R34) {
        assert(res2_out_channels == 64);  // "res2_out_channels must be 64 for R18/R34"
        assert(res5_dilation == 1);  // "res5_dilation must be 1 for R18/R34"
    }

    int out_channels = res2_out_channels;
    ITensor* out = nullptr;
    // stem
    auto stem = BasicStem(network, weightMap, "backbone.stem", input, stem_out_channels);
    out = stem->getOutput(0);

    // res
    for (int i = 0; i < 3; i++) {
        int dilation = (i == 3) ? res5_dilation : 1;
        int first_stride = (i == 0 || (i == 3 && dilation == 2)) ? 1 : 2;
        out = MakeStage(network, weightMap,
        "backbone.res" + std::to_string(i + 2), *out,
        num_blocks_per_stage.at(resnet_type)[i], stem_out_channels,
        bottleneck_channels, out_channels, first_stride, dilation);
        stem_out_channels = out_channels;
        bottleneck_channels *= 2;
        out_channels *= 2;
    }
    return out;
}
