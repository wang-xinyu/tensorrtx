#include "block.h"
#include "yololayer.h"
#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
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

IElementWiseLayer* convBnSilu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, int p, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{ k, k }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname + ".conv").c_str());
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });


    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);


    // silu = x * sigmoid(x)
    IActivationLayer* sig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig1);
    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *sig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
}

ILayer* ReOrg(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, kInputH / 2, kInputW / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, kInputH / 2, kInputW / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, kInputH / 2, kInputW / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, kInputH / 2, kInputW / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    return cat;
}

ILayer* DownC(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, const std::string& lname) {
    int c_ = int(c2 * 0.5);
    IElementWiseLayer* cv1 = convBnSilu(network, weightMap, input, c1, 1, 1, 0, lname + ".cv1");
    IElementWiseLayer* cv2 = convBnSilu(network, weightMap, *cv1->getOutput(0), c_, 3, 2, 1, lname + ".cv2");

    IPoolingLayer* m1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
    m1->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* cv3 = convBnSilu(network, weightMap, *m1->getOutput(0), c_, 1, 1, 0, lname + ".cv3");

    ITensor* input_tensors[] = { cv2->getOutput(0),  cv3->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(input_tensors, 2);

    return concat;

}

IElementWiseLayer* SPPCSPC(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, const std::string& lname) {
    int c_ = int(2 * c2 * 0.5);
    IElementWiseLayer* cv1 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, lname + ".cv1");
    IElementWiseLayer* cv2 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, lname + ".cv2");

    IElementWiseLayer* cv3 = convBnSilu(network, weightMap, *cv1->getOutput(0), c_, 3, 1, 1, lname + ".cv3");
    IElementWiseLayer* cv4 = convBnSilu(network, weightMap, *cv3->getOutput(0), c_, 1, 1, 0, lname + ".cv4");

    IPoolingLayer* m1 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
    m1->setStrideNd(DimsHW{ 1, 1 });
    m1->setPaddingNd(DimsHW{ 2, 2 });
    IPoolingLayer* m2 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{ 9, 9 });
    m2->setStrideNd(DimsHW{ 1, 1 });
    m2->setPaddingNd(DimsHW{ 4, 4 });
    IPoolingLayer* m3 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{ 13, 13 });
    m3->setStrideNd(DimsHW{ 1, 1 });
    m3->setPaddingNd(DimsHW{ 6, 6 });

    ITensor* input_tensors[] = { cv4->getOutput(0), m1->getOutput(0), m2->getOutput(0), m3->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(input_tensors, 4);
    // 0U
    concat->setAxis(0);

    IElementWiseLayer* cv5 = convBnSilu(network, weightMap, *concat->getOutput(0), c_, 1, 1, 0, lname + ".cv5");
    IElementWiseLayer* cv6 = convBnSilu(network, weightMap, *cv5->getOutput(0), c_, 3, 1, 1, lname + ".cv6");

    ITensor* input_tensors2[] = { cv6->getOutput(0), cv2->getOutput(0) };
    IConcatenationLayer* concat1 = network->addConcatenation(input_tensors2, 2);
    // 0U
    concat1->setAxis(0);


    IElementWiseLayer* cv7 = convBnSilu(network, weightMap, *concat1->getOutput(0), c2, 1, 1, 0, lname + ".cv7");
    return cv7;
}

IElementWiseLayer* RepConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, const std::string& lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // 256 * 128 * 3 *3
    IConvolutionLayer* rbr_dense_conv = network->addConvolutionNd(input, c2, DimsHW{ k, k }, weightMap[lname + ".rbr_dense.0.weight"], emptywts);
    assert(rbr_dense_conv);
    rbr_dense_conv->setPaddingNd(DimsHW{ k / 2, k / 2 });
    rbr_dense_conv->setStrideNd(DimsHW{ s, s });
    rbr_dense_conv->setName((lname + ".rbr_dense.0").c_str());
    IScaleLayer* rbr_dense_bn = addBatchNorm2d(network, weightMap, *rbr_dense_conv->getOutput(0), lname + ".rbr_dense.1", 1e-3);

    IConvolutionLayer* rbr_1x1_conv = network->addConvolutionNd(input, c2, DimsHW{ 1, 1 }, weightMap[lname + ".rbr_1x1.0.weight"], emptywts);
    assert(rbr_1x1_conv);
    rbr_1x1_conv->setStrideNd(DimsHW{ s, s });
    rbr_1x1_conv->setName((lname + ".rbr_1x1.0").c_str());
    IScaleLayer* rbr_1x1_bn = addBatchNorm2d(network, weightMap, *rbr_1x1_conv->getOutput(0), lname + ".rbr_1x1.1", 1e-3);

    IElementWiseLayer* ew1 = network->addElementWise(*rbr_dense_bn->getOutput(0), *rbr_1x1_bn->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    // silu
    IActivationLayer* sigmoid = network->addActivation(*ew1->getOutput(0), ActivationType::kSIGMOID);
    IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    return ew2;
}

IActivationLayer* convBlockLeakRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname + ".conv").c_str());
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    //conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);

    auto ew1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    ew1->setAlpha(0.1);
    return ew1;
}

static std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = kNumAnchor * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);

    PluginField plugin_fields[2];
    int netinfo[4] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;

    std::vector<YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        YoloKernel kernel;
        kernel.width = kInputW / scale;
        kernel.height = kInputH / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}

