#include "model.h"
#include "calibrator.h"
#include "config.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
static std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

static IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                   ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

static ILayer* convBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                         int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 3;
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    conv1->setName((lname + ".conv").c_str());
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

static ILayer* focus(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch,
                     int outch, int ksize, std::string lname) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{inch, kInputH / 2, kInputW / 2}, Dims3{1, 2, 2});
    ISliceLayer* s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{inch, kInputH / 2, kInputW / 2}, Dims3{1, 2, 2});
    ISliceLayer* s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{inch, kInputH / 2, kInputW / 2}, Dims3{1, 2, 2});
    ISliceLayer* s4 = network->addSlice(input, Dims3{0, 1, 1}, Dims3{inch, kInputH / 2, kInputW / 2}, Dims3{1, 2, 2});
    ITensor* inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

static ILayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                          int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

static ILayer* bottleneckCSP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                             int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{1, 1}, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{1, 1}, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

static ILayer* C3(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                  int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor* y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = {y1, cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

static ILayer* SPP(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                   int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k1, k1});
    pool1->setPaddingNd(DimsHW{k1 / 2, k1 / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k2, k2});
    pool2->setPaddingNd(DimsHW{k2 / 2, k2 / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k3, k3});
    pool3->setPaddingNd(DimsHW{k3 / 2, k3 / 2});
    pool3->setStrideNd(DimsHW{1, 1});

    ITensor* inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

static ILayer* SPPF(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                    int c2, int k, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool1->setPaddingNd(DimsHW{k / 2, k / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool2->setPaddingNd(DimsHW{k / 2, k / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool3->setPaddingNd(DimsHW{k / 2, k / 2});
    pool3->setStrideNd(DimsHW{1, 1});
    ITensor* inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

static ILayer* Proto(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c_,
                     int c2, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, c_, 3, 1, 1, lname + ".cv1");

    auto upsample = network->addResize(*cv1->getOutput(0));
    assert(upsample);
    upsample->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    const float scales[] = {1, 1, 2, 2};
    upsample->setScales(scales, 4);

    auto cv2 = convBlock(network, weightMap, *upsample->getOutput(0), c_, 3, 1, 1, lname + ".cv2");
    auto cv3 = convBlock(network, weightMap, *cv2->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    assert(cv3);
    return cv3;
}

static std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = kNumAnchor * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto* p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

nvinfer1::IHostMemory* build_cls_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,
                                        DataType dt, float& gd, float& gw, std::string& wts_name) {
    //  INetworkDefinition *network = builder->createNetworkV2(0U);
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // Create input tensor
    ITensor* data = network->addInput(kInputTensorName, dt, Dims4{maxBatchSize, 3, kClsInputH, kClsInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // Backbone
    auto conv0 = convBlock(network, weightMap, *data, get_width(64, gw), 6, 2, 1, "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw),
                              get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw),
                              get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw),
                              get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw),
                              get_depth(3, gd), true, 1, 0.5, "model.8");

    // Head
    auto conv_class = convBlock(network, weightMap, *bottleneck_csp8->getOutput(0), 1280, 1, 1, 1, "model.9.conv");
    int k = kClsInputH / 32;
    IPoolingLayer* pool2 = network->addPoolingNd(*conv_class->getOutput(0), PoolingType::kAVERAGE, DimsHW{k, k});
    assert(pool2);
    auto shuffle_0 = network->addShuffle(*pool2->getOutput(0));
    shuffle_0->setReshapeDimensions(nvinfer1::Dims2{kBatchSize, 1280});
    auto linear_weight = weightMap["model.9.linear.weight"];
    auto constant_weight = network->addConstant(nvinfer1::Dims2{kClsNumClass, 1280}, linear_weight);
    auto constant_bias =
            network->addConstant(nvinfer1::Dims2{kBatchSize, kClsNumClass}, weightMap["model.9.linear.bias"]);
    auto linear_matrix_multipy =
            network->addMatrixMultiply(*shuffle_0->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                       *constant_weight->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    auto yolo = network->addElementWise(*linear_matrix_multipy->getOutput(0), *constant_bias->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kSUM);
    assert(yolo);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    // Engine config
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator =
            new Int8EntropyCalibrator2(1, kClsInputW, kClsInputW, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}
