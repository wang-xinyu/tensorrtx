#include "model.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "config.h"

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

void printNetworkLayers(INetworkDefinition* network) {
    int numLayers = network->getNbLayers();
    // std::cout << "currently num of layers: " << numLayers << std::endl;

    auto dataTypeToString = [](DataType type) {
        switch (type) {
            case DataType::kFLOAT:
                return "kFLOAT";
            case DataType::kHALF:
                return "kHALF";
            case DataType::kINT8:
                return "kINT8";
            case DataType::kINT32:
                return "kINT32";
            case DataType::kBOOL:
                return "kBOOL";
            default:
                return "Unknown";
        }
    };

    for (int i = 0; i < numLayers; ++i) {
        ILayer* layer = network->getLayer(i);
        std::cout << "--- Layer" << i << " = " << layer->getName() << std::endl;
        std::cout << "input & output tensor type: " << dataTypeToString(layer->getInput(0)->getType()) << "\t"
                  << dataTypeToString(layer->getOutput(0)->getType()) << std::endl;

        // input
        int inTensorNum = layer->getNbInputs();
        for (int j = 0; j < inTensorNum; ++j) {
            // std::cout << layer->getInput(j)->getDimensions().nbDims;
            Dims dims_in = layer->getInput(j)->getDimensions();
            std::cout << "input shape[" << j << "]: (";
            for (int k = 0; k < dims_in.nbDims; ++k) {
                std::cout << dims_in.d[k];
                if (k < dims_in.nbDims - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")\t";
        }
        std::cout << std::endl;

        // output
        int outTensorNum = layer->getNbOutputs();
        for (int j = 0; j < outTensorNum; ++j) {
            // std::cout << layer->getOutput(j)->getName();
            Dims dims_out = layer->getOutput(j)->getDimensions();
            std::cout << "output shape: (";
            for (int k = 0; k < dims_out.nbDims; ++k) {
                std::cout << dims_out.d[k];
                if (k < dims_out.nbDims - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")";
        }
        std::cout << "\n" << std::endl;
    }
}

static IScaleLayer* NormalizeInput(INetworkDefinition* network, ITensor& input) {
    float meanValues[3] = {-0.485f, -0.456f, -0.406f};
    float stdValues[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    Weights meanWeights{DataType::kFLOAT, meanValues, 3};
    Weights stdWeights{DataType::kFLOAT, stdValues, 3};

    IScaleLayer* NormaLayer = network->addScale(input, ScaleMode::kCHANNEL, meanWeights, stdWeights, Weights{});
    assert(NormaLayer != nullptr);

    return NormaLayer;
}

static IScaleLayer* NormalizeTeacherMap(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                        ITensor& input) {
    float* mean = (float*)weightMap["mean_std.mean"].values;
    float* std = (float*)weightMap["mean_std.std"].values;
    int len = weightMap["mean_std.mean"].count;

    // 1.scale
    float* scaleVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scaleVal[i] = 1.0 / std[i];
    }
    Weights scale{DataType::kFLOAT, scaleVal, len};

    // 2.shift
    float* shiftVal = nullptr;
    shiftVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shiftVal[i] = -mean[i];
    }
    Weights shift{DataType::kFLOAT, shiftVal, len};

    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, Weights{}, Weights{});
    assert(scale_1);
    IScaleLayer* scale_2 = network->addScale(*scale_1->getOutput(0), ScaleMode::kCHANNEL, Weights{}, scale, Weights{});
    assert(scale_2);

    return scale_2;
}

static ILayer* NormalizeFinalMap(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 std::string name) {
    float* qa = (float*)weightMap["quantiles.qa_" + name].values;
    float* qb = (float*)weightMap["quantiles.qb_" + name].values;
    int len = weightMap["quantiles.qa_" + name].count;

    Weights qbWeight_2{DataType::kFLOAT, qb, len};

    // fmap_st - qa_st
    float* shiftVal_1 = nullptr;
    shiftVal_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shiftVal_1[i] = -qa[i];
    }
    Weights qa_shiftWeight_1{DataType::kFLOAT, shiftVal_1, len};
    IScaleLayer* mapNorm_subLayer_1 =
            network->addScale(input, ScaleMode::kUNIFORM, qa_shiftWeight_1, Weights{}, Weights{});
    assert(mapNorm_subLayer_1);

    // qb_st - qa_st
    float* shiftVal_2 = nullptr;
    shiftVal_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shiftVal_2[i] = qb[i] - qa[i];
    }

    // (fmap_st - qa_st) / (qb_st - qa_st)
    float* scaleVal_1 = nullptr;
    scaleVal_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scaleVal_1[i] = 1.0f / shiftVal_2[i];
    }
    Weights scaleWeight_1{DataType::kFLOAT, scaleVal_1, len};
    IScaleLayer* mapNorm_divLayer_1 = network->addScale(*mapNorm_subLayer_1->getOutput(0), ScaleMode::kUNIFORM,
                                                        Weights{}, scaleWeight_1, Weights{});
    assert(mapNorm_divLayer_1);

    // ((fmap_st - qa_st) / (qb_st - qa_st)) * 0.1
    float* scaleVal_2 = nullptr;
    scaleVal_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scaleVal_2[i] = 0.1f;
    }
    Weights scaleWeight_2{DataType::kFLOAT, scaleVal_2, 1};
    IScaleLayer* mapNorm_Layer = network->addScale(*mapNorm_divLayer_1->getOutput(0), ScaleMode::kUNIFORM, Weights{},
                                                   scaleWeight_2, Weights{});
    assert(mapNorm_Layer);

    return mapNorm_Layer;
}

static ILayer* convRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                        int outch, int ksize, int s, int p, int g, std::string lname, bool withRelu) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(
            input, outch, DimsHW{ksize, ksize}, weightMap[lname + ".weight"],
            weightMap[lname + ".bias"]);  // if without bias weights, the results won't match with torch version
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    conv1->setName((lname).c_str());

    if (!withRelu)
        return conv1;

    auto relu = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
}

static IResizeLayer* interpolate(INetworkDefinition* network, ITensor& input, Dims upsampleScale,
                                 ResizeMode resizeMode) {
    IResizeLayer* interpolateLayer = network->addResize(input);
    assert(interpolateLayer);
    interpolateLayer->setOutputDimensions(upsampleScale);
    interpolateLayer->setResizeMode(resizeMode);

    return interpolateLayer;
}

static ILayer* interpConvRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              int outch, int ksize, int s, int p, int g, std::string lname, int dim) {
    IResizeLayer* interpolateLayer = network->addResize(input);
    assert(interpolateLayer != nullptr);
    interpolateLayer->setOutputDimensions(Dims3{input.getDimensions().d[0], dim, dim});
    interpolateLayer->setResizeMode(ResizeMode::kLINEAR);

    IConvolutionLayer* conv1 = network->addConvolutionNd(*interpolateLayer->getOutput(0), outch, DimsHW{ksize, ksize},
                                                         weightMap[lname + ".weight"], weightMap[lname + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    conv1->setName((lname + ".conv").c_str());

    auto relu = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
}

static IPoolingLayer* avgPool2d(INetworkDefinition* network, ITensor& input, int kernelSize, int stride, int padding) {
    IPoolingLayer* poolLayer = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{kernelSize, kernelSize});
    assert(poolLayer);
    poolLayer->setStride(DimsHW{stride, stride});
    poolLayer->setPadding(DimsHW{padding, padding});

    return poolLayer;
}

static void slice(INetworkDefinition* network, ITensor& input, std::vector<ITensor*>& layer_vec) {
    Dims inputDims = input.getDimensions();
    ISliceLayer* slice1 = network->addSlice(input, Dims3{0, 0, 0},
                                            Dims3{inputDims.d[0] / 2, inputDims.d[1], inputDims.d[2]}, Dims3{1, 1, 1});
    assert(slice1);

    ISliceLayer* slice2 = network->addSlice(input, Dims3{inputDims.d[0] / 2, 0, 0},
                                            Dims3{inputDims.d[0] / 2, inputDims.d[1], inputDims.d[2]}, Dims3{1, 1, 1});
    assert(slice2);

    layer_vec.push_back(slice1->getOutput(0));
    layer_vec.push_back(slice2->getOutput(0));
}

static IElementWiseLayer* mergeMap(INetworkDefinition* network, ITensor& input1, ITensor& input2) {
    float* scaleVal = nullptr;
    scaleVal = reinterpret_cast<float*>(malloc(sizeof(float) * 1));
    for (int i = 0; i < 1; i++) {
        scaleVal[i] = 0.5f;
    }
    Weights scaleWeight{DataType::kFLOAT, scaleVal, 1};
    IScaleLayer* mergeMapLayer1 = network->addScale(input1, ScaleMode::kUNIFORM, Weights{}, scaleWeight, Weights{});
    assert(mergeMapLayer1);

    IScaleLayer* mergeMapLayer2 = network->addScale(input2, ScaleMode::kUNIFORM, Weights{}, scaleWeight, Weights{});
    assert(mergeMapLayer2);

    IElementWiseLayer* mergedMapLayer = network->addElementWise(
            *mergeMapLayer1->getOutput(0), *mergeMapLayer2->getOutput(0), ElementWiseOperation::kSUM);
    assert(mergedMapLayer);

    return mergedMapLayer;
}

ICudaEngine* build_efficientAD_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                      float& gd, float& gw, std::string& wts_name) {
    /* create network object */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    /* create input tensor {3, kInputH, kInputW} */
    ITensor* InputData = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(InputData);

    /* create weight map */
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* AE */
    // auto BN1 = NormalizeInput(network, *InputData);
    // encoder
    auto enconv1 = convRelu(network, weightMap, *InputData, 32, 4, 2, 1, 1, "ae.encoder.enconv1", true);
    auto enconv2 = convRelu(network, weightMap, *enconv1->getOutput(0), 32, 4, 2, 1, 1, "ae.encoder.enconv2", true);
    auto enconv3 = convRelu(network, weightMap, *enconv2->getOutput(0), 64, 4, 2, 1, 1, "ae.encoder.enconv3", true);
    auto enconv4 = convRelu(network, weightMap, *enconv3->getOutput(0), 64, 4, 2, 1, 1, "ae.encoder.enconv4", true);
    auto enconv5 = convRelu(network, weightMap, *enconv4->getOutput(0), 64, 4, 2, 1, 1, "ae.encoder.enconv5", true);
    auto enconv6 = convRelu(network, weightMap, *enconv5->getOutput(0), 64, 8, 1, 0, 1, "ae.encoder.enconv6", false);
    // decoder
    auto deconv1 = interpConvRelu(network, weightMap, *enconv6->getOutput(0), 64, 4, 1, 2, 1, "ae.decoder.deconv1", 3);
    auto deconv2 = interpConvRelu(network, weightMap, *deconv1->getOutput(0), 64, 4, 1, 2, 1, "ae.decoder.deconv2", 8);
    auto deconv3 = interpConvRelu(network, weightMap, *deconv2->getOutput(0), 64, 4, 1, 2, 1, "ae.decoder.deconv3", 15);
    auto deconv4 = interpConvRelu(network, weightMap, *deconv3->getOutput(0), 64, 4, 1, 2, 1, "ae.decoder.deconv4", 32);
    auto deconv5 = interpConvRelu(network, weightMap, *deconv4->getOutput(0), 64, 4, 1, 2, 1, "ae.decoder.deconv5", 63);
    auto deconv6 =
            interpConvRelu(network, weightMap, *deconv5->getOutput(0), 64, 4, 1, 2, 1, "ae.decoder.deconv6", 127);
    auto deconv7 = interpConvRelu(network, weightMap, *deconv6->getOutput(0), 64, 3, 1, 1, 1, "ae.decoder.deconv7", 56);
    auto deconv8 = convRelu(network, weightMap, *deconv7->getOutput(0), 384, 3, 1, 1, 1, "ae.decoder.deconv8", false);

    /* PDN_medium_teacher */
    // no BN added after the convolutional layer
    auto teacher1 = convRelu(network, weightMap, *InputData, 256, 4, 1, 0, 1, "teacher.conv1", true);
    auto avgPool1 = avgPool2d(network, *teacher1->getOutput(0), 2, 2, 0);
    auto teacher2 = convRelu(network, weightMap, *avgPool1->getOutput(0), 512, 4, 1, 0, 1, "teacher.conv2", true);
    auto avgPool2 = avgPool2d(network, *teacher2->getOutput(0), 2, 2, 0);
    auto teacher3 = convRelu(network, weightMap, *avgPool2->getOutput(0), 512, 1, 1, 0, 1, "teacher.conv3", true);
    auto teacher4 = convRelu(network, weightMap, *teacher3->getOutput(0), 512, 3, 1, 0, 1, "teacher.conv4", true);
    auto teacher5 = convRelu(network, weightMap, *teacher4->getOutput(0), 384, 4, 1, 0, 1, "teacher.conv5", true);
    auto teacher6 = convRelu(network, weightMap, *teacher5->getOutput(0), 384, 1, 1, 0, 1, "teacher.conv6", false);

    /* PDN_medium_student */
    auto student1 = convRelu(network, weightMap, *InputData, 256, 4, 1, 0, 1, "student.conv1", true);
    auto avgPool3 = avgPool2d(network, *student1->getOutput(0), 2, 2, 0);
    auto student2 = convRelu(network, weightMap, *avgPool3->getOutput(0), 512, 4, 1, 0, 1, "student.conv2", true);
    auto avgPool4 = avgPool2d(network, *student2->getOutput(0), 2, 2, 0);
    auto student3 = convRelu(network, weightMap, *avgPool4->getOutput(0), 512, 1, 1, 0, 1, "student.conv3", true);
    auto student4 = convRelu(network, weightMap, *student3->getOutput(0), 512, 3, 1, 0, 1, "student.conv4", true);
    auto student5 = convRelu(network, weightMap, *student4->getOutput(0), 768, 4, 1, 0, 1, "student.conv5", true);
    auto student6 = convRelu(network, weightMap, *student5->getOutput(0), 768, 1, 1, 0, 1, "student.conv6", false);

    /* postCalculate */
    auto normal_teacher_output = NormalizeTeacherMap(network, weightMap, *teacher6->getOutput(0));
    std::vector<ITensor*> layer_vec{};
    slice(network, *student6->getOutput(0), layer_vec);
    ITensor* y_st = layer_vec[0];
    ITensor* y_stae = layer_vec[1];

    // distance_st
    IElementWiseLayer* sub_st =
            network->addElementWise(*normal_teacher_output->getOutput(0), *y_st, ElementWiseOperation::kSUB);
    assert(sub_st);
    IElementWiseLayer* distance_st =
            network->addElementWise(*sub_st->getOutput(0), *sub_st->getOutput(0), ElementWiseOperation::kPROD);
    assert(distance_st);

    // distance_stae
    IElementWiseLayer* sub_stae = network->addElementWise(*deconv8->getOutput(0), *y_stae, ElementWiseOperation::kSUB);
    assert(sub_stae);
    IElementWiseLayer* distance_stae =
            network->addElementWise(*sub_stae->getOutput(0), *sub_stae->getOutput(0), ElementWiseOperation::kPROD);
    assert(distance_stae);

    IReduceLayer* map_st = network->addReduce(*distance_st->getOutput(0), ReduceOperation::kAVG, 1, true);
    assert(map_st);
    IReduceLayer* map_stae = network->addReduce(*distance_stae->getOutput(0), ReduceOperation::kAVG, 1, true);
    assert(map_stae);

    IPaddingLayer* padMap_st = network->addPadding(*map_st->getOutput(0), DimsHW{4, 4}, DimsHW{4, 4});
    assert(padMap_st);
    IPaddingLayer* padMap_stae = network->addPadding(*map_stae->getOutput(0), DimsHW{4, 4}, DimsHW{4, 4});
    assert(padMap_stae);

    IResizeLayer* interpMap_st =
            interpolate(network, *padMap_st->getOutput(0),
                        Dims3{padMap_st->getOutput(0)->getDimensions().d[0], 256, 256}, ResizeMode::kLINEAR);
    assert(interpMap_st);
    IResizeLayer* interpMap_stae =
            interpolate(network, *padMap_stae->getOutput(0),
                        Dims3{padMap_stae->getOutput(0)->getDimensions().d[0], 256, 256}, ResizeMode::kLINEAR);
    assert(interpMap_stae);

    ILayer* normalizedMap_st = NormalizeFinalMap(network, weightMap, *interpMap_st->getOutput(0), "st");
    assert(normalizedMap_st);
    ILayer* normalizedMap_stae = NormalizeFinalMap(network, weightMap, *interpMap_stae->getOutput(0), "ae");
    assert(normalizedMap_stae);

    IElementWiseLayer* mergedMapLayer =
            mergeMap(network, *normalizedMap_st->getOutput(0), *normalizedMap_st->getOutput(0));
    printNetworkLayers(network);

    /* ouput */
    mergedMapLayer->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*mergedMapLayer->getOutput(0));

    /* Engine config */
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}
