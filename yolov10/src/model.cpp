#include <cmath>
#include <iostream>

#include "block.h"
#include "calibrator.h"
#include "config.h"
#include "model.h"

static int get_width(int x, float gw, int max_channels, int divisor = 8) {
    int c = std::min(x, max_channels);
    auto channel = int(ceil((c * gw) / divisor)) * divisor;
    return channel;
}

static int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
        --r;
    return std::max<int>(r, 1);
}

void calculateStrides(nvinfer1::ILayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

nvinfer1::IHostMemory* buildEngineYolov10DetN(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLOV10 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV10 BACKBONE  ********************************************
    *******************************************************************************************************/
    auto* conv0 = convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, "model.0");
    auto* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, "model.1");
    // 11233
    auto* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                      get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    auto* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.3");
    // 22466
    auto* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                      get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    auto* conv5 = SCDown(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.5");
    // 22466
    auto* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                      get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    auto* conv7 = SCDown(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, "model.7");
    // 11233
    auto* conv8 = C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                      get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");
    auto* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                       get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels), "model.10");
    /*******************************************************************************************************
    *********************************************  YOLOV10 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    auto* conv13 = C2F(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                       get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    auto* conv16 = C2F(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                       get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.16");
    auto* conv17 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    auto* conv19 = C2F(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                       get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.19");
    auto* conv20 =
            SCDown(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    auto* conv22 = C2fCIB(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                          get_width(1024, gw, max_channels), get_depth(3, gd), true, true, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLOV10 OUTPUT  ******************************************
    *******************************************************************************************************/
    auto d = conv16->getOutput(0)->getDimensions();
    assert(d.nbDims == 4);
    int ch_0 = d.d[1];
    int base_in_channel = std::max(16, std::max(ch_0 / 4, 16 * 4));
    int base_out_channel = std::max(ch_0, std::min(kNumClass, 100));

    // output0
    auto* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.0.0");
    auto* conv23_cv2_0_1 = convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 = network->addConvolutionNd(
            *conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.0.2.weight"],
            weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.0.0.0", get_width(256, gw, max_channels));
    auto* conv23_cv3_0_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.0.1");
    auto* conv23_cv3_0_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.0.1.0", base_out_channel);
    auto* conv23_cv3_0_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.1.1");
    auto* conv23_cv3_0_2 = network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.0.2.weight"],
                                                     weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    auto* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.1.0");
    auto* conv23_cv2_1_1 = convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 = network->addConvolutionNd(
            *conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.1.2.weight"],
            weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = convBnSiLU(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.1.0.0", get_width(512, gw, max_channels));
    auto* conv23_cv3_1_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.0.1");
    auto* conv23_cv3_1_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.1.1.0", base_out_channel);
    auto* conv23_cv3_1_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.1.1");
    auto* conv23_cv3_1_2 = network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.1.2.weight"],
                                                     weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    auto* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.2.0");
    auto* conv23_cv2_2_1 = convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 = network->addConvolutionNd(
            *conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.2.2.weight"],
            weightMap["model.23.one2one_cv2.2.2.bias"]);
    auto* conv23_cv3_2_0_0 = convBnSiLU(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.2.0.0", get_width(1024, gw, max_channels));
    auto* conv23_cv3_2_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.0.1");
    auto* conv23_cv3_2_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.2.1.0", base_out_channel);
    auto* conv23_cv3_2_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.1.1");
    auto* conv23_cv3_2_2 = network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.2.2.weight"],
                                                     weightMap["model.23.one2one_cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV10 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::ILayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 =
            network->addSlice(*shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 2);
    cat23_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 =
            network->addSlice(*shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 2);
    cat23_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 =
            network->addSlice(*shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 2);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::ILayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2}, strides, stridesLength);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
                                                  kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}

nvinfer1::IHostMemory* buildEngineYolov10DetS(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLOV10 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV10 BACKBONE  ********************************************
    *******************************************************************************************************/
    auto* conv0 = convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, "model.0");
    auto* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, "model.1");
    // 11233
    auto* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                      get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    auto* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.3");
    // 22466
    auto* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                      get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    auto* conv5 = SCDown(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.5");
    // 22466
    auto* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                      get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    auto* conv7 = SCDown(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, "model.7");
    // 11233
    auto* conv8 = C2fCIB(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(3, gd), true, true, 0.5, "model.8");
    auto* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                       get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels), "model.10");
    /*******************************************************************************************************
    *********************************************  YOLOV10 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    auto* conv13 = C2F(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                       get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    auto* conv16 = C2F(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                       get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.16");
    auto* conv17 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    auto* conv19 = C2F(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                       get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.19");
    auto* conv20 =
            SCDown(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    auto* conv22 = C2fCIB(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                          get_width(1024, gw, max_channels), get_depth(3, gd), true, true, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLOV10 OUTPUT  ******************************************
    *******************************************************************************************************/
    auto d = conv16->getOutput(0)->getDimensions();
    assert(d.nbDims == 4);
    int ch_0 = d.d[1];
    int base_in_channel = std::max(16, std::max(ch_0 / 4, 16 * 4));
    int base_out_channel = std::max(ch_0, std::min(kNumClass, 100));

    // output0
    auto* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.0.0");
    auto* conv23_cv2_0_1 = convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 = network->addConvolutionNd(
            *conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.0.2.weight"],
            weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.0.0.0", get_width(256, gw, max_channels));
    auto* conv23_cv3_0_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.0.1");
    auto* conv23_cv3_0_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.0.1.0", base_out_channel);
    auto* conv23_cv3_0_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.1.1");
    auto* conv23_cv3_0_2 = network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.0.2.weight"],
                                                     weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    auto* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.1.0");
    auto* conv23_cv2_1_1 = convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 = network->addConvolutionNd(
            *conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.1.2.weight"],
            weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = convBnSiLU(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.1.0.0", get_width(512, gw, max_channels));
    auto* conv23_cv3_1_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.0.1");
    auto* conv23_cv3_1_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.1.1.0", base_out_channel);
    auto* conv23_cv3_1_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.1.1");
    auto* conv23_cv3_1_2 = network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.1.2.weight"],
                                                     weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    auto* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.2.0");
    auto* conv23_cv2_2_1 = convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 = network->addConvolutionNd(
            *conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.2.2.weight"],
            weightMap["model.23.one2one_cv2.2.2.bias"]);
    auto* conv23_cv3_2_0_0 = convBnSiLU(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.2.0.0", get_width(1024, gw, max_channels));
    auto* conv23_cv3_2_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.0.1");
    auto* conv23_cv3_2_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.2.1.0", base_out_channel);
    auto* conv23_cv3_2_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.1.1");
    auto* conv23_cv3_2_2 = network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.2.2.weight"],
                                                     weightMap["model.23.one2one_cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV10 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::ILayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 =
            network->addSlice(*shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 2);
    cat23_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 =
            network->addSlice(*shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 2);
    cat23_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 =
            network->addSlice(*shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 2);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::ILayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2}, strides, stridesLength);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
                                                  kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}

nvinfer1::IHostMemory* buildEngineYolov10DetM(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLOV10 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV10 BACKBONE  ********************************************
    *******************************************************************************************************/
    auto* conv0 = convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, "model.0");
    auto* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, "model.1");
    // 11233
    auto* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                      get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    auto* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.3");
    // 22466
    auto* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                      get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    auto* conv5 = SCDown(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.5");
    // 22466
    auto* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                      get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    auto* conv7 = SCDown(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, "model.7");
    // 11233
    auto* conv8 = C2fCIB(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.8");
    auto* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                       get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels), "model.10");
    /*******************************************************************************************************
    *********************************************  YOLOV10 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    auto* conv13 = C2F(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                       get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    auto* conv16 = C2F(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                       get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.16");
    auto* conv17 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    auto* conv19 = C2fCIB(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                          get_width(512, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.19");
    auto* conv20 =
            SCDown(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    auto* conv22 = C2fCIB(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                          get_width(1024, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLOV10 OUTPUT  ******************************************
    *******************************************************************************************************/
    auto d = conv16->getOutput(0)->getDimensions();
    assert(d.nbDims == 4);
    int ch_0 = d.d[1];
    int base_in_channel = std::max(16, std::max(ch_0 / 4, 16 * 4));
    int base_out_channel = std::max(ch_0, std::min(kNumClass, 100));

    // output0
    auto* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.0.0");
    auto* conv23_cv2_0_1 = convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 = network->addConvolutionNd(
            *conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.0.2.weight"],
            weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.0.0.0", get_width(256, gw, max_channels));
    auto* conv23_cv3_0_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.0.1");
    auto* conv23_cv3_0_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.0.1.0", base_out_channel);
    auto* conv23_cv3_0_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.1.1");
    auto* conv23_cv3_0_2 = network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.0.2.weight"],
                                                     weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    auto* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.1.0");
    auto* conv23_cv2_1_1 = convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 = network->addConvolutionNd(
            *conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.1.2.weight"],
            weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = convBnSiLU(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.1.0.0", get_width(512, gw, max_channels));
    auto* conv23_cv3_1_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.0.1");
    auto* conv23_cv3_1_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.1.1.0", base_out_channel);
    auto* conv23_cv3_1_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.1.1");
    auto* conv23_cv3_1_2 = network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.1.2.weight"],
                                                     weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    auto* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.2.0");
    auto* conv23_cv2_2_1 = convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 = network->addConvolutionNd(
            *conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.2.2.weight"],
            weightMap["model.23.one2one_cv2.2.2.bias"]);
    auto* conv23_cv3_2_0_0 = convBnSiLU(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.2.0.0", get_width(1024, gw, max_channels));
    auto* conv23_cv3_2_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.0.1");
    auto* conv23_cv3_2_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.2.1.0", base_out_channel);
    auto* conv23_cv3_2_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.1.1");
    auto* conv23_cv3_2_2 = network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.2.2.weight"],
                                                     weightMap["model.23.one2one_cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV10 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::ILayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 =
            network->addSlice(*shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 2);
    cat23_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 =
            network->addSlice(*shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 2);
    cat23_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 =
            network->addSlice(*shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 2);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::ILayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2}, strides, stridesLength);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
                                                  kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}

nvinfer1::IHostMemory* buildEngineYolov10DetBL(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                               nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                               int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLOV10 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV10 BACKBONE  ********************************************
    *******************************************************************************************************/
    auto* conv0 = convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, "model.0");
    auto* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, "model.1");
    // 11233
    auto* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                      get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    auto* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.3");
    // 22466
    auto* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                      get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    auto* conv5 = SCDown(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.5");
    // 22466
    auto* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                      get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    auto* conv7 = SCDown(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, "model.7");
    // 11233
    auto* conv8 = C2fCIB(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.8");
    auto* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                       get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels), "model.10");
    /*******************************************************************************************************
    *********************************************  YOLOV10 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    auto* conv13 = C2fCIB(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                          get_width(512, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    auto* conv16 = C2F(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                       get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.16");
    auto* conv17 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    auto* conv19 = C2fCIB(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                          get_width(512, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.19");
    auto* conv20 =
            SCDown(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    auto* conv22 = C2fCIB(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                          get_width(1024, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLOV10 OUTPUT  ******************************************
    *******************************************************************************************************/
    auto d = conv16->getOutput(0)->getDimensions();
    assert(d.nbDims == 4);
    int ch_0 = d.d[1];
    int base_in_channel = std::max(16, std::max(ch_0 / 4, 16 * 4));
    int base_out_channel = std::max(ch_0, std::min(kNumClass, 100));

    // output0
    auto* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.0.0");
    auto* conv23_cv2_0_1 = convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 = network->addConvolutionNd(
            *conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.0.2.weight"],
            weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.0.0.0", get_width(256, gw, max_channels));
    auto* conv23_cv3_0_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.0.1");
    auto* conv23_cv3_0_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.0.1.0", base_out_channel);
    auto* conv23_cv3_0_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.1.1");
    auto* conv23_cv3_0_2 = network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.0.2.weight"],
                                                     weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    auto* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.1.0");
    auto* conv23_cv2_1_1 = convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 = network->addConvolutionNd(
            *conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.1.2.weight"],
            weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = convBnSiLU(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.1.0.0", get_width(512, gw, max_channels));
    auto* conv23_cv3_1_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.0.1");
    auto* conv23_cv3_1_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.1.1.0", base_out_channel);
    auto* conv23_cv3_1_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.1.1");
    auto* conv23_cv3_1_2 = network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.1.2.weight"],
                                                     weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    auto* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.2.0");
    auto* conv23_cv2_2_1 = convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 = network->addConvolutionNd(
            *conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.2.2.weight"],
            weightMap["model.23.one2one_cv2.2.2.bias"]);
    auto* conv23_cv3_2_0_0 = convBnSiLU(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.2.0.0", get_width(1024, gw, max_channels));
    auto* conv23_cv3_2_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.0.1");
    auto* conv23_cv3_2_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.2.1.0", base_out_channel);
    auto* conv23_cv3_2_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.1.1");
    auto* conv23_cv3_2_2 = network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.2.2.weight"],
                                                     weightMap["model.23.one2one_cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV10 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::ILayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 =
            network->addSlice(*shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 2);
    cat23_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 =
            network->addSlice(*shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 2);
    cat23_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 =
            network->addSlice(*shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 2);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::ILayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2}, strides, stridesLength);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
                                                  kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}

nvinfer1::IHostMemory* buildEngineYolov10DetX(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLOV10 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV10 BACKBONE  ********************************************
    *******************************************************************************************************/
    auto* conv0 = convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, "model.0");
    auto* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, "model.1");
    // 11233
    auto* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                      get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    auto* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.3");
    // 22466
    auto* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                      get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    auto* conv5 = SCDown(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.5");
    // 22466
    auto* conv6 = C2fCIB(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                         get_width(512, gw, max_channels), get_depth(6, gd), true, false, 0.5, "model.6");
    auto* conv7 = SCDown(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, "model.7");
    // 11233
    auto* conv8 = C2fCIB(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.8");
    auto* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                       get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels), "model.10");
    /*******************************************************************************************************
    *********************************************  YOLOV10 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    auto* conv13 = C2fCIB(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                          get_width(512, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    auto* conv16 = C2F(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                       get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.16");
    auto* conv17 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    auto* conv19 = C2fCIB(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                          get_width(512, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.19");
    auto* conv20 =
            SCDown(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    auto* conv22 = C2fCIB(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                          get_width(1024, gw, max_channels), get_depth(3, gd), true, false, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLOV10 OUTPUT  ******************************************
    *******************************************************************************************************/
    auto d = conv16->getOutput(0)->getDimensions();
    assert(d.nbDims == 4);
    int ch_0 = d.d[1];
    int base_in_channel = std::max(16, std::max(ch_0 / 4, 16 * 4));
    int base_out_channel = std::max(ch_0, std::min(kNumClass, 100));

    // output0
    auto* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.0.0");
    auto* conv23_cv2_0_1 = convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 = network->addConvolutionNd(
            *conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.0.2.weight"],
            weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = convBnSiLU(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.0.0.0", get_width(256, gw, max_channels));
    auto* conv23_cv3_0_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.0.1");
    auto* conv23_cv3_0_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.0.1.0", base_out_channel);
    auto* conv23_cv3_0_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.0.1.1");
    auto* conv23_cv3_0_2 = network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.0.2.weight"],
                                                     weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    auto* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.1.0");
    auto* conv23_cv2_1_1 = convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 = network->addConvolutionNd(
            *conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.1.2.weight"],
            weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = convBnSiLU(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.1.0.0", get_width(512, gw, max_channels));
    auto* conv23_cv3_1_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.0.1");
    auto* conv23_cv3_1_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.1.1.0", base_out_channel);
    auto* conv23_cv3_1_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.1.1.1");
    auto* conv23_cv3_1_2 = network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.1.2.weight"],
                                                     weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    auto* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), base_in_channel, 3, 1, "model.23.one2one_cv2.2.0");
    auto* conv23_cv2_2_1 = convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), base_in_channel, 3, 1,
                                      "model.23.one2one_cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 = network->addConvolutionNd(
            *conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1}, weightMap["model.23.one2one_cv2.2.2.weight"],
            weightMap["model.23.one2one_cv2.2.2.bias"]);
    auto* conv23_cv3_2_0_0 = convBnSiLU(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels), 3,
                                        1, "model.23.one2one_cv3.2.0.0", get_width(1024, gw, max_channels));
    auto* conv23_cv3_2_0_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.0.1");
    auto* conv23_cv3_2_1_0 = convBnSiLU(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), base_out_channel, 3, 1,
                                        "model.23.one2one_cv3.2.1.0", base_out_channel);
    auto* conv23_cv3_2_1_1 = convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), base_out_channel, 1, 1,
                                        "model.23.one2one_cv3.2.1.1");
    auto* conv23_cv3_2_2 = network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                                     weightMap["model.23.one2one_cv3.2.2.weight"],
                                                     weightMap["model.23.one2one_cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV10 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::ILayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 =
            network->addSlice(*shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 2);
    cat23_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 =
            network->addSlice(*shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 2);
    cat23_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 =
            network->addSlice(*shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.23.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 2);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::ILayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2}, strides, stridesLength);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, kInputQuantizationFolder, "int8calib.table",
                                                  kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}
