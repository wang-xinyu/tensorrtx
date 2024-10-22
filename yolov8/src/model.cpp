#include <math.h>
#include <iostream>

#include "block.h"
#include "calibrator.h"
#include "config.h"
#include "model.h"

static int get_width_5u(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_width(int x, float gw, int max_channels, int divisor = 8) {
    auto channel = int(ceil((x * gw) / divisor)) * divisor;
    return channel >= max_channels ? max_channels : channel;
}

static int get_depth(int x, float gd) {
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
        --r;
    return std::max<int>(r, 1);
}

void calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[1];
        strides[i] = reference_size / feature_map_size;
    }
}

static nvinfer1::IElementWiseLayer* Proto(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          std::string lname, float gw, int max_channels) {
    int mid_channel = get_width(256, gw, max_channels);
    auto cv1 = convBnSiLU(network, weightMap, input, mid_channel, 3, 1, 1, "model.22.proto.cv1");
    float* convTranpsose_bais = (float*)weightMap["model.22.proto.upsample.bias"].values;
    int convTranpsose_bais_len = weightMap["model.22.proto.upsample.bias"].count;
    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, convTranpsose_bais, convTranpsose_bais_len};
    auto convTranpsose = network->addDeconvolutionNd(*cv1->getOutput(0), mid_channel, nvinfer1::DimsHW{2, 2},
                                                     weightMap["model.22.proto.upsample.weight"], bias);
    assert(convTranpsose);
    convTranpsose->setStrideNd(nvinfer1::DimsHW{2, 2});
    auto cv2 = convBnSiLU(network, weightMap, *convTranpsose->getOutput(0), mid_channel, 3, 1, 1, "model.22.proto.cv2");
    auto cv3 = convBnSiLU(network, weightMap, *cv2->getOutput(0), 32, 1, 1, 0, "model.22.proto.cv3");
    assert(cv3);
    return cv3;
}

static nvinfer1::IShuffleLayer* cv4_conv_combined(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, std::string lname, int grid_shape, float gw,
                                                  std::string algo_type) {
    int mid_channle = 0;
    int output_channel = 0;

    if (algo_type == "seg") {
        if (gw == 0.25 || gw == 0.5) {
            mid_channle = 32;
        } else if (gw == 0.75) {
            mid_channle = 48;
        } else if (gw == 1.00) {
            mid_channle = 64;
        } else if (gw == 1.25) {
            mid_channle = 80;
        }

        output_channel = 32;

    } else if (algo_type == "pose") {
        std::string bn_weight_key = lname + ".0.bn.weight";
        mid_channle = weightMap[bn_weight_key].count;
        output_channel = kNumberOfPoints * 3;
    } else if (algo_type == "obb") {
        std::string bn_weight_key = lname + ".0.bn.weight";
        mid_channle = weightMap[bn_weight_key].count;
        output_channel = 1;
    }

    auto cv0 = convBnSiLU(network, weightMap, input, mid_channle, 3, 1, 1, lname + ".0");
    auto cv1 = convBnSiLU(network, weightMap, *cv0->getOutput(0), mid_channle, 3, 1, 1, lname + ".1");
    float* cv2_bais_value = (float*)weightMap[lname + ".2" + ".bias"].values;
    int cv2_bais_len = weightMap[lname + ".2" + ".bias"].count;
    nvinfer1::Weights cv2_bais{nvinfer1::DataType::kFLOAT, cv2_bais_value, cv2_bais_len};
    auto cv2 = network->addConvolutionNd(*cv1->getOutput(0), output_channel, nvinfer1::DimsHW{1, 1},
                                         weightMap[lname + ".2" + ".weight"], cv2_bais);
    cv2->setStrideNd(nvinfer1::DimsHW{1, 1});
    nvinfer1::IShuffleLayer* cv2_shuffle = network->addShuffle(*cv2->getOutput(0));
    cv2_shuffle->setReshapeDimensions(nvinfer1::Dims2{output_channel, grid_shape});

    return cv2_shuffle;
}

nvinfer1::IHostMemory* buildEngineYolov8Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
  ******************************************  YOLOV8 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
  *****************************************  YOLOV8 BACKBONE
  *********************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    // 11233
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");
    /*******************************************************************************************************
  *********************************************  YOLOV8 HEAD
  *********************************************
  *******************************************************************************************************/
    float scale[] = {1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor11[] = {upsample10->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);

    nvinfer1::IElementWiseLayer* conv12 =
            C2F(network, weightMap, *cat11->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.12");

    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);

    nvinfer1::IElementWiseLayer* conv15 =
            C2F(network, weightMap, *cat14->getOutput(0), get_width(256, gw, max_channels),
                get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.15");
    nvinfer1::IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.16");
    nvinfer1::ITensor* inputTensor17[] = {conv16->getOutput(0), conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    nvinfer1::IElementWiseLayer* conv18 =
            C2F(network, weightMap, *cat17->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.18");
    nvinfer1::IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.19");
    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0), conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    nvinfer1::IElementWiseLayer* conv21 =
            C2F(network, weightMap, *cat20->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.21");

    /*******************************************************************************************************
  *********************************************  YOLOV8 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kNumClass, 100)) : get_width(256, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 =
            network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 =
            network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv22_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_0[] = {conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 =
            network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 =
            network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_1[] = {conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 =
            network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);
    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 =
            network->addConvolution(*conv22_cv3_2_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor22_2[] = {conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split22_0_0 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_0_1 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_0 =
            DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl22_0->getOutput(0), split22_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);

    nvinfer1::IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split22_1_0 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_1_1 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_1 =
            DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_1[] = {dfl22_1->getOutput(0), split22_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);

    nvinfer1::IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split22_2_0 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_2_1 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_2 =
            DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.22.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor22_dfl_2[] = {dfl22_2->getOutput(0), split22_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
                         strides, stridesLength, kNumClass, false, false, false);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8DetP6(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
    /*******************************************************************************************************
  ******************************************  YOLOV8 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);
    /*******************************************************************************************************
  *****************************************  YOLOV8 BACKBONE
  *********************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    // 11233
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");

    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(768, gw, max_channels), 3, 2, 1, "model.7");
    nvinfer1::IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), get_width(768, gw, max_channels),
                                             get_width(768, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* conv9 =
            convBnSiLU(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.9");
    nvinfer1::IElementWiseLayer* conv10 =
            C2F(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.10");

    nvinfer1::IElementWiseLayer* conv11 =
            SPPF(network, weightMap, *conv10->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.11");

    /*******************************************************************************************************
  *********************************************  YOLOV8 HEAD
  *********************************************
  *******************************************************************************************************/
    // Head
    float scale[] = {1.0, 2.0, 2.0};  // scale used for upsampling

    // P5
    nvinfer1::IResizeLayer* upsample12 = network->addResize(*conv11->getOutput(0));
    upsample12->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample12->setScales(scale, 3);
    nvinfer1::ITensor* concat13_inputs[] = {upsample12->getOutput(0), conv8->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat13 = network->addConcatenation(concat13_inputs, 2);
    nvinfer1::IElementWiseLayer* conv14 =
            C2(network, weightMap, *concat13->getOutput(0), get_width(768, gw, max_channels),
               get_width(768, gw, max_channels), get_depth(3, gd), false, 0.5, "model.14");

    // P4
    nvinfer1::IResizeLayer* upsample15 = network->addResize(*conv14->getOutput(0));
    upsample15->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample15->setScales(scale, 3);
    nvinfer1::ITensor* concat16_inputs[] = {upsample15->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat16 = network->addConcatenation(concat16_inputs, 2);
    nvinfer1::IElementWiseLayer* conv17 =
            C2(network, weightMap, *concat16->getOutput(0), get_width(512, gw, max_channels),
               get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.17");

    // P3
    nvinfer1::IResizeLayer* upsample18 = network->addResize(*conv17->getOutput(0));
    upsample18->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample18->setScales(scale, 3);
    nvinfer1::ITensor* concat19_inputs[] = {upsample18->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat19 = network->addConcatenation(concat19_inputs, 2);
    nvinfer1::IElementWiseLayer* conv20 =
            C2(network, weightMap, *concat19->getOutput(0), get_width(256, gw, max_channels),
               get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.20");

    // Additional layers for P4, P5, P6
    // P4/16-medium
    nvinfer1::IElementWiseLayer* conv21 = convBnSiLU(network, weightMap, *conv20->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.21");
    nvinfer1::ITensor* concat22_inputs[] = {conv21->getOutput(0), conv17->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat22 = network->addConcatenation(concat22_inputs, 2);
    nvinfer1::IElementWiseLayer* conv23 =
            C2(network, weightMap, *concat22->getOutput(0), get_width(512, gw, max_channels),
               get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.23");

    // P5/32-large
    nvinfer1::IElementWiseLayer* conv24 = convBnSiLU(network, weightMap, *conv23->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.24");
    nvinfer1::ITensor* concat25_inputs[] = {conv24->getOutput(0), conv14->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat25 = network->addConcatenation(concat25_inputs, 2);
    nvinfer1::IElementWiseLayer* conv26 =
            C2(network, weightMap, *concat25->getOutput(0), get_width(768, gw, max_channels),
               get_width(768, gw, max_channels), get_depth(3, gd), false, 0.5, "model.26");

    // P6/64-xlarge
    nvinfer1::IElementWiseLayer* conv27 = convBnSiLU(network, weightMap, *conv26->getOutput(0),
                                                     get_width(768, gw, max_channels), 3, 2, 1, "model.27");
    nvinfer1::ITensor* concat28_inputs[] = {conv27->getOutput(0), conv11->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat28 = network->addConcatenation(concat28_inputs, 2);
    nvinfer1::IElementWiseLayer* conv29 =
            C2(network, weightMap, *concat28->getOutput(0), get_width(1024, gw, max_channels),
               get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.29");

    /*******************************************************************************************************
  *********************************************  YOLOV8 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kNumClass, 100)) : get_width(256, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv30_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv20->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_0_2 =
            network->addConvolutionNd(*conv30_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv2.0.2.weight"], weightMap["model.30.cv2.0.2.bias"]);
    conv30_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});

    conv30_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    nvinfer1::IElementWiseLayer* conv30_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv20->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.0.0");

    nvinfer1::IElementWiseLayer* conv30_cv3_0_1 = convBnSiLU(network, weightMap, *conv30_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_0_2 =
            network->addConvolutionNd(*conv30_cv3_0_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv3.0.2.weight"], weightMap["model.30.cv3.0.2.bias"]);
    conv30_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv30_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_0[] = {conv30_cv2_0_2->getOutput(0), conv30_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_0 = network->addConcatenation(inputTensor30_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv30_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv23->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_1_2 =
            network->addConvolutionNd(*conv30_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv2.1.2.weight"], weightMap["model.30.cv2.1.2.bias"]);
    conv30_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv30_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv23->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv30_cv3_1_1 = convBnSiLU(network, weightMap, *conv30_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_1_2 =
            network->addConvolutionNd(*conv30_cv3_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv3.1.2.weight"], weightMap["model.30.cv3.1.2.bias"]);
    conv30_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_1[] = {conv30_cv2_1_2->getOutput(0), conv30_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_1 = network->addConcatenation(inputTensor30_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv30_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv26->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_2_2 =
            network->addConvolution(*conv30_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv2.2.2.weight"], weightMap["model.30.cv2.2.2.bias"]);
    conv30_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv30_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv26->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv30_cv3_2_1 = convBnSiLU(network, weightMap, *conv30_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_2_2 =
            network->addConvolution(*conv30_cv3_2_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv3.2.2.weight"], weightMap["model.30.cv3.2.2.bias"]);
    conv30_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_2[] = {conv30_cv2_2_2->getOutput(0), conv30_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_2 = network->addConcatenation(inputTensor30_2, 2);

    // output3
    nvinfer1::IElementWiseLayer* conv30_cv2_3_0 =
            convBnSiLU(network, weightMap, *conv29->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.3.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_3_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_3_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.3.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_3_2 =
            network->addConvolution(*conv30_cv2_3_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv2.3.2.weight"], weightMap["model.30.cv2.3.2.bias"]);
    conv30_cv2_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv2_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv30_cv3_3_0 =
            convBnSiLU(network, weightMap, *conv29->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.3.0");
    nvinfer1::IElementWiseLayer* conv30_cv3_3_1 = convBnSiLU(network, weightMap, *conv30_cv3_3_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.3.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_3_2 =
            network->addConvolution(*conv30_cv3_3_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv3.3.2.weight"], weightMap["model.30.cv3.3.2.bias"]);
    conv30_cv3_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv3_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_3[] = {conv30_cv2_3_2->getOutput(0), conv30_cv3_3_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_3 = network->addConcatenation(inputTensor30_3, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7, conv9};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    // P3 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle30_0 =
            network->addShuffle(*cat30_0->getOutput(0));  // Reusing the previous cat30_0 as P3 concatenation layer
    shuffle30_0->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split30_0_0 = network->addSlice(
            *shuffle30_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_0_1 = network->addSlice(
            *shuffle30_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_0 =
            DFL(network, weightMap, *split30_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.30.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor30_dfl_0[] = {dfl30_0->getOutput(0), split30_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_0 = network->addConcatenation(inputTensor30_dfl_0, 2);

    // P4 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle30_1 =
            network->addShuffle(*cat30_1->getOutput(0));  // Reusing the previous cat30_1 as P4 concatenation layer
    shuffle30_1->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split30_1_0 = network->addSlice(
            *shuffle30_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_1_1 = network->addSlice(
            *shuffle30_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_1 =
            DFL(network, weightMap, *split30_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.30.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor30_dfl_1[] = {dfl30_1->getOutput(0), split30_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_1 = network->addConcatenation(inputTensor30_dfl_1, 2);

    // P5 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle30_2 =
            network->addShuffle(*cat30_2->getOutput(0));  // Reusing the previous cat30_2 as P5 concatenation layer
    shuffle30_2->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split30_2_0 = network->addSlice(
            *shuffle30_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_2_1 = network->addSlice(
            *shuffle30_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_2 =
            DFL(network, weightMap, *split30_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.30.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor30_dfl_2[] = {dfl30_2->getOutput(0), split30_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_2 = network->addConcatenation(inputTensor30_dfl_2, 2);

    // P6 processing steps
    nvinfer1::IShuffleLayer* shuffle30_3 = network->addShuffle(*cat30_3->getOutput(0));
    shuffle30_3->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[3]) * (kInputW / strides[3])});
    nvinfer1::ISliceLayer* split30_3_0 = network->addSlice(
            *shuffle30_3->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_3_1 = network->addSlice(
            *shuffle30_3->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_3 =
            DFL(network, weightMap, *split30_3_0->getOutput(0), 4, (kInputH / strides[3]) * (kInputW / strides[3]), 1,
                1, 0, "model.30.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor30_dfl_3[] = {dfl30_3->getOutput(0), split30_3_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_3 = network->addConcatenation(inputTensor30_dfl_3, 2);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::IConcatenationLayer*>{cat30_dfl_0, cat30_dfl_1, cat30_dfl_2, cat30_dfl_3},
            strides, stridesLength, kNumClass, false, false, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8DetP2(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
  ******************************************  YOLOV8 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
  *****************************************  YOLOV8 BACKBONE
  *********************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    // 11233
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");

    /*******************************************************************************************************
  *********************************************  YOLOV8 HEAD
  *********************************************
  *******************************************************************************************************/
    // Head
    float scale[] = {1.0, 2.0, 2.0};  // scale used for upsampling

    // P4
    nvinfer1::IResizeLayer* upsample10 =
            network->addResize(*conv9->getOutput(0));  // Assuming conv9 is the last layer of the backbone
                                                       // as per P5 in your first section.
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);
    nvinfer1::ITensor* concat11_inputs[] = {upsample10->getOutput(0),
                                            conv6->getOutput(0)};  // Assuming conv6 corresponds to "backbone P4" as
                                                                   // per your pseudocode
    nvinfer1::IConcatenationLayer* concat11 = network->addConcatenation(concat11_inputs, 2);
    nvinfer1::IElementWiseLayer* conv12 =
            C2F(network, weightMap, *concat11->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.12");

    // P3
    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);
    nvinfer1::ITensor* concat14_inputs[] = {upsample13->getOutput(0),
                                            conv4->getOutput(0)};  // Assuming conv4 corresponds to "backbone P3"
    nvinfer1::IConcatenationLayer* concat14 = network->addConcatenation(concat14_inputs, 2);
    nvinfer1::IElementWiseLayer* conv15 =
            C2F(network, weightMap, *concat14->getOutput(0), get_width(256, gw, max_channels),
                get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.15");

    // P2
    nvinfer1::IResizeLayer* upsample16 = network->addResize(*conv15->getOutput(0));
    upsample16->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample16->setScales(scale, 3);
    nvinfer1::ITensor* concat17_inputs[] = {upsample16->getOutput(0),
                                            conv2->getOutput(0)};  // Assuming conv2 corresponds to "backbone P2"
    nvinfer1::IConcatenationLayer* concat17 = network->addConcatenation(concat17_inputs, 2);
    nvinfer1::IElementWiseLayer* conv18 =
            C2F(network, weightMap, *concat17->getOutput(0), get_width(128, gw, max_channels),
                get_width(128, gw, max_channels), get_depth(3, gd), false, 0.5, "model.18");

    // Additional layers for P3, P4, P5
    // Downsample and concatenate for P3
    nvinfer1::IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0),
                                                     get_width(128, gw, max_channels), 3, 2, 1, "model.19");
    nvinfer1::ITensor* concat20_inputs[] = {
            conv19->getOutput(0), conv15->getOutput(0)};  // concatenate with higher-resolution feature map from P3
    nvinfer1::IConcatenationLayer* concat20 = network->addConcatenation(concat20_inputs, 2);
    nvinfer1::IElementWiseLayer* conv21 =
            C2F(network, weightMap, *concat20->getOutput(0), get_width(256, gw, max_channels),
                get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.21");

    // Downsample and concatenate for P4
    nvinfer1::IElementWiseLayer* conv22 = convBnSiLU(network, weightMap, *conv21->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.22");
    nvinfer1::ITensor* concat23_inputs[] = {
            conv22->getOutput(0), conv12->getOutput(0)};  // concatenate with higher-resolution feature map from P4
    nvinfer1::IConcatenationLayer* concat23 = network->addConcatenation(concat23_inputs, 2);
    nvinfer1::IElementWiseLayer* conv24 =
            C2F(network, weightMap, *concat23->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.24");

    // Downsample and concatenate for P5
    nvinfer1::IElementWiseLayer* conv25 = convBnSiLU(network, weightMap, *conv24->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.25");
    nvinfer1::ITensor* concat26_inputs[] = {
            conv25->getOutput(0), conv9->getOutput(0)};  // concatenate with higher-resolution feature map from P5
    nvinfer1::IConcatenationLayer* concat26 = network->addConcatenation(concat26_inputs, 2);
    nvinfer1::IElementWiseLayer* conv27 =
            C2F(network, weightMap, *concat26->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.27");

    /*******************************************************************************************************
  *********************************************  YOLOV8 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kNumClass, 100)) : get_width(128, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv28_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv28_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv28_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv28_cv2_0_2 =
            network->addConvolutionNd(*conv28_cv2_0_1->getOutput(0), base_in_channel, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.28.cv2.0.2.weight"], weightMap["model.28.cv2.0.2.bias"]);
    conv28_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv28_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_out_channel, 3, 1, 1, "model.28.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv28_cv3_0_1 = convBnSiLU(network, weightMap, *conv28_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.28.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv28_cv3_0_2 =
            network->addConvolutionNd(*conv28_cv3_0_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.28.cv3.0.2.weight"], weightMap["model.28.cv3.0.2.bias"]);
    conv28_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv28_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor28_0[] = {conv28_cv2_0_2->getOutput(0), conv28_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_0 = network->addConcatenation(inputTensor28_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv28_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv28_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv28_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv28_cv2_1_2 =
            network->addConvolutionNd(*conv28_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.28.cv2.1.2.weight"], weightMap["model.28.cv2.1.2.bias"]);
    conv28_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv28_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_out_channel, 3, 1, 1, "model.28.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv28_cv3_1_1 = convBnSiLU(network, weightMap, *conv28_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.28.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv28_cv3_1_2 =
            network->addConvolutionNd(*conv28_cv3_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.28.cv3.1.2.weight"], weightMap["model.28.cv3.1.2.bias"]);
    conv28_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor28_1[] = {conv28_cv2_1_2->getOutput(0), conv28_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_1 = network->addConcatenation(inputTensor28_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv28_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv24->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv28_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv28_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv28_cv2_2_2 =
            network->addConvolution(*conv28_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.28.cv2.2.2.weight"], weightMap["model.28.cv2.2.2.bias"]);
    conv28_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv28_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv24->getOutput(0), base_out_channel, 3, 1, 1, "model.28.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv28_cv3_2_1 = convBnSiLU(network, weightMap, *conv28_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.28.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv28_cv3_2_2 =
            network->addConvolution(*conv28_cv3_2_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.28.cv3.2.2.weight"], weightMap["model.28.cv3.2.2.bias"]);
    conv28_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor28_2[] = {conv28_cv2_2_2->getOutput(0), conv28_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_2 = network->addConcatenation(inputTensor28_2, 2);

    // output3
    nvinfer1::IElementWiseLayer* conv28_cv2_3_0 =
            convBnSiLU(network, weightMap, *conv27->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.3.0");
    nvinfer1::IElementWiseLayer* conv28_cv2_3_1 =
            convBnSiLU(network, weightMap, *conv28_cv2_3_0->getOutput(0), base_in_channel, 3, 1, 1, "model.28.cv2.3.1");
    nvinfer1::IConvolutionLayer* conv28_cv2_3_2 =
            network->addConvolution(*conv28_cv2_3_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.28.cv2.3.2.weight"], weightMap["model.28.cv2.3.2.bias"]);
    conv28_cv2_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv2_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv28_cv3_3_0 =
            convBnSiLU(network, weightMap, *conv27->getOutput(0), base_out_channel, 3, 1, 1, "model.28.cv3.3.0");
    nvinfer1::IElementWiseLayer* conv28_cv3_3_1 = convBnSiLU(network, weightMap, *conv28_cv3_3_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.28.cv3.3.1");
    nvinfer1::IConvolutionLayer* conv28_cv3_3_2 =
            network->addConvolution(*conv28_cv3_3_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.28.cv3.3.2.weight"], weightMap["model.28.cv3.3.2.bias"]);
    conv28_cv3_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv28_cv3_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor28_3[] = {conv28_cv2_3_2->getOutput(0), conv28_cv3_3_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_3 = network->addConcatenation(inputTensor28_3, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv1, conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    // P2 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle28_0 = network->addShuffle(*cat28_0->getOutput(0));
    shuffle28_0->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split28_0_0 = network->addSlice(
            *shuffle28_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split28_0_1 = network->addSlice(
            *shuffle28_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl28_0 =
            DFL(network, weightMap, *split28_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.28.dfl.conv.weight");

    nvinfer1::ITensor* inputTensor28_dfl_0[] = {dfl28_0->getOutput(0), split28_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_dfl_0 = network->addConcatenation(inputTensor28_dfl_0, 2);

    // P3 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle28_1 = network->addShuffle(*cat28_1->getOutput(0));
    shuffle28_1->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split28_1_0 = network->addSlice(
            *shuffle28_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split28_1_1 = network->addSlice(
            *shuffle28_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl28_1 =
            DFL(network, weightMap, *split28_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.28.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor28_dfl_1[] = {dfl28_1->getOutput(0), split28_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_dfl_1 = network->addConcatenation(inputTensor28_dfl_1, 2);

    // P4 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle28_2 = network->addShuffle(*cat28_2->getOutput(0));
    shuffle28_2->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split28_2_0 = network->addSlice(
            *shuffle28_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split28_2_1 = network->addSlice(
            *shuffle28_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl28_2 =
            DFL(network, weightMap, *split28_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.28.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor28_dfl_2[] = {dfl28_2->getOutput(0), split28_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_dfl_2 = network->addConcatenation(inputTensor28_dfl_2, 2);

    // P5 processing steps
    nvinfer1::IShuffleLayer* shuffle28_3 = network->addShuffle(*cat28_3->getOutput(0));
    shuffle28_3->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[3]) * (kInputW / strides[3])});
    nvinfer1::ISliceLayer* split28_3_0 = network->addSlice(
            *shuffle28_3->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split28_3_1 = network->addSlice(
            *shuffle28_3->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl28_3 =
            DFL(network, weightMap, *split28_3_0->getOutput(0), 4, (kInputH / strides[3]) * (kInputW / strides[3]), 1,
                1, 0, "model.28.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor28_dfl_3[] = {dfl28_3->getOutput(0), split28_3_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28_dfl_3 = network->addConcatenation(inputTensor28_dfl_3, 2);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::IConcatenationLayer*>{cat28_dfl_0, cat28_dfl_1, cat28_dfl_2, cat28_dfl_3},
            strides, stridesLength, kNumClass, false, false, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8Cls(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
    int max_channels = 1280;
    // ****************************************** YOLOV8 INPUT
    // **********************************************
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kClsInputH, kClsInputW});
    assert(data);

    // ***************************************** YOLOV8 BACKBONE
    // ********************************************
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    // C2 Block (11233)
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    // C2 Block Sequence (22466)
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    // C2 Block Sequence (22466)
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.7");
    // C2 Block (11233)
    nvinfer1::IElementWiseLayer* conv8 =
            C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");

    // ********************************************* YOLOV8 HEAD
    // *********************************************

    auto conv_class = convBnSiLU(network, weightMap, *conv8->getOutput(0), 1280, 1, 1, 1, "model.9.conv");
    // Adjusted code
    nvinfer1::Dims dims = conv_class->getOutput(0)->getDimensions();  // Obtain the dimensions of the
                                                                      // output of conv_class
    assert(dims.nbDims == 3);  // Make sure there are exactly 3 dimensions (channels, height, width)

    nvinfer1::IPoolingLayer* pool2 = network->addPoolingNd(*conv_class->getOutput(0), nvinfer1::PoolingType::kAVERAGE,
                                                           nvinfer1::DimsHW{dims.d[1], dims.d[2]});
    assert(pool2);

    // Fully connected layer declaration
    nvinfer1::IFullyConnectedLayer* yolo = network->addFullyConnected(
            *pool2->getOutput(0), kClsNumClass, weightMap["model.9.linear.weight"], weightMap["model.9.linear.bias"]);
    assert(yolo);

    // Set the name for the output tensor and mark it as network output
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    // Set the maximum batch size and workspace size
    builder->setMaxBatchSize(kBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

    // Configuration according to the precision mode being used
#if defined(USE_FP16)
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
    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Cleanup the network definition and allocated weights
    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}

nvinfer1::IHostMemory* buildEngineYolov8Seg(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
  ******************************************  YOLOV8 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
  *****************************************  YOLOV8 BACKBONE
  *********************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.7");
    nvinfer1::IElementWiseLayer* conv8 =
            C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");

    /*******************************************************************************************************
  *********************************************  YOLOV8 HEAD
  *********************************************
  *******************************************************************************************************/
    float scale[] = {1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor11[] = {upsample10->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);
    nvinfer1::IElementWiseLayer* conv12 =
            C2F(network, weightMap, *cat11->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.12");

    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    nvinfer1::IElementWiseLayer* conv15 =
            C2F(network, weightMap, *cat14->getOutput(0), get_width(256, gw, max_channels),
                get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.15");
    nvinfer1::IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.16");
    nvinfer1::ITensor* inputTensor17[] = {conv16->getOutput(0), conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    nvinfer1::IElementWiseLayer* conv18 =
            C2F(network, weightMap, *cat17->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.18");
    nvinfer1::IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.19");
    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0), conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    nvinfer1::IElementWiseLayer* conv21 =
            C2F(network, weightMap, *cat20->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.21");

    /*******************************************************************************************************
  *********************************************  YOLOV8 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kNumClass, 100)) : get_width(256, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 =
            network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 =
            network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv22_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_0[] = {conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 =
            network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 =
            network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_1[] = {conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 =
            network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);
    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 =
            network->addConvolution(*conv22_cv3_2_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor22_2[] = {conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split22_0_0 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_0_1 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_0 =
            DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.22.dfl.conv.weight");

    nvinfer1::IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split22_1_0 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_1_1 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_1 =
            DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.22.dfl.conv.weight");

    nvinfer1::IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split22_2_0 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_2_1 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_2 =
            DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.22.dfl.conv.weight");

    // det0
    auto proto_coef_0 = cv4_conv_combined(network, weightMap, *conv15->getOutput(0), "model.22.cv4.0",
                                          (kInputH / strides[0]) * (kInputW / strides[0]), gw, "seg");
    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl22_0->getOutput(0), split22_0_1->getOutput(0),
                                                proto_coef_0->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 3);

    // det1
    auto proto_coef_1 = cv4_conv_combined(network, weightMap, *conv18->getOutput(0), "model.22.cv4.1",
                                          (kInputH / strides[1]) * (kInputW / strides[1]), gw, "seg");
    nvinfer1::ITensor* inputTensor22_dfl_1[] = {dfl22_1->getOutput(0), split22_1_1->getOutput(0),
                                                proto_coef_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 3);

    // det2
    auto proto_coef_2 = cv4_conv_combined(network, weightMap, *conv21->getOutput(0), "model.22.cv4.2",
                                          (kInputH / strides[2]) * (kInputW / strides[2]), gw, "seg");
    nvinfer1::ITensor* inputTensor22_dfl_2[] = {dfl22_2->getOutput(0), split22_2_1->getOutput(0),
                                                proto_coef_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 3);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
                         strides, stridesLength, kNumClass, true, false, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    auto proto = Proto(network, weightMap, *conv15->getOutput(0), "model.22.proto", gw, max_channels);
    proto->getOutput(0)->setName("proto");
    network->markOutput(*proto->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8Pose(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                             int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
  ******************************************  YOLOV8 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
  *****************************************  YOLOV8 BACKBONE
  *********************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.7");
    nvinfer1::IElementWiseLayer* conv8 =
            C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");
    /*******************************************************************************************************
  *********************************************  YOLOV8 HEAD
  *********************************************
  *******************************************************************************************************/
    float scale[] = {1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor11[] = {upsample10->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);
    nvinfer1::IElementWiseLayer* conv12 =
            C2F(network, weightMap, *cat11->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.12");

    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    nvinfer1::IElementWiseLayer* conv15 =
            C2F(network, weightMap, *cat14->getOutput(0), get_width(256, gw, max_channels),
                get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.15");
    nvinfer1::IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.16");
    nvinfer1::ITensor* inputTensor17[] = {conv16->getOutput(0), conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    nvinfer1::IElementWiseLayer* conv18 =
            C2F(network, weightMap, *cat17->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.18");
    nvinfer1::IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.19");
    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0), conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    nvinfer1::IElementWiseLayer* conv21 =
            C2F(network, weightMap, *cat20->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.21");

    /*******************************************************************************************************
  *********************************************  YOLOV8 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kPoseNumClass, 100)) : get_width(256, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 =
            network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 =
            network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv22_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_0[] = {conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 =
            network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 =
            network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_1[] = {conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 =
            network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);
    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 =
            network->addConvolution(*conv22_cv3_2_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor22_2[] = {conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);
    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    /**************************************************************************************P3****************************************************************************************************************************************/
    nvinfer1::IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split22_0_0 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_0_1 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_0 =
            DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.22.dfl.conv.weight");

    // det0
    auto shuffle_conv15 = cv4_conv_combined(network, weightMap, *conv15->getOutput(0), "model.22.cv4.0",
                                            (kInputH / strides[0]) * (kInputW / strides[0]), gw, "pose");

    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl22_0->getOutput(0), split22_0_1->getOutput(0),
                                                shuffle_conv15->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 3);

    /********************************************************************************************P4**********************************************************************************************************************************/
    nvinfer1::IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split22_1_0 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_1_1 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_1 =
            DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.22.dfl.conv.weight");

    // det1
    auto shuffle_conv18 = cv4_conv_combined(network, weightMap, *conv18->getOutput(0), "model.22.cv4.1",
                                            (kInputH / strides[1]) * (kInputW / strides[1]), gw, "pose");

    nvinfer1::ITensor* inputTensor22_dfl_1[] = {dfl22_1->getOutput(0), split22_1_1->getOutput(0),
                                                shuffle_conv18->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 3);

    /********************************************************************************************P5**********************************************************************************************************************************/
    nvinfer1::IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split22_2_0 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_2_1 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_2 =
            DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.22.dfl.conv.weight");

    // det2
    auto shuffle_conv21 = cv4_conv_combined(network, weightMap, *conv21->getOutput(0), "model.22.cv4.2",
                                            (kInputH / strides[2]) * (kInputW / strides[2]), gw, "pose");
    nvinfer1::ITensor* inputTensor22_dfl_2[] = {dfl22_2->getOutput(0), split22_2_1->getOutput(0),
                                                shuffle_conv21->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 3);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
                         strides, stridesLength, kPoseNumClass, false, true, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8PoseP6(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                               nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                               int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
    /*******************************************************************************************************
  ******************************************  YOLOV8 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);
    /*******************************************************************************************************
  *****************************************  YOLOV8 BACKBONE
  *********************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    // 11233
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");

    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(768, gw, max_channels), 3, 2, 1, "model.7");
    nvinfer1::IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), get_width(768, gw, max_channels),
                                             get_width(768, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* conv9 =
            convBnSiLU(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.9");
    nvinfer1::IElementWiseLayer* conv10 =
            C2F(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.10");

    nvinfer1::IElementWiseLayer* conv11 =
            SPPF(network, weightMap, *conv10->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.11");

    /*******************************************************************************************************
  *********************************************  YOLOV8 HEAD
  *********************************************
  *******************************************************************************************************/
    // Head
    float scale[] = {1.0, 2.0, 2.0};  // scale used for upsampling

    // P5
    nvinfer1::IResizeLayer* upsample12 = network->addResize(*conv11->getOutput(0));
    upsample12->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample12->setScales(scale, 3);
    nvinfer1::ITensor* concat13_inputs[] = {upsample12->getOutput(0), conv8->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat13 = network->addConcatenation(concat13_inputs, 2);
    nvinfer1::IElementWiseLayer* conv14 =
            C2(network, weightMap, *concat13->getOutput(0), get_width(768, gw, max_channels),
               get_width(768, gw, max_channels), get_depth(3, gd), false, 0.5, "model.14");

    // P4
    nvinfer1::IResizeLayer* upsample15 = network->addResize(*conv14->getOutput(0));
    upsample15->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample15->setScales(scale, 3);
    nvinfer1::ITensor* concat16_inputs[] = {upsample15->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat16 = network->addConcatenation(concat16_inputs, 2);
    nvinfer1::IElementWiseLayer* conv17 =
            C2(network, weightMap, *concat16->getOutput(0), get_width(512, gw, max_channels),
               get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.17");

    // P3
    nvinfer1::IResizeLayer* upsample18 = network->addResize(*conv17->getOutput(0));
    upsample18->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample18->setScales(scale, 3);
    nvinfer1::ITensor* concat19_inputs[] = {upsample18->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat19 = network->addConcatenation(concat19_inputs, 2);
    nvinfer1::IElementWiseLayer* conv20 =
            C2(network, weightMap, *concat19->getOutput(0), get_width(256, gw, max_channels),
               get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.20");

    // Additional layers for P4, P5, P6
    // P4/16-medium
    nvinfer1::IElementWiseLayer* conv21 = convBnSiLU(network, weightMap, *conv20->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.21");
    nvinfer1::ITensor* concat22_inputs[] = {conv21->getOutput(0), conv17->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat22 = network->addConcatenation(concat22_inputs, 2);
    nvinfer1::IElementWiseLayer* conv23 =
            C2(network, weightMap, *concat22->getOutput(0), get_width(512, gw, max_channels),
               get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.23");

    // P5/32-large
    nvinfer1::IElementWiseLayer* conv24 = convBnSiLU(network, weightMap, *conv23->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.24");
    nvinfer1::ITensor* concat25_inputs[] = {conv24->getOutput(0), conv14->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat25 = network->addConcatenation(concat25_inputs, 2);
    nvinfer1::IElementWiseLayer* conv26 =
            C2(network, weightMap, *concat25->getOutput(0), get_width(768, gw, max_channels),
               get_width(768, gw, max_channels), get_depth(3, gd), false, 0.5, "model.26");

    // P6/64-xlarge
    nvinfer1::IElementWiseLayer* conv27 = convBnSiLU(network, weightMap, *conv26->getOutput(0),
                                                     get_width(768, gw, max_channels), 3, 2, 1, "model.27");
    nvinfer1::ITensor* concat28_inputs[] = {conv27->getOutput(0), conv11->getOutput(0)};
    nvinfer1::IConcatenationLayer* concat28 = network->addConcatenation(concat28_inputs, 2);
    nvinfer1::IElementWiseLayer* conv29 =
            C2(network, weightMap, *concat28->getOutput(0), get_width(1024, gw, max_channels),
               get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.29");

    /*******************************************************************************************************
  *********************************************  YOLOV8 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kPoseNumClass, 100)) : get_width(256, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv30_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv20->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_0_2 =
            network->addConvolutionNd(*conv30_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv2.0.2.weight"], weightMap["model.30.cv2.0.2.bias"]);
    conv30_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});

    conv30_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    nvinfer1::IElementWiseLayer* conv30_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv20->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.0.0");

    nvinfer1::IElementWiseLayer* conv30_cv3_0_1 = convBnSiLU(network, weightMap, *conv30_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_0_2 =
            network->addConvolutionNd(*conv30_cv3_0_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv3.0.2.weight"], weightMap["model.30.cv3.0.2.bias"]);
    conv30_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv30_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_0[] = {conv30_cv2_0_2->getOutput(0), conv30_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_0 = network->addConcatenation(inputTensor30_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv30_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv23->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_1_2 =
            network->addConvolutionNd(*conv30_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv2.1.2.weight"], weightMap["model.30.cv2.1.2.bias"]);
    conv30_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv30_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv23->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv30_cv3_1_1 = convBnSiLU(network, weightMap, *conv30_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_1_2 =
            network->addConvolutionNd(*conv30_cv3_1_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.30.cv3.1.2.weight"], weightMap["model.30.cv3.1.2.bias"]);
    conv30_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_1[] = {conv30_cv2_1_2->getOutput(0), conv30_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_1 = network->addConcatenation(inputTensor30_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv30_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv26->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_2_2 =
            network->addConvolution(*conv30_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv2.2.2.weight"], weightMap["model.30.cv2.2.2.bias"]);
    conv30_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv30_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv26->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv30_cv3_2_1 = convBnSiLU(network, weightMap, *conv30_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_2_2 =
            network->addConvolution(*conv30_cv3_2_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv3.2.2.weight"], weightMap["model.30.cv3.2.2.bias"]);
    conv30_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_2[] = {conv30_cv2_2_2->getOutput(0), conv30_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_2 = network->addConcatenation(inputTensor30_2, 2);

    // output3
    nvinfer1::IElementWiseLayer* conv30_cv2_3_0 =
            convBnSiLU(network, weightMap, *conv29->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.3.0");
    nvinfer1::IElementWiseLayer* conv30_cv2_3_1 =
            convBnSiLU(network, weightMap, *conv30_cv2_3_0->getOutput(0), base_in_channel, 3, 1, 1, "model.30.cv2.3.1");
    nvinfer1::IConvolutionLayer* conv30_cv2_3_2 =
            network->addConvolution(*conv30_cv2_3_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv2.3.2.weight"], weightMap["model.30.cv2.3.2.bias"]);
    conv30_cv2_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv2_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv30_cv3_3_0 =
            convBnSiLU(network, weightMap, *conv29->getOutput(0), base_out_channel, 3, 1, 1, "model.30.cv3.3.0");
    nvinfer1::IElementWiseLayer* conv30_cv3_3_1 = convBnSiLU(network, weightMap, *conv30_cv3_3_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.30.cv3.3.1");
    nvinfer1::IConvolutionLayer* conv30_cv3_3_2 =
            network->addConvolution(*conv30_cv3_3_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.30.cv3.3.2.weight"], weightMap["model.30.cv3.3.2.bias"]);
    conv30_cv3_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv30_cv3_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor30_3[] = {conv30_cv2_3_2->getOutput(0), conv30_cv3_3_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_3 = network->addConcatenation(inputTensor30_3, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV8 DETECT
  *******************************************
  *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7, conv9};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    // P3 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle30_0 =
            network->addShuffle(*cat30_0->getOutput(0));  // Reusing the previous cat30_0 as P3 concatenation layer
    shuffle30_0->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split30_0_0 = network->addSlice(
            *shuffle30_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_0_1 = network->addSlice(
            *shuffle30_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_0 =
            DFL(network, weightMap, *split30_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.30.dfl.conv.weight");

    // det0
    auto shuffle_conv20 = cv4_conv_combined(network, weightMap, *conv20->getOutput(0), "model.30.cv4.0",
                                            (kInputH / strides[0]) * (kInputW / strides[0]), gw, "pose");
    nvinfer1::ITensor* inputTensor30_dfl_0[] = {dfl30_0->getOutput(0), split30_0_1->getOutput(0),
                                                shuffle_conv20->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_0 = network->addConcatenation(inputTensor30_dfl_0, 3);

    // P4 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle30_1 =
            network->addShuffle(*cat30_1->getOutput(0));  // Reusing the previous cat30_1 as P4 concatenation layer
    shuffle30_1->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split30_1_0 = network->addSlice(
            *shuffle30_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_1_1 = network->addSlice(
            *shuffle30_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_1 =
            DFL(network, weightMap, *split30_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.30.dfl.conv.weight");

    // det1
    auto shuffle_conv23 = cv4_conv_combined(network, weightMap, *conv23->getOutput(0), "model.30.cv4.1",
                                            (kInputH / strides[1]) * (kInputW / strides[1]), gw, "pose");
    nvinfer1::ITensor* inputTensor30_dfl_1[] = {dfl30_1->getOutput(0), split30_1_1->getOutput(0),
                                                shuffle_conv23->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_1 = network->addConcatenation(inputTensor30_dfl_1, 3);

    // P5 processing steps (remains unchanged)
    nvinfer1::IShuffleLayer* shuffle30_2 =
            network->addShuffle(*cat30_2->getOutput(0));  // Reusing the previous cat30_2 as P5 concatenation layer
    shuffle30_2->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split30_2_0 = network->addSlice(
            *shuffle30_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_2_1 = network->addSlice(
            *shuffle30_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_2 =
            DFL(network, weightMap, *split30_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.30.dfl.conv.weight");

    // det2
    auto shuffle_conv26 = cv4_conv_combined(network, weightMap, *conv26->getOutput(0), "model.30.cv4.2",
                                            (kInputH / strides[2]) * (kInputW / strides[2]), gw, "pose");
    nvinfer1::ITensor* inputTensor30_dfl_2[] = {dfl30_2->getOutput(0), split30_2_1->getOutput(0),
                                                shuffle_conv26->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_2 = network->addConcatenation(inputTensor30_dfl_2, 3);

    // P6 processing steps
    nvinfer1::IShuffleLayer* shuffle30_3 = network->addShuffle(*cat30_3->getOutput(0));
    shuffle30_3->setReshapeDimensions(
            nvinfer1::Dims2{64 + kPoseNumClass, (kInputH / strides[3]) * (kInputW / strides[3])});
    nvinfer1::ISliceLayer* split30_3_0 = network->addSlice(
            *shuffle30_3->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split30_3_1 = network->addSlice(
            *shuffle30_3->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kPoseNumClass, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl30_3 =
            DFL(network, weightMap, *split30_3_0->getOutput(0), 4, (kInputH / strides[3]) * (kInputW / strides[3]), 1,
                1, 0, "model.30.dfl.conv.weight");

    // det3
    auto shuffle_conv29 = cv4_conv_combined(network, weightMap, *conv29->getOutput(0), "model.30.cv4.3",
                                            (kInputH / strides[3]) * (kInputW / strides[3]), gw, "pose");
    nvinfer1::ITensor* inputTensor30_dfl_3[] = {dfl30_3->getOutput(0), split30_3_1->getOutput(0),
                                                shuffle_conv29->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat30_dfl_3 = network->addConcatenation(inputTensor30_dfl_3, 3);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::IConcatenationLayer*>{cat30_dfl_0, cat30_dfl_1, cat30_dfl_2, cat30_dfl_3},
            strides, stridesLength, kPoseNumClass, false, true, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8_5uDet(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                               nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                               int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
  ******************************************  YOLOV5U INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
  *****************************************  YOLOV5U BACKBONE
  *********************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width_5u(64, gw), 6, 2, calculateP(6), "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width_5u(128, gw), 3, 2, calculateP(3), "model.1");
    // 11233
    nvinfer1::IElementWiseLayer* conv2 = C3(network, weightMap, *conv1->getOutput(0), get_width_5u(128, gw),
                                            get_width_5u(128, gw), get_depth(3, gd), true, 0.5, "model.2");

    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width_5u(256, gw), 3, 2, calculateP(3), "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 = C3(network, weightMap, *conv3->getOutput(0), get_width_5u(256, gw),
                                            get_width_5u(256, gw), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width_5u(512, gw), 3, 2, calculateP(3), "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 = C3(network, weightMap, *conv5->getOutput(0), get_width_5u(512, gw),
                                            get_width_5u(512, gw), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width_5u(1024, gw), 3,
                                                    2, calculateP(3), "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 = C3(network, weightMap, *conv7->getOutput(0), get_width_5u(1024, gw),
                                            get_width_5u(1024, gw), get_depth(3, gd), true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), get_width_5u(1024, gw),
                                              get_width_5u(1024, gw), 5, "model.9");
    /*******************************************************************************************************
  *********************************************  YOLOV5U HEAD
  *********************************************
  *******************************************************************************************************/

    //    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0),
    //    get_width_5u(512, gw), 1, 1, 1, "model.10");

    //*********************************************  cat backbone P4
    //********************************************
    nvinfer1::IElementWiseLayer* conv10 = convBnSiLU(network, weightMap, *conv9->getOutput(0), get_width_5u(512, gw), 1,
                                                     1, calculateP(1), "model.10");
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(conv6->getOutput(0)->getDimensions());
    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);
    nvinfer1::IElementWiseLayer* conv13 = C3(network, weightMap, *cat12->getOutput(0), get_width_5u(512, gw),
                                             get_width_5u(512, gw), get_depth(3, gd), false, 0.5, "model.13");
    //*********************************************  cat backbone P4
    //********************************************

    //*********************************************  cat backbone P3
    //********************************************
    nvinfer1::IElementWiseLayer* conv14 = convBnSiLU(network, weightMap, *conv13->getOutput(0), get_width_5u(256, gw),
                                                     1, 1, calculateP(1), "model.14");
    nvinfer1::IResizeLayer* upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(conv4->getOutput(0)->getDimensions());
    nvinfer1::ITensor* inputTensor16[] = {upsample15->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat16 = network->addConcatenation(inputTensor16, 2);
    nvinfer1::IElementWiseLayer* conv17 = C3(network, weightMap, *cat16->getOutput(0), get_width_5u(256, gw),
                                             get_width_5u(256, gw), get_depth(3, gd), false, 0.5, "model.17");
    //*********************************************  cat backbone P3
    //********************************************

    //*********************************************  cat head P4
    //********************************************
    nvinfer1::IElementWiseLayer* conv18 = convBnSiLU(network, weightMap, *conv17->getOutput(0), get_width_5u(256, gw),
                                                     3, 2, calculateP(3), "model.18");
    nvinfer1::ITensor* inputTensor19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat19 = network->addConcatenation(inputTensor19, 2);
    nvinfer1::IElementWiseLayer* conv20 = C3(network, weightMap, *cat19->getOutput(0), get_width_5u(512, gw),
                                             get_width_5u(512, gw), get_depth(3, gd), false, 0.5, "model.20");
    //*********************************************  cat head P4
    //********************************************

    //*********************************************  cat head P3
    //********************************************
    nvinfer1::IElementWiseLayer* conv21 = convBnSiLU(network, weightMap, *conv20->getOutput(0), get_width_5u(512, gw),
                                                     3, 2, calculateP(3), "model.21");
    nvinfer1::ITensor* inputTensor22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22 = network->addConcatenation(inputTensor22, 2);
    nvinfer1::IElementWiseLayer* conv23 = C3(network, weightMap, *cat22->getOutput(0), get_width_5u(1024, gw),
                                             get_width_5u(1024, gw), get_depth(3, gd), false, 0.5, "model.23");
    //*********************************************  cat head P3
    //********************************************

    /*******************************************************************************************************
  *********************************************  YOLOV5U OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kNumClass, 100)) : get_width_5u(256, gw);

    // output0
    nvinfer1::IElementWiseLayer* conv24_cv2_0_0 = convBnSiLU(network, weightMap, *conv17->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.24.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv24_cv2_0_1 = convBnSiLU(network, weightMap, *conv24_cv2_0_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.24.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv24_cv2_0_2 =
            network->addConvolutionNd(*conv24_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.24.cv2.0.2.weight"], weightMap["model.24.cv2.0.2.bias"]);
    conv24_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv24_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv24_cv3_0_0 = convBnSiLU(network, weightMap, *conv17->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.24.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv24_cv3_0_1 = convBnSiLU(network, weightMap, *conv24_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.24.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv24_cv3_0_2 =
            network->addConvolutionNd(*conv24_cv3_0_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.24.cv3.0.2.weight"], weightMap["model.24.cv3.0.2.bias"]);
    conv24_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv24_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor24_0[] = {conv24_cv2_0_2->getOutput(0), conv24_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat24_0 = network->addConcatenation(inputTensor24_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv24_cv2_1_0 = convBnSiLU(network, weightMap, *conv20->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.24.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv24_cv2_1_1 = convBnSiLU(network, weightMap, *conv24_cv2_1_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.24.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv24_cv2_1_2 =
            network->addConvolutionNd(*conv24_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.24.cv2.1.2.weight"], weightMap["model.24.cv2.1.2.bias"]);
    conv24_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv24_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv24_cv3_1_0 = convBnSiLU(network, weightMap, *conv20->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.24.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv24_cv3_1_1 = convBnSiLU(network, weightMap, *conv24_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.24.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv24_cv3_1_2 =
            network->addConvolutionNd(*conv24_cv3_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.24.cv3.1.2.weight"], weightMap["model.24.cv3.1.2.bias"]);
    conv24_cv3_1_2->setStride(nvinfer1::DimsHW{1, 1});
    conv24_cv3_1_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor24_1[] = {conv24_cv2_1_2->getOutput(0), conv24_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat24_1 = network->addConcatenation(inputTensor24_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv24_cv2_2_0 = convBnSiLU(network, weightMap, *conv23->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.24.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv24_cv2_2_1 = convBnSiLU(network, weightMap, *conv24_cv2_2_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.24.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv24_cv2_2_2 =
            network->addConvolutionNd(*conv24_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.24.cv2.2.2.weight"], weightMap["model.24.cv2.2.2.bias"]);
    conv24_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv24_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv24_cv3_2_0 = convBnSiLU(network, weightMap, *conv23->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.24.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv24_cv3_2_1 = convBnSiLU(network, weightMap, *conv24_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.24.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv24_cv3_2_2 =
            network->addConvolutionNd(*conv24_cv3_2_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.24.cv3.2.2.weight"], weightMap["model.24.cv3.2.2.bias"]);
    conv24_cv3_2_2->setStride(nvinfer1::DimsHW{1, 1});
    conv24_cv3_2_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor24_2[] = {conv24_cv2_2_2->getOutput(0), conv24_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat24_2 = network->addConcatenation(inputTensor24_2, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV5U DETECT
  *******************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    // det0
    nvinfer1::IShuffleLayer* shuffle24_0 = network->addShuffle(*cat24_0->getOutput(0));
    shuffle24_0->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split24_0_0 = network->addSlice(
            *shuffle24_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split24_0_1 = network->addSlice(
            *shuffle24_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl24_0 =
            DFL(network, weightMap, *split24_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.24.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor24_dfl_0[] = {dfl24_0->getOutput(0), split24_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat24_dfl_0 = network->addConcatenation(inputTensor24_dfl_0, 2);

    // det1
    nvinfer1::IShuffleLayer* shuffle24_1 = network->addShuffle(*cat24_1->getOutput(0));
    shuffle24_1->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split24_1_0 = network->addSlice(
            *shuffle24_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split24_1_1 = network->addSlice(
            *shuffle24_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl24_1 =
            DFL(network, weightMap, *split24_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.24.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor24_dfl_1[] = {dfl24_1->getOutput(0), split24_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat24_dfl_1 = network->addConcatenation(inputTensor24_dfl_1, 2);

    // det2
    nvinfer1::IShuffleLayer* shuffle24_2 = network->addShuffle(*cat24_2->getOutput(0));
    shuffle24_2->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split24_2_0 = network->addSlice(
            *shuffle24_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split24_2_1 = network->addSlice(
            *shuffle24_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl24_2 =
            DFL(network, weightMap, *split24_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.24.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor24_dfl_2[] = {dfl24_2->getOutput(0), split24_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat24_dfl_2 = network->addConcatenation(inputTensor24_dfl_2, 2);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat24_dfl_0, cat24_dfl_1, cat24_dfl_2},
                         strides, stridesLength, kNumClass, false, false, false);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8_5uDetP6(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                                 nvinfer1::DataType dt, const std::string& wts_path, float& gd,
                                                 float& gw, int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
  ******************************************  YOLOV5U-P6 INPUT
  ***********************************************
  *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
  *****************************************  YOLOV5U-P6 BACKBONE
  *********************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width_5u(64, gw), 6, 2, calculateP(6), "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width_5u(128, gw), 3, 2, calculateP(3), "model.1");
    // 11233
    nvinfer1::IElementWiseLayer* conv2 = C3(network, weightMap, *conv1->getOutput(0), get_width_5u(128, gw),
                                            get_width_5u(128, gw), get_depth(3, gd), true, 0.5, "model.2");

    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width_5u(256, gw), 3, 2, calculateP(3), "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 = C3(network, weightMap, *conv3->getOutput(0), get_width_5u(256, gw),
                                            get_width_5u(256, gw), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width_5u(512, gw), 3, 2, calculateP(3), "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 = C3(network, weightMap, *conv5->getOutput(0), get_width_5u(512, gw),
                                            get_width_5u(512, gw), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width_5u(768, gw), 3, 2, calculateP(3), "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 = C3(network, weightMap, *conv7->getOutput(0), get_width_5u(768, gw),
                                            get_width_5u(768, gw), get_depth(3, gd), true, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* conv9 = convBnSiLU(network, weightMap, *conv8->getOutput(0), get_width_5u(1024, gw), 3,
                                                    2, calculateP(3), "model.9");
    // 11233
    nvinfer1::IElementWiseLayer* conv10 = C3(network, weightMap, *conv9->getOutput(0), get_width_5u(1024, gw),
                                             get_width_5u(1024, gw), get_depth(3, gd), true, 0.5, "model.10");

    nvinfer1::IElementWiseLayer* conv11 = SPPF(network, weightMap, *conv10->getOutput(0), get_width_5u(1024, gw),
                                               get_width_5u(1024, gw), 5, "model.11");
    /*******************************************************************************************************
  *********************************************  YOLOV5U-P6 HEAD
  *********************************************
  *******************************************************************************************************/

    //*********************************************  cat backbone P5
    //********************************************
    nvinfer1::IElementWiseLayer* conv12 = convBnSiLU(network, weightMap, *conv11->getOutput(0), get_width_5u(768, gw),
                                                     1, 1, calculateP(1), "model.12");
    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(conv8->getOutput(0)->getDimensions());
    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0), conv8->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    nvinfer1::IElementWiseLayer* conv15 = C3(network, weightMap, *cat14->getOutput(0), get_width_5u(768, gw),
                                             get_width_5u(768, gw), get_depth(3, gd), false, 0.5, "model.15");
    //*********************************************  cat backbone P5
    //********************************************

    //*********************************************  cat backbone P4
    //********************************************
    nvinfer1::IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0), get_width_5u(512, gw),
                                                     1, 1, calculateP(1), "model.16");
    nvinfer1::IResizeLayer* upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(conv6->getOutput(0)->getDimensions());
    nvinfer1::ITensor* inputTensor18[] = {upsample17->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    nvinfer1::IElementWiseLayer* conv19 = C3(network, weightMap, *cat18->getOutput(0), get_width_5u(512, gw),
                                             get_width_5u(512, gw), get_depth(3, gd), false, 0.5, "model.19");
    //*********************************************  cat backbone P4
    //********************************************

    //*********************************************  cat backbone P3
    //********************************************
    nvinfer1::IElementWiseLayer* conv20 = convBnSiLU(network, weightMap, *conv19->getOutput(0), get_width_5u(256, gw),
                                                     1, 1, calculateP(1), "model.20");
    nvinfer1::IResizeLayer* upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(conv4->getOutput(0)->getDimensions());
    nvinfer1::ITensor* inputTensor22[] = {upsample21->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22 = network->addConcatenation(inputTensor22, 2);
    nvinfer1::IElementWiseLayer* conv23 = C3(network, weightMap, *cat22->getOutput(0), get_width_5u(256, gw),
                                             get_width_5u(256, gw), get_depth(3, gd), false, 0.5, "model.23");
    //*********************************************  cat backbone P3
    //********************************************

    //*********************************************  cat head P4
    //********************************************
    nvinfer1::IElementWiseLayer* conv24 = convBnSiLU(network, weightMap, *conv23->getOutput(0), get_width_5u(256, gw),
                                                     3, 2, calculateP(3), "model.24");
    nvinfer1::ITensor* inputTensor25[] = {conv24->getOutput(0), conv20->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat25 = network->addConcatenation(inputTensor25, 2);
    nvinfer1::IElementWiseLayer* conv26 = C3(network, weightMap, *cat25->getOutput(0), get_width_5u(512, gw),
                                             get_width_5u(512, gw), get_depth(3, gd), false, 0.5, "model.26");
    //*********************************************  cat head P4
    //********************************************

    //*********************************************  cat head P5
    //********************************************
    nvinfer1::IElementWiseLayer* conv27 = convBnSiLU(network, weightMap, *conv26->getOutput(0), get_width_5u(512, gw),
                                                     3, 2, calculateP(3), "model.27");
    nvinfer1::ITensor* inputTensor28[] = {conv27->getOutput(0), conv16->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat28 = network->addConcatenation(inputTensor28, 2);
    nvinfer1::IElementWiseLayer* conv29 = C3(network, weightMap, *cat28->getOutput(0), get_width_5u(768, gw),
                                             get_width_5u(768, gw), get_depth(3, gd), false, 0.5, "model.29");
    //*********************************************  cat head P5
    //********************************************

    //*********************************************  cat head P6
    //********************************************
    nvinfer1::IElementWiseLayer* conv30 = convBnSiLU(network, weightMap, *conv29->getOutput(0), get_width_5u(768, gw),
                                                     3, 2, calculateP(3), "model.30");
    nvinfer1::ITensor* inputTensor31[] = {conv30->getOutput(0), conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat31 = network->addConcatenation(inputTensor31, 2);
    nvinfer1::IElementWiseLayer* conv32 = C3(network, weightMap, *cat31->getOutput(0), get_width_5u(768, gw),
                                             get_width_5u(1024, gw), get_depth(3, gd), false, 0.5, "model.32");
    //*********************************************  cat head P6
    //********************************************

    /*******************************************************************************************************
  *********************************************  YOLOV5U-P6 OUTPUT
  *******************************************
  *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kNumClass, 100)) : get_width_5u(256, gw);

    // output0
    nvinfer1::IElementWiseLayer* conv33_cv2_0_0 = convBnSiLU(network, weightMap, *conv23->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.33.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv33_cv2_0_1 = convBnSiLU(network, weightMap, *conv33_cv2_0_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.33.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv33_cv2_0_2 =
            network->addConvolutionNd(*conv33_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv2.0.2.weight"], weightMap["model.33.cv2.0.2.bias"]);
    conv33_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv33_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv33_cv3_0_0 = convBnSiLU(network, weightMap, *conv23->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv33_cv3_0_1 = convBnSiLU(network, weightMap, *conv33_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.0.1");
    nvinfer1::IConvolutionLayer* conv33_cv3_0_2 =
            network->addConvolutionNd(*conv33_cv3_0_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv3.0.2.weight"], weightMap["model.33.cv3.0.2.bias"]);
    conv33_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv33_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor33_0[] = {conv33_cv2_0_2->getOutput(0), conv33_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_0 = network->addConcatenation(inputTensor33_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv33_cv2_1_0 = convBnSiLU(network, weightMap, *conv26->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.33.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv33_cv2_1_1 = convBnSiLU(network, weightMap, *conv33_cv2_1_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.33.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv33_cv2_1_2 =
            network->addConvolutionNd(*conv33_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv2.1.2.weight"], weightMap["model.33.cv2.1.2.bias"]);
    conv33_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv33_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv33_cv3_1_0 = convBnSiLU(network, weightMap, *conv26->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv33_cv3_1_1 = convBnSiLU(network, weightMap, *conv33_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv33_cv3_1_2 =
            network->addConvolutionNd(*conv33_cv3_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv3.1.2.weight"], weightMap["model.33.cv3.1.2.bias"]);
    conv33_cv3_1_2->setStride(nvinfer1::DimsHW{1, 1});
    conv33_cv3_1_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor33_1[] = {conv33_cv2_1_2->getOutput(0), conv33_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_1 = network->addConcatenation(inputTensor33_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv33_cv2_2_0 = convBnSiLU(network, weightMap, *conv29->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.33.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv33_cv2_2_1 = convBnSiLU(network, weightMap, *conv33_cv2_2_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.33.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv33_cv2_2_2 =
            network->addConvolutionNd(*conv33_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv2.2.2.weight"], weightMap["model.33.cv2.2.2.bias"]);
    conv33_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv33_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv33_cv3_2_0 = convBnSiLU(network, weightMap, *conv29->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv33_cv3_2_1 = convBnSiLU(network, weightMap, *conv33_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv33_cv3_2_2 =
            network->addConvolutionNd(*conv33_cv3_2_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv3.2.2.weight"], weightMap["model.33.cv3.2.2.bias"]);
    conv33_cv3_2_2->setStride(nvinfer1::DimsHW{1, 1});
    conv33_cv3_2_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor33_2[] = {conv33_cv2_2_2->getOutput(0), conv33_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_2 = network->addConcatenation(inputTensor33_2, 2);

    // output3
    nvinfer1::IElementWiseLayer* conv33_cv2_3_0 = convBnSiLU(network, weightMap, *conv32->getOutput(0), base_in_channel,
                                                             3, 1, calculateP(3), "model.33.cv2.3.0");
    nvinfer1::IElementWiseLayer* conv33_cv2_3_1 = convBnSiLU(network, weightMap, *conv33_cv2_3_0->getOutput(0),
                                                             base_in_channel, 3, 1, calculateP(3), "model.33.cv2.3.1");
    nvinfer1::IConvolutionLayer* conv33_cv2_3_2 =
            network->addConvolutionNd(*conv33_cv2_3_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv2.3.2.weight"], weightMap["model.33.cv2.3.2.bias"]);
    conv33_cv2_3_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv33_cv2_3_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv33_cv3_3_0 = convBnSiLU(network, weightMap, *conv32->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.3.0");
    nvinfer1::IElementWiseLayer* conv33_cv3_3_1 = convBnSiLU(network, weightMap, *conv33_cv3_3_0->getOutput(0),
                                                             base_out_channel, 3, 1, calculateP(3), "model.33.cv3.3.1");
    nvinfer1::IConvolutionLayer* conv33_cv3_3_2 =
            network->addConvolutionNd(*conv33_cv3_3_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.33.cv3.3.2.weight"], weightMap["model.33.cv3.3.2.bias"]);
    conv33_cv3_3_2->setStride(nvinfer1::DimsHW{1, 1});
    conv33_cv3_3_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor33_3[] = {conv33_cv2_3_2->getOutput(0), conv33_cv3_3_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_3 = network->addConcatenation(inputTensor33_3, 2);

    /*******************************************************************************************************
  *********************************************  YOLOV5U-P6 DETECT
  *******************************************
  *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7, conv9};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    // det0
    nvinfer1::IShuffleLayer* shuffle33_0 = network->addShuffle(*cat33_0->getOutput(0));
    shuffle33_0->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split33_0_0 = network->addSlice(
            *shuffle33_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split33_0_1 = network->addSlice(
            *shuffle33_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl33_0 =
            DFL(network, weightMap, *split33_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.33.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor33_dfl_0[] = {dfl33_0->getOutput(0), split33_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_dfl_0 = network->addConcatenation(inputTensor33_dfl_0, 2);

    // det1
    nvinfer1::IShuffleLayer* shuffle33_1 = network->addShuffle(*cat33_1->getOutput(0));
    shuffle33_1->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split33_1_0 = network->addSlice(
            *shuffle33_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split33_1_1 = network->addSlice(
            *shuffle33_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl33_1 =
            DFL(network, weightMap, *split33_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.33.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor33_dfl_1[] = {dfl33_1->getOutput(0), split33_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_dfl_1 = network->addConcatenation(inputTensor33_dfl_1, 2);

    // det2
    nvinfer1::IShuffleLayer* shuffle33_2 = network->addShuffle(*cat33_2->getOutput(0));
    shuffle33_2->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split33_2_0 = network->addSlice(
            *shuffle33_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split33_2_1 = network->addSlice(
            *shuffle33_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl33_2 =
            DFL(network, weightMap, *split33_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.33.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor33_dfl_2[] = {dfl33_2->getOutput(0), split33_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_dfl_2 = network->addConcatenation(inputTensor33_dfl_2, 2);

    // det3
    nvinfer1::IShuffleLayer* shuffle33_3 = network->addShuffle(*cat33_3->getOutput(0));
    shuffle33_3->setReshapeDimensions(nvinfer1::Dims2{64 + kNumClass, (kInputH / strides[3]) * (kInputW / strides[3])});
    nvinfer1::ISliceLayer* split33_3_0 = network->addSlice(
            *shuffle33_3->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split33_3_1 = network->addSlice(
            *shuffle33_3->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kNumClass, (kInputH / strides[3]) * (kInputW / strides[3])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl33_3 =
            DFL(network, weightMap, *split33_3_0->getOutput(0), 4, (kInputH / strides[3]) * (kInputW / strides[3]), 1,
                1, 0, "model.33.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor33_dfl_3[] = {dfl33_3->getOutput(0), split33_3_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat33_dfl_3 = network->addConcatenation(inputTensor33_dfl_3, 2);

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(
            network, std::vector<nvinfer1::IConcatenationLayer*>{cat33_dfl_0, cat33_dfl_1, cat33_dfl_2, cat33_dfl_3},
            strides, stridesLength, kNumClass, false, false, false);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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

nvinfer1::IHostMemory* buildEngineYolov8Obb(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
    ******************************************  YOLOV8 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims3{3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV8 BACKBONE  ********************************************
    *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), 3, 2, 1, "model.0");
    nvinfer1::IElementWiseLayer* conv1 =
            convBnSiLU(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), 3, 2, 1, "model.1");
    nvinfer1::IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                                             get_width(128, gw, max_channels), get_depth(3, gd), true, 0.5, "model.2");
    nvinfer1::IElementWiseLayer* conv3 =
            convBnSiLU(network, weightMap, *conv2->getOutput(0), get_width(256, gw, max_channels), 3, 2, 1, "model.3");
    nvinfer1::IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                                             get_width(256, gw, max_channels), get_depth(6, gd), true, 0.5, "model.4");
    nvinfer1::IElementWiseLayer* conv5 =
            convBnSiLU(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 3, 2, 1, "model.5");
    nvinfer1::IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                             get_width(512, gw, max_channels), get_depth(6, gd), true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 =
            convBnSiLU(network, weightMap, *conv6->getOutput(0), get_width(1024, gw, max_channels), 3, 2, 1, "model.7");
    nvinfer1::IElementWiseLayer* conv8 =
            C2F(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");

    /*******************************************************************************************************
    *********************************************  YOLOV8 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor11[] = {upsample10->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);
    nvinfer1::IElementWiseLayer* conv12 =
            C2F(network, weightMap, *cat11->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.12");

    nvinfer1::IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    nvinfer1::ITensor* inputTensor14[] = {upsample13->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    nvinfer1::IElementWiseLayer* conv15 =
            C2F(network, weightMap, *cat14->getOutput(0), get_width(256, gw, max_channels),
                get_width(256, gw, max_channels), get_depth(3, gd), false, 0.5, "model.15");
    nvinfer1::IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0),
                                                     get_width(256, gw, max_channels), 3, 2, 1, "model.16");
    nvinfer1::ITensor* inputTensor17[] = {conv16->getOutput(0), conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    nvinfer1::IElementWiseLayer* conv18 =
            C2F(network, weightMap, *cat17->getOutput(0), get_width(512, gw, max_channels),
                get_width(512, gw, max_channels), get_depth(3, gd), false, 0.5, "model.18");
    nvinfer1::IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0),
                                                     get_width(512, gw, max_channels), 3, 2, 1, "model.19");
    nvinfer1::ITensor* inputTensor20[] = {conv19->getOutput(0), conv9->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    nvinfer1::IElementWiseLayer* conv21 =
            C2F(network, weightMap, *cat20->getOutput(0), get_width(1024, gw, max_channels),
                get_width(1024, gw, max_channels), get_depth(3, gd), false, 0.5, "model.21");

    /*******************************************************************************************************
    *********************************************  YOLOV8 OUTPUT  ******************************************
    *******************************************************************************************************/
    int base_in_channel = (gw == 1.25) ? 80 : 64;
    int base_out_channel = (gw == 0.25) ? std::max(64, std::min(kObbNumClass, 100)) : get_width(256, gw, max_channels);

    // output0
    nvinfer1::IElementWiseLayer* conv22_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.0");

    nvinfer1::IElementWiseLayer* conv22_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.0.1");

    nvinfer1::IConvolutionLayer* conv22_cv2_0_2 =
            network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    nvinfer1::IElementWiseLayer* conv22_cv3_0_0 =
            convBnSiLU(network, weightMap, *conv15->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.0.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.0.1");

    nvinfer1::IConvolutionLayer* conv22_cv3_0_2 =
            network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(nvinfer1::DimsHW{1, 1});
    conv22_cv3_0_2->setPadding(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_0[] = {conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv22_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_1_2 =
            network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IElementWiseLayer* conv22_cv3_1_0 =
            convBnSiLU(network, weightMap, *conv18->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.1.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.1.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_1_2 =
            network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv22_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor22_1[] = {conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv22_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), base_in_channel, 3, 1, 1, "model.22.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv2_2_2 =
            network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);
    nvinfer1::IElementWiseLayer* conv22_cv3_2_0 =
            convBnSiLU(network, weightMap, *conv21->getOutput(0), base_out_channel, 3, 1, 1, "model.22.cv3.2.0");
    nvinfer1::IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0),
                                                             base_out_channel, 3, 1, 1, "model.22.cv3.2.1");
    nvinfer1::IConvolutionLayer* conv22_cv3_2_2 =
            network->addConvolution(*conv22_cv3_2_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
                                    weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);
    nvinfer1::ITensor* inputTensor22_2[] = {conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV8 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(
            nvinfer1::Dims2{64 + kObbNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});

    nvinfer1::ISliceLayer* split22_0_0 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_0_1 = network->addSlice(
            *shuffle22_0->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kObbNumClass, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_0 =
            DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.22.dfl.conv.weight");

    nvinfer1::IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(
            nvinfer1::Dims2{64 + kObbNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split22_1_0 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_1_1 = network->addSlice(
            *shuffle22_1->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kObbNumClass, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_1 =
            DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.22.dfl.conv.weight");

    nvinfer1::IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(
            nvinfer1::Dims2{64 + kObbNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split22_2_0 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{0, 0},
            nvinfer1::Dims2{64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::ISliceLayer* split22_2_1 = network->addSlice(
            *shuffle22_2->getOutput(0), nvinfer1::Dims2{64, 0},
            nvinfer1::Dims2{kObbNumClass, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims2{1, 1});
    nvinfer1::IShuffleLayer* dfl22_2 =
            DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.22.dfl.conv.weight");

    // det0
    auto shuffle_conv15 = cv4_conv_combined(network, weightMap, *conv15->getOutput(0), "model.22.cv4.0",
                                            (kInputH / strides[0]) * (kInputW / strides[0]), gw, "obb");
    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl22_0->getOutput(0), split22_0_1->getOutput(0),
                                                shuffle_conv15->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 3);

    // det1
    auto shuffle_conv18 = cv4_conv_combined(network, weightMap, *conv18->getOutput(0), "model.22.cv4.1",
                                            (kInputH / strides[1]) * (kInputW / strides[1]), gw, "obb");
    nvinfer1::ITensor* inputTensor22_dfl_1[] = {dfl22_1->getOutput(0), split22_1_1->getOutput(0),
                                                shuffle_conv18->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 3);

    // det2
    auto shuffle_conv21 = cv4_conv_combined(network, weightMap, *conv21->getOutput(0), "model.22.cv4.2",
                                            (kInputH / strides[2]) * (kInputW / strides[2]), gw, "obb");
    nvinfer1::ITensor* inputTensor22_dfl_2[] = {dfl22_2->getOutput(0), split22_2_1->getOutput(0),
                                                shuffle_conv21->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 3);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
                         strides, stridesLength, kObbNumClass, false, false, true);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
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
