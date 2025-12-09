#include <math.h>
#include <iostream>

#include "block.h"
#include "calibrator.h"
#include "config.h"
#include "model.h"


static int get_width(int x, float gw, int max_channels, int divisor = 8) {
    auto channel = std::min(x, max_channels);
    channel = int(ceil((channel * gw) / divisor)) * divisor;
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
static nvinfer1::IElementWiseLayer* convBnSiLUProto(nvinfer1::INetworkDefinition* network,
                                                    std::map<std::string, nvinfer1::Weights> weightMap,
                                                    nvinfer1::ITensor& input, int ch, int k, int s, int p,
                                                    std::string lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"], bias_empty);
    
    conv->setStrideNd(nvinfer1::DimsHW{s, s});
    conv->setPaddingNd(nvinfer1::DimsHW{p, p});
    conv->setName((lname + ".conv").c_str());

    nvinfer1::IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    bn->setName((lname + ".bn").c_str());
    // This concat operator is not used for calculation, in order to prevent the operator fusion unrealized error when int8 is quantized.
    // Error Code 10: Internal Error (Could not find any implementation for node
    // model.22.proto.cv3.conv + model.22.proto.cv3.sigmoid + PWN(PWN((Unnamed Layer* 353) [Activation]), PWN(model.22.proto.cv3.silu)).)

#if defined(USE_INT8)
    nvinfer1::ITensor* inputTensors[] = {bn->getOutput(0)};
    auto concat = network->addConcatenation(inputTensors, 1);
    nvinfer1::IActivationLayer* sigmoid =
            network->addActivation(*concat->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid);
    bn->setName((lname + ".sigmoid").c_str());
    nvinfer1::IElementWiseLayer* ew = network->addElementWise(*concat->getOutput(0), *sigmoid->getOutput(0),
                                                              nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    ew->setName((lname + ".silu").c_str());
#else
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid);
    bn->setName((lname + ".sigmoid").c_str());
    nvinfer1::IElementWiseLayer* ew =
            network->addElementWise(*bn->getOutput(0), *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    ew->setName((lname + ".silu").c_str());

#endif
    return ew;
}

static nvinfer1::IElementWiseLayer* Proto(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          std::string lname, float gw, int max_channels) {
    int mid_channel = get_width(256, gw, max_channels);
    auto cv1 = convBnSiLU(network, weightMap, input, mid_channel, {3, 3}, 1, lname + ".cv1");
    //    float *convTranpsose_bais = (float *) weightMap["model.23.proto.upsample.bias"].values;
    //    int convTranpsose_bais_len = weightMap["model.23.proto.upsample.bias"].count;
    //    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, convTranpsose_bais, convTranpsose_bais_len};
    auto convTranpsose = network->addDeconvolutionNd(*cv1->getOutput(0), mid_channel, nvinfer1::DimsHW{2, 2},
                                                     weightMap[lname + ".upsample.weight"],
                                                     weightMap[lname + ".upsample.bias"]);

    assert(convTranpsose);
    convTranpsose->setStrideNd(nvinfer1::DimsHW{2, 2});
    convTranpsose->setPadding(nvinfer1::DimsHW{0, 0});
    auto cv2 =
            convBnSiLU(network, weightMap, *convTranpsose->getOutput(0), mid_channel, {3, 3}, 1, lname + ".cv2");
    auto cv3 = convBnSiLUProto(network, weightMap, *cv2->getOutput(0), 32, 1, 1, 0, lname + ".cv3");
    assert(cv3);
    return cv3;
}

static nvinfer1::IShuffleLayer* cv4_conv_combined(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, std::string lname, int grid_shape, float gw,
                                                  const std::string& algo_type, int max_channels) {
    int nm_nk = 0;
    int c4 = 0;

    if (algo_type == "seg") {
        nm_nk = 32;
        c4 = std::max(get_width(256, gw, max_channels) / 4, nm_nk);
    } else if (algo_type == "pose") {
        nm_nk = kNumberOfPoints * 3;
        c4 = std::max(get_width(256, gw, max_channels) / 4, kNumberOfPoints * 3);
    }

    auto cv0 = convBnSiLU(network, weightMap, input, c4, {3, 3}, 1, lname + ".0");
    auto cv1 = convBnSiLU(network, weightMap, *cv0->getOutput(0), c4, {3, 3}, 1, lname + ".1");
    float* cv2_bais_value = (float*)weightMap[lname + ".2" + ".bias"].values;
    int cv2_bais_len = weightMap[lname + ".2" + ".bias"].count;
    nvinfer1::Weights cv2_bais{nvinfer1::DataType::kFLOAT, cv2_bais_value, cv2_bais_len};
    auto cv2 = network->addConvolutionNd(*cv1->getOutput(0), nm_nk, nvinfer1::DimsHW{1, 1},
                                         weightMap[lname + ".2" + ".weight"], cv2_bais);
    cv2->setStrideNd(nvinfer1::DimsHW{1, 1});
    nvinfer1::IShuffleLayer* cv2_shuffle = network->addShuffle(*cv2->getOutput(0));
    cv2_shuffle->setReshapeDimensions(nvinfer1::Dims3{kBatchSize, nm_nk, grid_shape});

    return cv2_shuffle;
}


void calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

void calculateStrides(nvinfer1::ILayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

nvinfer1::IHostMemory* buildEngineYolov13Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                             int& max_channels, std::string& type) {

    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // =====================   input   ===================================================
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    // =====================   backbone   ===================================================
    nvinfer1::ILayer* conv0 = Conv(network, weightMap, *data, get_width(64, gw, max_channels), "model.0", 3, 2);
    nvinfer1::ILayer* conv1 = Conv(network, weightMap, *conv0->getOutput(0), get_width(128, gw, max_channels), "model.1", 3, 2, 1, 2);

    bool dsc3k = false;
    float mlp_ratio = 2.0;
    bool residual = false;
    bool channel_adjust = true;
    if (type == "l" || type == "x") {
        mlp_ratio = 1.5;  // see the yolov12-seg/ultralytics/nn/tasks.py/parse_model()
        residual = true;
        dsc3k = true;
        channel_adjust = false;
    }
    nvinfer1::ILayer* conv2 = DSC3K2(network, weightMap, *conv1->getOutput(0), get_width(256, gw, max_channels), 
                                                   "model.2", get_depth(2, gd), dsc3k, 0.25);
    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3", 1, 4);
    nvinfer1::ILayer* conv4 = DSC3K2(network, weightMap, *conv3->getOutput(0), get_width(512, gw, max_channels),
                                        "model.4", get_depth(2, gd), dsc3k, 0.25);
    nvinfer1::IElementWiseLayer* conv5 =
            DSConv(network, weightMap, *conv4->getOutput(0), get_width(512, gw, max_channels), 
                                       get_width(512, gw, max_channels), "model.5", 3, 2);
    nvinfer1::ILayer* conv6 = A2C2f(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                    get_depth(4, gd), "model.6", true, 4, residual, mlp_ratio);
    nvinfer1::IElementWiseLayer* conv7 = DSConv(network, weightMap, *conv6->getOutput(0), get_width(512, gw, max_channels),
                          get_width(1024, gw, max_channels), "model.7", 3, 2);

    nvinfer1::ILayer* conv8 = A2C2f(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                                    get_depth(4, gd), "model.8", true, 1, residual, mlp_ratio);

    //=========================  neck ====================================================================
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    int num_hyperedges = 8;
    if (type == "n") {
        num_hyperedges *= 0.5;
    } else if (type == "x") {
        num_hyperedges *= 1.5;
    }

    nvinfer1::ILayer* conv9 =
            HyperACE(network, weightMap, {conv4->getOutput(0), conv6->getOutput(0), conv8->getOutput(0)},
                     get_width(512, gw, max_channels), get_width(512, gw, max_channels), "model.9", get_depth(2, gd),
                     num_hyperedges, true, true, 0.5, 1, "both", channel_adjust);

    nvinfer1::IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample10->setScales(scale, 4);

    nvinfer1::ILayer* downsample11 = DownsampleConv(network, weightMap, *conv9->getOutput(0),
                                                    get_width(512, gw, max_channels), "model.11", channel_adjust);


    nvinfer1::IElementWiseLayer* conv12 =  // conv6:(1, 128, 40, 40) conv9: (1, 128, 40, 40)
            FullPad_Tunnel(network, weightMap, {conv6->getOutput(0), conv9->getOutput(0)}, "model.12");
    nvinfer1::IElementWiseLayer* conv13 =
            FullPad_Tunnel(network, weightMap, {conv4->getOutput(0), upsample10->getOutput(0)}, "model.13");

    nvinfer1::IElementWiseLayer* conv14 =
            FullPad_Tunnel(network, weightMap, {conv8->getOutput(0), downsample11->getOutput(0)}, "model.14");


    nvinfer1::IResizeLayer* upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample15->setScales(scale, 4);
    nvinfer1::ITensor* inputTensors16[] = {upsample15->getOutput(0), conv12->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat16 = network->addConcatenation(inputTensors16, 2);
    nvinfer1::ILayer* conv17 = DSC3K2(network, weightMap, *cat16->getOutput(0), get_width(512, gw, max_channels),
                                      "model.17", get_depth(2, gd), true);

    nvinfer1::IElementWiseLayer* conv18 =
            FullPad_Tunnel(network, weightMap, {conv17->getOutput(0), conv9->getOutput(0)}, "model.18");

    nvinfer1::IResizeLayer* upsample19 = network->addResize(*conv17->getOutput(0));
    assert(upsample19);
    upsample19->setScales(scale, 4);
    upsample19->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    nvinfer1::ITensor* inputTensors20[] = {upsample19->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat20 = network->addConcatenation(inputTensors20, 2);
    nvinfer1::ILayer* conv21 = DSC3K2(network, weightMap, *cat20->getOutput(0), get_width(256, gw, max_channels),
                                      "model.21", get_depth(2, gd), true);


    nvinfer1::ILayer* conv22 =
            Conv(network, weightMap, *upsample10->getOutput(0), get_width(256, gw, max_channels), "model.22");
    nvinfer1::IElementWiseLayer* conv23 =
            FullPad_Tunnel(network, weightMap, {conv21->getOutput(0), conv22->getOutput(0)}, "model.23");

    nvinfer1::ILayer* conv24 =
            Conv(network, weightMap, *conv23->getOutput(0), get_width(256, gw, max_channels), "model.24", 3, 2);
    nvinfer1::ITensor* inputTensors25[] = {conv24->getOutput(0), conv18->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat25 = network->addConcatenation(inputTensors25, 2);
    nvinfer1::ILayer* conv26 = DSC3K2(network, weightMap, *cat25->getOutput(0), get_width(512, gw, max_channels),
                                      "model.26", get_depth(2, gd), true);
    nvinfer1::IElementWiseLayer* conv27 =
            FullPad_Tunnel(network, weightMap, {conv26->getOutput(0), conv9->getOutput(0)}, "model.27");


    nvinfer1::ILayer* conv28 =
            Conv(network, weightMap, *conv26->getOutput(0), get_width(512, gw, max_channels), "model.28", 3, 2);
    nvinfer1::ITensor* inputTensors29[] = {conv28->getOutput(0), conv14->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat29 = network->addConcatenation(inputTensors29, 2);
    nvinfer1::ILayer* conv30 = DSC3K2(network, weightMap, *cat29->getOutput(0), get_width(1024, gw, max_channels),
                                      "model.30", get_depth(2, gd), true);
    nvinfer1::IElementWiseLayer* conv31 =
            FullPad_Tunnel(network, weightMap, {conv30->getOutput(0), downsample11->getOutput(0)}, "model.31");


    // =============================== output ===================================================================
    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kNumClass, 100));

    // output0   location
    nvinfer1::IElementWiseLayer* conv32_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv23->getOutput(0), c2, {3, 3}, 1, "model.32.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv32_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv32_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.32.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv32_cv2_0_2 =
            network->addConvolutionNd(*conv32_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.32.cv2.0.2.weight"], weightMap["model.32.cv2.0.2.bias"]);
    conv32_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv32_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    // output0 classes
    auto* conv32_cv3_0_0_0 = DWConv(network, weightMap, *conv23->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.32.cv3.0.0.0");
    nvinfer1::IElementWiseLayer* conv32_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv32_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.32.cv3.0.0.1");

    auto* conv32_cv3_0_1_0 =
            DWConv(network, weightMap, *conv32_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.32.cv3.0.1.0");
    nvinfer1::IElementWiseLayer* conv32_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv32_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.32.cv3.0.1.1");
    nvinfer1::IConvolutionLayer* conv32_cv3_0_1_2 =
            network->addConvolutionNd(*conv32_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.32.cv3.0.2.weight"], weightMap["model.32.cv3.0.2.bias"]);
    conv32_cv3_0_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv32_cv3_0_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});

    nvinfer1::ITensor* inputTensors32_0[] = {conv32_cv2_0_2->getOutput(0), conv32_cv3_0_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat32_0 = network->addConcatenation(inputTensors32_0, 2);

    // out1 location
    nvinfer1::IElementWiseLayer* conv32_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv27->getOutput(0), c2, {3, 3}, 1, "model.32.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv32_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv32_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.32.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv32_cv2_1_2 =
            network->addConvolutionNd(*conv32_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.32.cv2.1.2.weight"], weightMap["model.32.cv2.1.2.bias"]);
    conv32_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv32_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    // out1 classes
    auto* conv32_cv3_1_0_0 = DWConv(network, weightMap, *conv27->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.32.cv3.1.0.0");
    nvinfer1::IElementWiseLayer* conv32_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv32_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.32.cv3.1.0.1");
    auto* conv32_cv3_1_1_0 =
            DWConv(network, weightMap, *conv32_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.32.cv3.1.1.0");
    nvinfer1::IElementWiseLayer* conv32_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv32_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.32.cv3.1.1.1");
    nvinfer1::IConvolutionLayer* conv32_cv3_1_1_2 =
            network->addConvolutionNd(*conv32_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.32.cv3.1.2.weight"], weightMap["model.32.cv3.1.2.bias"]);
    conv32_cv3_1_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv32_cv3_1_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});

    nvinfer1::ITensor* inputTensors32_1[] = {conv32_cv2_1_2->getOutput(0), conv32_cv3_1_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat32_1 = network->addConcatenation(inputTensors32_1, 2);

    // out2 location
    nvinfer1::IElementWiseLayer* conv32_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv31->getOutput(0), c2, {3, 3}, 1, "model.32.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv32_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv32_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.32.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv32_cv2_2_2 =
            network->addConvolutionNd(*conv32_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.32.cv2.2.2.weight"], weightMap["model.32.cv2.2.2.bias"]);
    conv32_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv32_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    // out2 classes
    auto* conv32_cv3_2_0_0 = DWConv(network, weightMap, *conv31->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.32.cv3.2.0.0");
    nvinfer1::IElementWiseLayer* conv32_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv32_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.32.cv3.2.0.1");
    auto* conv32_cv3_2_1_0 =
            DWConv(network, weightMap, *conv32_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.32.cv3.2.1.0");
    nvinfer1::IElementWiseLayer* conv32_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv32_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.32.cv3.2.1.1");
    nvinfer1::IConvolutionLayer* conv32_cv3_2_1_2 =
            network->addConvolutionNd(*conv32_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.32.cv3.2.2.weight"], weightMap["model.32.cv3.2.2.bias"]);
    conv32_cv3_2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv32_cv3_2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    nvinfer1::ITensor* inputTensor32_2[] = {conv32_cv2_2_2->getOutput(0), conv32_cv3_2_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat32_2 = network->addConcatenation(inputTensor32_2, 2);

    // ============================================ yolov12  detect =========================================
    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle32_0 = network->addShuffle(*cat32_0->getOutput(0));
    shuffle32_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split32_0_0 = network->addSlice(
            *shuffle32_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split32_0_1 =
            network->addSlice(*shuffle32_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl32_0 =
            DFL(network, weightMap, *split32_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.32.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor32_dfl_0[] = {dfl32_0->getOutput(0), split32_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat32_dfl_0 = network->addConcatenation(inputTensor32_dfl_0, 2);
    cat32_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle32_1 = network->addShuffle(*cat32_1->getOutput(0));
    shuffle32_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split32_1_0 = network->addSlice(
            *shuffle32_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split32_1_1 =
            network->addSlice(*shuffle32_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl32_1 =
            DFL(network, weightMap, *split32_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.32.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor32_dfl_1[] = {dfl32_1->getOutput(0), split32_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat32_dfl_1 = network->addConcatenation(inputTensor32_dfl_1, 2);
    cat32_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer* shuffle32_2 = network->addShuffle(*cat32_2->getOutput(0));
    shuffle32_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split32_2_0 = network->addSlice(
            *shuffle32_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split32_2_1 =
            network->addSlice(*shuffle32_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                              nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl32_2 =
            DFL(network, weightMap, *split32_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.32.dfl.conv.weight");
    nvinfer1::ITensor* inputTensor32_dfl_2[] = {dfl32_2->getOutput(0), split32_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat32_dfl_2 = network->addConcatenation(inputTensor32_dfl_2, 2);
    cat32_dfl_2->setAxis(1);
    std::cout << " There are  " << weightMap.size() << "  layers parameters in the network!!!" << endl;

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat32_dfl_0, cat32_dfl_1, cat32_dfl_2},
                         strides, stridesLength, false, false, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(kBatchSize, kInputW, kInputH, kInputQuantizationFolder,
                                                  "int8calib.table", kInputTensorName);
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


