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

void calculateStrides(nvinfer1::IElementWiseLayer* conv_layers[], int size, int reference_size, int strides[]) {
    for (int i = 0; i < size; ++i) {
        nvinfer1::ILayer* layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

nvinfer1::IHostMemory* buildEngineYolo11Cls(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            std::string& type, int max_channels) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // ****************************************** YOLO11 INPUT **********************************************
    nvinfer1::ITensor* data =
            network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kClsInputH, kClsInputW});
    assert(data);

    // ***************************************** YOLO11 BACKBONE ********************************************
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");
    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }
    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");
    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");
    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 =
            C3K2(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C3K2(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.8");
    auto* conv9 = C2PSA(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                        get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.9");

    // ********************************************* YOLO11 HEAD *********************************************

    auto conv_class = convBnSiLU(network, weightMap, *conv9->getOutput(0), 1280, {1, 1}, 1, "model.10.conv");
    // Adjusted code
    nvinfer1::Dims dims =
            conv_class->getOutput(0)->getDimensions();  // Obtain the dimensions of the output of conv_class
    assert(dims.nbDims == 4);  // Make sure there are exactly 3 dimensions (channels, height, width)

    nvinfer1::IPoolingLayer* pool2 = network->addPoolingNd(*conv_class->getOutput(0), nvinfer1::PoolingType::kAVERAGE,
                                                           nvinfer1::DimsHW{dims.d[2], dims.d[3]});
    assert(pool2);

    // Fully connected layer declaration
    auto shuffle_0 = network->addShuffle(*pool2->getOutput(0));
    shuffle_0->setReshapeDimensions(nvinfer1::Dims2{kBatchSize, 1280});
    auto linear_weight = weightMap["model.10.linear.weight"];
    auto constant_weight = network->addConstant(nvinfer1::Dims2{kClsNumClass, 1280}, linear_weight);
    auto constant_bias =
            network->addConstant(nvinfer1::Dims2{kBatchSize, kClsNumClass}, weightMap["model.10.linear.bias"]);
    auto linear_matrix_multipy =
            network->addMatrixMultiply(*shuffle_0->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                       *constant_weight->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    auto yolo = network->addElementWise(*linear_matrix_multipy->getOutput(0), *constant_bias->getOutput(0),
                                        nvinfer1::ElementWiseOperation::kSUM);
    assert(yolo);

    // Set the name for the output tensor and mark it as network output
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    // Set the maximum batch size and workspace size
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

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

nvinfer1::IHostMemory* buildEngineYolo11Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLO11 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO11 BACKBONE  ********************************************
    *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");
    // 11233
    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }
    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");
    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");
    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 =
            C3K2(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C3K2(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = C2PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");
    /*******************************************************************************************************
    *********************************************  YOLO11 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    nvinfer1::IElementWiseLayer* conv13 =
            C3K2(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    nvinfer1::IElementWiseLayer* conv16 =
            C3K2(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.16");

    nvinfer1::IElementWiseLayer* conv17 = convBnSiLU(network, weightMap, *conv16->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    nvinfer1::IElementWiseLayer* conv19 =
            C3K2(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.19");

    nvinfer1::IElementWiseLayer* conv20 = convBnSiLU(network, weightMap, *conv19->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    nvinfer1::IElementWiseLayer* conv22 =
            C3K2(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLO11 OUTPUT  ******************************************
    *******************************************************************************************************/
    // c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kNumClass, 100));

    // output0
    nvinfer1::IElementWiseLayer* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 =
            network->addConvolutionNd(*conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.0.2.weight"], weightMap["model.23.cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = DWConv(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.0.0.0");
    auto* conv23_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.0.1");
    auto* conv23_cv3_0_1_0 =
            DWConv(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.0.1.0");
    auto* conv23_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_0_2 =
            network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.0.2.weight"], weightMap["model.23.cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 =
            network->addConvolutionNd(*conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.1.2.weight"], weightMap["model.23.cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = DWConv(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.1.0.0");
    auto* conv23_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.0.1");
    auto* conv23_cv3_1_1_0 =
            DWConv(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.1.1.0");
    auto* conv23_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_1_2 =
            network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.1.2.weight"], weightMap["model.23.cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 =
            network->addConvolutionNd(*conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.2.2.weight"], weightMap["model.23.cv2.2.2.bias"]);
    conv23_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_2_0_0 = DWConv(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.23.cv3.2.0.0");
    auto* conv23_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.0.1");
    auto* conv23_cv3_2_1_0 =
            DWConv(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.2.1.0");
    auto* conv23_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_2_2 =
            network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.2.2.weight"], weightMap["model.23.cv3.2.2.bias"]);
    conv23_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLO11 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
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
    nvinfer1::ITensor* inputTensor22_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);
    cat22_dfl_0->setAxis(1);

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
    nvinfer1::ITensor* inputTensor22_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);
    cat22_dfl_1->setAxis(1);

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
    nvinfer1::ITensor* inputTensor22_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);
    cat22_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
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

static nvinfer1::IElementWiseLayer* convBnSiLUProto(nvinfer1::INetworkDefinition* network,
                                                    std::map<std::string, nvinfer1::Weights> weightMap,
                                                    nvinfer1::ITensor& input, int ch, int k, int s, int p,
                                                    std::string lname) {
    nvinfer1::Weights bias_empty{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer* conv =
            network->addConvolutionNd(input, ch, nvinfer1::DimsHW{k, k}, weightMap[lname + ".conv.weight"], bias_empty);
    assert(conv);
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
    auto cv1 = convBnSiLU(network, weightMap, input, mid_channel, {3, 3}, 1, "model.23.proto.cv1");
    //    float *convTranpsose_bais = (float *) weightMap["model.23.proto.upsample.bias"].values;
    //    int convTranpsose_bais_len = weightMap["model.23.proto.upsample.bias"].count;
    //    nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, convTranpsose_bais, convTranpsose_bais_len};
    auto convTranpsose = network->addDeconvolutionNd(*cv1->getOutput(0), mid_channel, nvinfer1::DimsHW{2, 2},
                                                     weightMap["model.23.proto.upsample.weight"],
                                                     weightMap["model.23.proto.upsample.bias"]);
    assert(convTranpsose);
    convTranpsose->setStrideNd(nvinfer1::DimsHW{2, 2});
    convTranpsose->setPadding(nvinfer1::DimsHW{0, 0});
    auto cv2 =
            convBnSiLU(network, weightMap, *convTranpsose->getOutput(0), mid_channel, {3, 3}, 1, "model.23.proto.cv2");
    auto cv3 = convBnSiLUProto(network, weightMap, *cv2->getOutput(0), 32, 1, 1, 0, "model.23.proto.cv3");
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
    } else if (algo_type == "obb") {
        nm_nk = kObbNe;
        c4 = std::max(get_width(256, gw, max_channels) / 4, nm_nk);
    } else {
        std::cerr << "Unknown algo type: " << algo_type << std::endl;
        return nullptr;
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

nvinfer1::IHostMemory* buildEngineYolo11Seg(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLO11 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO11 BACKBONE  ********************************************
    *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");
    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }
    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");
    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");
    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 =
            C3K2(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C3K2(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = C2PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");

    /*******************************************************************************************************
    *********************************************  YOLO11 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    nvinfer1::IElementWiseLayer* conv13 =
            C3K2(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    nvinfer1::IElementWiseLayer* conv16 =
            C3K2(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.16");

    nvinfer1::IElementWiseLayer* conv17 = convBnSiLU(network, weightMap, *conv16->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    nvinfer1::IElementWiseLayer* conv19 =
            C3K2(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.19");

    nvinfer1::IElementWiseLayer* conv20 = convBnSiLU(network, weightMap, *conv19->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    nvinfer1::IElementWiseLayer* conv22 =
            C3K2(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLO11 OUTPUT  ******************************************
    *******************************************************************************************************/
    // c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kNumClass, 100));

    // output0
    nvinfer1::IElementWiseLayer* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 =
            network->addConvolutionNd(*conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.0.2.weight"], weightMap["model.23.cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = DWConv(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.0.0.0");
    auto* conv23_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.0.1");
    auto* conv23_cv3_0_1_0 =
            DWConv(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.0.1.0");
    auto* conv23_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_0_2 =
            network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.0.2.weight"], weightMap["model.23.cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 =
            network->addConvolutionNd(*conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.1.2.weight"], weightMap["model.23.cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = DWConv(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.1.0.0");
    auto* conv23_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.0.1");
    auto* conv23_cv3_1_1_0 =
            DWConv(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.1.1.0");
    auto* conv23_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_1_2 =
            network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.1.2.weight"], weightMap["model.23.cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 =
            network->addConvolutionNd(*conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.2.2.weight"], weightMap["model.23.cv2.2.2.bias"]);
    conv23_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_2_0_0 = DWConv(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.23.cv3.2.0.0");
    auto* conv23_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.0.1");
    auto* conv23_cv3_2_1_0 =
            DWConv(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.2.1.0");
    auto* conv23_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_2_2 =
            network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.2.2.weight"], weightMap["model.23.cv3.2.2.bias"]);
    conv23_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLO11 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
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

    // det0
    auto proto_coef_0 = cv4_conv_combined(network, weightMap, *conv16->getOutput(0), "model.23.cv4.0",
                                          (kInputH / strides[0]) * (kInputW / strides[0]), gw, "seg", max_channels);
    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0),
                                                proto_coef_0->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 3);
    cat23_dfl_0->setAxis(1);

    // det1
    auto proto_coef_1 = cv4_conv_combined(network, weightMap, *conv19->getOutput(0), "model.23.cv4.1",
                                          (kInputH / strides[1]) * (kInputW / strides[1]), gw, "seg", max_channels);
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0),
                                                proto_coef_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 3);
    cat23_dfl_1->setAxis(1);

    // det2
    auto proto_coef_2 = cv4_conv_combined(network, weightMap, *conv22->getOutput(0), "model.23.cv4.2",
                                          (kInputH / strides[2]) * (kInputW / strides[2]), gw, "seg", max_channels);
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0),
                                                proto_coef_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 3);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2},
                         strides, stridesLength, true, false, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    auto proto = Proto(network, weightMap, *conv16->getOutput(0), "model.23.proto", gw, max_channels);
    proto->getOutput(0)->setName(kProtoTensorName);
    network->markOutput(*proto->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

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

nvinfer1::IHostMemory* buildEngineYolo11Pose(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                             int& max_channels, std::string& type) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLO11 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO11 BACKBONE  ********************************************
    *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");
    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }
    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");
    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");
    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 =
            C3K2(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C3K2(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = C2PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");
    /*******************************************************************************************************
    *********************************************  YOLO11 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    nvinfer1::IElementWiseLayer* conv13 =
            C3K2(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    nvinfer1::IElementWiseLayer* conv16 =
            C3K2(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.16");

    nvinfer1::IElementWiseLayer* conv17 = convBnSiLU(network, weightMap, *conv16->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    nvinfer1::IElementWiseLayer* conv19 =
            C3K2(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.19");

    nvinfer1::IElementWiseLayer* conv20 = convBnSiLU(network, weightMap, *conv19->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    nvinfer1::IElementWiseLayer* conv22 =
            C3K2(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLO11 OUTPUT  ******************************************
    *******************************************************************************************************/
    // c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kPoseNumClass, 100));

    // output0
    nvinfer1::IElementWiseLayer* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 =
            network->addConvolutionNd(*conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.0.2.weight"], weightMap["model.23.cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = DWConv(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.0.0.0");
    auto* conv23_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.0.1");
    auto* conv23_cv3_0_1_0 =
            DWConv(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.0.1.0");
    auto* conv23_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_0_2 =
            network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.0.2.weight"], weightMap["model.23.cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 =
            network->addConvolutionNd(*conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.1.2.weight"], weightMap["model.23.cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = DWConv(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.1.0.0");
    auto* conv23_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.0.1");
    auto* conv23_cv3_1_1_0 =
            DWConv(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.1.1.0");
    auto* conv23_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_1_2 =
            network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.1.2.weight"], weightMap["model.23.cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 =
            network->addConvolutionNd(*conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.2.2.weight"], weightMap["model.23.cv2.2.2.bias"]);
    conv23_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_2_0_0 = DWConv(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.23.cv3.2.0.0");
    auto* conv23_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.0.1");
    auto* conv23_cv3_2_1_0 =
            DWConv(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.2.1.0");
    auto* conv23_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_2_2 =
            network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kPoseNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.2.2.weight"], weightMap["model.23.cv3.2.2.bias"]);
    conv23_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);
    /*******************************************************************************************************
    *********************************************  YOLO11 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    /**************************************************************************************P3****************************************************************************************************************************************/
    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kPoseNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
            nvinfer1::Dims3{kBatchSize, kPoseNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
            nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
                1, 0, "model.23.dfl.conv.weight");

    // det0
    auto shuffle_conv16 = cv4_conv_combined(network, weightMap, *conv16->getOutput(0), "model.23.cv4.0",
                                            (kInputH / strides[0]) * (kInputW / strides[0]), gw, "pose", max_channels);

    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0),
                                                shuffle_conv16->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 3);
    cat23_dfl_0->setAxis(1);

    /********************************************************************************************P4**********************************************************************************************************************************/
    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kPoseNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
            nvinfer1::Dims3{kBatchSize, kPoseNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
                1, 0, "model.23.dfl.conv.weight");

    // det1
    auto shuffle_conv19 = cv4_conv_combined(network, weightMap, *conv19->getOutput(0), "model.23.cv4.1",
                                            (kInputH / strides[1]) * (kInputW / strides[1]), gw, "pose", max_channels);

    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0),
                                                shuffle_conv19->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 3);
    cat23_dfl_1->setAxis(1);

    /********************************************************************************************P5**********************************************************************************************************************************/
    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kPoseNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
            nvinfer1::Dims3{kBatchSize, kPoseNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
                1, 0, "model.23.dfl.conv.weight");

    // det2
    auto shuffle_conv22 = cv4_conv_combined(network, weightMap, *conv22->getOutput(0), "model.23.cv4.2",
                                            (kInputH / strides[2]) * (kInputW / strides[2]), gw, "pose", max_channels);
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0),
                                                shuffle_conv22->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 3);
    cat23_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2},
                         strides, stridesLength, false, true, false);
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

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

nvinfer1::IHostMemory* buildEngineYolo11Obb(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type) {
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);
    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLO11 INPUT  **********************************************
    *******************************************************************************************************/
    nvinfer1::ITensor* data =
            network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kObbInputH, kObbInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO11 BACKBONE  ********************************************
    *******************************************************************************************************/
    nvinfer1::IElementWiseLayer* conv0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");
    nvinfer1::IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");
    // 11233
    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }
    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");
    nvinfer1::IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");
    // 22466
    nvinfer1::IElementWiseLayer* conv4 =
            C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");
    nvinfer1::IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");
    // 22466
    nvinfer1::IElementWiseLayer* conv6 =
            C3K2(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.6");
    nvinfer1::IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");
    // 11233
    nvinfer1::IElementWiseLayer* conv8 =
            C3K2(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.8");
    nvinfer1::IElementWiseLayer* conv9 =
            SPPF(network, weightMap, *conv8->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 5, "model.9");
    auto* conv10 = C2PSA(network, weightMap, *conv9->getOutput(0), get_width(1024, gw, max_channels),
                         get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");
    /*******************************************************************************************************
    *********************************************  YOLO11 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample11->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensor12, 2);

    nvinfer1::IElementWiseLayer* conv13 =
            C3K2(network, weightMap, *cat12->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*conv13->getOutput(0));
    assert(upsample14);
    upsample14->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensor15[] = {upsample14->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensor15, 2);

    nvinfer1::IElementWiseLayer* conv16 =
            C3K2(network, weightMap, *cat15->getOutput(0), get_width(256, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.16");

    nvinfer1::IElementWiseLayer* conv17 = convBnSiLU(network, weightMap, *conv16->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.17");
    nvinfer1::ITensor* inputTensor18[] = {conv17->getOutput(0), conv13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensor18, 2);
    nvinfer1::IElementWiseLayer* conv19 =
            C3K2(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.5, "model.19");

    nvinfer1::IElementWiseLayer* conv20 = convBnSiLU(network, weightMap, *conv19->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.20");
    nvinfer1::ITensor* inputTensor21[] = {conv20->getOutput(0), conv10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensor21, 2);
    nvinfer1::IElementWiseLayer* conv22 =
            C3K2(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.22");

    /*******************************************************************************************************
    *********************************************  YOLO11 OUTPUT  ******************************************
    *******************************************************************************************************/
    // c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
    // c4 = max(ch[0] // 4, self.ne)
    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kObbNumClass, 100));
    int c4 = std::max(get_width(256, gw, max_channels) / 4, kObbNe);

    // output0
    nvinfer1::IElementWiseLayer* conv23_cv2_0_0 =
            convBnSiLU(network, weightMap, *conv16->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.0.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_0_2 =
            network->addConvolutionNd(*conv23_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.0.2.weight"], weightMap["model.23.cv2.0.2.bias"]);
    conv23_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_0_0_0 = DWConv(network, weightMap, *conv16->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.0.0.0");
    auto* conv23_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.0.1");
    auto* conv23_cv3_0_1_0 =
            DWConv(network, weightMap, *conv23_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.0.1.0");
    auto* conv23_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.0.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_0_2 =
            network->addConvolutionNd(*conv23_cv3_0_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.0.2.weight"], weightMap["model.23.cv3.0.2.bias"]);
    conv23_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_0[] = {conv23_cv2_0_2->getOutput(0), conv23_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_0 = network->addConcatenation(inputTensor23_0, 2);

    // output1
    nvinfer1::IElementWiseLayer* conv23_cv2_1_0 =
            convBnSiLU(network, weightMap, *conv19->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_1_2 =
            network->addConvolutionNd(*conv23_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.1.2.weight"], weightMap["model.23.cv2.1.2.bias"]);
    conv23_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_1_0_0 = DWConv(network, weightMap, *conv19->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.23.cv3.1.0.0");
    auto* conv23_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.0.1");
    auto* conv23_cv3_1_1_0 =
            DWConv(network, weightMap, *conv23_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.1.1.0");
    auto* conv23_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.1.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_1_2 =
            network->addConvolutionNd(*conv23_cv3_1_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.1.2.weight"], weightMap["model.23.cv3.1.2.bias"]);
    conv23_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_1[] = {conv23_cv2_1_2->getOutput(0), conv23_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensor23_1, 2);

    // output2
    nvinfer1::IElementWiseLayer* conv23_cv2_2_0 =
            convBnSiLU(network, weightMap, *conv22->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.0");
    nvinfer1::IElementWiseLayer* conv23_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv23_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.23.cv2.2.1");
    nvinfer1::IConvolutionLayer* conv23_cv2_2_2 =
            network->addConvolutionNd(*conv23_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv2.2.2.weight"], weightMap["model.23.cv2.2.2.bias"]);
    conv23_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto* conv23_cv3_2_0_0 = DWConv(network, weightMap, *conv22->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.23.cv3.2.0.0");
    auto* conv23_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.0.1");
    auto* conv23_cv3_2_1_0 =
            DWConv(network, weightMap, *conv23_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.23.cv3.2.1.0");
    auto* conv23_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv23_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.23.cv3.2.1.1");
    nvinfer1::IConvolutionLayer* conv23_cv3_2_2 =
            network->addConvolutionNd(*conv23_cv3_2_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
                                      weightMap["model.23.cv3.2.2.weight"], weightMap["model.23.cv3.2.2.bias"]);
    conv23_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor* inputTensor23_2[] = {conv23_cv2_2_2->getOutput(0), conv23_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensor23_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLO11 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kObbInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer* shuffle23_0 = network->addShuffle(*cat23_0->getOutput(0));
    shuffle23_0->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kObbNumClass, (kObbInputH / strides[0]) * (kObbInputW / strides[0])});
    nvinfer1::ISliceLayer* split23_0_0 =
            network->addSlice(*shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
                              nvinfer1::Dims3{kBatchSize, 64, (kObbInputH / strides[0]) * (kObbInputW / strides[0])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_0_1 = network->addSlice(
            *shuffle23_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
            nvinfer1::Dims3{kBatchSize, kObbNumClass, (kObbInputH / strides[0]) * (kObbInputW / strides[0])},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_0 =
            DFL(network, weightMap, *split23_0_0->getOutput(0), 4,
                (kObbInputH / strides[0]) * (kObbInputW / strides[0]), 1, 1, 0, "model.23.dfl.conv.weight");

    nvinfer1::IShuffleLayer* shuffle23_1 = network->addShuffle(*cat23_1->getOutput(0));
    shuffle23_1->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kObbNumClass, (kObbInputH / strides[1]) * (kObbInputW / strides[1])});
    nvinfer1::ISliceLayer* split23_1_0 =
            network->addSlice(*shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
                              nvinfer1::Dims3{kBatchSize, 64, (kObbInputH / strides[1]) * (kObbInputW / strides[1])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_1_1 = network->addSlice(
            *shuffle23_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
            nvinfer1::Dims3{kBatchSize, kObbNumClass, (kObbInputH / strides[1]) * (kObbInputW / strides[1])},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_1 =
            DFL(network, weightMap, *split23_1_0->getOutput(0), 4,
                (kObbInputH / strides[1]) * (kObbInputW / strides[1]), 1, 1, 0, "model.23.dfl.conv.weight");

    nvinfer1::IShuffleLayer* shuffle23_2 = network->addShuffle(*cat23_2->getOutput(0));
    shuffle23_2->setReshapeDimensions(
            nvinfer1::Dims3{kBatchSize, 64 + kObbNumClass, (kObbInputH / strides[2]) * (kObbInputW / strides[2])});
    nvinfer1::ISliceLayer* split23_2_0 =
            network->addSlice(*shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
                              nvinfer1::Dims3{kBatchSize, 64, (kObbInputH / strides[2]) * (kObbInputW / strides[2])},
                              nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23_2_1 = network->addSlice(
            *shuffle23_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
            nvinfer1::Dims3{kBatchSize, kObbNumClass, (kObbInputH / strides[2]) * (kObbInputW / strides[2])},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer* dfl23_2 =
            DFL(network, weightMap, *split23_2_0->getOutput(0), 4,
                (kObbInputH / strides[2]) * (kObbInputW / strides[2]), 1, 1, 0, "model.23.dfl.conv.weight");

    // det0
    auto shuffle_conv16 =
            cv4_conv_combined(network, weightMap, *conv16->getOutput(0), "model.23.cv4.0",
                              (kObbInputH / strides[0]) * (kObbInputW / strides[0]), gw, "obb", max_channels);

    nvinfer1::ITensor* inputTensor23_dfl_0[] = {dfl23_0->getOutput(0), split23_0_1->getOutput(0),
                                                shuffle_conv16->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_0 = network->addConcatenation(inputTensor23_dfl_0, 3);
    cat23_dfl_0->setAxis(1);

    // det1
    auto shuffle_conv19 =
            cv4_conv_combined(network, weightMap, *conv19->getOutput(0), "model.23.cv4.1",
                              (kObbInputH / strides[1]) * (kObbInputW / strides[1]), gw, "obb", max_channels);
    nvinfer1::ITensor* inputTensor23_dfl_1[] = {dfl23_1->getOutput(0), split23_1_1->getOutput(0),
                                                shuffle_conv19->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_1 = network->addConcatenation(inputTensor23_dfl_1, 3);
    cat23_dfl_1->setAxis(1);

    // det2
    auto shuffle_conv22 =
            cv4_conv_combined(network, weightMap, *conv22->getOutput(0), "model.23.cv4.2",
                              (kObbInputH / strides[2]) * (kObbInputW / strides[2]), gw, "obb", max_channels);
    nvinfer1::ITensor* inputTensor23_dfl_2[] = {dfl23_2->getOutput(0), split23_2_1->getOutput(0),
                                                shuffle_conv22->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_dfl_2 = network->addConcatenation(inputTensor23_dfl_2, 3);
    cat23_dfl_2->setAxis(1);

    // yolo layer
    nvinfer1::IPluginV2Layer* yolo =
            addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer*>{cat23_dfl_0, cat23_dfl_1, cat23_dfl_2},
                         strides, stridesLength, false, false, true);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    auto* calibrator = new Int8EntropyCalibrator2(1, kObbInputW, kObbInputH, kInputQuantizationFolder,
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
