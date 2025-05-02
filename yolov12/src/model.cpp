#include <math.h>
#include <iostream>
#include "block.h"
// Todo: Add the followings
/*
#include "calibrator.h"
*/
#include "config.h"
#include "model.h"

static int get_width(int x, float gw, int max_channels, int divisor = 8)
{
    auto channel = std::min(x, max_channels);
    channel = int(ceil((channel * gw) / divisor)) * divisor;
    return channel;
}

static int get_depth(int x, float gd)
{
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
        --r;
    return std::max<int>(r, 1);
}

void calculateStrides(nvinfer1::IElementWiseLayer *conv_layers[], int size, int reference_size, int strides[])
{
    for (int i = 0; i < size; ++i)
    {
        nvinfer1::ILayer *layer = conv_layers[i];
        nvinfer1::Dims dims = layer->getOutput(0)->getDimensions();
        int feature_map_size = dims.d[2];
        strides[i] = reference_size / feature_map_size;
    }
}

nvinfer1::IHostMemory *buildEngineYolo12Det(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config,
                                            nvinfer1::DataType dt, const std::string &wts_path, float &gd, float &gw,
                                            int &max_channels, std::string &type)
{
    std::cout << "Building Yolo12 Det engine..." << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);

    //	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
    ******************************************  YOLO12 INPUT  **********************************************
    *******************************************************************************************************/

    nvinfer1::ITensor *data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO12 BACKBONE  ********************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer *conv0 =
        convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");

    nvinfer1::IElementWiseLayer *conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0),
                                                    get_width(128, gw, max_channels), {3, 3}, 2, "model.1");

    bool c3k = false;
    if (type == "m" || type == "l" || type == "x")
    {
        c3k = true;
    }

    nvinfer1::IElementWiseLayer *conv2 =
        C3K2(network, weightMap, *conv1->getOutput(0), get_width(128, gw, max_channels),
             get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.2");

    nvinfer1::IElementWiseLayer *conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                    get_width(256, gw, max_channels), {3, 3}, 2, "model.3");

    nvinfer1::IElementWiseLayer *conv4 =
        C3K2(network, weightMap, *conv3->getOutput(0), get_width(256, gw, max_channels),
             get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, 0.25, "model.4");

    nvinfer1::IElementWiseLayer *conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0),
                                                    get_width(512, gw, max_channels), {3, 3}, 2, "model.5");

    nvinfer1::ILayer *conv6 = A2C2f(network, weightMap, *conv5->getOutput(0), get_width(512, gw, max_channels),
                                    get_width(512, gw, max_channels), 4, true, 4, true, 2.0, 0.25, 1, true, "model.6");

    nvinfer1::IElementWiseLayer *conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0),
                                                    get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");

    nvinfer1::ILayer *conv8 = A2C2f(network, weightMap, *conv7->getOutput(0), get_width(1024, gw, max_channels),
                                    get_width(1024, gw, max_channels), 4, true, 1, true, 2.0, 0.25, 1, true, "model.8");

    /*******************************************************************************************************
    *********************************************  YOLO12 HEAD  ********************************************
    *******************************************************************************************************/

    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer *upsample9 = network->addResize(*conv8->getOutput(0));
    assert(upsample9);
    upsample9->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample9->setScales(scale, 4);

    nvinfer1::ITensor *inputTensors10[] = {upsample9->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat10 = network->addConcatenation(inputTensors10, 2);

    nvinfer1::ILayer *conv11 =
        A2C2f(network, weightMap, *cat10->getOutput(0), get_width(1024, gw, max_channels),
              get_width(512, gw, max_channels), 4, false, 1, true, 2.0, 0.25, 1, true, "model.11");

    nvinfer1::IResizeLayer *upsample12 = network->addResize(*conv11->getOutput(0));
    assert(upsample12);
    upsample12->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
    upsample12->setScales(scale, 4);
    nvinfer1::ITensor *inputTensors13[] = {upsample12->getOutput(0), conv4->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat13 = network->addConcatenation(inputTensors13, 2);
    nvinfer1::ILayer *conv14 =
        A2C2f(network, weightMap, *cat13->getOutput(0), get_width(256, gw, max_channels),
              get_width(256, gw, max_channels), 4, false, 1, true, 2.0, 0.25, 1, true, "model.14");

    nvinfer1::IElementWiseLayer *conv15 = convBnSiLU(network, weightMap, *conv14->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.15");
    nvinfer1::ITensor *inputTensors16[] = {conv15->getOutput(0), conv11->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat16 = network->addConcatenation(inputTensors16, 2);

    nvinfer1::ILayer *conv17 =
        A2C2f(network, weightMap, *cat16->getOutput(0), get_width(512, gw, max_channels),
              get_width(512, gw, max_channels), 4, false, 1, true, 2.0, 0.25, 1, true, "model.17");

    nvinfer1::IElementWiseLayer *conv18 = convBnSiLU(network, weightMap, *conv17->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.18");
    nvinfer1::ITensor *inputTensors19[] = {conv18->getOutput(0), conv8->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat19 = network->addConcatenation(inputTensors19, 2);

    nvinfer1::IElementWiseLayer *conv20 =
        C3K2(network, weightMap, *cat19->getOutput(0), get_width(1024, gw, max_channels),
             get_width(1024, gw, max_channels), get_depth(2, gd), true, true, 0.5, "model.20");

    /*******************************************************************************************************
    *********************************************  YOLO12 OUTPUT  ******************************************
    *******************************************************************************************************/

    int c2 = std::max(std::max(16, get_width(256, gw, max_channels) / 4), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kNumClass, 100));

    // output 0
    nvinfer1::IElementWiseLayer *conv21_cv2_0_0 =
        convBnSiLU(network, weightMap, *conv14->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.0.0");
    nvinfer1::IElementWiseLayer *conv21_cv2_0_1 =
        convBnSiLU(network, weightMap, *conv21_cv2_0_0->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.0.1");

    nvinfer1::IConvolutionLayer *conv21_cv2_0_2 =
        network->addConvolutionNd(*conv21_cv2_0_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                  weightMap["model.21.cv2.0.2.weight"], weightMap["model.21.cv2.0.2.bias"]);
    conv21_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    auto *conv21_cv3_0_0_0 = DWConv(network, weightMap, *conv14->getOutput(0), get_width(256, gw, max_channels), {3, 3},
                                    1, "model.21.cv3.0.0.0");
    auto *conv21_cv3_0_0_1 =
        convBnSiLU(network, weightMap, *conv21_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.0.0.1");
    auto *conv21_cv3_0_1_0 =
        DWConv(network, weightMap, *conv21_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1, "model.21.cv3.0.1.0");
    auto *conv21_cv3_0_1_1 =
        convBnSiLU(network, weightMap, *conv21_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.0.1.1");
    nvinfer1::IConvolutionLayer *conv21_cv3_0_2 =
        network->addConvolutionNd(*conv21_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                  weightMap["model.21.cv3.0.2.weight"], weightMap["model.21.cv3.0.2.bias"]);
    conv21_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});

    nvinfer1::ITensor *inputTensor21_0[] = {conv21_cv2_0_2->getOutput(0), conv21_cv3_0_2->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat21_0 = network->addConcatenation(inputTensor21_0, 2);

    // output 1
    nvinfer1::IElementWiseLayer *conv21_cv2_1_0 =
        convBnSiLU(network, weightMap, *conv17->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.1.0");
    nvinfer1::IElementWiseLayer *conv21_cv2_1_1 =
        convBnSiLU(network, weightMap, *conv21_cv2_1_0->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.1.1");
    nvinfer1::IConvolutionLayer *conv21_cv2_1_2 =
        network->addConvolutionNd(*conv21_cv2_1_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                  weightMap["model.21.cv2.1.2.weight"], weightMap["model.21.cv2.1.2.bias"]);
    conv21_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto *conv21_cv3_1_0_0 = DWConv(network, weightMap, *conv17->getOutput(0), get_width(512, gw, max_channels), {3, 3},
                                    1, "model.21.cv3.1.0.0");
    auto *conv21_cv3_1_0_1 =
        convBnSiLU(network, weightMap, *conv21_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.1.0.1");
    auto *conv21_cv3_1_1_0 =
        DWConv(network, weightMap, *conv21_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1, "model.21.cv3.1.1.0");
    auto *conv21_cv3_1_1_1 =
        convBnSiLU(network, weightMap, *conv21_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.1.1.1");
    nvinfer1::IConvolutionLayer *conv21_cv3_1_2 =
        network->addConvolutionNd(*conv21_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                  weightMap["model.21.cv3.1.2.weight"], weightMap["model.21.cv3.1.2.bias"]);
    conv21_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor *inputTensor21_1[] = {conv21_cv2_1_2->getOutput(0), conv21_cv3_1_2->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat21_1 = network->addConcatenation(inputTensor21_1, 2);

    // output 2
    nvinfer1::IElementWiseLayer *conv21_cv2_2_0 =
        convBnSiLU(network, weightMap, *conv20->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.2.0");
    nvinfer1::IElementWiseLayer *conv21_cv2_2_1 =
        convBnSiLU(network, weightMap, *conv21_cv2_2_0->getOutput(0), c2, {3, 3}, 1, "model.21.cv2.2.1");
    nvinfer1::IConvolutionLayer *conv21_cv2_2_2 =
        network->addConvolutionNd(*conv21_cv2_2_1->getOutput(0), 64, nvinfer1::DimsHW{1, 1},
                                  weightMap["model.21.cv2.2.2.weight"], weightMap["model.21.cv2.2.2.bias"]);
    conv21_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    auto *conv21_cv3_2_0_0 = DWConv(network, weightMap, *conv20->getOutput(0), get_width(1024, gw, max_channels),
                                    {3, 3}, 1, "model.21.cv3.2.0.0");
    auto *conv21_cv3_2_0_1 =
        convBnSiLU(network, weightMap, *conv21_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.2.0.1");
    auto *conv21_cv3_2_1_0 =
        DWConv(network, weightMap, *conv21_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1, "model.21.cv3.2.1.0");
    auto *conv21_cv3_2_1_1 =
        convBnSiLU(network, weightMap, *conv21_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1, "model.21.cv3.2.1.1");
    nvinfer1::IConvolutionLayer *conv21_cv3_2_2 =
        network->addConvolutionNd(*conv21_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
                                  weightMap["model.21.cv3.2.2.weight"], weightMap["model.21.cv3.2.2.bias"]);
    conv21_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv21_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::ITensor *inputTensor21_2[] = {conv21_cv2_2_2->getOutput(0), conv21_cv3_2_2->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat21_2 = network->addConcatenation(inputTensor21_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLO12 DETECT  ******************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer *conv_layers[] = {conv3, conv5, conv7};
    int strides[sizeof(conv_layers) / sizeof(conv_layers[0])];
    calculateStrides(conv_layers, sizeof(conv_layers) / sizeof(conv_layers[0]), kInputH, strides);
    int stridesLength = sizeof(strides) / sizeof(int);

    nvinfer1::IShuffleLayer *shuffle21_0 = network->addShuffle(*cat21_0->getOutput(0));
    shuffle21_0->setReshapeDimensions(
        nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])});
    nvinfer1::ISliceLayer *split21_0_0 = network->addSlice(
        *shuffle21_0->getOutput(0), nvinfer1::Dims3{0, 0, 0},
        nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[0]) * (kInputW / strides[0])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer *split21_0_1 =
        network->addSlice(*shuffle21_0->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                          nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[0]) * (kInputW / strides[0])},
                          nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IShuffleLayer *dfl21_0 =
        DFL(network, weightMap, *split21_0_0->getOutput(0), 4, (kInputH / strides[0]) * (kInputW / strides[0]), 1,
            1, 0, "model.21.dfl.conv.weight");
    nvinfer1::ITensor *inputTensor22_dfl_0[] = {dfl21_0->getOutput(0), split21_0_1->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);
    cat22_dfl_0->setAxis(1);

    nvinfer1::IShuffleLayer *shuffle21_1 = network->addShuffle(*cat21_1->getOutput(0));
    shuffle21_1->setReshapeDimensions(
        nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])});
    nvinfer1::ISliceLayer *split21_1_0 = network->addSlice(
        *shuffle21_1->getOutput(0), nvinfer1::Dims3{0, 0, 0},
        nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[1]) * (kInputW / strides[1])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer *split21_1_1 =
        network->addSlice(*shuffle21_1->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                          nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[1]) * (kInputW / strides[1])},
                          nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer *dfl21_1 =
        DFL(network, weightMap, *split21_1_0->getOutput(0), 4, (kInputH / strides[1]) * (kInputW / strides[1]), 1,
            1, 0, "model.21.dfl.conv.weight");
    nvinfer1::ITensor *inputTensor22_dfl_1[] = {dfl21_1->getOutput(0), split21_1_1->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);
    cat22_dfl_1->setAxis(1);

    nvinfer1::IShuffleLayer *shuffle21_2 = network->addShuffle(*cat21_2->getOutput(0));
    shuffle21_2->setReshapeDimensions(
        nvinfer1::Dims3{kBatchSize, 64 + kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])});
    nvinfer1::ISliceLayer *split21_2_0 = network->addSlice(
        *shuffle21_2->getOutput(0), nvinfer1::Dims3{0, 0, 0},
        nvinfer1::Dims3{kBatchSize, 64, (kInputH / strides[2]) * (kInputW / strides[2])}, nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer *split21_2_1 =
        network->addSlice(*shuffle21_2->getOutput(0), nvinfer1::Dims3{0, 64, 0},
                          nvinfer1::Dims3{kBatchSize, kNumClass, (kInputH / strides[2]) * (kInputW / strides[2])},
                          nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IShuffleLayer *dfl21_2 =
        DFL(network, weightMap, *split21_2_0->getOutput(0), 4, (kInputH / strides[2]) * (kInputW / strides[2]), 1,
            1, 0, "model.21.dfl.conv.weight");
    nvinfer1::ITensor *inputTensor22_dfl_2[] = {dfl21_2->getOutput(0), split21_2_1->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);
    cat22_dfl_2->setAxis(1);

    nvinfer1::IPluginV2Layer *yolo = addYoLoLayer(network, std::vector<nvinfer1::IConcatenationLayer *>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2},
                                                  strides, stridesLength, false, false, false);

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::IHostMemory *serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return serialized_model;
}