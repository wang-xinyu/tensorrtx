#include <math.h>
#include <iostream>

#include "block.h"
// #include "calibrator.h"
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

nvinfer1::IHostMemory* buildEngineYolo26Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type)

{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
     ******************************************  YOLO26 INPUT  **********************************************
     *******************************************************************************************************/

    nvinfer1::ITensor* data = network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kInputH, kInputW});
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLO26 BACKBONE  ********************************************
    *******************************************************************************************************/

    nvinfer1::IElementWiseLayer* block0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");

    nvinfer1::IElementWiseLayer* block1 = convBnSiLU(network, weightMap, *block0->getOutput(0),
                                                     get_width(128, gw, max_channels), {3, 3}, 2, "model.1");

    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }

    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *block1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.2");

    nvinfer1::IElementWiseLayer* block3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.3");

    nvinfer1::IElementWiseLayer* block4 =
            C3K2(network, weightMap, *block3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.4");

    nvinfer1::IElementWiseLayer* block5 = convBnSiLU(network, weightMap, *block4->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.5");

    nvinfer1::IElementWiseLayer* block6 =
            C3K2(network, weightMap, *block5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.6");

    nvinfer1::IElementWiseLayer* block7 = convBnSiLU(network, weightMap, *block6->getOutput(0),
                                                     get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");

    nvinfer1::IElementWiseLayer* block8 =
            C3K2(network, weightMap, *block7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* block9 = SPPF(network, weightMap, *block8->getOutput(0),
                                               get_width(1024, gw, max_channels), get_width(1024, gw, max_channels), 5,
                                               true, "model.9");  // TODO: VERIFY THIS BLOCK FOR OTHER YOLO26 MODELS

    nvinfer1::IElementWiseLayer* block10 =
            C2PSA(network, weightMap, *block9->getOutput(0), get_width(1024, gw, max_channels),
                  get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");
    /*******************************************************************************************************
    *********************************************  YOLO26 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*block10->getOutput(0));
    assert(upsample11);

    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);
    nvinfer1::ITensor* inputTensors12[] = {upsample11->getOutput(0), block6->getOutput(0)};

    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensors12, 2);

    nvinfer1::IElementWiseLayer* block13 =
            C3K2(network, weightMap, *cat12->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*block13->getOutput(0));
    assert(upsample14);

    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensors15[] = {upsample14->getOutput(0), block4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensors15, 2);

    nvinfer1::IElementWiseLayer* block16 =
            C3K2(network, weightMap, *cat15->getOutput(0), get_width(512, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.16");

    nvinfer1::IElementWiseLayer* block17 = convBnSiLU(network, weightMap, *block16->getOutput(0),
                                                      get_width(256, gw, max_channels), {3, 3}, 2, "model.17");

    nvinfer1::ITensor* inputTensors18[] = {block17->getOutput(0), block13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensors18, 2);

    nvinfer1::IElementWiseLayer* block19 =
            C3K2(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.19");

    nvinfer1::IElementWiseLayer* block20 = convBnSiLU(network, weightMap, *block19->getOutput(0),
                                                      get_width(512, gw, max_channels), {3, 3}, 2, "model.20");

    nvinfer1::ITensor* inputTensors21[] = {block20->getOutput(0), block10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensors21, 2);

    nvinfer1::IElementWiseLayer* block22 =
            C3K2(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 1, true, true, true, 0.5,
                 "model.22");  // WARN: get_depth(2, gd) changed to 1.

    /*******************************************************************************************************
    *********************************************  YOLO26 OUTPUT  ********************************************
    *******************************************************************************************************/

    int c2 = std::max(std::max(16, get_width(256, gw, max_channels)), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kNumClass, 100));

    /////////////////////////////////////////////////////

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_0_0 =
            convBnSiLU(network, weightMap, *block16->getOutput(0), c2, {3, 3}, 1, "model.23.one2one_cv3.0.0.0", c2);

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.0.0.1", 1);

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_1_0 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1,
                       "model.23.one2one_cv3.0.1.0", c3);

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.0.1.1", 1);

    nvinfer1::IConvolutionLayer* conv23_one2one_cv3_0_2 = network->addConvolutionNd(
            *conv23_one2one_cv3_0_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv3.0.2.weight"], weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_one2one_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv3_0_2->setNbGroups(1);

    nvinfer1::IShuffleLayer* reshape23_3 = network->addShuffle(*conv23_one2one_cv3_0_2->getOutput(0));
    reshape23_3->setReshapeDimensions(nvinfer1::Dims3{1, kNumClass, -1});

    /////////////////////////////////////////////////////

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_0_0 = convBnSiLU(
            network, weightMap, *block19->getOutput(0), c2 * 2, {3, 3}, 1, "model.23.one2one_cv3.1.0.0", c2 * 2);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.1.0.1", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_1_0 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1,
                       "model.23.one2one_cv3.1.1.0", c3);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.1.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv3_1_2 = network->addConvolutionNd(
            *conv23_one2one_cv3_1_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv3.1.2.weight"], weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_one2one_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv3_1_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_4 = network->addShuffle(*conv23_one2one_cv3_1_2->getOutput(0));
    reshape23_4->setReshapeDimensions(nvinfer1::Dims3{1, kNumClass, -1});

    /////////////////////////////////////////////////////
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_0_0;
    if (type == "m" || type == "l" || type == "x") {
        conv23_one2one_cv3_2_0_0 = convBnSiLU(network, weightMap, *block22->getOutput(0), c2 * 2, {3, 3}, 1,
                                              "model.23.one2one_cv3.2.0.0", c2 * 2);
    } else {
        conv23_one2one_cv3_2_0_0 = convBnSiLU(network, weightMap, *block22->getOutput(0), c2 * 4, {3, 3}, 1,
                                              "model.23.one2one_cv3.2.0.0", c2 * 4);
    }

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.2.0.1", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_1_0 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1,
                       "model.23.one2one_cv3.2.1.0", c3);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.2.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv3_2_2 = network->addConvolutionNd(
            *conv23_one2one_cv3_2_1_1->getOutput(0), kNumClass, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv3.2.2.weight"], weightMap["model.23.one2one_cv3.2.2.bias"]);
    conv23_one2one_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv3_2_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_5 = network->addShuffle(*conv23_one2one_cv3_2_2->getOutput(0));
    reshape23_5->setReshapeDimensions(nvinfer1::Dims3{1, kNumClass, -1});

    /////////////////////////////////////////////////////

    nvinfer1::ITensor* inputTensors23_1[] = {reshape23_3->getOutput(0), reshape23_4->getOutput(0),
                                             reshape23_5->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(inputTensors23_1, 3);
    cat23_1->setAxis(2);
    nvinfer1::IActivationLayer* sigmoid23 = network->addActivation(
            *cat23_1->getOutput(0),
            nvinfer1::ActivationType::kSIGMOID);  // TODO: THIS IS UNNESSARY, REMOVE AFTER PLUGIN IS READY

    /////////////////////////////////////////////////////

    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_0_0 =
            convBnSiLU(network, weightMap, *block16->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv2.0.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv2_0_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv2.0.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv2_0_2 = network->addConvolutionNd(
            *conv23_one2one_cv2_0_1->getOutput(0), 4, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv2.0.2.weight"], weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_one2one_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv2_0_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23 = network->addShuffle(*conv23_one2one_cv2_0_2->getOutput(0));
    reshape23->setReshapeDimensions(nvinfer1::Dims3{1, 4, -1});

    /////////////////////////////////////////////////////

    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_1_0 =
            convBnSiLU(network, weightMap, *block19->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv2.1.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv2_1_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv2.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv2_1_2 = network->addConvolutionNd(
            *conv23_one2one_cv2_1_1->getOutput(0), 4, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv2.1.2.weight"], weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_one2one_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv2_1_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_1 = network->addShuffle(*conv23_one2one_cv2_1_2->getOutput(0));
    reshape23_1->setReshapeDimensions(nvinfer1::Dims3{1, 4, -1});

    /////////////////////////////////////////////////////

    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_2_0 =
            convBnSiLU(network, weightMap, *block22->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv2.2.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv2_2_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv2.2.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv2_2_2 = network->addConvolutionNd(
            *conv23_one2one_cv2_2_1->getOutput(0), 4, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv2.2.2.weight"], weightMap["model.23.one2one_cv2.2.2.bias"]);
    conv23_one2one_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv2_2_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_2 = network->addShuffle(*conv23_one2one_cv2_2_2->getOutput(0));
    reshape23_2->setReshapeDimensions(nvinfer1::Dims3{1, 4, -1});

    /////////////////////////////////////////////////////

    nvinfer1::ITensor* inputTensors23[] = {reshape23->getOutput(0), reshape23_1->getOutput(0),
                                           reshape23_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23 = network->addConcatenation(inputTensors23, 3);
    cat23->setAxis(2);

    /////////////////////////////////////////////////////

    nvinfer1::ISliceLayer* slice23_1 = network->addSlice(
            *cat23->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{cat23->getOutput(0)->getDimensions().d[0], cat23->getOutput(0)->getDimensions().d[1] / 2,
                            cat23->getOutput(0)->getDimensions().d[2]},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* slice23 = network->addSlice(
            *cat23->getOutput(0), nvinfer1::Dims3{0, cat23->getOutput(0)->getDimensions().d[1] / 2, 0},
            nvinfer1::Dims3{cat23->getOutput(0)->getDimensions().d[0], cat23->getOutput(0)->getDimensions().d[1] / 2,
                            cat23->getOutput(0)->getDimensions().d[2]},
            nvinfer1::Dims3{1, 1, 1});

    // TODO: MAKE HARDCODED TO AUTOMATIC
    const int anchor_num = cat23->getOutput(0)->getDimensions().d[2];

    std::vector<int> fm_sizes;
    int fm_h_0 = block16->getOutput(0)->getDimensions().d[2];  // P3
    int fm_h_1 = block19->getOutput(0)->getDimensions().d[2];  // P4
    int fm_h_2 = block22->getOutput(0)->getDimensions().d[2];  // P5

    fm_sizes.push_back(fm_h_0);
    fm_sizes.push_back(fm_h_1);
    fm_sizes.push_back(fm_h_2);

    std::vector<int> strides = {kInputH / fm_h_0, kInputH / fm_h_1, kInputH / fm_h_2};
    std::vector<float> grid(anchor_num * 2);
    std::vector<float> stride_vec(anchor_num);
    std::fill(stride_vec.begin(), stride_vec.begin() + fm_sizes[0] * fm_sizes[0], strides[0]);
    std::fill(stride_vec.begin() + fm_sizes[0] * fm_sizes[0],
              stride_vec.begin() + fm_sizes[0] * fm_sizes[0] + fm_sizes[1] * fm_sizes[1], strides[1]);
    std::fill(stride_vec.begin() + fm_sizes[0] * fm_sizes[0] + fm_sizes[1] * fm_sizes[1], stride_vec.end(), strides[2]);

    int idx = 0;
    for (int s = 0; s < fm_sizes.size(); ++s) {
        int h = fm_sizes[s];
        int w = fm_sizes[s];

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                grid[idx] = x + 0.5f;
                grid[idx + anchor_num] = y + 0.5f;

                idx++;
            }
        }
    }

    nvinfer1::Dims gridDims;
    gridDims.nbDims = 3;
    gridDims.d[0] = 1;
    gridDims.d[1] = 2;
    gridDims.d[2] = anchor_num;

    nvinfer1::IConstantLayer* constant_grid = network->addConstant(
            gridDims, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, grid.data(), (int64_t)grid.size()});

    nvinfer1::IElementWiseLayer* conv23_add_1 = network->addElementWise(
            *constant_grid->getOutput(0), *slice23->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::IElementWiseLayer* conv23_sub_1 = network->addElementWise(
            *constant_grid->getOutput(0), *slice23_1->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);

    nvinfer1::ITensor* tensor23[] = {conv23_sub_1->getOutput(0), conv23_add_1->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(tensor23, 2);
    cat23_2->setAxis(1);

    nvinfer1::IConstantLayer* constant_stride = network->addConstant(
            nvinfer1::Dims3{1, 1, anchor_num},
            nvinfer1::Weights{nvinfer1::DataType::kFLOAT, stride_vec.data(), (int64_t)stride_vec.size()});

    nvinfer1::IElementWiseLayer* mul23_2 = network->addElementWise(
            *cat23_2->getOutput(0), *constant_stride->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    ///////////////////////////////////////////////////////////

    nvinfer1::IConcatenationLayer* cat23_3 = network->addConcatenation(
            std::array<nvinfer1::ITensor*, 2>{mul23_2->getOutput(0), sigmoid23->getOutput(0)}.data(), 2);
    cat23_3->setAxis(1);

    nvinfer1::IShuffleLayer* transpose = network->addShuffle(*cat23_3->getOutput(0));
    transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
    // transpose->setReshapeDimensions(nvinfer1::Dims3{1, anchor_num, kNumClass + 4});

    ///////////////////////////////////////////////////////////

    int stridesLength = strides.size();
    nvinfer1::IPluginV2Layer* yolo = addYoloLayer(network, *transpose->getOutput(0), strides, fm_sizes, stridesLength,
                                                  true, false, false, false, anchor_num);
    assert(yolo);

    ///////////////////////////////////////////////////////////

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    // Use setMemoryPoolLimit instead of deprecated setMaxWorkspaceSize
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cerr << "INT8 not supported for YOLO26 model yet." << std::endl;
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

nvinfer1::IHostMemory* buildEngineYolo26Obb(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type)

{
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_path);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    /*******************************************************************************************************
     ******************************************  YOLO26-Obb INPUT  **********************************************
     *******************************************************************************************************/

    nvinfer1::ITensor* data =
            network->addInput(kInputTensorName, dt, nvinfer1::Dims4{kBatchSize, 3, kObbInputH, kObbInputW});
    assert(data);

    nvinfer1::IElementWiseLayer* block0 =
            convBnSiLU(network, weightMap, *data, get_width(64, gw, max_channels), {3, 3}, 2, "model.0");

    nvinfer1::IElementWiseLayer* block1 = convBnSiLU(network, weightMap, *block0->getOutput(0),
                                                     get_width(128, gw, max_channels), {3, 3}, 2, "model.1");

    bool c3k = false;
    if (type == "m" || type == "l" || type == "x") {
        c3k = true;
    }

    nvinfer1::IElementWiseLayer* conv2 =
            C3K2(network, weightMap, *block1->getOutput(0), get_width(128, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.2");

    nvinfer1::IElementWiseLayer* block3 = convBnSiLU(network, weightMap, *conv2->getOutput(0),
                                                     get_width(256, gw, max_channels), {3, 3}, 2, "model.3");

    nvinfer1::IElementWiseLayer* block4 =
            C3K2(network, weightMap, *block3->getOutput(0), get_width(256, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), c3k, true, false, 0.25, "model.4");

    nvinfer1::IElementWiseLayer* block5 = convBnSiLU(network, weightMap, *block4->getOutput(0),
                                                     get_width(512, gw, max_channels), {3, 3}, 2, "model.5");

    nvinfer1::IElementWiseLayer* block6 =
            C3K2(network, weightMap, *block5->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.6");

    nvinfer1::IElementWiseLayer* block7 = convBnSiLU(network, weightMap, *block6->getOutput(0),
                                                     get_width(1024, gw, max_channels), {3, 3}, 2, "model.7");

    nvinfer1::IElementWiseLayer* block8 =
            C3K2(network, weightMap, *block7->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.8");

    nvinfer1::IElementWiseLayer* block9 = SPPF(network, weightMap, *block8->getOutput(0),
                                               get_width(1024, gw, max_channels), get_width(1024, gw, max_channels), 5,
                                               true, "model.9");  // TODO: VERIFY THIS BLOCK FOR OTHER YOLO26 MODELS

    nvinfer1::IElementWiseLayer* block10 =
            C2PSA(network, weightMap, *block9->getOutput(0), get_width(1024, gw, max_channels),
                  get_width(1024, gw, max_channels), get_depth(2, gd), 0.5, "model.10");

    /*******************************************************************************************************
    *********************************************  YOLO26-Obb HEAD  ********************************************
    *******************************************************************************************************/

    float scale[] = {1.0, 1.0, 2.0, 2.0};
    nvinfer1::IResizeLayer* upsample11 = network->addResize(*block10->getOutput(0));
    assert(upsample11);

    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 4);
    nvinfer1::ITensor* inputTensors12[] = {upsample11->getOutput(0), block6->getOutput(0)};

    nvinfer1::IConcatenationLayer* cat12 = network->addConcatenation(inputTensors12, 2);

    nvinfer1::IElementWiseLayer* block13 =
            C3K2(network, weightMap, *cat12->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.13");

    nvinfer1::IResizeLayer* upsample14 = network->addResize(*block13->getOutput(0));
    assert(upsample14);

    upsample14->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample14->setScales(scale, 4);

    nvinfer1::ITensor* inputTensors15[] = {upsample14->getOutput(0), block4->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat15 = network->addConcatenation(inputTensors15, 2);

    nvinfer1::IElementWiseLayer* block16 =
            C3K2(network, weightMap, *cat15->getOutput(0), get_width(512, gw, max_channels),
                 get_width(256, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.16");

    nvinfer1::IElementWiseLayer* block17 = convBnSiLU(network, weightMap, *block16->getOutput(0),
                                                      get_width(256, gw, max_channels), {3, 3}, 2, "model.17");

    nvinfer1::ITensor* inputTensors18[] = {block17->getOutput(0), block13->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat18 = network->addConcatenation(inputTensors18, 2);

    nvinfer1::IElementWiseLayer* block19 =
            C3K2(network, weightMap, *cat18->getOutput(0), get_width(512, gw, max_channels),
                 get_width(512, gw, max_channels), get_depth(2, gd), true, true, false, 0.5, "model.19");

    nvinfer1::IElementWiseLayer* block20 = convBnSiLU(network, weightMap, *block19->getOutput(0),
                                                      get_width(512, gw, max_channels), {3, 3}, 2, "model.20");

    nvinfer1::ITensor* inputTensors21[] = {block20->getOutput(0), block10->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat21 = network->addConcatenation(inputTensors21, 2);

    nvinfer1::IElementWiseLayer* block22 =
            C3K2(network, weightMap, *cat21->getOutput(0), get_width(1024, gw, max_channels),
                 get_width(1024, gw, max_channels), 1, true, true, true, 0.5,
                 "model.22");  // WARN: get_depth(2, gd) changed to 1.

    /*******************************************************************************************************
    *********************************************  YOLO26-Obb OUTPUT  ********************************************
    *******************************************************************************************************/

    int c2 = std::max(std::max(16, get_width(256, gw, max_channels)), 16 * 4);
    int c3 = std::max(get_width(256, gw, max_channels), std::min(kObbNumClass, 100));

    //cv.2.*.*
    /////////////////////////////////////////////////////

    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_0_0 =
            convBnSiLU(network, weightMap, *block16->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv2.0.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv2_0_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv2.0.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv2_0_2 = network->addConvolutionNd(
            *conv23_one2one_cv2_0_1->getOutput(0), 4, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv2.0.2.weight"], weightMap["model.23.one2one_cv2.0.2.bias"]);
    conv23_one2one_cv2_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv2_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv2_0_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23 = network->addShuffle(*conv23_one2one_cv2_0_2->getOutput(0));
    reshape23->setReshapeDimensions(nvinfer1::Dims3{1, 4, -1});

    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_1_0 =
            convBnSiLU(network, weightMap, *block19->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv2.1.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv2_1_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv2.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv2_1_2 = network->addConvolutionNd(
            *conv23_one2one_cv2_1_1->getOutput(0), 4, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv2.1.2.weight"], weightMap["model.23.one2one_cv2.1.2.bias"]);
    conv23_one2one_cv2_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv2_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv2_1_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_1 = network->addShuffle(*conv23_one2one_cv2_1_2->getOutput(0));
    reshape23_1->setReshapeDimensions(nvinfer1::Dims3{1, 4, -1});

    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_2_0 =
            convBnSiLU(network, weightMap, *block22->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv2.2.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv2_2_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv2_2_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv2.2.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv2_2_2 = network->addConvolutionNd(
            *conv23_one2one_cv2_2_1->getOutput(0), 4, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv2.2.2.weight"], weightMap["model.23.one2one_cv2.2.2.bias"]);
    conv23_one2one_cv2_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv2_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv2_2_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_2 = network->addShuffle(*conv23_one2one_cv2_2_2->getOutput(0));
    reshape23_2->setReshapeDimensions(nvinfer1::Dims3{1, 4, -1});

    nvinfer1::ITensor* inputTensors23[] = {reshape23->getOutput(0), reshape23_1->getOutput(0),
                                           reshape23_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23 = network->addConcatenation(inputTensors23, 3);
    cat23->setAxis(2);

    //cv.4.*.*
    /////////////////////////////////////////////////////
    nvinfer1::IElementWiseLayer* conv23_one2one_cv4_0_0 =
            convBnSiLU(network, weightMap, *block16->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv4.0.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv4_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv4_0_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv4.0.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv4_0_2 = network->addConvolutionNd(
            *conv23_one2one_cv4_0_1->getOutput(0), 1, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv4.0.2.weight"], weightMap["model.23.one2one_cv4.0.2.bias"]);
    conv23_one2one_cv4_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv4_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv4_0_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_6 = network->addShuffle(*conv23_one2one_cv4_0_2->getOutput(0));
    reshape23_6->setReshapeDimensions(nvinfer1::Dims3{1, 1, -1});

    nvinfer1::IElementWiseLayer* conv23_one2one_cv4_1_0 =
            convBnSiLU(network, weightMap, *block19->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv4.1.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv4_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv4_1_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv4.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv4_1_2 = network->addConvolutionNd(
            *conv23_one2one_cv4_1_1->getOutput(0), 1, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv4.1.2.weight"], weightMap["model.23.one2one_cv4.1.2.bias"]);
    conv23_one2one_cv4_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv4_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv4_1_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_7 = network->addShuffle(*conv23_one2one_cv4_1_2->getOutput(0));
    reshape23_7->setReshapeDimensions(nvinfer1::Dims3{1, 1, -1});

    nvinfer1::IElementWiseLayer* conv23_one2one_cv4_2_0 =
            convBnSiLU(network, weightMap, *block22->getOutput(0), c2 / 4, {3, 3}, 1, "model.23.one2one_cv4.2.0", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv4_2_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv4_2_0->getOutput(0), c2 / 4, {3, 3}, 1,
                       "model.23.one2one_cv4.2.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv4_2_2 = network->addConvolutionNd(
            *conv23_one2one_cv4_2_1->getOutput(0), 1, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv4.2.2.weight"], weightMap["model.23.one2one_cv4.2.2.bias"]);
    conv23_one2one_cv4_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv4_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv4_2_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_8 = network->addShuffle(*conv23_one2one_cv4_2_2->getOutput(0));
    reshape23_8->setReshapeDimensions(nvinfer1::Dims3{1, 1, -1});

    nvinfer1::ITensor* inputTensors23_2[] = {reshape23_6->getOutput(0), reshape23_7->getOutput(0),
                                             reshape23_8->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_2 = network->addConcatenation(inputTensors23_2, 3);
    cat23_2->setAxis(2);

    /////////////////////////////////////////////////////
    nvinfer1::ISliceLayer* split23__0 = network->addSlice(
            *cat23->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{cat23->getOutput(0)->getDimensions().d[0], cat23->getOutput(0)->getDimensions().d[1] / 2,
                            cat23->getOutput(0)->getDimensions().d[2]},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::ISliceLayer* split23__1 = network->addSlice(
            *cat23->getOutput(0), nvinfer1::Dims3{0, cat23->getOutput(0)->getDimensions().d[1] / 2, 0},
            nvinfer1::Dims3{cat23->getOutput(0)->getDimensions().d[0], cat23->getOutput(0)->getDimensions().d[1] / 2,
                            cat23->getOutput(0)->getDimensions().d[2]},
            nvinfer1::Dims3{1, 1, 1});
    nvinfer1::IElementWiseLayer* sub23 = network->addElementWise(*split23__1->getOutput(0), *split23__0->getOutput(0),
                                                                 nvinfer1::ElementWiseOperation::kSUB);

    // Divide by 2
    static float two = 2.0f;
    nvinfer1::Weights two_weights{nvinfer1::DataType::kFLOAT, &two, 1};
    nvinfer1::IConstantLayer* const_two = network->addConstant(nvinfer1::Dims3{1, 1, 1}, two_weights);
    nvinfer1::IElementWiseLayer* div23 = network->addElementWise(*sub23->getOutput(0), *const_two->getOutput(0),
                                                                 nvinfer1::ElementWiseOperation::kDIV);

    nvinfer1::ISliceLayer* split23_1__0 = network->addSlice(
            *div23->getOutput(0), nvinfer1::Dims3{0, 0, 0},
            nvinfer1::Dims3{div23->getOutput(0)->getDimensions().d[0], div23->getOutput(0)->getDimensions().d[1] / 2,
                            div23->getOutput(0)->getDimensions().d[2]},
            nvinfer1::Dims3{1, 1, 1});

    nvinfer1::ISliceLayer* split23_1__1 = network->addSlice(
            *div23->getOutput(0), nvinfer1::Dims3{0, div23->getOutput(0)->getDimensions().d[1] / 2, 0},
            nvinfer1::Dims3{div23->getOutput(0)->getDimensions().d[0], div23->getOutput(0)->getDimensions().d[1] / 2,
                            div23->getOutput(0)->getDimensions().d[2]},
            nvinfer1::Dims3{1, 1, 1});

    nvinfer1::IUnaryLayer* cos23 = network->addUnary(*cat23_2->getOutput(0), nvinfer1::UnaryOperation::kCOS);
    nvinfer1::IUnaryLayer* sin23 = network->addUnary(*cat23_2->getOutput(0), nvinfer1::UnaryOperation::kSIN);

    nvinfer1::IElementWiseLayer* mul23 = network->addElementWise(*split23_1__0->getOutput(0), *cos23->getOutput(0),
                                                                 nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::IElementWiseLayer* mul23_1 = network->addElementWise(*split23_1__1->getOutput(0), *sin23->getOutput(0),
                                                                   nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::IElementWiseLayer* sub23_1 =
            network->addElementWise(*mul23->getOutput(0), *mul23_1->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);

    nvinfer1::IElementWiseLayer* mul23_2 = network->addElementWise(*split23_1__0->getOutput(0), *sin23->getOutput(0),
                                                                   nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::IElementWiseLayer* mul23_3 = network->addElementWise(*split23_1__1->getOutput(0), *cos23->getOutput(0),
                                                                   nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::IElementWiseLayer* add23 = network->addElementWise(*mul23_2->getOutput(0), *mul23_3->getOutput(0),
                                                                 nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::ITensor* tensor23[] = {sub23_1->getOutput(0), add23->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_3 = network->addConcatenation(tensor23, 2);
    cat23_3->setAxis(1);

    std::vector<int> fm_sizes;
    int fm_h_0 = block16->getOutput(0)->getDimensions().d[2];  // P3
    int fm_h_1 = block19->getOutput(0)->getDimensions().d[2];  // P4
    int fm_h_2 = block22->getOutput(0)->getDimensions().d[2];  // P5

    fm_sizes.push_back(fm_h_0);
    fm_sizes.push_back(fm_h_1);
    fm_sizes.push_back(fm_h_2);

    int grid_num = fm_h_0 * fm_h_0 + fm_h_1 * fm_h_1 + fm_h_2 * fm_h_2;

    assert((kObbInputH % fm_h_0) == 0 && (kObbInputH % fm_h_1) == 0 && (kObbInputH % fm_h_2) == 0);
    assert((fm_h_0 == block16->getOutput(0)->getDimensions().d[3]) &&
           (fm_h_1 == block19->getOutput(0)->getDimensions().d[3]) &&
           (fm_h_2 == block22->getOutput(0)->getDimensions().d[3]));  // verify fm_w == fm_h

    assert(cat23_3->getOutput(0)->getDimensions().d[2] == grid_num);

    int idx = 0;
    std::vector<float> grid(grid_num * 2);
    auto fill_grid = [&](int fm_h) {
        for (int y = 0; y < fm_h; ++y) {
            for (int x = 0; x < fm_h; ++x) {
                grid[idx] = x + 0.5f;
                grid[idx + grid_num] = y + 0.5f;
                idx++;
            }
        }
    };
    fill_grid(fm_h_0);
    fill_grid(fm_h_1);
    fill_grid(fm_h_2);

    std::vector<float> stride_vec(grid_num);
    idx = 0;
    auto fill_stride = [&](int fm_h, int fm_w, int stride) {
        for (int y = 0; y < fm_h; ++y) {
            for (int x = 0; x < fm_w; ++x) {
                stride_vec[idx] = static_cast<float>(stride);
                idx++;
            }
        }
    };

    std::vector<int> strides = {kObbInputH / fm_h_0, kObbInputH / fm_h_1, kObbInputH / fm_h_2};
    fill_stride(fm_h_0, fm_h_0, strides[0]);
    fill_stride(fm_h_1, fm_h_1, strides[1]);
    fill_stride(fm_h_2, fm_h_2, strides[2]);

    nvinfer1::Dims gridDims{3, {1, 2, grid_num}};
    nvinfer1::IConstantLayer* constant_grid = network->addConstant(
            gridDims, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, grid.data(), (int64_t)grid.size()});

    nvinfer1::Dims strideDims{3, {1, 1, grid_num}};
    nvinfer1::IConstantLayer* constant_stride = network->addConstant(
            strideDims, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, stride_vec.data(), (int64_t)stride_vec.size()});

    nvinfer1::IElementWiseLayer* add23_1 = network->addElementWise(*cat23_3->getOutput(0), *constant_grid->getOutput(0),
                                                                   nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::IElementWiseLayer* add23_2 = network->addElementWise(*split23__0->getOutput(0), *split23__1->getOutput(0),
                                                                   nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::ITensor* tensor23_4[] = {add23_1->getOutput(0), add23_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_4 = network->addConcatenation(tensor23_4, 2);
    cat23_4->setAxis(1);

    nvinfer1::IElementWiseLayer* mul23_4 = network->addElementWise(
            *cat23_4->getOutput(0), *constant_stride->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    /////////////////////////////////////////////////////
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_0_0 =
            convBnSiLU(network, weightMap, *block16->getOutput(0), c2, {3, 3}, 1, "model.23.one2one_cv3.0.0.0", c2);

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_0_0_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.0.0.1", 1);

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_1_0 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_0_0_1->getOutput(0), c3, {3, 3}, 1,
                       "model.23.one2one_cv3.0.1.0", c3);

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_0_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_0_1_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.0.1.1", 1);

    nvinfer1::IConvolutionLayer* conv23_one2one_cv3_0_2 = network->addConvolutionNd(
            *conv23_one2one_cv3_0_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv3.0.2.weight"], weightMap["model.23.one2one_cv3.0.2.bias"]);
    conv23_one2one_cv3_0_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv3_0_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv3_0_2->setNbGroups(1);

    nvinfer1::IShuffleLayer* reshape23_3 = network->addShuffle(*conv23_one2one_cv3_0_2->getOutput(0));
    reshape23_3->setReshapeDimensions(nvinfer1::Dims3{1, kObbNumClass, -1});

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_0_0 = convBnSiLU(
            network, weightMap, *block19->getOutput(0), c2 * 2, {3, 3}, 1, "model.23.one2one_cv3.1.0.0", c2 * 2);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_1_0_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.1.0.1", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_1_0 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_1_0_1->getOutput(0), c3, {3, 3}, 1,
                       "model.23.one2one_cv3.1.1.0", c3);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_1_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_1_1_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.1.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv3_1_2 = network->addConvolutionNd(
            *conv23_one2one_cv3_1_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv3.1.2.weight"], weightMap["model.23.one2one_cv3.1.2.bias"]);
    conv23_one2one_cv3_1_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv3_1_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv3_1_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_4 = network->addShuffle(*conv23_one2one_cv3_1_2->getOutput(0));
    reshape23_4->setReshapeDimensions(nvinfer1::Dims3{1, kObbNumClass, -1});

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_0_0;
    if (type == "m" || type == "l" || type == "x") {
        conv23_one2one_cv3_2_0_0 = convBnSiLU(network, weightMap, *block22->getOutput(0), c2 * 2, {3, 3}, 1,
                                              "model.23.one2one_cv3.2.0.0", c2 * 2);
    } else {
        conv23_one2one_cv3_2_0_0 = convBnSiLU(network, weightMap, *block22->getOutput(0), c2 * 4, {3, 3}, 1,
                                              "model.23.one2one_cv3.2.0.0", c2 * 4);
    }

    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_0_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_2_0_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.2.0.1", 1);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_1_0 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_2_0_1->getOutput(0), c3, {3, 3}, 1,
                       "model.23.one2one_cv3.2.1.0", c3);
    nvinfer1::IElementWiseLayer* conv23_one2one_cv3_2_1_1 =
            convBnSiLU(network, weightMap, *conv23_one2one_cv3_2_1_0->getOutput(0), c3, {1, 1}, 1,
                       "model.23.one2one_cv3.2.1.1", 1);
    nvinfer1::IConvolutionLayer* conv23_one2one_cv3_2_2 = network->addConvolutionNd(
            *conv23_one2one_cv3_2_1_1->getOutput(0), kObbNumClass, nvinfer1::DimsHW{1, 1},
            weightMap["model.23.one2one_cv3.2.2.weight"], weightMap["model.23.one2one_cv3.2.2.bias"]);
    conv23_one2one_cv3_2_2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv23_one2one_cv3_2_2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    conv23_one2one_cv3_2_2->setNbGroups(1);
    nvinfer1::IShuffleLayer* reshape23_5 = network->addShuffle(*conv23_one2one_cv3_2_2->getOutput(0));
    reshape23_5->setReshapeDimensions(nvinfer1::Dims3{1, kObbNumClass, -1});

    nvinfer1::ITensor* tensor23_1[] = {reshape23_3->getOutput(0), reshape23_4->getOutput(0), reshape23_5->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_1 = network->addConcatenation(tensor23_1, 3);
    cat23_1->setAxis(2);
    nvinfer1::IActivationLayer* sigmoid23 = network->addActivation(
            *cat23_1->getOutput(0),
            nvinfer1::ActivationType::kSIGMOID);  // TODO: THIS IS UNNESSARY, REMOVE AFTER PLUGIN IS READY
    /////////////////////////////////////////////////////

    nvinfer1::ITensor* tensor23_5[] = {mul23_4->getOutput(0), sigmoid23->getOutput(0), cat23_2->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat23_5 = network->addConcatenation(tensor23_5, 3);
    cat23_5->setAxis(1);

    nvinfer1::IShuffleLayer* transpose = network->addShuffle(*cat23_5->getOutput(0));
    transpose->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});

    nvinfer1::IPluginV2Layer* yolo = addYoloLayer(network, *transpose->getOutput(0), strides, fm_sizes, strides.size(),
                                                  false, false, false, true, grid_num);

    /////////////////////////////////////////////////////

    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));
    // Use setMemoryPoolLimit instead of deprecated setMaxWorkspaceSize
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

#if defined(USE_FP16)
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cerr << "INT8 not supported for YOLO26 model yet." << std::endl;
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