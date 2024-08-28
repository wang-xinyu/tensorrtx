#include "model.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include "block.h"
#include "calibrator.h"
#include "config.h"
#include "yololayer.h"

using namespace nvinfer1;
#ifdef USE_INT8
void Calibrator(IBuilder* builder, IBuilderConfig* config) {
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
}
#endif

IHostMemory* build_engine_yolov9_t(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                   std::string& wts_name, bool isConvert) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // # conv down
    auto conv_1 = convBnSiLU(network, weightMap, *data, 16, 3, 2, 1, "model.0", 1);
    // # conv down
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 32, 3, 2, 1, "model.1");
    // # elan-1 block
    auto repncspelan_3 = ELAN1(network, weightMap, *conv_2->getOutput(0), 32, 32, 32, 16, "model.2");
    // # avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = AConv(network, weightMap, *repncspelan_3->getOutput(0), 64, "model.3");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 5
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 64, 64, 64, 32, 3, "model.4");
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = AConv(network, weightMap, *repncspelan_5->getOutput(0), 96, "model.5");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 7
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 96, 96, 96, 48, 3, "model.6");
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 8-P5/32
    auto adown_8 = AConv(network, weightMap, *repncspelan_7->getOutput(0), 128, "model.7");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 9
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 128, 128, 128, 64, 3, "model.8");
    // # elan-spp block
    // [-1, 1, SPPELAN, [512, 256]],  # 10
    auto sppelan_10 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 128, 128, 64, "model.9");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_11 = network->addResize(*sppelan_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    const float scales_11[] = {1.0, 2.0, 2.0};
    upsample_11->setScales(scales_11, 3);
    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_12[] = {upsample_11->getOutput(0), repncspelan_7->getOutput(0)};
    auto cat_12 = network->addConcatenation(input_tensor_12, 2);

    // # elan-2 block
    auto repncspelan_13 = RepNCSPELAN4(network, weightMap, *cat_12->getOutput(0), 288, 96, 96, 48, 3, "model.12");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_14 = network->addResize(*repncspelan_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    const float scales_14[] = {1.0, 2.0, 2.0};
    upsample_14->setScales(scales_14, 3);
    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_15[] = {upsample_14->getOutput(0), repncspelan_5->getOutput(0)};
    auto cat_15 = network->addConcatenation(input_tensor_15, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16 (P3/8-small)
    auto repncspelan_16 = RepNCSPELAN4(network, weightMap, *cat_15->getOutput(0), 192, 64, 64, 32, 3, "model.15");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_17 = AConv(network, weightMap, *repncspelan_16->getOutput(0), 48, "model.16");
    // [[-1, 13], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_18[] = {adown_17->getOutput(0), repncspelan_13->getOutput(0)};
    auto cat_18 = network->addConcatenation(input_tensor_18, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19 (P4/16-medium)
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cat_18->getOutput(0), 144, 96, 96, 48, 3, "model.18");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_20 = AConv(network, weightMap, *repncspelan_19->getOutput(0), 64, "model.19");
    // [[-1, 10], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_21[] = {adown_20->getOutput(0), sppelan_10->getOutput(0)};
    auto cat_21 = network->addConcatenation(input_tensor_21, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 22 (P5/32-large)
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cat_21->getOutput(0), 256, 128, 128, 64, 3, "model.21");

    std::vector<IConcatenationLayer*> head;
    if (!isConvert) {
        // # elan-spp block
        auto sppelan_23 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 512, 128, 64, "model.22");

        // # up-concat merge
        auto upsample_24 = network->addResize(*sppelan_23->getOutput(0));
        upsample_24->setResizeMode(ResizeMode::kNEAREST);
        const float scales_24[] = {1.0, 2.0, 2.0};
        upsample_24->setScales(scales_24, 3);
        // [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        ITensor* input_tensor_25[] = {upsample_24->getOutput(0), repncspelan_7->getOutput(0)};
        auto cat_25 = network->addConcatenation(input_tensor_25, 2);

        // # elan-2 block
        auto repncspelan_26 = RepNCSPELAN4(network, weightMap, *cat_25->getOutput(0), 384, 96, 96, 48, 3, "model.25");

        // # up-concat merge
        auto upsample_27 = network->addResize(*repncspelan_26->getOutput(0));
        upsample_27->setResizeMode(ResizeMode::kNEAREST);
        const float scales_27[] = {1.0, 2.0, 2.0};
        upsample_27->setScales(scales_27, 3);
        // [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        ITensor* input_tensor_28[] = {upsample_27->getOutput(0), repncspelan_5->getOutput(0)};
        auto cat_28 = network->addConcatenation(input_tensor_28, 2);

        // # elan-2 block
        auto repncspelan_29 = RepNCSPELAN4(network, weightMap, *cat_28->getOutput(0), 256, 64, 64, 32, 3, "model.28");
        head = DualDDetect(network, weightMap, std::vector<ILayer*>{repncspelan_16, repncspelan_19, repncspelan_22},
                           kNumClass, {64, 96, 128}, "model.29");
    } else {
        head = DDetect(network, weightMap, std::vector<ILayer*>{repncspelan_16, repncspelan_19, repncspelan_22},
                       kNumClass, {64, 96, 128}, "model.22");
    }

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, head, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}

IHostMemory* build_engine_yolov9_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                   std::string& wts_name, bool isConvert) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // # conv down
    auto conv_1 = convBnSiLU(network, weightMap, *data, 32, 3, 2, 1, "model.0", 1);
    // # conv down
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 64, 3, 2, 1, "model.1");
    // # elan-1 block
    auto repncspelan_3 = ELAN1(network, weightMap, *conv_2->getOutput(0), 32, 64, 64, 32, "model.2");
    // # avg-conv down
    auto adown_4 = AConv(network, weightMap, *repncspelan_3->getOutput(0), 128, "model.3");
    // # elan-2 block
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 128, 128, 128, 64, 3, "model.4");
    // # avg-conv down
    auto adown_6 = AConv(network, weightMap, *repncspelan_5->getOutput(0), 192, "model.5");
    // # elan-2 block
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 192, 192, 192, 96, 3, "model.6");
    // # avg-conv down
    auto adown_8 = AConv(network, weightMap, *repncspelan_7->getOutput(0), 256, "model.7");
    // # elan-2 block
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 256, 256, 256, 128, 3, "model.8");
    // # elan-spp block
    auto sppelan_10 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 512, 256, 128, "model.9");

    // # up-concat merge
    auto upsample_11 = network->addResize(*sppelan_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    const float scales_11[] = {1.0, 2.0, 2.0};
    upsample_11->setScales(scales_11, 3);
    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_12[] = {upsample_11->getOutput(0), repncspelan_7->getOutput(0)};
    auto cat_12 = network->addConcatenation(input_tensor_12, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 13
    auto repncspelan_13 = RepNCSPELAN4(network, weightMap, *cat_12->getOutput(0), 192, 192, 192, 96, 3, "model.12");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_14 = network->addResize(*repncspelan_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    const float scales_14[] = {1.0, 2.0, 2.0};
    upsample_14->setScales(scales_14, 3);
    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_15[] = {upsample_14->getOutput(0), repncspelan_5->getOutput(0)};
    auto cat_15 = network->addConcatenation(input_tensor_15, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16 (P3/8-small)
    auto repncspelan_16 = RepNCSPELAN4(network, weightMap, *cat_15->getOutput(0), 128, 128, 128, 64, 3, "model.15");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_17 = AConv(network, weightMap, *repncspelan_16->getOutput(0), 96, "model.16");
    // [[-1, 13], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_18[] = {adown_17->getOutput(0), repncspelan_13->getOutput(0)};
    auto cat_18 = network->addConcatenation(input_tensor_18, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19 (P4/16-medium)
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cat_18->getOutput(0), 768, 192, 192, 96, 3, "model.18");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_20 = AConv(network, weightMap, *repncspelan_19->getOutput(0), 128, "model.19");
    // [[-1, 10], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_21[] = {adown_20->getOutput(0), sppelan_10->getOutput(0)};
    auto cat_21 = network->addConcatenation(input_tensor_21, 2);

    // # elan-2 block
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cat_21->getOutput(0), 1024, 256, 256, 128, 1, "model.21");
    std::vector<IConcatenationLayer*> head;
    if (!isConvert) {
        // # elan-spp block
        auto sppelan_23 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 512, 256, 128, "model.22");

        // # up-concat merge
        auto upsample_24 = network->addResize(*sppelan_23->getOutput(0));
        upsample_24->setResizeMode(ResizeMode::kNEAREST);
        const float scales_24[] = {1.0, 2.0, 2.0};
        upsample_24->setScales(scales_24, 3);
        // [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        ITensor* input_tensor_25[] = {upsample_24->getOutput(0), repncspelan_7->getOutput(0)};
        auto cat_25 = network->addConcatenation(input_tensor_25, 2);

        // # elan-2 block
        auto repncspelan_26 = RepNCSPELAN4(network, weightMap, *cat_25->getOutput(0), 384, 192, 192, 96, 3, "model.25");

        // # up-concat merge
        auto upsample_27 = network->addResize(*repncspelan_26->getOutput(0));
        upsample_27->setResizeMode(ResizeMode::kNEAREST);
        const float scales_27[] = {1.0, 2.0, 2.0};
        upsample_27->setScales(scales_27, 3);
        // [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        ITensor* input_tensor_28[] = {upsample_27->getOutput(0), repncspelan_5->getOutput(0)};
        auto cat_28 = network->addConcatenation(input_tensor_28, 2);

        // # elan-2 block
        auto repncspelan_29 = RepNCSPELAN4(network, weightMap, *cat_28->getOutput(0), 256, 128, 128, 64, 3, "model.28");
        head = DualDDetect(network, weightMap, std::vector<ILayer*>{repncspelan_16, repncspelan_19, repncspelan_22},
                           kNumClass, {128, 192, 256}, "model.29");
    } else {
        head = DDetect(network, weightMap, std::vector<ILayer*>{repncspelan_16, repncspelan_19, repncspelan_22},
                       kNumClass, {128, 192, 256}, "model.22");
    }

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, head, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}
IHostMemory* build_engine_yolov9_m(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                   std::string& wts_name, bool isConvert) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    int begin = isConvert ? 0 : 1;

    // # conv down
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 32, 3, 2, 1, "model." + std::to_string(begin), 1);
    begin += 1;
    // # conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 64, 3, 2, 1, "model." + std::to_string(begin));
    begin += 1;
    // # elan-1 block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 3
    auto repncspelan_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 128, 128, 64, 1,
                                      "model." + std::to_string(begin));
    begin += 1;
    // # avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = AConv(network, weightMap, *repncspelan_3->getOutput(0), 240, "model." + std::to_string(begin));
    begin += 1;
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 5
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 240, 240, 120, 1,
                                      "model." + std::to_string(begin));
    begin += 1;
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = AConv(network, weightMap, *repncspelan_5->getOutput(0), 360, "model." + std::to_string(begin));
    begin += 1;
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 7
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 360, 360, 180, 1,
                                      "model." + std::to_string(begin));
    begin += 1;
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 8-P5/32
    auto adown_8 = AConv(network, weightMap, *repncspelan_7->getOutput(0), 480, "model." + std::to_string(begin));
    begin += 1;
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 9
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 480, 480, 240, 1,
                                      "model." + std::to_string(begin));
    begin += 1;
    // # elan-spp block
    // [-1, 1, SPPELAN, [512, 256]],  # 10
    auto sppelan_10 =
            SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 512, 480, 240, "model." + std::to_string(begin));
    begin += 3;

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_11 = network->addResize(*sppelan_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    const float scales_11[] = {1.0, 2.0, 2.0};
    upsample_11->setScales(scales_11, 3);
    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_12[] = {upsample_11->getOutput(0), repncspelan_7->getOutput(0)};
    auto cat_12 = network->addConcatenation(input_tensor_12, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 13
    auto repncspelan_13 = RepNCSPELAN4(network, weightMap, *cat_12->getOutput(0), 1536, 360, 360, 180, 1,
                                       "model." + std::to_string(begin));
    begin += 3;

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_14 = network->addResize(*repncspelan_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    const float scales_14[] = {1.0, 2.0, 2.0};
    upsample_14->setScales(scales_14, 3);
    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_15[] = {upsample_14->getOutput(0), repncspelan_5->getOutput(0)};
    auto cat_15 = network->addConcatenation(input_tensor_15, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16 (P3/8-small)
    auto repncspelan_16 = RepNCSPELAN4(network, weightMap, *cat_15->getOutput(0), 1024, 240, 240, 120, 1,
                                       "model." + std::to_string(begin));
    begin += 1;

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_17 = AConv(network, weightMap, *repncspelan_16->getOutput(0), 184, "model." + std::to_string(begin));
    begin += 2;
    // [[-1, 13], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_18[] = {adown_17->getOutput(0), repncspelan_13->getOutput(0)};
    auto cat_18 = network->addConcatenation(input_tensor_18, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19 (P4/16-medium)
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cat_18->getOutput(0), 768, 360, 360, 180, 1,
                                       "model." + std::to_string(begin));
    begin += 1;

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_20 = AConv(network, weightMap, *repncspelan_19->getOutput(0), 240, "model." + std::to_string(begin));
    begin += 2;
    // [[-1, 10], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_21[] = {adown_20->getOutput(0), sppelan_10->getOutput(0)};
    auto cat_21 = network->addConcatenation(input_tensor_21, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 22 (P5/32-large)
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cat_21->getOutput(0), 1024, 480, 480, 240, 1,
                                       "model." + std::to_string(begin));
    begin += 1;
    std::vector<IConcatenationLayer*> head;
    if (!isConvert) {
        // # routing
        // [5, 1, CBLinear, [[256]]], # 23
        auto cblinear_23 = CBLinear(network, weightMap, *repncspelan_5->getOutput(0), {240}, 1, 1, 0, 1,
                                    "model." + std::to_string(begin));
        begin += 1;
        // [7, 1, CBLinear, [[256, 512]]], # 24
        auto cblinear_24 = CBLinear(network, weightMap, *repncspelan_7->getOutput(0), {240, 360}, 1, 1, 0, 1,
                                    "model." + std::to_string(begin));
        begin += 1;
        // [9, 1, CBLinear, [[256, 512, 512]]], # 25
        auto cblinear_25 = CBLinear(network, weightMap, *repncspelan_9->getOutput(0), {240, 360, 480}, 1, 1, 0, 1,
                                    "model." + std::to_string(begin));
        begin += 1;

        // # conv down
        // [0, 1, Conv, [64, 3, 2]],  # 26-P1/2
        auto conv_26 = convBnSiLU(network, weightMap, *data, 32, 3, 2, 1, "model." + std::to_string(begin), 1);
        begin += 1;

        // # conv down
        // [-1, 1, Conv, [128, 3, 2]],  # 27-P2/4
        auto conv_27 =
                convBnSiLU(network, weightMap, *conv_26->getOutput(0), 64, 3, 2, 1, "model." + std::to_string(begin));
        begin += 1;

        // # elan-1 block
        // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 28
        auto repncspelan_28 = RepNCSPELAN4(network, weightMap, *conv_27->getOutput(0), 128, 128, 128, 64, 1,
                                           "model." + std::to_string(begin));
        begin += 1;

        // # avg-conv down fuse
        // [-1, 1, ADown, [256]],  # 29-P3/8
        auto adown_29 = AConv(network, weightMap, *repncspelan_28->getOutput(0), 240, "model." + std::to_string(begin));
        begin += 2;
        // [[23, 24, 25, -1], 1, CBFuse, [[0, 0, 0]]], # 30
        auto cbfuse = CBFuse(network, {cblinear_23, cblinear_24, cblinear_25, std::vector<ILayer*>{adown_29}},
                             {0, 0, 0, 0}, {8, 16, 32, 8});

        // # elan-2 block
        // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 31
        auto repncspelan_31 = RepNCSPELAN4(network, weightMap, *cbfuse->getOutput(0), 256, 240, 240, 120, 1,
                                           "model." + std::to_string(begin));
        begin += 1;

        // # avg-conv down fuse
        // [-1, 1, ADown, [512]],  # 32-P4/16
        auto adown_32 = AConv(network, weightMap, *repncspelan_31->getOutput(0), 360, "model." + std::to_string(begin));
        begin += 2;
        // [[24, 25, -1], 1, CBFuse, [[1, 1]]], # 33
        auto cbfuse_33 =
                CBFuse(network, {cblinear_24, cblinear_25, std::vector<ILayer*>{adown_32}}, {1, 1, 0}, {16, 32, 16});

        // # elan-2 block
        // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 34
        auto repncspelan_34 = RepNCSPELAN4(network, weightMap, *cbfuse_33->getOutput(0), 512, 360, 360, 180, 1,
                                           "model." + std::to_string(begin));
        begin += 1;

        // # avg-conv down fuse
        // [-1, 1, ADown, [512]],  # 35-P5/32
        auto adown_35 = AConv(network, weightMap, *repncspelan_34->getOutput(0), 480, "model." + std::to_string(begin));
        begin += 2;

        // [[25, -1], 1, CBFuse, [[2]]], # 36
        auto cbfuse_36 = CBFuse(network, {cblinear_25, std::vector<ILayer*>{adown_35}}, {2, 0}, {32, 32});

        // # elan-2 block
        // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 37
        auto repncspelan_37 = RepNCSPELAN4(network, weightMap, *cbfuse_36->getOutput(0), 512, 480, 480, 240, 1,
                                           "model." + std::to_string(begin));
        begin += 1;

        // # detection head
        // # detect
        // [[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]],  # DualDDetect(A3, A4, A5, P3, P4, P5)
        head = DualDDetect(network, weightMap, std::vector<ILayer*>{repncspelan_31, repncspelan_34, repncspelan_37},
                           kNumClass, {240, 360, 480}, "model." + std::to_string(begin));
    } else {
        // # detection head
        // # detect
        // [[16, 19, 22], 1, DDetect, [nc]],  # DDetect(P3, P4, P5)
        head = DDetect(network, weightMap, std::vector<ILayer*>{repncspelan_16, repncspelan_19, repncspelan_22},
                       kNumClass, {240, 360, 480}, "model." + std::to_string(begin));
    }

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, head, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}

IHostMemory* build_engine_yolov9_c(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                   std::string& wts_name) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // # conv down
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.1", 1);
    // # conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 128, 3, 2, 1, "model.2");
    // # elan-1 block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 3
    auto repncspelan_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 256, 128, 64, 1, "model.3");
    // # avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = ADown(network, weightMap, *repncspelan_3->getOutput(0), 256, "model.4");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 5
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 512, 256, 128, 1, "model.5");
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = ADown(network, weightMap, *repncspelan_5->getOutput(0), 512, "model.6");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 7
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 512, 512, 256, 1, "model.7");
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 8-P5/32
    auto adown_8 = ADown(network, weightMap, *repncspelan_7->getOutput(0), 512, "model.8");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 9
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 512, 512, 256, 1, "model.9");
    // # elan-spp block
    // [-1, 1, SPPELAN, [512, 256]],  # 10
    auto sppelan_10 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 512, 512, 256, "model.10");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_11 = network->addResize(*sppelan_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    const float scales_11[] = {1.0, 2.0, 2.0};
    upsample_11->setScales(scales_11, 3);
    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_12[] = {upsample_11->getOutput(0), repncspelan_7->getOutput(0)};
    auto cat_12 = network->addConcatenation(input_tensor_12, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 13
    auto repncspelan_13 = RepNCSPELAN4(network, weightMap, *cat_12->getOutput(0), 1536, 512, 512, 256, 1, "model.13");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_14 = network->addResize(*repncspelan_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    const float scales_14[] = {1.0, 2.0, 2.0};
    upsample_14->setScales(scales_14, 3);
    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_15[] = {upsample_14->getOutput(0), repncspelan_5->getOutput(0)};
    auto cat_15 = network->addConcatenation(input_tensor_15, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16 (P3/8-small)
    auto repncspelan_16 = RepNCSPELAN4(network, weightMap, *cat_15->getOutput(0), 1024, 256, 256, 128, 1, "model.16");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_17 = ADown(network, weightMap, *repncspelan_16->getOutput(0), 256, "model.17");
    // [[-1, 13], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_18[] = {adown_17->getOutput(0), repncspelan_13->getOutput(0)};
    auto cat_18 = network->addConcatenation(input_tensor_18, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19 (P4/16-medium)
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cat_18->getOutput(0), 768, 512, 512, 256, 1, "model.19");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_20 = ADown(network, weightMap, *repncspelan_19->getOutput(0), 512, "model.20");
    // [[-1, 10], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_21[] = {adown_20->getOutput(0), sppelan_10->getOutput(0)};
    auto cat_21 = network->addConcatenation(input_tensor_21, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 22 (P5/32-large)
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cat_21->getOutput(0), 1024, 512, 512, 256, 1, "model.22");

    // # multi-level reversible auxiliary branch

    // # routing
    // [5, 1, CBLinear, [[256]]], # 23
    auto cblinear_23 = CBLinear(network, weightMap, *repncspelan_5->getOutput(0), {256}, 1, 1, 0, 1, "model.23");
    // [7, 1, CBLinear, [[256, 512]]], # 24
    auto cblinear_24 = CBLinear(network, weightMap, *repncspelan_7->getOutput(0), {256, 512}, 1, 1, 0, 1, "model.24");
    // [9, 1, CBLinear, [[256, 512, 512]]], # 25
    auto cblinear_25 =
            CBLinear(network, weightMap, *repncspelan_9->getOutput(0), {256, 512, 512}, 1, 1, 0, 1, "model.25");

    // # conv down
    // [0, 1, Conv, [64, 3, 2]],  # 26-P1/2
    auto conv_26 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.26", 1);

    // # conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 27-P2/4
    auto conv_27 = convBnSiLU(network, weightMap, *conv_26->getOutput(0), 128, 3, 2, 1, "model.27");

    // # elan-1 block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 28
    auto repncspelan_28 = RepNCSPELAN4(network, weightMap, *conv_27->getOutput(0), 128, 256, 128, 64, 1, "model.28");

    // # avg-conv down fuse
    // [-1, 1, ADown, [256]],  # 29-P3/8
    auto adown_29 = ADown(network, weightMap, *repncspelan_28->getOutput(0), 256, "model.29");
    // [[23, 24, 25, -1], 1, CBFuse, [[0, 0, 0]]], # 30
    auto cbfuse = CBFuse(network, {cblinear_23, cblinear_24, cblinear_25, std::vector<ILayer*>{adown_29}}, {0, 0, 0, 0},
                         {8, 16, 32, 8});

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 31
    auto repncspelan_31 = RepNCSPELAN4(network, weightMap, *cbfuse->getOutput(0), 256, 512, 256, 128, 1, "model.31");

    // # avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 32-P4/16
    auto adown_32 = ADown(network, weightMap, *repncspelan_31->getOutput(0), 512, "model.32");
    // [[24, 25, -1], 1, CBFuse, [[1, 1]]], # 33
    auto cbfuse_33 =
            CBFuse(network, {cblinear_24, cblinear_25, std::vector<ILayer*>{adown_32}}, {1, 1, 0}, {16, 32, 16});

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 34
    auto repncspelan_34 = RepNCSPELAN4(network, weightMap, *cbfuse_33->getOutput(0), 512, 512, 512, 256, 1, "model.34");

    // # avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 35-P5/32
    auto adown_35 = ADown(network, weightMap, *repncspelan_34->getOutput(0), 512, "model.35");

    // [[25, -1], 1, CBFuse, [[2]]], # 36
    auto cbfuse_36 = CBFuse(network, {cblinear_25, std::vector<ILayer*>{adown_35}}, {2, 0}, {32, 32});

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 37
    auto repncspelan_37 = RepNCSPELAN4(network, weightMap, *cbfuse_36->getOutput(0), 512, 512, 512, 256, 1, "model.37");

    // # detection head
    // # detect
    // [[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]],  # DualDDetect(A3, A4, A5, P3, P4, P5)
    auto dualddetect_38 =
            DualDDetect(network, weightMap, std::vector<ILayer*>{repncspelan_31, repncspelan_34, repncspelan_37},
                        kNumClass, {512, 512, 512}, "model.38");

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, dualddetect_38, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}

IHostMemory* build_engine_yolov9_e(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                   std::string& wts_name) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------backbone------ */
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.1", 1);
    assert(conv_1);
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 128, 3, 2, 1, "model.2");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 2]],  # 3
    auto repncspelan_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 256, 128, 64, 2, "model.3");
    // avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = ADown(network, weightMap, *repncspelan_3->getOutput(0), 256, "model.4");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 2]],  # 5
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 512, 256, 128, 2, "model.5");
    // avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = ADown(network, weightMap, *repncspelan_5->getOutput(0), 512, "model.6");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 7
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 1024, 512, 256, 2, "model.7");
    // avg-conv down
    // [-1, 1, ADown, [1024]],  # 8-P5/32
    auto adown_8 = ADown(network, weightMap, *repncspelan_7->getOutput(0), 1024, "model.8");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 9
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 1024, 512, 256, 2, "model.9");

    // [1, 1, CBLinear, [[64]]], # 10
    auto cblinear_10 = CBLinear(network, weightMap, *conv_1->getOutput(0), {64}, 1, 1, 0, 1, "model.10");
    // [3, 1, CBLinear, [[64, 128]]], # 11
    auto cblinear_11 = CBLinear(network, weightMap, *repncspelan_3->getOutput(0), {64, 128}, 1, 1, 0, 1, "model.11");
    // [5, 1, CBLinear, [[64, 128, 256]]], # 12
    auto cblinear_12 =
            CBLinear(network, weightMap, *repncspelan_5->getOutput(0), {64, 128, 256}, 1, 1, 0, 1, "model.12");
    // [7, 1, CBLinear, [[64, 128, 256, 512]]], # 13
    auto cblinear_13 =
            CBLinear(network, weightMap, *repncspelan_7->getOutput(0), {64, 128, 256, 512}, 1, 1, 0, 1, "model.13");
    // [9, 1, CBLinear, [[64, 128, 256, 512, 1024]]], # 14
    auto cblinear_14 = CBLinear(network, weightMap, *repncspelan_9->getOutput(0), {64, 128, 256, 512, 1024}, 1, 1, 0, 1,
                                "model.14");

    // conv down
    // [0, 1, Conv, [64, 3, 2]],  # 15-P1/2
    auto conv_15 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.15", 1);
    // [[10, 11, 12, 13, 14, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 16
    auto cbfuse_16 = CBFuse(
            network, {cblinear_10, cblinear_11, cblinear_12, cblinear_13, cblinear_14, std::vector<ILayer*>{conv_15}},
            {0, 0, 0, 0, 0, 0}, {2, 4, 8, 16, 32, 2});

    // conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 17-P2/4
    auto conv_17 = convBnSiLU(network, weightMap, *cbfuse_16->getOutput(0), 128, 3, 2, 1, "model.17");
    // [[11, 12, 13, 14, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 18
    auto cbfuse_18 =
            CBFuse(network, {cblinear_11, cblinear_12, cblinear_13, cblinear_14, std::vector<ILayer*>{conv_17}},
                   {1, 1, 1, 1, 0}, {4, 8, 16, 32, 4});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 2]],  # 19
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cbfuse_18->getOutput(0), 128, 256, 128, 64, 2, "model.19");

    // avg-conv down fuse
    // [-1, 1, ADown, [256]],  # 20-P3/8
    auto adown_20 = ADown(network, weightMap, *repncspelan_19->getOutput(0), 256, "model.20");
    // [[12, 13, 14, -1], 1, CBFuse, [[2, 2, 2]]], # 21
    auto cbfuse_21 = CBFuse(network, {cblinear_12, cblinear_13, cblinear_14, std::vector<ILayer*>{adown_20}},
                            {2, 2, 2, 0}, {8, 16, 32, 8});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 2]],  # 22
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cbfuse_21->getOutput(0), 256, 512, 256, 128, 2, "model.22");

    // avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 23-P4/16
    auto adown_23 = ADown(network, weightMap, *repncspelan_22->getOutput(0), 512, "model.23");
    // [[13, 14, -1], 1, CBFuse, [[3, 3]]], # 24
    auto cbfuse_24 =
            CBFuse(network, {cblinear_13, cblinear_14, std::vector<ILayer*>{adown_23}}, {3, 3, 0}, {16, 32, 16});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 25
    auto repncspelan_25 =
            RepNCSPELAN4(network, weightMap, *cbfuse_24->getOutput(0), 512, 1024, 512, 256, 2, "model.25");

    // avg-conv down fuse
    // [-1, 1, ADown, [1024]],  # 26-P5/32
    auto adown_26 = ADown(network, weightMap, *repncspelan_25->getOutput(0), 1024, "model.26");
    // [[14, -1], 1, CBFuse, [[4]]], # 27
    auto cbfuse_27 = CBFuse(network, {cblinear_14, std::vector<ILayer*>{adown_26}}, {4, 0}, {32, 32});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 28
    auto repncspelan_28 =
            RepNCSPELAN4(network, weightMap, *cbfuse_27->getOutput(0), 512, 1024, 512, 256, 2, "model.28");

    // elan-spp block
    // [9, 1, SPPELAN, [512, 256]],  # 29
    auto sppelan_29 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 1024, 512, 256, "model.29");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_30 = network->addResize(*sppelan_29->getOutput(0));
    upsample_30->setResizeMode(ResizeMode::kNEAREST);
    const float scales_30[] = {1.0, 2.0, 2.0};
    upsample_30->setScales(scales_30, 3);
    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_31[] = {upsample_30->getOutput(0), repncspelan_7->getOutput(0)};
    auto cat_31 = network->addConcatenation(input_tensor_31, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 32
    auto repncspelan_32 = RepNCSPELAN4(network, weightMap, *cat_31->getOutput(0), 1536, 512, 512, 256, 2, "model.32");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_33 = network->addResize(*repncspelan_32->getOutput(0));
    upsample_33->setResizeMode(ResizeMode::kNEAREST);
    const float scales_33[] = {1.0, 2.0, 2.0};
    upsample_33->setScales(scales_33, 3);
    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_34[] = {upsample_33->getOutput(0), repncspelan_5->getOutput(0)};
    auto cat_34 = network->addConcatenation(input_tensor_34, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 2]],  # 35
    auto repncspelan_35 = RepNCSPELAN4(network, weightMap, *cat_34->getOutput(0), 1024, 256, 256, 128, 2, "model.35");

    // # elan-spp block
    // [28, 1, SPPELAN, [512, 256]],  # 36
    auto sppelan_36 = SPPELAN(network, weightMap, *repncspelan_28->getOutput(0), 1024, 512, 256, "model.36");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_37 = network->addResize(*sppelan_36->getOutput(0));
    upsample_37->setResizeMode(ResizeMode::kNEAREST);
    const float scales_37[] = {1.0, 2.0, 2.0};
    upsample_37->setScales(scales_37, 3);
    // [[-1, 25], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_38[] = {upsample_37->getOutput(0), repncspelan_25->getOutput(0)};
    auto cat_38 = network->addConcatenation(input_tensor_38, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 39
    auto repncspelan_39 = RepNCSPELAN4(network, weightMap, *cat_38->getOutput(0), 1536, 512, 512, 256, 2, "model.39");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_40 = network->addResize(*repncspelan_39->getOutput(0));
    upsample_40->setResizeMode(ResizeMode::kNEAREST);
    const float scales_40[] = {1.0, 2.0, 2.0};
    upsample_40->setScales(scales_40, 3);
    // [[-1, 22], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_41[] = {upsample_40->getOutput(0), repncspelan_22->getOutput(0)};
    auto cat_41 = network->addConcatenation(input_tensor_41, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 2]],  # 42 (P3/8-small)
    auto repncspelan_42 = RepNCSPELAN4(network, weightMap, *cat_41->getOutput(0), 1024, 256, 256, 128, 2, "model.42");
    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_43 = ADown(network, weightMap, *repncspelan_42->getOutput(0), 256, "model.43");
    // [[-1, 39], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_44[] = {adown_43->getOutput(0), repncspelan_39->getOutput(0)};
    auto cat_44 = network->addConcatenation(input_tensor_44, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 45 (P4/16-medium)
    auto repncspelan_45 = RepNCSPELAN4(network, weightMap, *cat_44->getOutput(0), 768, 512, 512, 256, 2, "model.45");
    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_46 = ADown(network, weightMap, *repncspelan_45->getOutput(0), 512, "model.46");
    // [[-1, 36], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_47[] = {adown_46->getOutput(0), sppelan_36->getOutput(0)};
    auto cat_47 = network->addConcatenation(input_tensor_47, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 1024, 512, 2]],  # 48 (P5/32-large)
    auto repncspelan_48 = RepNCSPELAN4(network, weightMap, *cat_47->getOutput(0), 1024, 512, 1024, 512, 2, "model.48");

    // auto DualDDetect_49 = DualDDetect(network, weightMap, std::vector<ILayer*>{RepNCSPELAN_42, RepNCSPELAN_45, RepNCSPELAN_48}, kNumClass, {256, 512, 512}, "model.49");
    auto dualddetect_49 =
            DualDDetect(network, weightMap, std::vector<ILayer*>{repncspelan_35, repncspelan_32, sppelan_29}, kNumClass,
                        {256, 512, 512}, "model.49");

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, dualddetect_49, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}

IHostMemory* build_engine_gelan_e(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                  std::string& wts_name) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------backbone------ */
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.1", 1);
    assert(conv_1);
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 128, 3, 2, 1, "model.2");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 2]],  # 3
    auto repncspelan_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 256, 128, 64, 2, "model.3");
    // avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = ADown(network, weightMap, *repncspelan_3->getOutput(0), 256, "model.4");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 2]],  # 5
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 512, 256, 128, 2, "model.5");
    // avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = ADown(network, weightMap, *repncspelan_5->getOutput(0), 512, "model.6");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 7
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 1024, 512, 256, 2, "model.7");
    // avg-conv down
    // [-1, 1, ADown, [1024]],  # 8-P5/32
    auto adown_8 = ADown(network, weightMap, *repncspelan_7->getOutput(0), 1024, "model.8");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 9
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 1024, 512, 256, 2, "model.9");

    // [1, 1, CBLinear, [[64]]], # 10
    auto cblinear_10 = CBLinear(network, weightMap, *conv_1->getOutput(0), {64}, 1, 1, 0, 1, "model.10");
    // [3, 1, CBLinear, [[64, 128]]], # 11
    auto cblinear_11 = CBLinear(network, weightMap, *repncspelan_3->getOutput(0), {64, 128}, 1, 1, 0, 1, "model.11");
    // [5, 1, CBLinear, [[64, 128, 256]]], # 12
    auto cblinear_12 =
            CBLinear(network, weightMap, *repncspelan_5->getOutput(0), {64, 128, 256}, 1, 1, 0, 1, "model.12");
    // [7, 1, CBLinear, [[64, 128, 256, 512]]], # 13
    auto cblinear_13 =
            CBLinear(network, weightMap, *repncspelan_7->getOutput(0), {64, 128, 256, 512}, 1, 1, 0, 1, "model.13");
    // [9, 1, CBLinear, [[64, 128, 256, 512, 1024]]], # 14
    auto cblinear_14 = CBLinear(network, weightMap, *repncspelan_9->getOutput(0), {64, 128, 256, 512, 1024}, 1, 1, 0, 1,
                                "model.14");

    // conv down
    // [0, 1, Conv, [64, 3, 2]],  # 15-P1/2
    auto conv_15 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.15", 1);
    // [[10, 11, 12, 13, 14, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 16
    auto cbfuse_16 = CBFuse(
            network, {cblinear_10, cblinear_11, cblinear_12, cblinear_13, cblinear_14, std::vector<ILayer*>{conv_15}},
            {0, 0, 0, 0, 0, 0}, {2, 4, 8, 16, 32, 2});

    // conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 17-P2/4
    auto conv_17 = convBnSiLU(network, weightMap, *cbfuse_16->getOutput(0), 128, 3, 2, 1, "model.17");
    // [[11, 12, 13, 14, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 18
    auto cbfuse_18 =
            CBFuse(network, {cblinear_11, cblinear_12, cblinear_13, cblinear_14, std::vector<ILayer*>{conv_17}},
                   {1, 1, 1, 1, 0}, {4, 8, 16, 32, 4});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 2]],  # 19
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cbfuse_18->getOutput(0), 128, 256, 128, 64, 2, "model.19");

    // avg-conv down fuse
    // [-1, 1, ADown, [256]],  # 20-P3/8
    auto adown_20 = ADown(network, weightMap, *repncspelan_19->getOutput(0), 256, "model.20");
    // [[12, 13, 14, -1], 1, CBFuse, [[2, 2, 2]]], # 21
    auto cbfuse_21 = CBFuse(network, {cblinear_12, cblinear_13, cblinear_14, std::vector<ILayer*>{adown_20}},
                            {2, 2, 2, 0}, {8, 16, 32, 8});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 2]],  # 22
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cbfuse_21->getOutput(0), 256, 512, 256, 128, 2, "model.22");

    // avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 23-P4/16
    auto adown_23 = ADown(network, weightMap, *repncspelan_22->getOutput(0), 512, "model.23");
    // [[13, 14, -1], 1, CBFuse, [[3, 3]]], # 24
    auto cbfuse_24 =
            CBFuse(network, {cblinear_13, cblinear_14, std::vector<ILayer*>{adown_23}}, {3, 3, 0}, {16, 32, 16});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 25
    auto repncspelan_25 =
            RepNCSPELAN4(network, weightMap, *cbfuse_24->getOutput(0), 512, 1024, 512, 256, 2, "model.25");

    // avg-conv down fuse
    // [-1, 1, ADown, [1024]],  # 26-P5/32
    auto adown_26 = ADown(network, weightMap, *repncspelan_25->getOutput(0), 1024, "model.26");
    // [[14, -1], 1, CBFuse, [[4]]], # 27
    auto cbfuse_27 = CBFuse(network, {cblinear_14, std::vector<ILayer*>{adown_26}}, {4, 0}, {32, 32});

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 28
    auto repncspelan_28 =
            RepNCSPELAN4(network, weightMap, *cbfuse_27->getOutput(0), 512, 1024, 512, 256, 2, "model.28");

    // elan-spp block
    // [28, 1, SPPELAN, [512, 256]],  # 29
    auto sppelan_29 = SPPELAN(network, weightMap, *repncspelan_28->getOutput(0), 1024, 512, 256, "model.29");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_30 = network->addResize(*sppelan_29->getOutput(0));
    upsample_30->setResizeMode(ResizeMode::kNEAREST);
    const float scales_30[] = {1.0, 2.0, 2.0};
    upsample_30->setScales(scales_30, 3);
    // [[-1, 25], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_31[] = {upsample_30->getOutput(0), repncspelan_25->getOutput(0)};
    auto cat_31 = network->addConcatenation(input_tensor_31, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 32
    auto repncspelan_32 = RepNCSPELAN4(network, weightMap, *cat_31->getOutput(0), 1536, 512, 512, 256, 2, "model.32");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_33 = network->addResize(*repncspelan_32->getOutput(0));
    upsample_33->setResizeMode(ResizeMode::kNEAREST);
    const float scales_33[] = {1.0, 2.0, 2.0};
    upsample_33->setScales(scales_33, 3);
    // [[-1, 22], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_34[] = {upsample_33->getOutput(0), repncspelan_22->getOutput(0)};
    auto cat_34 = network->addConcatenation(input_tensor_34, 2);

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 2]],  # 35
    auto repncspelan_35 = RepNCSPELAN4(network, weightMap, *cat_34->getOutput(0), 1024, 256, 256, 128, 2, "model.35");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_36 = ADown(network, weightMap, *repncspelan_35->getOutput(0), 256, "model.36");
    // [[-1, 32], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_37[] = {adown_36->getOutput(0), repncspelan_32->getOutput(0)};
    auto cat_37 = network->addConcatenation(input_tensor_37, 2);

    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 38 (P4/16-medium)
    auto repncspelan_38 = RepNCSPELAN4(network, weightMap, *cat_37->getOutput(0), 768, 512, 512, 256, 2, "model.38");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_39 = ADown(network, weightMap, *repncspelan_38->getOutput(0), 512, "model.39");
    // [[-1, 29], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_40[] = {adown_39->getOutput(0), sppelan_29->getOutput(0)};
    auto cat_40 = network->addConcatenation(input_tensor_40, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 1024, 512, 2]],  # 41 (P5/32-large)
    auto repncspelan_41 = RepNCSPELAN4(network, weightMap, *cat_40->getOutput(0), 1024, 512, 1024, 512, 2, "model.41");

    auto ddetect_42 = DDetect(network, weightMap, std::vector<ILayer*>{repncspelan_35, repncspelan_38, repncspelan_41},
                              kNumClass, {256, 512, 512}, "model.42");

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, ddetect_42, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}
IHostMemory* build_engine_gelan_c(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt,
                                  std::string& wts_name) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    // # conv down
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.0", 1);
    // # conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 128, 3, 2, 1, "model.1");
    // # elan-1 block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 3
    auto repncspelan_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 256, 128, 64, 1, "model.2");
    // # avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = ADown(network, weightMap, *repncspelan_3->getOutput(0), 256, "model.3");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 5
    auto repncspelan_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 512, 256, 128, 1, "model.4");
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = ADown(network, weightMap, *repncspelan_5->getOutput(0), 512, "model.5");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 7
    auto repncspelan_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 512, 512, 256, 1, "model.6");
    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 8-P5/32
    auto adown_8 = ADown(network, weightMap, *repncspelan_7->getOutput(0), 512, "model.7");
    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 9
    auto repncspelan_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 512, 512, 256, 1, "model.8");
    // # elan-spp block
    // [-1, 1, SPPELAN, [512, 256]],  # 10
    auto sppelan_10 = SPPELAN(network, weightMap, *repncspelan_9->getOutput(0), 512, 512, 256, "model.9");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_11 = network->addResize(*sppelan_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    const float scales_11[] = {1.0, 2.0, 2.0};
    upsample_11->setScales(scales_11, 3);
    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_12[] = {upsample_11->getOutput(0), repncspelan_7->getOutput(0)};
    auto cat_12 = network->addConcatenation(input_tensor_12, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 13
    auto repncspelan_13 = RepNCSPELAN4(network, weightMap, *cat_12->getOutput(0), 1536, 512, 512, 256, 1, "model.12");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_14 = network->addResize(*repncspelan_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    const float scales_14[] = {1.0, 2.0, 2.0};
    upsample_14->setScales(scales_14, 3);
    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_15[] = {upsample_14->getOutput(0), repncspelan_5->getOutput(0)};
    auto cat_15 = network->addConcatenation(input_tensor_15, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16 (P3/8-small)
    auto repncspelan_16 = RepNCSPELAN4(network, weightMap, *cat_15->getOutput(0), 1024, 256, 256, 128, 1, "model.15");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_17 = ADown(network, weightMap, *repncspelan_16->getOutput(0), 256, "model.16");
    // [[-1, 13], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_18[] = {adown_17->getOutput(0), repncspelan_13->getOutput(0)};
    auto cat_18 = network->addConcatenation(input_tensor_18, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19 (P4/16-medium)
    auto repncspelan_19 = RepNCSPELAN4(network, weightMap, *cat_18->getOutput(0), 768, 512, 512, 256, 1, "model.18");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_20 = ADown(network, weightMap, *repncspelan_19->getOutput(0), 512, "model.19");
    // [[-1, 10], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_21[] = {adown_20->getOutput(0), sppelan_10->getOutput(0)};
    auto cat_21 = network->addConcatenation(input_tensor_21, 2);

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 22 (P5/32-large)
    auto repncspelan_22 = RepNCSPELAN4(network, weightMap, *cat_21->getOutput(0), 1024, 512, 512, 256, 1, "model.21");

    // # detection head
    // # detect
    // [[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]],  # DualDDetect(A3, A4, A5, P3, P4, P5)
    auto ddetect_23 = DDetect(network, weightMap, std::vector<ILayer*>{repncspelan_16, repncspelan_19, repncspelan_22},
                              kNumClass, {256, 512, 512}, "model.22");

    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, ddetect_23, false);
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
    auto* calibrator =
            new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return serialized_model;
}
