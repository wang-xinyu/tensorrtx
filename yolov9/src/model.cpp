#include "model.h"
#include "calibrator.h"
#include "config.h"
#include "yololayer.h"
#include "block.h"
#include <iostream>
#include <fstream>
#include <map>
#include <cassert>
#include <cmath>
#include <cstring>

using namespace nvinfer1;

#ifdef USE_INT8
void Calibrator(IBuilder* builder, IBuilderConfig* config){
      std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, gCalibTablePath, "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
}
#endif

IHostMemory* build_engine_yolov9_e(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& wts_name) {
    

    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov7-tiny backbone------ */
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.1", 1);
    assert(conv_1);
    PrintDim(conv_1, "conv1:");
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4   
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 128, 3, 2, 1, "model.2");
    PrintDim(conv_2, "conv2:");
    //    # csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 2]],  # 3
    auto RepNCSPELAN_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 256, 128, 64, 2, "model.3");
    PrintDim(RepNCSPELAN_3, "RepNCSPELAN_3:");
    // avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = ADown(network, weightMap, *RepNCSPELAN_3->getOutput(0), 256, "model.4");
    PrintDim(adown_4, "adown_4:");

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 2]],  # 5
    auto RepNCSPELAN_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 512, 256, 128, 2, "model.5");
    PrintDim(RepNCSPELAN_5, "RepNCSPELAN_5:");
    // avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = ADown(network, weightMap, *RepNCSPELAN_5->getOutput(0), 512, "model.6");
    PrintDim(adown_6, "adown_6:");
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 7
    auto RepNCSPELAN_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 1024, 512, 256, 2, "model.7");
    PrintDim(RepNCSPELAN_7, "RepNCSPELAN_7:");
    // avg-conv down
    // [-1, 1, ADown, [1024]],  # 8-P5/32
    auto adown_8 = ADown(network, weightMap, *RepNCSPELAN_7->getOutput(0), 1024, "model.8");
    PrintDim(adown_8, "adown_8:");

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 9
    auto RepNCSPELAN_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 1024, 512, 256, 2, "model.9");
    PrintDim(RepNCSPELAN_9, "RepNCSPELAN_9:");
    // [1, 1, CBLinear, [[64]]], # 10
    auto CBlinear_10 = CBLinear(network, weightMap, *conv_1->getOutput(0), { 64}, 1, 1, 0, 1, "model.10");
    for(int i = 0; i < CBlinear_10.size(); i++)
    {
        PrintDim(CBlinear_10[i], "CBlinear_10:" + std::to_string(i));
    }
    // [3, 1, CBLinear, [[64, 128]]], # 11
    auto CBlinear_11 = CBLinear(network, weightMap, *RepNCSPELAN_3->getOutput(0), { 64, 128 }, 1, 1, 0, 1, "model.11");
    for(int i = 0; i < CBlinear_11.size(); i++)
    {
        PrintDim(CBlinear_11[i], "CBlinear_11:" + std::to_string(i));
    }
    // [5, 1, CBLinear, [[64, 128, 256]]], # 12
    auto CBlinear_12 = CBLinear(network, weightMap, *RepNCSPELAN_5->getOutput(0), { 64, 128, 256 }, 1, 1, 0, 1, "model.12");
    for(int i = 0; i < CBlinear_12.size(); i++)
    {
        PrintDim(CBlinear_12[i], "CBlinear_12:" + std::to_string(i));
    }
    // [7, 1, CBLinear, [[64, 128, 256, 512]]], # 13
    auto CBlinear_13 = CBLinear(network, weightMap, *RepNCSPELAN_7->getOutput(0), { 64, 128, 256, 512 }, 1, 1, 0, 1, "model.13");
    for(int i = 0; i < CBlinear_13.size(); i++)
    {
        PrintDim(CBlinear_13[i], "CBlinear_13:" + std::to_string(i));
    }
    // [9, 1, CBLinear, [[64, 128, 256, 512, 1024]]], # 14
    auto CBlinear_14 = CBLinear(network, weightMap, *RepNCSPELAN_9->getOutput(0), { 64, 128, 256, 512, 1024 }, 1, 1, 0, 1, "model.14");
    for(int i = 0; i < CBlinear_14.size(); i++)
    {
        PrintDim(CBlinear_14[i], "CBlinear_14:" + std::to_string(i));
    }

    // conv down
    // [0, 1, Conv, [64, 3, 2]],  # 15-P1/2
    auto conv_15 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.15", 1);
    PrintDim(conv_15, "conv_15:");
    // [[10, 11, 12, 13, 14, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 16
    auto cbfuse_16 = CBFuse(network, {CBlinear_10, CBlinear_11, CBlinear_12, CBlinear_13, CBlinear_14, std::vector<ILayer*>{conv_15}}, 
                            {0, 0, 0, 0, 0, 0}, 
                            {2, 4, 8, 16, 32, 2});
    PrintDim(cbfuse_16, "cbfuse_16:");
    // conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 17-P2/4
    auto conv_17 = convBnSiLU(network, weightMap, *cbfuse_16->getOutput(0), 128, 3, 2, 1, "model.17");
    PrintDim(conv_17, "conv_17:");
    // [[11, 12, 13, 14, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 18  
    auto cbfuse_18 = CBFuse(network, {CBlinear_11, CBlinear_12, CBlinear_13, CBlinear_14, std::vector<ILayer*>{conv_17}},
                            {1, 1, 1, 1, 0},
                            {4, 8, 16, 32, 4});
    PrintDim(cbfuse_18, "cbfuse_18:");
    
    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 2]],  # 19
    auto RepNCSPELAN_19 = RepNCSPELAN4(network, weightMap, *cbfuse_18->getOutput(0), 128, 256, 128, 64, 2, "model.19");
    PrintDim(RepNCSPELAN_19, "RepNCSPELAN_19:");

    // avg-conv down fuse
    // [-1, 1, ADown, [256]],  # 20-P3/8
    auto adown_20 = ADown(network, weightMap, *RepNCSPELAN_19->getOutput(0), 256, "model.20");
    PrintDim(adown_20, "adown_20:");
    // [[12, 13, 14, -1], 1, CBFuse, [[2, 2, 2]]], # 21  
    auto cbfuse_21 = CBFuse(network, {CBlinear_12, CBlinear_13, CBlinear_14, std::vector<ILayer*>{adown_20}},
                            {2, 2, 2, 0},
                            {8, 16, 32, 8});
    PrintDim(cbfuse_21, "cbfuse_21:");

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 2]],  # 22
    auto RepNCSPELAN_22 = RepNCSPELAN4(network, weightMap, *cbfuse_21->getOutput(0), 256, 512, 256, 128, 2, "model.22");
    PrintDim(RepNCSPELAN_22, "RepNCSPELAN_22:");

    // avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 23-P4/16
    auto adown_23 = ADown(network, weightMap, *RepNCSPELAN_22->getOutput(0), 512, "model.23");
    PrintDim(adown_23, "adown_23:");
    // [[13, 14, -1], 1, CBFuse, [[3, 3]]], # 24 
    auto cbfuse_24 = CBFuse(network, {CBlinear_13, CBlinear_14, std::vector<ILayer*>{adown_23}},
                            {3, 3, 0},
                            {16, 32, 16});
    PrintDim(cbfuse_24, "cbfuse_24:");

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 25
    auto RepNCSPELAN_25 = RepNCSPELAN4(network, weightMap, *cbfuse_24->getOutput(0), 512, 1024, 512, 256, 2, "model.25");
    PrintDim(RepNCSPELAN_25, "RepNCSPELAN_25:");
    // avg-conv down fuse
    // [-1, 1, ADown, [1024]],  # 26-P5/32
    auto adown_26 = ADown(network, weightMap, *RepNCSPELAN_25->getOutput(0), 1024, "model.26");
    PrintDim(adown_26, "adown_26:");
    // [[14, -1], 1, CBFuse, [[4]]], # 27
    auto cbfuse_27 = CBFuse(network, {CBlinear_14, std::vector<ILayer*>{adown_26}},
                            {4, 0},
                            {32, 32});
    PrintDim(cbfuse_27, "cbfuse_27:");

    // csp-elan block
    // [-1, 1, RepNCSPELAN4, [1024, 512, 256, 2]],  # 28
    auto RepNCSPELAN_28 = RepNCSPELAN4(network, weightMap, *cbfuse_27->getOutput(0), 512, 1024, 512, 256, 2, "model.28");
    PrintDim(RepNCSPELAN_28, "RepNCSPELAN_28:");
    
    //     # elan-spp block
    // [9, 1, SPPELAN, [512, 256]],  # 29
    auto SPPELAN_29 = SPPELAN(network, weightMap, *RepNCSPELAN_9->getOutput(0), 1024, 512, 256, "model.29");
    PrintDim(SPPELAN_29, "SPPELAN_29:");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_30 = network->addResize(*SPPELAN_29->getOutput(0));
    upsample_30->setResizeMode(ResizeMode::kNEAREST);
    const float scales_30[] = { 1.0, 2.0, 2.0 };
    upsample_30->setScales(scales_30, 3);
    PrintDim(upsample_30, "upsample_30:");

    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_31[] = { upsample_30->getOutput(0), RepNCSPELAN_7->getOutput(0) };
    auto cat_31 = network->addConcatenation(input_tensor_31, 2);
    PrintDim(cat_31, "cat_31:");

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 32
    auto RepNCSPELAN_32 = RepNCSPELAN4(network, weightMap, *cat_31->getOutput(0), 1536, 512, 512, 256, 2, "model.32");
    PrintDim(RepNCSPELAN_32, "RepNCSPELAN_32:");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_33 = network->addResize(*RepNCSPELAN_32->getOutput(0));
    upsample_33->setResizeMode(ResizeMode::kNEAREST);
    const float scales_33[] = { 1.0, 2.0, 2.0 };
    upsample_33->setScales(scales_33, 3);
    PrintDim(upsample_33, "upsample_33:");

    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_34[] = { upsample_33->getOutput(0), RepNCSPELAN_5->getOutput(0) };
    auto cat_34 = network->addConcatenation(input_tensor_34, 2);
    PrintDim(cat_34, "cat_34:");

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 2]],  # 35
    auto RepNCSPELAN_35 = RepNCSPELAN4(network, weightMap, *cat_34->getOutput(0), 1024, 256, 256, 128, 2, "model.35");
    PrintDim(RepNCSPELAN_35, "RepNCSPELAN_35:");

    // # elan-spp block
    // [28, 1, SPPELAN, [512, 256]],  # 36
    auto SPPELAN_36 = SPPELAN(network, weightMap, *RepNCSPELAN_28->getOutput(0), 1024, 512, 256, "model.36");
    PrintDim(SPPELAN_36, "SPPELAN_36:");
    
    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_37 = network->addResize(*SPPELAN_36->getOutput(0));
    upsample_37->setResizeMode(ResizeMode::kNEAREST);
    const float scales_37[] = { 1.0, 2.0, 2.0 };
    upsample_37->setScales(scales_37, 3);
    PrintDim(upsample_37, "upsample_37:");

    // [[-1, 25], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_38[] = { upsample_37->getOutput(0), RepNCSPELAN_25->getOutput(0) };
    auto cat_38 = network->addConcatenation(input_tensor_38, 2);
    PrintDim(cat_38, "cat_38:");

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 39
    auto RepNCSPELAN_39 = RepNCSPELAN4(network, weightMap, *cat_38->getOutput(0), 1536, 512, 512, 256, 2, "model.39");
    PrintDim(RepNCSPELAN_39, "RepNCSPELAN_39:");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_40 = network->addResize(*RepNCSPELAN_39->getOutput(0));
    upsample_40->setResizeMode(ResizeMode::kNEAREST);
    const float scales_40[] = { 1.0, 2.0, 2.0 };
    upsample_40->setScales(scales_40, 3);
    PrintDim(upsample_40, "upsample_40:");

    // [[-1, 22], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_41[] = { upsample_40->getOutput(0), RepNCSPELAN_22->getOutput(0) };
    auto cat_41 = network->addConcatenation(input_tensor_41, 2);
    PrintDim(cat_41, "cat_41:");

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 2]],  # 42 (P3/8-small)
    auto RepNCSPELAN_42 = RepNCSPELAN4(network, weightMap, *cat_41->getOutput(0), 1024, 256, 256, 128, 2, "model.42");
    PrintDim(RepNCSPELAN_42, "RepNCSPELAN_42:");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_43 = ADown(network, weightMap, *RepNCSPELAN_42->getOutput(0), 256, "model.43");
    PrintDim(adown_43, "adown_43:");

    // [[-1, 39], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_44[] = { adown_43->getOutput(0), RepNCSPELAN_39->getOutput(0) };
    auto cat_44 = network->addConcatenation(input_tensor_44, 2);
    PrintDim(cat_44, "cat_44:");

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 2]],  # 45 (P4/16-medium)
    auto RepNCSPELAN_45 = RepNCSPELAN4(network, weightMap, *cat_44->getOutput(0), 768, 512, 512, 256, 2, "model.45");
    PrintDim(RepNCSPELAN_45, "RepNCSPELAN_45:");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_46 = ADown(network, weightMap, *RepNCSPELAN_45->getOutput(0), 512, "model.46");
    PrintDim(adown_46, "adown_46:");

    // [[-1, 36], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_47[] = { adown_46->getOutput(0), SPPELAN_36->getOutput(0) };
    auto cat_47 = network->addConcatenation(input_tensor_47, 2);
    PrintDim(cat_47, "cat_47:");

    // # csp-elan block
    // [-1, 1, RepNCSPELAN4, [512, 1024, 512, 2]],  # 48 (P5/32-large)
    auto RepNCSPELAN_48 = RepNCSPELAN4(network, weightMap, *cat_47->getOutput(0), 1024, 512, 1024, 512, 2, "model.48");
    PrintDim(RepNCSPELAN_48, "RepNCSPELAN_48:");
    
    // auto DualDDetect_49 = DualDDetect(network, weightMap, std::vector<ILayer*>{RepNCSPELAN_42, RepNCSPELAN_45, RepNCSPELAN_48}, kNumClass, {256, 512, 512}, "model.49");
    auto DualDDetect_49 = DualDDetect(network, weightMap, std::vector<ILayer*>{RepNCSPELAN_35, RepNCSPELAN_32, SPPELAN_29}, kNumClass, {256, 512, 512}, "model.49");
    
    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, DualDDetect_49, false);
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
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "../coco_calib/", "int8calib.table", kInputTensorName);
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
IHostMemory* build_engine_yolov9_c(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& wts_name) {
    /* ------ Create the builder ------ */
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    
    // # conv down
    // [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    auto conv_1 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.1", 1);
    PrintDim(conv_1, "conv1:");
    // # conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
    auto conv_2 = convBnSiLU(network, weightMap, *conv_1->getOutput(0), 128, 3, 2, 1, "model.2");
    PrintDim(conv_2, "conv2:");

    // # elan-1 block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 3
    auto RepNCSPELAN_3 = RepNCSPELAN4(network, weightMap, *conv_2->getOutput(0), 128, 256, 128, 64, 1, "model.3");
    PrintDim(RepNCSPELAN_3, "RepNCSPELAN_3:");

    // # avg-conv down
    // [-1, 1, ADown, [256]],  # 4-P3/8
    auto adown_4 = ADown(network, weightMap, *RepNCSPELAN_3->getOutput(0), 256, "model.4");
    PrintDim(adown_4, "adown_4:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 5
    auto RepNCSPELAN_5 = RepNCSPELAN4(network, weightMap, *adown_4->getOutput(0), 256, 512, 256, 128, 1, "model.5");
    PrintDim(RepNCSPELAN_5, "RepNCSPELAN_5:");

    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 6-P4/16
    auto adown_6 = ADown(network, weightMap, *RepNCSPELAN_5->getOutput(0), 512, "model.6");
    PrintDim(adown_6, "adown_6:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 7
    auto RepNCSPELAN_7 = RepNCSPELAN4(network, weightMap, *adown_6->getOutput(0), 512, 512, 512, 256, 1, "model.7");
    PrintDim(RepNCSPELAN_7, "RepNCSPELAN_7:");

    // # avg-conv down
    // [-1, 1, ADown, [512]],  # 8-P5/32
    auto adown_8 = ADown(network, weightMap, *RepNCSPELAN_7->getOutput(0), 512, "model.8");
    PrintDim(adown_8, "adown_8:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 9
    auto RepNCSPELAN_9 = RepNCSPELAN4(network, weightMap, *adown_8->getOutput(0), 512, 512, 512, 256, 1, "model.9");
    PrintDim(RepNCSPELAN_9, "RepNCSPELAN_9:");


    // # elan-spp block
    // [-1, 1, SPPELAN, [512, 256]],  # 10
    auto SPPELAN_10 = SPPELAN(network, weightMap, *RepNCSPELAN_9->getOutput(0), 512, 512, 256, "model.10");
    PrintDim(SPPELAN_10, "SPPELAN_10:");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_11 = network->addResize(*SPPELAN_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    const float scales_11[] = { 1.0, 2.0, 2.0 };
    upsample_11->setScales(scales_11, 3);
    PrintDim(upsample_11, "upsample_11:");


    // [[-1, 7], 1, Concat, [1]],  # cat backbone P4
    ITensor* input_tensor_12[] = { upsample_11->getOutput(0), RepNCSPELAN_7->getOutput(0) };
    auto cat_12 = network->addConcatenation(input_tensor_12, 2);
    PrintDim(cat_12, "cat_12:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 13
    auto RepNCSPELAN_13 = RepNCSPELAN4(network, weightMap, *cat_12->getOutput(0), 1536, 512, 512, 256, 1, "model.13");
    PrintDim(RepNCSPELAN_13, "RepNCSPELAN_13:");

    // # up-concat merge
    // [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    auto upsample_14 = network->addResize(*RepNCSPELAN_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    const float scales_14[] = { 1.0, 2.0, 2.0 };
    upsample_14->setScales(scales_14, 3);
    PrintDim(upsample_14, "upsample_14:");

    // [[-1, 5], 1, Concat, [1]],  # cat backbone P3
    ITensor* input_tensor_15[] = { upsample_14->getOutput(0), RepNCSPELAN_5->getOutput(0) };
    auto cat_15 = network->addConcatenation(input_tensor_15, 2);
    PrintDim(cat_15, "cat_15:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16 (P3/8-small)
    auto RepNCSPELAN_16 = RepNCSPELAN4(network, weightMap, *cat_15->getOutput(0), 1024, 256, 256, 128, 1, "model.16");
    PrintDim(RepNCSPELAN_16, "RepNCSPELAN_16:");

    // # avg-conv-down merge
    // [-1, 1, ADown, [256]],
    auto adown_17 = ADown(network, weightMap, *RepNCSPELAN_16->getOutput(0), 256, "model.17");
    // [[-1, 13], 1, Concat, [1]],  # cat head P4
    ITensor* input_tensor_18[] = { adown_17->getOutput(0), RepNCSPELAN_13->getOutput(0) };
    auto cat_18 = network->addConcatenation(input_tensor_18, 2);
    PrintDim(cat_18, "cat_18:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19 (P4/16-medium)
    auto RepNCSPELAN_19 = RepNCSPELAN4(network, weightMap, *cat_18->getOutput(0), 768, 512, 512, 256, 1, "model.19");
    PrintDim(RepNCSPELAN_19, "RepNCSPELAN_19:");

    // # avg-conv-down merge
    // [-1, 1, ADown, [512]],
    auto adown_20 = ADown(network, weightMap, *RepNCSPELAN_19->getOutput(0), 512, "model.20");
    // [[-1, 10], 1, Concat, [1]],  # cat head P5
    ITensor* input_tensor_21[] = { adown_20->getOutput(0), SPPELAN_10->getOutput(0) };
    auto cat_21 = network->addConcatenation(input_tensor_21, 2);
    PrintDim(cat_21, "cat_21:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 22 (P5/32-large)
    auto RepNCSPELAN_22 = RepNCSPELAN4(network, weightMap, *cat_21->getOutput(0), 1024, 512, 512, 256, 1, "model.22");
    PrintDim(RepNCSPELAN_22, "RepNCSPELAN_22:");
    
    // # multi-level reversible auxiliary branch
    
    // # routing
    // [5, 1, CBLinear, [[256]]], # 23
    auto CBlinear_23 = CBLinear(network, weightMap, *RepNCSPELAN_5->getOutput(0), { 256}, 1, 1, 0, 1, "model.23");
    // [7, 1, CBLinear, [[256, 512]]], # 24
    auto CBlinear_24 = CBLinear(network, weightMap, *RepNCSPELAN_7->getOutput(0), { 256, 512 }, 1, 1, 0, 1, "model.24");
    // [9, 1, CBLinear, [[256, 512, 512]]], # 25
    auto CBlinear_25 = CBLinear(network, weightMap, *RepNCSPELAN_9->getOutput(0), { 256, 512, 512 }, 1, 1, 0, 1, "model.25");
    
    // # conv down
    // [0, 1, Conv, [64, 3, 2]],  # 26-P1/2
    auto conv_26 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.26", 1);
    PrintDim(conv_26, "conv_26:");

    // # conv down
    // [-1, 1, Conv, [128, 3, 2]],  # 27-P2/4
    auto conv_27 = convBnSiLU(network, weightMap, *conv_26->getOutput(0), 128, 3, 2, 1, "model.27");
    PrintDim(conv_27, "conv_27:");

    // # elan-1 block
    // [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 28
    auto RepNCSPELAN_28 = RepNCSPELAN4(network, weightMap, *conv_27->getOutput(0), 128, 256, 128, 64, 1, "model.28");
    PrintDim(RepNCSPELAN_28, "RepNCSPELAN_28:");

    // # avg-conv down fuse
    // [-1, 1, ADown, [256]],  # 29-P3/8
    auto adown_29 = ADown(network, weightMap, *RepNCSPELAN_28->getOutput(0), 256, "model.29");
    PrintDim(adown_29, "adown_29:");
    // [[23, 24, 25, -1], 1, CBFuse, [[0, 0, 0]]], # 30  
    auto cbfuse = CBFuse(network, {CBlinear_23, CBlinear_24, CBlinear_25, std::vector<ILayer*>{adown_29}},
                            {0, 0, 0, 0},
                            {8, 16, 32, 8});
    PrintDim(cbfuse, "cbfuse:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 31
    auto RepNCSPELAN_31 = RepNCSPELAN4(network, weightMap, *cbfuse->getOutput(0), 256, 512, 256, 128, 1, "model.31");
    PrintDim(RepNCSPELAN_31, "RepNCSPELAN_31:");

    // # avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 32-P4/16
    auto adown_32 = ADown(network, weightMap, *RepNCSPELAN_31->getOutput(0), 512, "model.32");
    PrintDim(adown_32, "adown_32:");
    // [[24, 25, -1], 1, CBFuse, [[1, 1]]], # 33 
    auto cbfuse_33 = CBFuse(network, {CBlinear_24, CBlinear_25, std::vector<ILayer*>{adown_32}},
                            {1, 1, 0},
                            {16, 32, 16});
    PrintDim(cbfuse_33, "cbfuse_33:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 34
    auto RepNCSPELAN_34 = RepNCSPELAN4(network, weightMap, *cbfuse_33->getOutput(0), 512, 512, 512, 256, 1, "model.34");
    PrintDim(RepNCSPELAN_34, "RepNCSPELAN_34:");
    
    // # avg-conv down fuse
    // [-1, 1, ADown, [512]],  # 35-P5/32
    auto adown_35 = ADown(network, weightMap, *RepNCSPELAN_34->getOutput(0), 512, "model.35");
    PrintDim(adown_35, "adown_35:");

    // [[25, -1], 1, CBFuse, [[2]]], # 36
    auto cbfuse_36 = CBFuse(network, {CBlinear_25, std::vector<ILayer*>{adown_35}},
                            {2, 0},
                            {32, 32});
    PrintDim(cbfuse_36, "cbfuse_36:");

    // # elan-2 block
    // [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 37
    auto RepNCSPELAN_37 = RepNCSPELAN4(network, weightMap, *cbfuse_36->getOutput(0), 512, 512, 512, 256, 1, "model.37");
    PrintDim(RepNCSPELAN_37, "RepNCSPELAN_37:");
    
    // # detection head
    // # detect
    // [[31, 34, 37, 16, 19, 22], 1, DualDDetect, [nc]],  # DualDDetect(A3, A4, A5, P3, P4, P5)
    auto DualDDetect_38 = DualDDetect(network, weightMap, std::vector<ILayer*>{RepNCSPELAN_31, RepNCSPELAN_34, RepNCSPELAN_37}, kNumClass, {512, 512, 512}, "model.38");
    
    nvinfer1::IPluginV2Layer* yolo = addYoLoLayer(network, DualDDetect_38, false);
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
    auto* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "../coco_calib/", "int8calib.table", kInputTensorName);
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