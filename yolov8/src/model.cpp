#include "model.h"
#include "block.h"
#include "calibrator.h"
#include <iostream>
#include "config.h"
using namespace nvinfer1;

IHostMemory* buildEngineYolov8n(const int& kBatchSize, IBuilder* builder,
IBuilderConfig* config, DataType dt, const std::string& wts_path){
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    INetworkDefinition* network = builder->createNetworkV2(0U);

    /*******************************************************************************************************
    ******************************************  YOLOV8 INPUT  **********************************************
    *******************************************************************************************************/
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{3, kInputH, kInputW});
    assert(data);
    
    /*******************************************************************************************************
    *****************************************  YOLOV8 BACKBONE  ********************************************
    *******************************************************************************************************/
    IElementWiseLayer* conv0 = convBnSiLU(network, weightMap, *data, 16, 3, 2, 1, "model.0");
    IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0), 32, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), 32, 32, 1, true, 0.5, "model.2");
    IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0), 64, 3, 2, 1, "model.3");
    IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), 64, 64, 2, true, 0.5, "model.4");
    IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0), 128, 3, 2, 1, "model.5");
    IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), 128, 128, 2, true, 0.5, "model.6");
    IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0), 256, 3, 2, 1, "model.7");
    IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), 256, 256, 1, true, 0.5, "model.8");
    IElementWiseLayer* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), 256, 256, 5, "model.9");
  
    /*******************************************************************************************************
    *********************************************  YOLOV8 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = {1.0, 2.0, 2.0};
    IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    ITensor* inputTensor11[] = {upsample10->getOutput(0), conv6->getOutput(0)};
    IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);

    IElementWiseLayer* conv12 = C2F(network, weightMap, *cat11->getOutput(0), 128, 128, 1, false, 0.5, "model.12");

    IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    ITensor* inputTensor14[] = {upsample13->getOutput(0), conv4->getOutput(0)};
    IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);

    IElementWiseLayer* conv15 = C2F(network, weightMap, *cat14->getOutput(0), 64, 64, 1, false, 0.5, "model.15");
    IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 64, 3, 2, 1, "model.16");
    ITensor* inputTensor17[] = {conv16->getOutput(0), conv12->getOutput(0)};
    IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    IElementWiseLayer* conv18 = C2F(network, weightMap, *cat17->getOutput(0), 128, 128, 1, false, 0.5, "model.18");
    IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 128, 3, 2, 1, "model.19");
    ITensor* inputTensor20[] = {conv19->getOutput(0), conv9->getOutput(0)};
    IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    IElementWiseLayer* conv21 = C2F(network, weightMap, *cat20->getOutput(0), 256, 256, 1, false, 0.5, "model.21");

    /*******************************************************************************************************
    *********************************************  YOLOV8 OUTPUT  ******************************************
    *******************************************************************************************************/
    // output0
    IElementWiseLayer* conv22_cv2_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.0");
    IElementWiseLayer* conv22_cv2_0_1 = convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.1");
    IConvolutionLayer* conv22_cv2_0_2 = network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, DimsHW{1,1}, weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(DimsHW{1, 1});
    conv22_cv2_0_2->setPaddingNd(DimsHW{0, 0});

    IElementWiseLayer* conv22_cv3_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 80, 3, 1, 1, "model.22.cv3.0.0");
    IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0), 80, 3, 1, 1, "model.22.cv3.0.1");
    IConvolutionLayer* conv22_cv3_0_2 = network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), 80, DimsHW{1,1}, weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(DimsHW{1, 1});
    conv22_cv3_0_2->setPadding(DimsHW{0, 0});
    ITensor* inputTensor22_0[] = {conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0)};
    IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    IElementWiseLayer* conv22_cv2_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.0");
    IElementWiseLayer* conv22_cv2_1_1 = convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.1");
    IConvolutionLayer* conv22_cv2_1_2 = network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, DimsHW{1, 1}, weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(DimsHW{1,1});
    conv22_cv2_1_2->setPaddingNd(DimsHW{0,0});

    IElementWiseLayer* conv22_cv3_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 80, 3, 1, 1, "model.22.cv3.1.0");
    IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0), 80, 3, 1, 1, "model.22.cv3.1.1");
    IConvolutionLayer* conv22_cv3_1_2 = network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), 80, DimsHW{1, 1}, weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(DimsHW{1,1});
    conv22_cv3_1_2->setPaddingNd(DimsHW{0,0});

    ITensor* inputTensor22_1[] = {conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0)};
    IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    IElementWiseLayer* conv22_cv2_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.0");
    IElementWiseLayer* conv22_cv2_2_1 = convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.1");
    IConvolutionLayer* conv22_cv2_2_2 = network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, DimsHW{1,1}, weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);

    IElementWiseLayer* conv22_cv3_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 80, 3, 1, 1, "model.22.cv3.2.0");
    IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0), 80, 3, 1, 1, "model.22.cv3.2.1");
    IConvolutionLayer* conv22_cv3_2_2 = network->addConvolution(*conv22_cv3_2_1->getOutput(0), 80, DimsHW{1,1}, weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);

    ITensor* inputTensor22_2[] = {conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0)};
    IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);


    /*******************************************************************************************************
    *********************************************  YOLOV8 DETECT  ******************************************
    *******************************************************************************************************/
    
    IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(Dims2{144, (kInputH / 8) * (kInputW / 8) });
    
    ISliceLayer* split22_0_0 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{0, 0}, Dims2{64, (kInputH / 8) * (kInputW / 8) }, Dims2{1,1});
    ISliceLayer* split22_0_1 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{64, 0}, Dims2{80, (kInputH / 8) * (kInputW / 8) }, Dims2{1,1});
    IShuffleLayer* dfl22_0 = DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / 8) * (kInputW / 8), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_0[] = {dfl22_0->getOutput(0), split22_0_1->getOutput(0)};
    IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);

    IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(Dims2{144, (kInputH / 16) * (kInputW / 16) });
    ISliceLayer* split22_1_0 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{0, 0}, Dims2{64, (kInputH / 16) * (kInputW / 16) }, Dims2{1,1});
    ISliceLayer* split22_1_1 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{64, 0}, Dims2{ 80, (kInputH / 16) * (kInputW / 16) }, Dims2{1,1});
    IShuffleLayer* dfl22_1 = DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / 16) * (kInputW / 16), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_1[] = {dfl22_1->getOutput(0), split22_1_1->getOutput(0)};
    IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);

    IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(Dims2{144, (kInputH / 32) * (kInputW / 32) });
    ISliceLayer* split22_2_0 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{0, 0}, Dims2{64, (kInputH / 32) * (kInputW / 32) }, Dims2{1,1});
    ISliceLayer* split22_2_1 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{64, 0}, Dims2{ 80, (kInputH / 32) * (kInputW / 32) }, Dims2{1,1});
    IShuffleLayer* dfl22_2 = DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / 32) * (kInputW / 32), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_2[] = {dfl22_2->getOutput(0), split22_2_1->getOutput(0)};
    IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);

    IPluginV2Layer* yolo = addYoLoLayer(network, std::vector<IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2});
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));
    
    builder->setMaxBatchSize(kBatchSize);
    config->setMaxWorkspaceSize(16* (1<<20));

#if defined(USE_FP16) 
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);


#endif
    
    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;

}


IHostMemory* buildEngineYolov8s(const int& kBatchSize, IBuilder* builder,
IBuilderConfig* config, DataType dt, const std::string& wts_path) {

    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    /*******************************************************************************************************
    ******************************************  YOLOV8 INPUT  **********************************************
    *******************************************************************************************************/
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV8 BACKBONE  ********************************************
    *******************************************************************************************************/
    IElementWiseLayer* conv0 = convBnSiLU(network, weightMap, *data, 32, 3, 2, 1, "model.0");
    IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0), 64, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 0.5, "model.2");
    IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0), 128, 3, 2, 1, "model.3");
    IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), 128, 128, 2, true, 0.5, "model.4");
    IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0), 256, 3, 2, 1, "model.5");
    IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), 256, 256, 2, true, 0.5, "model.6");
    IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0), 512, 3, 2, 1, "model.7");
    IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), 512, 512, 1, true, 0.5, "model.8");
    IElementWiseLayer* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), 512, 512, 5, "model.9");
    /*******************************************************************************************************
    *********************************************  YOLOV8 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    assert(upsample10);
    upsample10->setResizeMode(ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    ITensor* inputTensor11[] = { upsample10->getOutput(0), conv6->getOutput(0) };
    IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);

    IElementWiseLayer* conv12 = C2F(network, weightMap, *cat11->getOutput(0), 256, 256, 1, false, 0.5, "model.12");

    IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    ITensor* inputTensor14[] = { upsample13->getOutput(0), conv4->getOutput(0) };
    IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);

    IElementWiseLayer* conv15 = C2F(network, weightMap, *cat14->getOutput(0), 128, 128, 1, false, 0.5, "model.15");
    IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 128, 3, 2, 1, "model.16");
    ITensor* inputTensor17[] = { conv16->getOutput(0), conv12->getOutput(0) };
    IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    IElementWiseLayer* conv18 = C2F(network, weightMap, *cat17->getOutput(0), 256, 256, 1, false, 0.5, "model.18");
    IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 256, 3, 2, 1, "model.19");
    ITensor* inputTensor20[] = { conv19->getOutput(0), conv9->getOutput(0) };
    IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    IElementWiseLayer* conv21 = C2F(network, weightMap, *cat20->getOutput(0), 512, 512, 1, false, 0.5, "model.21");

    /*******************************************************************************************************
    *********************************************  YOLOV8 OUTPUT  ******************************************
    *******************************************************************************************************/
    // output0
    IElementWiseLayer* conv22_cv2_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.0");
    IElementWiseLayer* conv22_cv2_0_1 = convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.1");
    IConvolutionLayer* conv22_cv2_0_2 = network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(DimsHW{ 1, 1 });
    conv22_cv2_0_2->setPaddingNd(DimsHW{ 0, 0 });

    IElementWiseLayer* conv22_cv3_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 128, 3, 1, 1, "model.22.cv3.0.0");
    IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0), 128, 3, 1, 1, "model.22.cv3.0.1");
    IConvolutionLayer* conv22_cv3_0_2 = network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(DimsHW{ 1, 1 });
    conv22_cv3_0_2->setPadding(DimsHW{ 0, 0 });
    ITensor* inputTensor22_0[] = { conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0) };
    IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    IElementWiseLayer* conv22_cv2_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.0");
    IElementWiseLayer* conv22_cv2_1_1 = convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.1");
    IConvolutionLayer* conv22_cv2_1_2 = network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, DimsHW{ 1, 1 }, weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv2_1_2->setPaddingNd(DimsHW{ 0,0 });

    IElementWiseLayer* conv22_cv3_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 128, 3, 1, 1, "model.22.cv3.1.0");
    IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0), 128, 3, 1, 1, "model.22.cv3.1.1");
    IConvolutionLayer* conv22_cv3_1_2 = network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), 80, DimsHW{ 1, 1 }, weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv3_1_2->setPaddingNd(DimsHW{ 0,0 });

    ITensor* inputTensor22_1[] = { conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0) };
    IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    IElementWiseLayer* conv22_cv2_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.0");
    IElementWiseLayer* conv22_cv2_2_1 = convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.1");
    IConvolutionLayer* conv22_cv2_2_2 = network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);

    IElementWiseLayer* conv22_cv3_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 128, 3, 1, 1, "model.22.cv3.2.0");
    IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0), 128, 3, 1, 1, "model.22.cv3.2.1");
    IConvolutionLayer* conv22_cv3_2_2 = network->addConvolution(*conv22_cv3_2_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);

    ITensor* inputTensor22_2[] = { conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0) };
    IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);


    /*******************************************************************************************************
    *********************************************  YOLOV8 DETECT  ******************************************
    *******************************************************************************************************/
    IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(Dims2{ 144, (kInputH / 8) * (kInputW / 8) });

    ISliceLayer* split22_0_0 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    ISliceLayer* split22_0_1 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_0 = DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / 8) * (kInputW / 8), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_0[] = { dfl22_0->getOutput(0), split22_0_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);

    IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(Dims2{ 144, (kInputH / 16) * (kInputW / 16) });
    ISliceLayer* split22_1_0 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    ISliceLayer* split22_1_1 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_1 = DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / 16) * (kInputW / 16), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_1[] = { dfl22_1->getOutput(0), split22_1_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);

    IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(Dims2{ 144, (kInputH / 32) * (kInputW / 32) });
    ISliceLayer* split22_2_0 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    ISliceLayer* split22_2_1 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_2 = DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / 32) * (kInputW / 32), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_2[] = { dfl22_2->getOutput(0), split22_2_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);

    IPluginV2Layer* yolo = addYoLoLayer(network, std::vector<IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2});
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16) 
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}


IHostMemory* buildEngineYolov8m(const int& kBatchSize, IBuilder* builder,
IBuilderConfig* config, DataType dt, const std::string& wts_path) {
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    /*******************************************************************************************************
    ******************************************  YOLOV8 INPUT  **********************************************
    *******************************************************************************************************/
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV8 BACKBONE  ********************************************
    *******************************************************************************************************/
    IElementWiseLayer* conv0 = convBnSiLU(network, weightMap, *data, 48, 3, 2, 1, "model.0");
    IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0), 96, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 0.5, "model.2");
    IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0), 192, 3, 2, 1, "model.3");
    IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), 192, 192, 4, true, 0.5, "model.4");
    IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0), 384, 3, 2, 1, "model.5");
    IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), 384, 384, 4, true, 0.5, "model.6");
    IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0), 576, 3, 2, 1, "model.7");
    IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), 576, 576, 2, true, 0.5, "model.8");
    IElementWiseLayer* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), 576, 576, 5, "model.9");

    /*******************************************************************************************************
    *********************************************  YOLOV8 HEAD  ********************************************
    *******************************************************************************************************/
    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    upsample10->setResizeMode(ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    ITensor* inputTensor11[] = { upsample10->getOutput(0), conv6->getOutput(0) };
    IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);
    IElementWiseLayer* conv12 = C2F(network, weightMap, *cat11->getOutput(0), 384, 384, 2, false, 0.5, "model.12");
    
    IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    ITensor* inputTensor14[] = { upsample13->getOutput(0), conv4->getOutput(0) };
    IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    IElementWiseLayer* conv15 = C2F(network, weightMap, *cat14->getOutput(0), 192, 192, 2, false, 0.5, "model.15");
    IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 192, 3, 2, 1, "model.16");
    ITensor* inputTensor17[] = { conv16->getOutput(0), conv12->getOutput(0) };
    IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    IElementWiseLayer* conv18 = C2F(network, weightMap, *cat17->getOutput(0), 384, 384, 2, false, 0.5, "model.18");
    IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 384, 3, 2, 1, "model.19");
    ITensor* inputTensor20[] = { conv19->getOutput(0), conv9->getOutput(0) };
    IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    IElementWiseLayer* conv21 = C2F(network, weightMap, *cat20->getOutput(0), 576, 576, 2, false, 0.5, "model.21");
    /*******************************************************************************************************
    *********************************************  YOLOV8 OUTPUT  ******************************************
    *******************************************************************************************************/
    // output0
    IElementWiseLayer* conv22_cv2_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.0");
    IElementWiseLayer* conv22_cv2_0_1 = convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.1");
    IConvolutionLayer* conv22_cv2_0_2 = network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(DimsHW{ 1, 1 });
    conv22_cv2_0_2->setPaddingNd(DimsHW{ 0, 0 });

    IElementWiseLayer* conv22_cv3_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 192, 3, 1, 1, "model.22.cv3.0.0");
    IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0), 192, 3, 1, 1, "model.22.cv3.0.1");
    IConvolutionLayer* conv22_cv3_0_2 = network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(DimsHW{ 1, 1 });
    conv22_cv3_0_2->setPadding(DimsHW{ 0, 0 });
    ITensor* inputTensor22_0[] = { conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0) };
    IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    IElementWiseLayer* conv22_cv2_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.0");
    IElementWiseLayer* conv22_cv2_1_1 = convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.1");
    IConvolutionLayer* conv22_cv2_1_2 = network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, DimsHW{ 1, 1 }, weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv2_1_2->setPaddingNd(DimsHW{ 0,0 });

    IElementWiseLayer* conv22_cv3_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 192, 3, 1, 1, "model.22.cv3.1.0");
    IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0), 192, 3, 1, 1, "model.22.cv3.1.1");
    IConvolutionLayer* conv22_cv3_1_2 = network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), 80, DimsHW{ 1, 1 }, weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv3_1_2->setPaddingNd(DimsHW{ 0,0 });

    ITensor* inputTensor22_1[] = { conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0) };
    IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    IElementWiseLayer* conv22_cv2_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.0");
    IElementWiseLayer* conv22_cv2_2_1 = convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.1");
    IConvolutionLayer* conv22_cv2_2_2 = network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);

    IElementWiseLayer* conv22_cv3_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 192, 3, 1, 1, "model.22.cv3.2.0");
    IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0), 192, 3, 1, 1, "model.22.cv3.2.1");
    IConvolutionLayer* conv22_cv3_2_2 = network->addConvolution(*conv22_cv3_2_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);

    ITensor* inputTensor22_2[] = { conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0) };
    IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);
    
    /*******************************************************************************************************
    *********************************************  YOLOV8 DETECT  ******************************************
    *******************************************************************************************************/
    IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(Dims2{ 144, (kInputH / 8) * (kInputW / 8) });

    ISliceLayer* split22_0_0 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    ISliceLayer* split22_0_1 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_0 = DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / 8) * (kInputW / 8), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_0[] = { dfl22_0->getOutput(0), split22_0_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);

    IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(Dims2{ 144, (kInputH / 16) * (kInputW / 16) });
    ISliceLayer* split22_1_0 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    ISliceLayer* split22_1_1 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_1 = DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / 16) * (kInputW / 16), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_1[] = { dfl22_1->getOutput(0), split22_1_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);

    IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(Dims2{ 144, (kInputH / 32) * (kInputW / 32) });
    ISliceLayer* split22_2_0 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    ISliceLayer* split22_2_1 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_2 = DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / 32) * (kInputW / 32), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_2[] = { dfl22_2->getOutput(0), split22_2_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);

    IPluginV2Layer* yolo = addYoLoLayer(network, std::vector<IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2});
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16) 
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}


IHostMemory* buildEngineYolov8l(const int& kBatchSize, IBuilder* builder,
    IBuilderConfig* config, DataType dt, const std::string& wts_path) {
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    /*******************************************************************************************************
    ******************************************  YOLOV8 INPUT  **********************************************
    *******************************************************************************************************/
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV8 BACKBONE  ********************************************
    *******************************************************************************************************/
    IElementWiseLayer* conv0 = convBnSiLU(network, weightMap, *data, 64, 3, 2, 1, "model.0");
    IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0), 128, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), 128, 128, 3, true, 0.5, "model.2");
    IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0), 256, 3, 2, 1, "model.3");
    IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), 256, 256, 6, true, 0.5, "model.4");
    IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0), 512, 3, 2, 1, "model.5");
    IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), 512, 512, 6, true, 0.5, "model.6");
    IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0), 512, 3, 2, 1, "model.7");
    IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), 512, 512, 3, true, 0.5, "model.8");
    IElementWiseLayer* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), 512, 512, 5, "model.9");

    /*******************************************************************************************************
    ******************************************  YOLOV8 HEAD  ***********************************************
    *******************************************************************************************************/
    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    upsample10->setResizeMode(ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    ITensor* inputTensor11[] = { upsample10->getOutput(0), conv6->getOutput(0) };
    IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);
    IElementWiseLayer* conv12 = C2F(network, weightMap, *cat11->getOutput(0), 512, 512, 3, false, 0.5, "model.12");

    IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    ITensor* inputTensor14[] = { upsample13->getOutput(0), conv4->getOutput(0) };
    IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    IElementWiseLayer* conv15 = C2F(network, weightMap, *cat14->getOutput(0), 256, 256, 3, false, 0.5, "model.15");
    IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 256, 3, 2, 1, "model.16");
    ITensor* inputTensor17[] = { conv16->getOutput(0), conv12->getOutput(0) };
    IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    IElementWiseLayer* conv18 = C2F(network, weightMap, *cat17->getOutput(0), 512, 512, 3, false, 0.5, "model.18");
    IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 512, 3, 2, 1, "model.19");
    ITensor* inputTensor20[] = { conv19->getOutput(0), conv9->getOutput(0) };
    IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    IElementWiseLayer* conv21 = C2F(network, weightMap, *cat20->getOutput(0), 512, 512, 3, false, 0.5, "model.21");

    /*******************************************************************************************************
    *********************************************  YOLOV8 OUTPUT  ******************************************
    *******************************************************************************************************/
    // output0
    IElementWiseLayer* conv22_cv2_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.0");
    IElementWiseLayer* conv22_cv2_0_1 = convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.0.1");
    IConvolutionLayer* conv22_cv2_0_2 = network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(DimsHW{ 1, 1 });
    conv22_cv2_0_2->setPaddingNd(DimsHW{ 0, 0 });

    IElementWiseLayer* conv22_cv3_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 256, 3, 1, 1, "model.22.cv3.0.0");
    IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0), 256, 3, 1, 1, "model.22.cv3.0.1");
    IConvolutionLayer* conv22_cv3_0_2 = network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(DimsHW{ 1, 1 });
    conv22_cv3_0_2->setPadding(DimsHW{ 0, 0 });
    ITensor* inputTensor22_0[] = { conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0) };
    IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    IElementWiseLayer* conv22_cv2_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.0");
    IElementWiseLayer* conv22_cv2_1_1 = convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.1.1");
    IConvolutionLayer* conv22_cv2_1_2 = network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, DimsHW{ 1, 1 }, weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv2_1_2->setPaddingNd(DimsHW{ 0,0 });

    IElementWiseLayer* conv22_cv3_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 256, 3, 1, 1, "model.22.cv3.1.0");
    IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0), 256, 3, 1, 1, "model.22.cv3.1.1");
    IConvolutionLayer* conv22_cv3_1_2 = network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), 80, DimsHW{ 1, 1 }, weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv3_1_2->setPaddingNd(DimsHW{ 0,0 });

    ITensor* inputTensor22_1[] = { conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0) };
    IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    IElementWiseLayer* conv22_cv2_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.0");
    IElementWiseLayer* conv22_cv2_2_1 = convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), 64, 3, 1, 1, "model.22.cv2.2.1");
    IConvolutionLayer* conv22_cv2_2_2 = network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);

    IElementWiseLayer* conv22_cv3_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 256, 3, 1, 1, "model.22.cv3.2.0");
    IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0), 256, 3, 1, 1, "model.22.cv3.2.1");
    IConvolutionLayer* conv22_cv3_2_2 = network->addConvolution(*conv22_cv3_2_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);

    ITensor* inputTensor22_2[] = { conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0) };
    IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV8 DETECT  ******************************************
    *******************************************************************************************************/
    IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(Dims2{ 144, (kInputH / 8) * (kInputW / 8) });

    ISliceLayer* split22_0_0 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    ISliceLayer* split22_0_1 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_0 = DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / 8) * (kInputW / 8), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_0[] = { dfl22_0->getOutput(0), split22_0_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);

    IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(Dims2{ 144, (kInputH / 16) * (kInputW / 16) });
    ISliceLayer* split22_1_0 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    ISliceLayer* split22_1_1 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_1 = DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / 16) * (kInputW / 16), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_1[] = { dfl22_1->getOutput(0), split22_1_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);

    IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(Dims2{ 144, (kInputH / 32) * (kInputW / 32) });
    ISliceLayer* split22_2_0 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    ISliceLayer* split22_2_1 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_2 = DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / 32) * (kInputW / 32), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_2[] = { dfl22_2->getOutput(0), split22_2_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);

    IPluginV2Layer* yolo = addYoLoLayer(network, std::vector<IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2});
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16) 
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}


IHostMemory* buildEngineYolov8x(const int& kBatchSize, IBuilder* builder,
IBuilderConfig* config, DataType dt, const std::string& wts_path) {
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    INetworkDefinition* network = builder->createNetworkV2(0U);
    /*******************************************************************************************************
    ******************************************  YOLOV8 INPUT  **********************************************
    *******************************************************************************************************/
    ITensor* data = network->addInput(kInputTensorName, dt, Dims3{ 3, kInputH, kInputW });
    assert(data);

    /*******************************************************************************************************
    *****************************************  YOLOV8 BACKBONE  ********************************************
    *******************************************************************************************************/
    IElementWiseLayer* conv0 = convBnSiLU(network, weightMap, *data, 80, 3, 2, 1, "model.0");
    IElementWiseLayer* conv1 = convBnSiLU(network, weightMap, *conv0->getOutput(0), 160, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = C2F(network, weightMap, *conv1->getOutput(0), 160, 160, 3, true, 0.5, "model.2");
    IElementWiseLayer* conv3 = convBnSiLU(network, weightMap, *conv2->getOutput(0), 320, 3, 2, 1, "model.3");
    IElementWiseLayer* conv4 = C2F(network, weightMap, *conv3->getOutput(0), 320, 320, 6, true, 0.5, "model.4");
    IElementWiseLayer* conv5 = convBnSiLU(network, weightMap, *conv4->getOutput(0), 640, 3, 2, 1, "model.5");
    IElementWiseLayer* conv6 = C2F(network, weightMap, *conv5->getOutput(0), 640, 640, 6, true, 0.5, "model.6");
    IElementWiseLayer* conv7 = convBnSiLU(network, weightMap, *conv6->getOutput(0), 640, 3, 2, 1, "model.7");
    IElementWiseLayer* conv8 = C2F(network, weightMap, *conv7->getOutput(0), 640, 640, 3, true, 0.5, "model.8");
    IElementWiseLayer* conv9 = SPPF(network, weightMap, *conv8->getOutput(0), 640, 640, 5, "model.9");

    /*******************************************************************************************************
    ******************************************  YOLOV8 HEAD  ***********************************************
    *******************************************************************************************************/
    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* upsample10 = network->addResize(*conv9->getOutput(0));
    upsample10->setResizeMode(ResizeMode::kNEAREST);
    upsample10->setScales(scale, 3);

    ITensor* inputTensor11[] = { upsample10->getOutput(0), conv6->getOutput(0) };
    IConcatenationLayer* cat11 = network->addConcatenation(inputTensor11, 2);
    IElementWiseLayer* conv12 = C2F(network, weightMap, *cat11->getOutput(0), 640, 640, 3, false, 0.5, "model.12");

    IResizeLayer* upsample13 = network->addResize(*conv12->getOutput(0));
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setScales(scale, 3);

    ITensor* inputTensor14[] = { upsample13->getOutput(0), conv4->getOutput(0) };
    IConcatenationLayer* cat14 = network->addConcatenation(inputTensor14, 2);
    IElementWiseLayer* conv15 = C2F(network, weightMap, *cat14->getOutput(0), 320, 320, 3, false, 0.5, "model.15");
    IElementWiseLayer* conv16 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 320, 3, 2, 1, "model.16");
    ITensor* inputTensor17[] = { conv16->getOutput(0), conv12->getOutput(0) };
    IConcatenationLayer* cat17 = network->addConcatenation(inputTensor17, 2);
    IElementWiseLayer* conv18 = C2F(network, weightMap, *cat17->getOutput(0), 640, 640, 3, false, 0.5, "model.18");
    IElementWiseLayer* conv19 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 640, 3, 2, 1, "model.19");
    ITensor* inputTensor20[] = { conv19->getOutput(0), conv9->getOutput(0) };
    IConcatenationLayer* cat20 = network->addConcatenation(inputTensor20, 2);
    IElementWiseLayer* conv21 = C2F(network, weightMap, *cat20->getOutput(0), 640, 640, 3, false, 0.5, "model.21");

    /*******************************************************************************************************
    *********************************************  YOLOV8 OUTPUT  ******************************************
    *******************************************************************************************************/
    // output0
    IElementWiseLayer* conv22_cv2_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 80, 3, 1, 1, "model.22.cv2.0.0");
    IElementWiseLayer* conv22_cv2_0_1 = convBnSiLU(network, weightMap, *conv22_cv2_0_0->getOutput(0), 80, 3, 1, 1, "model.22.cv2.0.1");
    IConvolutionLayer* conv22_cv2_0_2 = network->addConvolutionNd(*conv22_cv2_0_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.0.2.weight"], weightMap["model.22.cv2.0.2.bias"]);
    conv22_cv2_0_2->setStrideNd(DimsHW{ 1, 1 });
    conv22_cv2_0_2->setPaddingNd(DimsHW{ 0, 0 });

    IElementWiseLayer* conv22_cv3_0_0 = convBnSiLU(network, weightMap, *conv15->getOutput(0), 320, 3, 1, 1, "model.22.cv3.0.0");
    IElementWiseLayer* conv22_cv3_0_1 = convBnSiLU(network, weightMap, *conv22_cv3_0_0->getOutput(0), 320, 3, 1, 1, "model.22.cv3.0.1");
    IConvolutionLayer* conv22_cv3_0_2 = network->addConvolutionNd(*conv22_cv3_0_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.0.2.weight"], weightMap["model.22.cv3.0.2.bias"]);
    conv22_cv3_0_2->setStride(DimsHW{ 1, 1 });
    conv22_cv3_0_2->setPadding(DimsHW{ 0, 0 });
    ITensor* inputTensor22_0[] = { conv22_cv2_0_2->getOutput(0), conv22_cv3_0_2->getOutput(0) };
    IConcatenationLayer* cat22_0 = network->addConcatenation(inputTensor22_0, 2);

    // output1
    IElementWiseLayer* conv22_cv2_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 80, 3, 1, 1, "model.22.cv2.1.0");
    IElementWiseLayer* conv22_cv2_1_1 = convBnSiLU(network, weightMap, *conv22_cv2_1_0->getOutput(0), 80, 3, 1, 1, "model.22.cv2.1.1");
    IConvolutionLayer* conv22_cv2_1_2 = network->addConvolutionNd(*conv22_cv2_1_1->getOutput(0), 64, DimsHW{ 1, 1 }, weightMap["model.22.cv2.1.2.weight"], weightMap["model.22.cv2.1.2.bias"]);
    conv22_cv2_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv2_1_2->setPaddingNd(DimsHW{ 0,0 });

    IElementWiseLayer* conv22_cv3_1_0 = convBnSiLU(network, weightMap, *conv18->getOutput(0), 320, 3, 1, 1, "model.22.cv3.1.0");
    IElementWiseLayer* conv22_cv3_1_1 = convBnSiLU(network, weightMap, *conv22_cv3_1_0->getOutput(0), 320, 3, 1, 1, "model.22.cv3.1.1");
    IConvolutionLayer* conv22_cv3_1_2 = network->addConvolutionNd(*conv22_cv3_1_1->getOutput(0), 80, DimsHW{ 1, 1 }, weightMap["model.22.cv3.1.2.weight"], weightMap["model.22.cv3.1.2.bias"]);
    conv22_cv3_1_2->setStrideNd(DimsHW{ 1,1 });
    conv22_cv3_1_2->setPaddingNd(DimsHW{ 0,0 });

    ITensor* inputTensor22_1[] = { conv22_cv2_1_2->getOutput(0), conv22_cv3_1_2->getOutput(0) };
    IConcatenationLayer* cat22_1 = network->addConcatenation(inputTensor22_1, 2);

    // output2
    IElementWiseLayer* conv22_cv2_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 80, 3, 1, 1, "model.22.cv2.2.0");
    IElementWiseLayer* conv22_cv2_2_1 = convBnSiLU(network, weightMap, *conv22_cv2_2_0->getOutput(0), 80, 3, 1, 1, "model.22.cv2.2.1");
    IConvolutionLayer* conv22_cv2_2_2 = network->addConvolution(*conv22_cv2_2_1->getOutput(0), 64, DimsHW{ 1,1 }, weightMap["model.22.cv2.2.2.weight"], weightMap["model.22.cv2.2.2.bias"]);

    IElementWiseLayer* conv22_cv3_2_0 = convBnSiLU(network, weightMap, *conv21->getOutput(0), 320, 3, 1, 1, "model.22.cv3.2.0");
    IElementWiseLayer* conv22_cv3_2_1 = convBnSiLU(network, weightMap, *conv22_cv3_2_0->getOutput(0), 320, 3, 1, 1, "model.22.cv3.2.1");
    IConvolutionLayer* conv22_cv3_2_2 = network->addConvolution(*conv22_cv3_2_1->getOutput(0), 80, DimsHW{ 1,1 }, weightMap["model.22.cv3.2.2.weight"], weightMap["model.22.cv3.2.2.bias"]);

    ITensor* inputTensor22_2[] = { conv22_cv2_2_2->getOutput(0), conv22_cv3_2_2->getOutput(0) };
    IConcatenationLayer* cat22_2 = network->addConcatenation(inputTensor22_2, 2);

    /*******************************************************************************************************
    *********************************************  YOLOV8 DETECT  ******************************************
    *******************************************************************************************************/
    IShuffleLayer* shuffle22_0 = network->addShuffle(*cat22_0->getOutput(0));
    shuffle22_0->setReshapeDimensions(Dims2{ 144, (kInputH / 8) * (kInputW / 8) });

    ISliceLayer* split22_0_0 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    ISliceLayer* split22_0_1 = network->addSlice(*shuffle22_0->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 8) * (kInputW / 8) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_0 = DFL(network, weightMap, *split22_0_0->getOutput(0), 4, (kInputH / 8) * (kInputW / 8), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_0[] = { dfl22_0->getOutput(0), split22_0_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_0 = network->addConcatenation(inputTensor22_dfl_0, 2);

    IShuffleLayer* shuffle22_1 = network->addShuffle(*cat22_1->getOutput(0));
    shuffle22_1->setReshapeDimensions(Dims2{ 144, (kInputH / 16) * (kInputW / 16) });
    ISliceLayer* split22_1_0 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    ISliceLayer* split22_1_1 = network->addSlice(*shuffle22_1->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 16) * (kInputW / 16) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_1 = DFL(network, weightMap, *split22_1_0->getOutput(0), 4, (kInputH / 16) * (kInputW / 16), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_1[] = { dfl22_1->getOutput(0), split22_1_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_1 = network->addConcatenation(inputTensor22_dfl_1, 2);

    IShuffleLayer* shuffle22_2 = network->addShuffle(*cat22_2->getOutput(0));
    shuffle22_2->setReshapeDimensions(Dims2{ 144, (kInputH / 32) * (kInputW / 32) });
    ISliceLayer* split22_2_0 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 0, 0 }, Dims2{ 64, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    ISliceLayer* split22_2_1 = network->addSlice(*shuffle22_2->getOutput(0), Dims2{ 64, 0 }, Dims2{ 80, (kInputH / 32) * (kInputW / 32) }, Dims2{ 1,1 });
    IShuffleLayer* dfl22_2 = DFL(network, weightMap, *split22_2_0->getOutput(0), 4, (kInputH / 32) * (kInputW / 32), 1, 1, 0, "model.22.dfl.conv.weight");
    ITensor* inputTensor22_dfl_2[] = { dfl22_2->getOutput(0), split22_2_1->getOutput(0) };
    IConcatenationLayer* cat22_dfl_2 = network->addConcatenation(inputTensor22_dfl_2, 2);

    IPluginV2Layer* yolo = addYoLoLayer(network, std::vector<IConcatenationLayer*>{cat22_dfl_0, cat22_dfl_1, cat22_dfl_2});
    yolo->getOutput(0)->setName(kOutputTensorName);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(kBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));

#if defined(USE_FP16) 
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    delete network;

    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }
    return serialized_model;
}
