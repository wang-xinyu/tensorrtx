#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include<cassert>


#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

// #define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32



static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static Logger gLogger;

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

inline int Get_channel(int x, int gw = 1, float divisor = 8.0){
  // std::cout << "=======" << (x*gw) / divisor << "===============" << std::endl;
  auto ch_out = int(ceil((x * gw) / divisor)) * divisor;
  return ch_out;
}

nvinfer1::ICudaEngine *build_det_v5_lite_c(unsigned int maxBatchSize, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, 
       nvinfer1::DataType dt, std::string wts_name)
{

  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
  nvinfer1::ITensor *data = network->addInput(Yolo::INPUT_BLOB_NAME, dt, nvinfer1::Dims3{3, Yolo::INPUT_W, Yolo::INPUT_H});
  std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_name);


  // backbone
  nvinfer1::IElementWiseLayer *conv0 = CBH(network, weightMap, *data, Get_channel(32), 3, 2, "model.0");
  nvinfer1::IElementWiseLayer *conv1 = LC_Block(network, weightMap, *conv0->getOutput(0), Get_channel(32), Get_channel(64), 2, 3, "model.1", false);
  nvinfer1::IElementWiseLayer *conv2 = LC_Block(network, weightMap, *conv1->getOutput(0), Get_channel(64), Get_channel(64), 1, 3, "model.2", false);
  nvinfer1::IElementWiseLayer *conv3 = LC_Block(network, weightMap, *conv2->getOutput(0), Get_channel(64), Get_channel(128), 2, 3, "model.3", false);
  nvinfer1::IElementWiseLayer *conv4 = LC_Block(network, weightMap, *conv3->getOutput(0), Get_channel(128), Get_channel(128), 1, 3, "model.4", false);
  nvinfer1::IElementWiseLayer *conv5 = LC_Block(network, weightMap, *conv4->getOutput(0), Get_channel(128), Get_channel(128), 1, 3, "model.5", false);
  nvinfer1::IElementWiseLayer *conv6 = LC_Block(network, weightMap, *conv5->getOutput(0), Get_channel(128), Get_channel(128), 1, 3, "model.6", false);
  nvinfer1::IElementWiseLayer *conv7 = LC_Block(network, weightMap, *conv6->getOutput(0), Get_channel(128), Get_channel(256), 2, 3, "model.7", false);
  nvinfer1::IElementWiseLayer *conv8 = LC_Block(network, weightMap, *conv7->getOutput(0), Get_channel(256), Get_channel(256), 1, 5, "model.8", false);
  nvinfer1::IElementWiseLayer *conv9 = LC_Block(network, weightMap, *conv8->getOutput(0), Get_channel(256), Get_channel(256), 1, 5, "model.9", false);
  nvinfer1::IElementWiseLayer *conv10 = LC_Block(network, weightMap, *conv9->getOutput(0), Get_channel(256), Get_channel(256), 1, 5, "model.10", false);
  nvinfer1::IElementWiseLayer *conv11 = LC_Block(network, weightMap, *conv10->getOutput(0), Get_channel(256), Get_channel(256), 1, 5, "model.11", false);
  nvinfer1::IElementWiseLayer *conv12 = LC_Block(network, weightMap, *conv11->getOutput(0), Get_channel(256), Get_channel(256), 1, 5, "model.12", false);
  nvinfer1::IElementWiseLayer *conv13 = LC_Block(network, weightMap, *conv12->getOutput(0), Get_channel(256), Get_channel(512), 2, 5, "model.13", true);
  nvinfer1::IElementWiseLayer *conv14 = LC_Block(network, weightMap, *conv13->getOutput(0), Get_channel(512), Get_channel(512), 1, 5, "model.14", true);
  nvinfer1::IElementWiseLayer *conv15 = LC_Block(network, weightMap, *conv14->getOutput(0), Get_channel(512), Get_channel(512), 1, 5, "model.15", true);
  nvinfer1::IElementWiseLayer *conv16 = LC_Block(network, weightMap, *conv15->getOutput(0), Get_channel(512), Get_channel(512), 1, 5, "model.16", true);
  nvinfer1::IElementWiseLayer *conv17 = Dense(network, weightMap, *conv16->getOutput(0), Get_channel(512), 1, "model.17");

  // neck
  float scale[] = {1.0, 2.0, 2.0};
  nvinfer1::IElementWiseLayer *conv18 = convBlock(network, weightMap, *conv17->getOutput(0), Get_channel(256), 1, 1, 1, "model.18");
  nvinfer1::IResizeLayer *upsample19 = network->addResize(*conv18->getOutput(0));
  upsample19->setScales(scale, 3);
  nvinfer1::ITensor *inputTensors20[] = {upsample19->getOutput(0), conv12->getOutput(0)}; // 256 + 256 = 512
  nvinfer1::IConcatenationLayer *cat20 = network->addConcatenation(inputTensors20, 2);
  nvinfer1::IElementWiseLayer *conv21 = C3(network, weightMap, *cat20->getOutput(0), 512, Get_channel(256), get_depth(1, 1), false, 1, 0.5, "model.21");

  nvinfer1::IElementWiseLayer *conv22 = convBlock(network, weightMap, *conv21->getOutput(0), Get_channel(128), 1, 1, 1, "model.22");
  nvinfer1::IResizeLayer *upsample23 = network->addResize(*conv22->getOutput(0));
  upsample23->setScales(scale, 3);
  nvinfer1::ITensor *inputTensors24[] = {upsample23->getOutput(0), conv6->getOutput(0)}; // 128 + 128 = 256
  nvinfer1::IConcatenationLayer *cat24 = network->addConcatenation(inputTensors24, 2);
  nvinfer1::IElementWiseLayer *conv25 = C3(network, weightMap, *cat24->getOutput(0), 256, Get_channel(128), get_depth(1, 1), false, 1, 0.5, "model.25");

  nvinfer1::IElementWiseLayer *conv26 = LC_Block(network, weightMap, *conv25->getOutput(0), Get_channel(128), Get_channel(128), 2, 5, "model.26", true);
  nvinfer1::ITensor *inputTensor27[] = {conv26->getOutput(0), conv22->getOutput(0)}; // 128 + 128 = 256
  nvinfer1::IConcatenationLayer *cat27 = network->addConcatenation(inputTensor27, 2);
  nvinfer1::IElementWiseLayer *conv28 = C3(network, weightMap, *cat27->getOutput(0), 256, Get_channel(256), get_depth(1, 1), false, 1, 0.5, "model.28");

  nvinfer1::IElementWiseLayer *conv29 = LC_Block(network, weightMap, *conv28->getOutput(0), Get_channel(256), Get_channel(256), 2, 5, "model.29", true);
  nvinfer1::ITensor *inputTensor30[] = {conv29->getOutput(0), conv18->getOutput(0)}; // 256 + 256 = 512
  nvinfer1::IConcatenationLayer *cat30 = network->addConcatenation(inputTensor30, 2);
  nvinfer1::IElementWiseLayer *conv31 = C3(network, weightMap, *cat30->getOutput(0), 512, Get_channel(512), get_depth(1, 1), false, 1, 0.5, "model.31");

    // detect
  nvinfer1::IConvolutionLayer *det0 = network->addConvolutionNd(*conv25->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.32.m.0.weight"], weightMap["model.32.m.0.bias"]);
    
  nvinfer1::IConvolutionLayer *det1 = network->addConvolutionNd(*conv28->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.32.m.1.weight"], weightMap["model.32.m.1.bias"]);
    
  nvinfer1::IConvolutionLayer *det2 = network->addConvolutionNd(*conv31->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
        nvinfer1::DimsHW{1, 1}, weightMap["model.32.m.2.weight"], weightMap["model.32.m.2.bias"]);
    
  auto yolo = addYoLoLayer(network, weightMap, "model.32", std::vector<nvinfer1::IConvolutionLayer*>{det0, det1, det2});
  yolo->getOutput(0)->setName(Yolo::OUTPUT_BLOB_NAME);
  network->markOutput(*yolo->getOutput(0));

      // Engine config
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(BuilderFlag::kINT8);
  std::string data_path = "tensorrtx-int8calib-data/coco_calib/";
  //Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
  Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, Yolo::INPUT_W, Yolo::INPUT_H, data_path.c_str(), "int8calib.table", Yolo::INPUT_BLOB_NAME);
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


nvinfer1::ICudaEngine *build_det_v5_lite_e(unsigned int maxBatchSize, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config,
    nvinfer1::DataType dt, std::string wts_name){
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);
  nvinfer1::ITensor *data = network->addInput(Yolo::INPUT_BLOB_NAME, dt, nvinfer1::Dims3{3, Yolo::INPUT_W, Yolo::INPUT_H});
  std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_name);

  // backbone
  nvinfer1::IPoolingLayer *conv0 = conv_bn_relu_maxpool(network, weightMap, *data, 32, "model.0."); //32
  // std::cout << "Get_channel: " << Get_channel(116) << std::endl;
  nvinfer1::IShuffleLayer *conv1 = shuffle_block(network, weightMap, *conv0->getOutput(0), "model.1.", 32, Get_channel(116), 2); //120
  nvinfer1::IShuffleLayer *conv2_0 = shuffle_block(network, weightMap, *conv1->getOutput(0), "model.2.0.", Get_channel(116), Get_channel(116), 1); //120
  nvinfer1::IShuffleLayer *conv2_1 = shuffle_block(network, weightMap, *conv2_0->getOutput(0), "model.2.1.", Get_channel(116), Get_channel(116), 1); // 120
  nvinfer1::IShuffleLayer *conv2_2 = shuffle_block(network, weightMap, *conv2_1->getOutput(0), "model.2.2.", Get_channel(116), Get_channel(116), 1); // 120
  nvinfer1::IShuffleLayer *conv3 = shuffle_block(network, weightMap, *conv2_2->getOutput(0), "model.3.", Get_channel(116), Get_channel(232), 2); // 232
  nvinfer1::IShuffleLayer *conv4_0 = shuffle_block(network, weightMap, *conv3->getOutput(0), "model.4.0.", Get_channel(232), Get_channel(232), 1); // 232 
  nvinfer1::IShuffleLayer *conv4_1 = shuffle_block(network, weightMap, *conv4_0->getOutput(0), "model.4.1.", Get_channel(232), Get_channel(232), 1); // 232
  nvinfer1::IShuffleLayer *conv4_2 = shuffle_block(network, weightMap, *conv4_1->getOutput(0), "model.4.2.", Get_channel(232), Get_channel(232), 1); // 232
  nvinfer1::IShuffleLayer *conv4_3 = shuffle_block(network, weightMap, *conv4_2->getOutput(0), "model.4.3.", Get_channel(232), Get_channel(232), 1); // 232
  nvinfer1::IShuffleLayer *conv4_4 = shuffle_block(network, weightMap, *conv4_3->getOutput(0), "model.4.4.", Get_channel(232), Get_channel(232), 1); //232
  nvinfer1::IShuffleLayer *conv4_5 = shuffle_block(network, weightMap, *conv4_4->getOutput(0), "model.4.5.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_6 = shuffle_block(network, weightMap, *conv4_5->getOutput(0), "model.4.6.", Get_channel(232), Get_channel(232), 1); // 232
  nvinfer1::IShuffleLayer *conv5 = shuffle_block(network, weightMap, *conv4_6->getOutput(0), "model.5.", Get_channel(232), Get_channel(464), 2); //464 
  nvinfer1::IShuffleLayer *conv6 = shuffle_block(network, weightMap, *conv5->getOutput(0), "model.6.", Get_channel(464), Get_channel(464), 1); // 464

  // neck
  float scale[] = {1.0, 2.0, 2.0};
  nvinfer1::IElementWiseLayer *conv7 = convBlock(network, weightMap, *conv6->getOutput(0), Get_channel(96), 1, 1, 1, "model.7"); // 96
  nvinfer1::IResizeLayer *upsample8 = network->addResize(*conv7->getOutput(0));
  upsample8->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  upsample8->setScales(scale, 3);
  nvinfer1::ITensor *inputTensors9[] = {upsample8->getOutput(0), conv4_6->getOutput(0)};
  nvinfer1::IConcatenationLayer *cat9 = network->addConcatenation(inputTensors9, 2); //  96 + 232 = 328
  nvinfer1::IActivationLayer *conv10 = DWConvblock(network, weightMap, *cat9->getOutput(0), "model.10", 328, Get_channel(96), 3, 1);

  nvinfer1::IElementWiseLayer *conv11 = convBlock(network, weightMap, *conv10->getOutput(0), Get_channel(96), 1, 1, 1, "model.11"); // 96
  nvinfer1::IResizeLayer *upsample12 = network->addResize(*conv11->getOutput(0));
  upsample12->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  upsample12->setScales(scale, 3);
  nvinfer1::ITensor *inputTensors13[] = {upsample12->getOutput(0), conv2_2->getOutput(0)}; // 96 + 120 
  nvinfer1::IConcatenationLayer *cat13 = network->addConcatenation(inputTensors13, 2);
  nvinfer1::IActivationLayer *conv14 = DWConvblock(network, weightMap, *cat13->getOutput(0), "model.14", 216, Get_channel(96), 3, 1);

  nvinfer1::IActivationLayer *conv15 = DWConvblock(network, weightMap, *conv14->getOutput(0), "model.15", Get_channel(96), Get_channel(96), 3, 2);
  nvinfer1::IElementWiseLayer *add16 = ADD(network, *conv15->getOutput(0), *conv11->getOutput(0), 1.0);
  nvinfer1::IActivationLayer *conv17 = DWConvblock(network, weightMap, *add16->getOutput(0), "model.17", Get_channel(96), Get_channel(96), 3, 1);

  nvinfer1::IActivationLayer *conv18 = DWConvblock(network, weightMap, *conv17->getOutput(0), "model.18", Get_channel(96), Get_channel(96), 3, 2);
  nvinfer1::IElementWiseLayer *add19 = ADD(network, *conv18->getOutput(0), *conv7->getOutput(0), 1.0);
  nvinfer1::IActivationLayer *conv20 = DWConvblock(network, weightMap, *add19->getOutput(0), "model.20", Get_channel(96), Get_channel(96), 3, 1);



  // detect
  nvinfer1::IConvolutionLayer *det0 = network->addConvolutionNd(*conv14->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.21.m.0.weight"], weightMap["model.21.m.0.bias"]);
    
  nvinfer1::IConvolutionLayer *det1 = network->addConvolutionNd(*conv17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.21.m.1.weight"], weightMap["model.21.m.1.bias"]);
    
  nvinfer1::IConvolutionLayer *det2 = network->addConvolutionNd(*conv20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
        nvinfer1::DimsHW{1, 1}, weightMap["model.21.m.2.weight"], weightMap["model.21.m.2.bias"]);
    
  auto yolo = addYoLoLayer(network, weightMap, "model.21", std::vector<nvinfer1::IConvolutionLayer*>{det0, det1, det2});
  yolo->getOutput(0)->setName(Yolo::OUTPUT_BLOB_NAME);
  network->markOutput(*yolo->getOutput(0));

      // Engine config
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(BuilderFlag::kINT8);
  std::string data_path = "tensorrtx-int8calib-data/coco_calib/";
  //Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
  Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, Yolo::INPUT_W, Yolo::INPUT_H, data_path.c_str(), "int8calib.table", Yolo::INPUT_BLOB_NAME);
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


nvinfer1::ICudaEngine *build_det_v5_lite_g(unsigned int maxBatchSize, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, 
                  nvinfer1::DataType dt,  std::string wts_name){
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);

    // backbone
    nvinfer1::ITensor *data = network->addInput(Yolo::INPUT_BLOB_NAME, dt, nvinfer1::Dims3{3, Yolo::INPUT_H, Yolo::INPUT_W});
    assert(data);
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_name);
    nvinfer1::IElementWiseLayer *conv0 = focus(network, weightMap, *data, 3, Get_channel(32), 3, "model.0"); // 32
    nvinfer1::IActivationLayer *conv1 = RepVGGBlock(network, weightMap, *conv0->getOutput(0), "model.1", Get_channel(64), 3, 2, 1); //64
    nvinfer1::IElementWiseLayer *conv2 = C3(network, weightMap, *conv1->getOutput(0), Get_channel(64), Get_channel(64), get_depth(1, 1), true, 1, 0.5, "model.2"); // 64
    nvinfer1::IActivationLayer *conv3 = RepVGGBlock(network, weightMap, *conv2->getOutput(0), "model.3", Get_channel(128), 3, 2, 1); // 128
    nvinfer1::IElementWiseLayer *conv4 = C3(network, weightMap, *conv3->getOutput(0), Get_channel(128), Get_channel(128), get_depth(3, 1), true, 1, 0.5, "model.4"); // 128
    nvinfer1::IActivationLayer *conv5 = RepVGGBlock(network, weightMap, *conv4->getOutput(0), "model.5", Get_channel(256), 3, 2, 1); // 256
    nvinfer1::IElementWiseLayer *conv6 = C3(network, weightMap, *conv5->getOutput(0), Get_channel(256), Get_channel(256), get_depth(3, 1), true, 1, 0.5, "model.6"); // 256
    nvinfer1::IActivationLayer *conv7 = RepVGGBlock(network, weightMap, *conv6->getOutput(0), "model.7", Get_channel(512), 3, 2, 1); // 512
    nvinfer1::IElementWiseLayer *conv8 = SPP(network, weightMap, *conv7->getOutput(0), Get_channel(512), Get_channel(512), 5, 9, 13, "model.8"); // 512
    nvinfer1::IElementWiseLayer *conv9 = C3(network, weightMap, *conv8->getOutput(0), Get_channel(512), Get_channel(512), get_depth(1, 1), false, 1, 0.5, "model.9"); // 512
    

    float scale[] = {1.0, 2.0, 2.0};
    nvinfer1::IElementWiseLayer *conv10 = convBlock(network, weightMap, *conv9->getOutput(0), Get_channel(128), 1, 1, 1, "model.10"); // 128
    nvinfer1::IResizeLayer *upsample11 = network->addResize(*conv10->getOutput(0));
    upsample11->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample11->setScales(scale, 3);
    nvinfer1::ITensor *inputTensors12[] = {upsample11->getOutput(0), conv6->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat12 = network->addConcatenation(inputTensors12, 2); // 384
    nvinfer1::IElementWiseLayer *conv13 = C3(network, weightMap, *cat12->getOutput(0), 384, Get_channel(128), get_depth(3, 1), false, 1, 0.5, "model.13");

    nvinfer1::IElementWiseLayer *conv14 = convBlock(network, weightMap, *conv13->getOutput(0), Get_channel(128), 1, 1, 1, "model.14"); // 128
    nvinfer1::IResizeLayer *upsample15 = network->addResize(*conv14->getOutput(0));
    upsample15->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
    upsample15->setScales(scale, 3);
    nvinfer1::ITensor *inputTensors16[] = {upsample15->getOutput(0), conv4->getOutput(0)}; //  128+128
    nvinfer1::IConcatenationLayer *cat16 = network->addConcatenation(inputTensors16, 2);
    nvinfer1::IElementWiseLayer *conv17 = C3(network, weightMap, *cat16->getOutput(0), 256, Get_channel(128), get_depth(3, 1), false, 1, 0.5, "model.17");

    nvinfer1::IElementWiseLayer *conv18 = convBlock(network, weightMap, *conv17->getOutput(0), Get_channel(128), 3, 2, 1, "model.18"); // 128
    nvinfer1::ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    nvinfer1::IConcatenationLayer *cat19 = network->addConcatenation(inputTensors19, 2); // 128 + 128
    nvinfer1::IElementWiseLayer *conv20 = C3(network, weightMap, *cat19->getOutput(0), 256, Get_channel(128), get_depth(3, 1), false, 1, 0.5, "model.20");

    nvinfer1::IElementWiseLayer *conv21 = convBlock(network, weightMap, *conv20->getOutput(0), Get_channel(128), 3, 2, 1, "model.21"); // 128
    nvinfer1::ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)}; 
    nvinfer1::IConcatenationLayer *cat22 = network->addConcatenation(inputTensors22, 2); // 128 + 128
    nvinfer1::IElementWiseLayer *conv23 = C3(network, weightMap, *cat22->getOutput(0), 256, Get_channel(128), get_depth(3, 1), false, 1, 0.5, "model.23");

      // detect
    nvinfer1::IConvolutionLayer *det0 = network->addConvolutionNd(*conv17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    
    nvinfer1::IConvolutionLayer *det1 = network->addConvolutionNd(*conv20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    
    nvinfer1::IConvolutionLayer *det2 = network->addConvolutionNd(*conv23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
        nvinfer1::DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
    
    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<nvinfer1::IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(Yolo::OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

      // Engine config
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(BuilderFlag::kINT8);
  std::string data_path = "tensorrtx-int8calib-data/coco_calib/";
  //Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
  Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, Yolo::INPUT_W, Yolo::INPUT_H, data_path.c_str(), "int8calib.table", Yolo::INPUT_BLOB_NAME);
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




nvinfer1::ICudaEngine *build_det_v5_lite_s(unsigned int maxBatchSize, nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, nvinfer1::DataType dt,std::string & wts_name){
  // backbone
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
  nvinfer1::ITensor *data = network->addInput(Yolo::INPUT_BLOB_NAME, dt, nvinfer1::Dims3{3, Yolo::INPUT_H, Yolo::INPUT_W});
  assert(data);
  std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(wts_name);
  nvinfer1::IPoolingLayer *conv0 = conv_bn_relu_maxpool(network, weightMap, *data, 32, "model.0.");
  std::cout << "Get_channel: " << Get_channel(116) << std::endl;
  nvinfer1::IShuffleLayer *conv1 = shuffle_block(network, weightMap, *conv0->getOutput(0), "model.1.", 32, Get_channel(116), 2);
  nvinfer1::IShuffleLayer *conv2_0 = shuffle_block(network, weightMap, *conv1->getOutput(0), "model.2.0.", Get_channel(116), Get_channel(116), 1);
  nvinfer1::IShuffleLayer *conv2_1 = shuffle_block(network, weightMap, *conv2_0->getOutput(0), "model.2.1.", Get_channel(116), Get_channel(116), 1);
  nvinfer1::IShuffleLayer *conv2_2 = shuffle_block(network, weightMap, *conv2_1->getOutput(0), "model.2.2.", Get_channel(116), Get_channel(116), 1);
  nvinfer1::IShuffleLayer *conv3 = shuffle_block(network, weightMap, *conv2_2->getOutput(0), "model.3.", Get_channel(116), Get_channel(232), 2);
  nvinfer1::IShuffleLayer *conv4_0 = shuffle_block(network, weightMap, *conv3->getOutput(0), "model.4.0.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_1 = shuffle_block(network, weightMap, *conv4_0->getOutput(0), "model.4.1.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_2 = shuffle_block(network, weightMap, *conv4_1->getOutput(0), "model.4.2.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_3 = shuffle_block(network, weightMap, *conv4_2->getOutput(0), "model.4.3.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_4 = shuffle_block(network, weightMap, *conv4_3->getOutput(0), "model.4.4.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_5 = shuffle_block(network, weightMap, *conv4_4->getOutput(0), "model.4.5.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv4_6 = shuffle_block(network, weightMap, *conv4_5->getOutput(0), "model.4.6.", Get_channel(232), Get_channel(232), 1);
  nvinfer1::IShuffleLayer *conv5 = shuffle_block(network, weightMap, *conv4_6->getOutput(0), "model.5.", Get_channel(232), Get_channel(464), 2);
  nvinfer1::IShuffleLayer *conv6_0 = shuffle_block(network, weightMap, *conv5->getOutput(0), "model.6.0.", Get_channel(464), Get_channel(464), 1);
  nvinfer1::IShuffleLayer *conv6_1 = shuffle_block(network, weightMap, *conv6_0->getOutput(0), "model.6.1.", Get_channel(464), Get_channel(464), 1);
  nvinfer1::IShuffleLayer *conv6_2 = shuffle_block(network, weightMap, *conv6_1->getOutput(0), "model.6.2.", Get_channel(464), Get_channel(464), 1);

  // head
  float scale[] = {1.0, 2.0, 2.0};
  nvinfer1::IElementWiseLayer *conv7 = convBlock(network, weightMap, *conv6_2->getOutput(0), Get_channel(128), 1, 1, 1, "model.7");
  nvinfer1::IResizeLayer *upsample8 = network->addResize(*conv7->getOutput(0));
  upsample8->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  upsample8->setScales(scale, 3);
  assert(upsample8);
  nvinfer1::ITensor *inputTensors9[] = {upsample8->getOutput(0), conv4_6->getOutput(0)}; // channels = 128 + 232 = 360
  nvinfer1::IConcatenationLayer *cat9 = network->addConcatenation(inputTensors9, 2);
  // std::cout << "The c3 's n is " << get_depth(3, 1) << std::endl;
  nvinfer1::IElementWiseLayer *conv10 = C3(network, weightMap, *cat9->getOutput(0), 360, Get_channel(128), get_depth(1, 1), false, 1, 0.5, "model.10");

  nvinfer1::IElementWiseLayer *conv11 = convBlock(network, weightMap, *conv10->getOutput(0), Get_channel(64), 1, 1, 1, "model.11");
  nvinfer1::IResizeLayer *upsample12 = network->addResize(*conv11->getOutput(0));
  upsample12->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
  upsample12->setScales(scale, 3);
  assert(upsample12);
  nvinfer1::ITensor *inputTensors13[] = {upsample12->getOutput(0), conv2_2->getOutput(0)}; // 64 + 120 = 184
  nvinfer1::IConcatenationLayer *cat13 = network->addConcatenation(inputTensors13, 2);
  nvinfer1::IElementWiseLayer *conv14 = C3(network, weightMap, *cat13->getOutput(0), 184, Get_channel(64), get_depth(1, 1), false, 1, 0.5, "model.14");

  nvinfer1::IElementWiseLayer *conv15 = convBlock(network, weightMap, *conv14->getOutput(0), Get_channel(64), 3, 2, 1, "model.15");
  nvinfer1::ITensor *inputTensors16[] = {conv15->getOutput(0), conv11->getOutput(0)}; // 64 + 64 = 128
  nvinfer1::IConcatenationLayer *cat16 = network->addConcatenation(inputTensors16, 2); 
  nvinfer1::IElementWiseLayer *conv17 = C3(network, weightMap, *cat16->getOutput(0), 128, Get_channel(128), get_depth(1, 1), false, 1, 0.5, "model.17");

  nvinfer1::IElementWiseLayer *conv18 = convBlock(network, weightMap, *conv17->getOutput(0), Get_channel(128), 3, 2, 1, "model.18");
  nvinfer1::ITensor *inputTensors19[] = {conv18->getOutput(0), conv7->getOutput(0)}; // 128 + 128 = 256
  nvinfer1::IConcatenationLayer *cat19 = network->addConcatenation(inputTensors19, 2); 
  nvinfer1::IElementWiseLayer *conv20 = C3(network, weightMap, *cat19->getOutput(0), 256, Get_channel(256), get_depth(1, 1), false, 1, 0.5, "model.20");

  // detect
  nvinfer1::IConvolutionLayer *det0 = network->addConvolutionNd(*conv14->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
     nvinfer1::DimsHW{1, 1}, weightMap["model.21.m.0.weight"], weightMap["model.21.m.0.bias"]);
  
  nvinfer1::IConvolutionLayer *det1 = network->addConvolutionNd(*conv17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
    nvinfer1::DimsHW{1, 1}, weightMap["model.21.m.1.weight"], weightMap["model.21.m.1.bias"]);
  
  nvinfer1::IConvolutionLayer *det2 = network->addConvolutionNd(*conv20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), 
      nvinfer1::DimsHW{1, 1}, weightMap["model.21.m.2.weight"], weightMap["model.21.m.2.bias"]);
  
  auto yolo = addYoLoLayer(network, weightMap, "model.21", std::vector<nvinfer1::IConvolutionLayer*>{det0, det1, det2});
  yolo->getOutput(0)->setName(Yolo::OUTPUT_BLOB_NAME);
  network->markOutput(*yolo->getOutput(0));

    // Engine config
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
  config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
  std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
  assert(builder->platformHasFastInt8());
  config->setFlag(BuilderFlag::kINT8);
  std::string data_path = "tensorrtx-int8calib-data/coco_calib/";
  //Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, kInputW, kInputH, "./coco_calib/", "int8calib.table", kInputTensorName);
  Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, Yolo::INPUT_W, Yolo::INPUT_H, data_path.c_str(), "int8calib.table", Yolo::INPUT_BLOB_NAME);
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




void serialize_engine(unsigned int max_batchsize, std::string& wts_name, std::string& engine_name, std::string & used_model){
  
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  ICudaEngine *engine = nullptr;
  if(used_model == "g"){
    engine = build_det_v5_lite_g(max_batchsize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
  }else if(used_model == "s"){
    engine = build_det_v5_lite_s(max_batchsize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
  }else if(used_model == "c"){
    engine = build_det_v5_lite_c(max_batchsize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
  }
  else{
    engine = build_det_v5_lite_e(max_batchsize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
  }
  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    // assert(false);

  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  config->destroy();
  serialized_engine->destroy();
  builder->destroy();
}




void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * Yolo::INPUT_H * Yolo::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char **argv, std::string & wts_name, std::string & engine_name,
                                   std::string & used_model, std::string & img_dir){
  if(argc < 4 || argc > 6) return false;
  if(std::string(argv[1]) == "-s" && (argc == 5)){
    wts_name = argv[2];
    engine_name = argv[3];
    used_model = argv[4];
  }else if(std::string(argv[1]) == "-d" && argc == 4){
    engine_name = std::string(argv[2]);
    img_dir = std::string(argv[3]);
  }else{
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(Yolo::DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir, used_model;
    

    if(!parse_args(argc, argv, wts_name, engine_name, used_model, img_dir)){
      std::cerr << "arguments not right!" << std::endl;
      std::cerr << "./v5lite -s [.wts] [.engine] [s/e/g/c] // serialize modeo to the plan" << std::endl;
      std::cerr << "./v5lite -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
      return -1;  
    }

    if (!wts_name.empty()) {
        serialize_engine(Yolo::BATCH_SIZE,  wts_name, engine_name, used_model);
        return 0;
    }

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[Yolo::BATCH_SIZE * 3 * Yolo::INPUT_H * Yolo::INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[Yolo::BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(Yolo::INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(Yolo::OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], Yolo::BATCH_SIZE * 3 * Yolo::INPUT_H * Yolo::INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], Yolo::BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < Yolo::BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img, Yolo::INPUT_W, Yolo::INPUT_H); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < Yolo::INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < Yolo::INPUT_W; ++col) {
                    data[b * 3 * Yolo::INPUT_H * Yolo::INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    data[b * 3 * Yolo::INPUT_H * Yolo::INPUT_W + i + Yolo::INPUT_H * Yolo::INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[b * 3 * Yolo::INPUT_H * Yolo::INPUT_W + i + 2 * Yolo::INPUT_H * Yolo::INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, Yolo::BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], Yolo::CONF_THRESH, Yolo::NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite(file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    // std::cout << "\nOutput:\n\n";
    // for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    // {
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    // }
    // std::cout << std::endl;

    return 0;
}
