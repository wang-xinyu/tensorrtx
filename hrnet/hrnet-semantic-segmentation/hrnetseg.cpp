#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "common.hpp"
#include "logging.h"

static Logger gLogger;
#define USE_FP16
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1 // only support 1

const char* INPUT_BLOB_NAME = "image";
const char* OUTPUT_BLOB_NAME = "output"; 
static const int INPUT_H = 512;
static const int INPUT_W = 1024;
static const int NUM_CLASSES = 19;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W;

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_H, INPUT_W, 3 });
    assert(data);

    // hwc to chw
    auto ps = network->addShuffle(*data);
    ps->setFirstTranspose(nvinfer1::Permutation{ 2, 0, 1 });
    //     mean = [0.485, 0.456, 0.406]
    //  std = [0.229, 0.224, 0.225]
    float mean[3] = { 0.406, 0.456, 0.485 };
    float std[3] = { 0.225, 0.224, 0.229 };
    ITensor* preinput = MeanStd(network, ps->getOutput(0), mean, std, true);

    // BGR to RGB
    ISliceLayer *B = network->addSlice(*preinput, Dims3{ 0, 0, 0 }, Dims3{ 1, INPUT_H, INPUT_W }, Dims3{ 1, 1, 1 });
    ISliceLayer *G = network->addSlice(*preinput, Dims3{ 1, 0, 0 }, Dims3{ 1, INPUT_H, INPUT_W }, Dims3{ 1, 1, 1 });
    ISliceLayer *R = network->addSlice(*preinput, Dims3{ 2, 0, 0 }, Dims3{ 1, INPUT_H, INPUT_W }, Dims3{ 1, 1, 1 });

    ITensor* inputTensors[] = { R->getOutput(0), G->getOutput(0), B->getOutput(0) };
    auto inputcat = network->addConcatenation(inputTensors, 3);

    std::map<std::string, Weights> weightMap = loadWeights("../HRNetSeg.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    auto id_876 = convBnLeaky(network, weightMap, *inputcat->getOutput(0), 64, 3, 2, 1, "conv1", "bn1");  
    auto id_879 = convBnLeaky(network, weightMap, *id_876->getOutput(0), 64, 3, 2, 1, "conv2", "bn2");                                                                                  //Res
    auto id_891 = ResBlock2Conv(network, weightMap, *id_879->getOutput(0), 64, 256, 1, "layer1.0");
    auto id_901 = ResBlock(network, weightMap, *id_891->getOutput(0), 256, 64, 1, "layer1.1");
    nvinfer1::Dims dim1 = id_901->getOutput(0)->getDimensions();

    auto id_904 = convBnLeaky(network, weightMap, *id_901->getOutput(0), 18, 3, 1, 1, "transition1.0.0", "transition1.0.1");
    auto id_914 = liteResBlock(network, weightMap, *id_904->getOutput(0), 18, "stage2.0.branches.0.0");
    auto id_921 = liteResBlock(network, weightMap, *id_914->getOutput(0), 18, "stage2.0.branches.0.1");

    auto id_907 = convBnLeaky(network, weightMap, *id_901->getOutput(0), 36, 3, 2, 1, "transition1.1.0.0", "transition1.1.0.1");
    auto id_928 = liteResBlock(network, weightMap, *id_907->getOutput(0), 36, "stage2.0.branches.1.0");
    auto id_935 = liteResBlock(network, weightMap, *id_928->getOutput(0), 36, "stage2.0.branches.1.1");

    auto id_957 = convBnUpAdd(network, weightMap, *id_935->getOutput(0), *id_921->getOutput(0), 18, 1, 1, 0, "stage2.0.fuse_layers.0.1.0", "stage2.0.fuse_layers.0.1.1", true);
    auto id_958 = network->addActivation(*id_957->getOutput(0), ActivationType::kRELU);
   
    dim1 = id_935->getOutput(0)->getDimensions();
    dim1 = id_921->getOutput(0)->getDimensions();
    auto id_961 = convBnUpAdd(network, weightMap, *id_921->getOutput(0), *id_935->getOutput(0), 36, 3, 2, 1, "stage2.0.fuse_layers.1.0.0.0", "stage2.0.fuse_layers.1.0.0.1", false);
    auto id_962 = network->addActivation(*id_961->getOutput(0), ActivationType::kRELU);
    dim1 = id_962->getOutput(0)->getDimensions();

    auto id_972 = liteResBlock(network, weightMap, *id_958->getOutput(0), 18, "stage3.0.branches.0.0");
    auto id_979 = liteResBlock(network, weightMap, *id_972->getOutput(0), 18, "stage3.0.branches.0.1");

    auto id_986 = liteResBlock(network, weightMap, *id_962->getOutput(0), 36, "stage3.0.branches.1.0");
    auto id_993 = liteResBlock(network, weightMap, *id_986->getOutput(0), 36, "stage3.0.branches.1.1");

    auto id_963 = convBnLeaky(network, weightMap, *id_962->getOutput(0), 72, 3, 2, 1, "transition2.2.0.0", "transition2.2.0.1");
    auto id_1000 = liteResBlock(network, weightMap, *id_963->getOutput(0), 72, "stage3.0.branches.2.0");
    auto id_1007 = liteResBlock(network, weightMap, *id_1000->getOutput(0), 72, "stage3.0.branches.2.1");

    auto id_1029 = convBnUpAdd(network, weightMap, *id_993->getOutput(0), *id_979->getOutput(0), 18, 1, 1, 0, "stage3.0.fuse_layers.0.1.0", "stage3.0.fuse_layers.0.1.1", true);
    auto id_1051 = convBnUpAdd(network, weightMap, *id_1007->getOutput(0), *id_1029->getOutput(0), 18, 1, 1, 0, "stage3.0.fuse_layers.0.2.0", "stage3.0.fuse_layers.0.2.1", true);
    auto id_1052 = network->addActivation(*id_1051->getOutput(0), ActivationType::kRELU);

    auto id_1055 = convBnUpAdd(network, weightMap, *id_979->getOutput(0), *id_993->getOutput(0), 36, 3, 2, 1, "stage3.0.fuse_layers.1.0.0.0", "stage3.0.fuse_layers.1.0.0.1", false);
    auto id_1077 = convBnUpAdd(network, weightMap, *id_1007->getOutput(0), *id_1055->getOutput(0), 36, 1, 1, 0, "stage3.0.fuse_layers.1.2.0", "stage3.0.fuse_layers.1.2.1", true);
    auto id_1078 = network->addActivation(*id_1077->getOutput(0), ActivationType::kRELU);

    auto id_1081 = convBnLeaky(network, weightMap, *id_979->getOutput(0), 18, 3, 2, 1, "stage3.0.fuse_layers.2.0.0.0", "stage3.0.fuse_layers.2.0.0.1");
    auto id_1083 = convBnLeaky(network, weightMap, *id_1081->getOutput(0), 72, 3, 2, 1, "stage3.0.fuse_layers.2.0.1.0", "stage3.0.fuse_layers.2.0.1.1",false);
    auto id_1086= convBnUpAdd(network, weightMap, *id_993->getOutput(0), *id_1083->getOutput(0), 72, 3, 2, 1, "stage3.0.fuse_layers.2.1.0.0", "stage3.0.fuse_layers.2.1.0.1", false);
    auto id_1087 = network->addElementWise(*id_1086->getOutput(0), *id_1007->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1088 = network->addActivation(*id_1087->getOutput(0), ActivationType::kRELU);

    auto id_1095 = liteResBlock(network, weightMap, *id_1052->getOutput(0), 18, "stage3.1.branches.0.0");
    auto id_1102 = liteResBlock(network, weightMap, *id_1095->getOutput(0), 18, "stage3.1.branches.0.1");
    auto id_1109 = liteResBlock(network, weightMap, *id_1078->getOutput(0), 36, "stage3.1.branches.1.0");
    auto id_1116 = liteResBlock(network, weightMap, *id_1109->getOutput(0), 36, "stage3.1.branches.1.1");
    auto id_1123 = liteResBlock(network, weightMap, *id_1088->getOutput(0), 72, "stage3.1.branches.2.0");
    auto id_1130 = liteResBlock(network, weightMap, *id_1123->getOutput(0), 72, "stage3.1.branches.2.1");

    auto id_1152 = convBnUpAdd(network, weightMap, *id_1116->getOutput(0), *id_1102->getOutput(0), 18, 1, 1, 0, "stage3.1.fuse_layers.0.1.0", "stage3.1.fuse_layers.0.1.1", true);
    auto id_1174 = convBnUpAdd(network, weightMap, *id_1130->getOutput(0), *id_1152->getOutput(0), 18, 1, 1, 0, "stage3.1.fuse_layers.0.2.0", "stage3.1.fuse_layers.0.2.1", true);
    auto id_1175 = network->addActivation(*id_1174->getOutput(0), ActivationType::kRELU);

    auto id_1178 = convBnUpAdd(network, weightMap, *id_1102->getOutput(0), *id_1116->getOutput(0), 36, 3, 2, 1, "stage3.1.fuse_layers.1.0.0.0", "stage3.1.fuse_layers.1.0.0.1", false);
    auto id_1200 = convBnUpAdd(network, weightMap, *id_1130->getOutput(0), *id_1178->getOutput(0), 36, 1, 1, 0, "stage3.1.fuse_layers.1.2.0", "stage3.1.fuse_layers.1.2.1", true);
    auto id_1201 = network->addActivation(*id_1200->getOutput(0), ActivationType::kRELU);

    auto id_1204 = convBnLeaky(network, weightMap, *id_1102->getOutput(0), 18, 3, 2, 1, "stage3.1.fuse_layers.2.0.0.0", "stage3.1.fuse_layers.2.0.0.1");
    auto id_1206 = convBnLeaky(network, weightMap, *id_1204->getOutput(0), 72, 3, 2, 1, "stage3.1.fuse_layers.2.0.1.0", "stage3.1.fuse_layers.2.0.1.1", false);
    auto id_1209 = convBnUpAdd(network, weightMap, *id_1116->getOutput(0), *id_1206->getOutput(0), 72, 3, 2, 1, "stage3.1.fuse_layers.2.1.0.0", "stage3.1.fuse_layers.2.1.0.1", false);
    auto id_1210 = network->addElementWise(*id_1209->getOutput(0), *id_1130->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1211 = network->addActivation(*id_1210->getOutput(0), ActivationType::kRELU);
    
    auto id_1218 = liteResBlock(network, weightMap, *id_1175->getOutput(0), 18, "stage3.2.branches.0.0");
    auto id_1225 = liteResBlock(network, weightMap, *id_1218->getOutput(0), 18, "stage3.2.branches.0.1");
    auto id_1232 = liteResBlock(network, weightMap, *id_1201->getOutput(0), 36, "stage3.2.branches.1.0");
    auto id_1239 = liteResBlock(network, weightMap, *id_1232->getOutput(0), 36, "stage3.2.branches.1.1");
    auto id_1246 = liteResBlock(network, weightMap, *id_1211->getOutput(0), 72, "stage3.2.branches.2.0");
    auto id_1253 = liteResBlock(network, weightMap, *id_1246->getOutput(0), 72, "stage3.2.branches.2.1");
    
    auto id_1275 = convBnUpAdd(network, weightMap, *id_1239->getOutput(0), *id_1225->getOutput(0), 18, 1, 1, 0, "stage3.2.fuse_layers.0.1.0", "stage3.2.fuse_layers.0.1.1", true);
    auto id_1297 = convBnUpAdd(network, weightMap, *id_1253->getOutput(0), *id_1275->getOutput(0), 18, 1, 1, 0, "stage3.2.fuse_layers.0.2.0", "stage3.2.fuse_layers.0.2.1", true);
    auto id_1298 = network->addActivation(*id_1297->getOutput(0), ActivationType::kRELU);

    auto id_1301 = convBnUpAdd(network, weightMap, *id_1225->getOutput(0), *id_1239->getOutput(0), 36, 3, 2, 1, "stage3.2.fuse_layers.1.0.0.0", "stage3.2.fuse_layers.1.0.0.1", false);
    auto id_1323 = convBnUpAdd(network, weightMap, *id_1253->getOutput(0), *id_1301->getOutput(0), 36, 1, 1, 0, "stage3.2.fuse_layers.1.2.0", "stage3.2.fuse_layers.1.2.1", true);
    auto id_1324 = network->addActivation(*id_1323->getOutput(0), ActivationType::kRELU);

    auto id_1327 = convBnLeaky(network, weightMap, *id_1225->getOutput(0), 18, 3, 2, 1, "stage3.2.fuse_layers.2.0.0.0", "stage3.2.fuse_layers.2.0.0.1");
    auto id_1329 = convBnLeaky(network, weightMap, *id_1327->getOutput(0), 72, 3, 2, 1, "stage3.2.fuse_layers.2.0.1.0", "stage3.2.fuse_layers.2.0.1.1", false);
    auto id_1332 = convBnUpAdd(network, weightMap, *id_1239->getOutput(0), *id_1329->getOutput(0), 72, 3, 2, 1, "stage3.2.fuse_layers.2.1.0.0", "stage3.2.fuse_layers.2.1.0.1", false);
    auto id_1333 = network->addElementWise(*id_1332->getOutput(0), *id_1253->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1334 = network->addActivation(*id_1333->getOutput(0), ActivationType::kRELU);

    auto id_1344 = liteResBlock(network, weightMap, *id_1298->getOutput(0), 18, "stage4.0.branches.0.0");
    auto id_1351 = liteResBlock(network, weightMap, *id_1344->getOutput(0), 18, "stage4.0.branches.0.1");
    auto id_1358 = liteResBlock(network, weightMap, *id_1324->getOutput(0), 36, "stage4.0.branches.1.0");
    auto id_1365 = liteResBlock(network, weightMap, *id_1358->getOutput(0), 36, "stage4.0.branches.1.1");
    auto id_1372 = liteResBlock(network, weightMap, *id_1334->getOutput(0), 72, "stage4.0.branches.2.0");
    auto id_1379 = liteResBlock(network, weightMap, *id_1372->getOutput(0), 72, "stage4.0.branches.2.1");

    auto id_1337 = convBnLeaky(network, weightMap, *id_1334->getOutput(0), 144, 3, 2, 1, "transition3.3.0.0", "transition3.3.0.1");
    auto id_1386 = liteResBlock(network, weightMap, *id_1337->getOutput(0), 144, "stage4.0.branches.3.0");
    auto id_1393 = liteResBlock(network, weightMap, *id_1386->getOutput(0), 144, "stage4.0.branches.3.1");

    auto id_1415 = convBnUpAdd(network, weightMap, *id_1365->getOutput(0), *id_1351->getOutput(0), 18, 1, 1, 0, "stage4.0.fuse_layers.0.1.0", "stage4.0.fuse_layers.0.1.1", true);
    auto id_1437 = convBnUpAdd(network, weightMap, *id_1379->getOutput(0), *id_1415->getOutput(0), 18, 1, 1, 0, "stage4.0.fuse_layers.0.2.0", "stage4.0.fuse_layers.0.2.1", true);
    auto id_1459 = convBnUpAdd(network, weightMap, *id_1393->getOutput(0), *id_1437->getOutput(0), 18, 1, 1, 0, "stage4.0.fuse_layers.0.3.0", "stage4.0.fuse_layers.0.3.1", true);
    auto id_1460 = network->addActivation(*id_1459->getOutput(0), ActivationType::kRELU);

    auto id_1463 = convBnUpAdd(network, weightMap, *id_1351->getOutput(0), *id_1365->getOutput(0), 36, 3, 2, 1, "stage4.0.fuse_layers.1.0.0.0", "stage4.0.fuse_layers.1.0.0.1", false);
    auto id_1458 = convBnUpAdd(network, weightMap, *id_1379->getOutput(0), *id_1463->getOutput(0), 36, 1, 1, 0, "stage4.0.fuse_layers.1.2.0", "stage4.0.fuse_layers.1.2.1", true);
    auto id_1507 = convBnUpAdd(network, weightMap, *id_1393->getOutput(0), *id_1458->getOutput(0), 36, 1, 1, 0, "stage4.0.fuse_layers.1.3.0", "stage4.0.fuse_layers.1.3.1", true);
    auto id_1508 = network->addActivation(*id_1507->getOutput(0), ActivationType::kRELU);

    auto id_1511 = convBnLeaky(network, weightMap, *id_1351->getOutput(0), 18, 3, 2, 1, "stage4.0.fuse_layers.2.0.0.0", "stage4.0.fuse_layers.2.0.0.1");
    auto id_1513 = convBnLeaky(network, weightMap, *id_1511->getOutput(0), 72, 3, 2, 1, "stage4.0.fuse_layers.2.0.1.0", "stage4.0.fuse_layers.2.0.1.1", false);
    auto id_1516 = convBnUpAdd(network, weightMap, *id_1365->getOutput(0), *id_1513->getOutput(0), 72, 3, 2, 1, "stage4.0.fuse_layers.2.1.0.0", "stage4.0.fuse_layers.2.1.0.1", true);
    auto id_1517 = network->addElementWise(*id_1516->getOutput(0), *id_1379->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1539= convBnUpAdd(network, weightMap, *id_1393->getOutput(0), *id_1517->getOutput(0), 72, 1, 1, 0, "stage4.0.fuse_layers.2.3.0", "stage4.0.fuse_layers.2.3.1", true);
    auto id_1540 = network->addActivation(*id_1539->getOutput(0), ActivationType::kRELU);

    auto id_1543 = convBnLeaky(network, weightMap, *id_1351->getOutput(0), 18, 3, 2, 1, "stage4.0.fuse_layers.3.0.0.0", "stage4.0.fuse_layers.3.0.0.1");
    auto id_1546 = convBnLeaky(network, weightMap, *id_1543->getOutput(0), 18, 3, 2, 1, "stage4.0.fuse_layers.3.0.1.0", "stage4.0.fuse_layers.3.0.1.1");
    auto id_1548 = convBnLeaky(network, weightMap, *id_1546->getOutput(0), 144, 3, 2, 1, "stage4.0.fuse_layers.3.0.2.0", "stage4.0.fuse_layers.3.0.2.1", false);
    auto id_1551 = convBnLeaky(network, weightMap, *id_1365->getOutput(0), 36, 3, 2, 1, "stage4.0.fuse_layers.3.1.0.0", "stage4.0.fuse_layers.3.1.0.1");
    auto id_1554 = convBnUpAdd(network, weightMap, *id_1551->getOutput(0), *id_1548->getOutput(0), 144, 3, 2, 1, "stage4.0.fuse_layers.3.1.1.0", "stage4.0.fuse_layers.3.1.1.1", false);
    auto id_1557 = convBnUpAdd(network, weightMap, *id_1379->getOutput(0), *id_1554->getOutput(0), 144, 3, 2, 1, "stage4.0.fuse_layers.3.2.0.0", "stage4.0.fuse_layers.3.2.0.1", false);
    auto id_1558 = network->addElementWise(*id_1557->getOutput(0), *id_1393->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1559 = network->addActivation(*id_1558->getOutput(0), ActivationType::kRELU);

    auto id_1566= liteResBlock(network, weightMap, *id_1460->getOutput(0), 18, "stage4.1.branches.0.0");
    auto id_1573 = liteResBlock(network, weightMap, *id_1566->getOutput(0), 18, "stage4.1.branches.0.1");
    auto id_1580 = liteResBlock(network, weightMap, *id_1508->getOutput(0), 36, "stage4.1.branches.1.0");
    auto id_1587 = liteResBlock(network, weightMap, *id_1580->getOutput(0), 36, "stage4.1.branches.1.1");
    auto id_1594 = liteResBlock(network, weightMap, *id_1540->getOutput(0), 72, "stage4.1.branches.2.0");
    auto id_1601 = liteResBlock(network, weightMap, *id_1594->getOutput(0), 72, "stage4.1.branches.2.1");
    auto id_1608 = liteResBlock(network, weightMap, *id_1559->getOutput(0), 144, "stage4.1.branches.3.0");
    auto id_1615 = liteResBlock(network, weightMap, *id_1608->getOutput(0), 144, "stage4.1.branches.3.1");

    auto id_1637 = convBnUpAdd(network, weightMap, *id_1587->getOutput(0), *id_1573->getOutput(0), 18, 1, 1, 0, "stage4.1.fuse_layers.0.1.0", "stage4.1.fuse_layers.0.1.1", true);
    auto id_1659 = convBnUpAdd(network, weightMap, *id_1601->getOutput(0), *id_1637->getOutput(0), 18, 1, 1, 0, "stage4.1.fuse_layers.0.2.0", "stage4.1.fuse_layers.0.2.1", true);
    auto id_1681 = convBnUpAdd(network, weightMap, *id_1615->getOutput(0), *id_1659->getOutput(0), 18, 1, 1, 0, "stage4.1.fuse_layers.0.3.0", "stage4.1.fuse_layers.0.3.1", true);
    auto id_1682 = network->addActivation(*id_1681->getOutput(0), ActivationType::kRELU);

    auto id_1685 = convBnUpAdd(network, weightMap, *id_1573->getOutput(0), *id_1587->getOutput(0), 36, 3, 2, 1, "stage4.1.fuse_layers.1.0.0.0", "stage4.1.fuse_layers.1.0.0.1", false);
    auto id_1707 = convBnUpAdd(network, weightMap, *id_1601->getOutput(0), *id_1685->getOutput(0), 36, 1, 1, 0, "stage4.1.fuse_layers.1.2.0", "stage4.1.fuse_layers.1.2.1", true);
    auto id_1729 = convBnUpAdd(network, weightMap, *id_1615->getOutput(0), *id_1707->getOutput(0), 36, 1, 1, 0, "stage4.1.fuse_layers.1.3.0", "stage4.1.fuse_layers.1.3.1", true);
    auto id_1730 = network->addActivation(*id_1729->getOutput(0), ActivationType::kRELU);

    auto id_1733 = convBnLeaky(network, weightMap, *id_1573->getOutput(0), 18, 3, 2, 1, "stage4.1.fuse_layers.2.0.0.0", "stage4.1.fuse_layers.2.0.0.1");
    auto id_1735 = convBnLeaky(network, weightMap, *id_1733->getOutput(0), 72, 3, 2, 1, "stage4.1.fuse_layers.2.0.1.0", "stage4.1.fuse_layers.2.0.1.1", false);
    auto id_1738 = convBnUpAdd(network, weightMap, *id_1587->getOutput(0), *id_1735->getOutput(0), 72, 3, 2, 1, "stage4.1.fuse_layers.2.1.0.0", "stage4.1.fuse_layers.2.1.0.1", false);
    auto id_1739 = network->addElementWise(*id_1601->getOutput(0), *id_1738->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1761 = convBnUpAdd(network, weightMap, *id_1615->getOutput(0), *id_1739->getOutput(0), 72, 1, 1, 0, "stage4.1.fuse_layers.2.3.0", "stage4.1.fuse_layers.2.3.1", true);
    auto id_1762 = network->addActivation(*id_1761->getOutput(0), ActivationType::kRELU);

    auto id_1765 = convBnLeaky(network, weightMap, *id_1573->getOutput(0), 18, 3, 2, 1, "stage4.1.fuse_layers.3.0.0.0", "stage4.1.fuse_layers.3.0.0.1");
    auto id_1768 = convBnLeaky(network, weightMap, *id_1765->getOutput(0), 18, 3, 2, 1, "stage4.1.fuse_layers.3.0.1.0", "stage4.1.fuse_layers.3.0.1.1");
    auto id_1770 = convBnLeaky(network, weightMap, *id_1768->getOutput(0), 144, 3, 2, 1, "stage4.1.fuse_layers.3.0.2.0", "stage4.1.fuse_layers.3.0.2.1",false);
    auto id_1773 = convBnLeaky(network, weightMap, *id_1587->getOutput(0), 36, 3, 2, 1, "stage4.1.fuse_layers.3.1.0.0", "stage4.1.fuse_layers.3.1.0.1");
    auto id_1776 = convBnUpAdd(network, weightMap, *id_1773->getOutput(0), *id_1770->getOutput(0), 144, 3, 2, 1, "stage4.1.fuse_layers.3.1.1.0", "stage4.1.fuse_layers.3.1.1.1", false);
    auto id_1779 = convBnUpAdd(network, weightMap, *id_1601->getOutput(0), *id_1776->getOutput(0), 144, 3, 2, 1, "stage4.1.fuse_layers.3.2.0.0", "stage4.1.fuse_layers.3.2.0.1", false);
    auto id_1780 = network->addElementWise(*id_1779->getOutput(0), *id_1615->getOutput(0), ElementWiseOperation::kSUM);
    auto id_1781 = network->addActivation(*id_1780->getOutput(0), ActivationType::kRELU);
    
    nvinfer1::Dims dim = id_1682->getOutput(0)->getDimensions();
    dim.d[0] = id_1730->getOutput(0)->getDimensions().d[0];
    auto id_1730_up = netAddUpsampleBi(network, id_1730->getOutput(0), dim);
    dim.d[0] = id_1762->getOutput(0)->getDimensions().d[0];
    auto id_1762_up = netAddUpsampleBi(network, id_1762->getOutput(0), dim);
    dim.d[0] = id_1781->getOutput(0)->getDimensions().d[0];
    auto id_1781_up = netAddUpsampleBi(network, id_1781->getOutput(0), dim);

    ITensor* concatTensors[] = { id_1682->getOutput(0), id_1730_up ->getOutput(0), id_1762_up->getOutput(0), id_1781_up->getOutput(0) };
    auto id_1827 = network->addConcatenation(concatTensors, 4);

    dim1 = id_1827->getOutput(0)->getDimensions();
    auto id_1830 = convBnLeaky(network, weightMap, *id_1827->getOutput(0), 270, 1, 1, 0, "last_layer.0", "last_layer.1", true, true);
    auto id_1831 = network->addConvolutionNd(*id_1830->getOutput(0), NUM_CLASSES, DimsHW{ 1,1 },weightMap["last_layer.3.weight"],weightMap["last_layer.3.bias"]);
    id_1831->setStrideNd(DimsHW{ 1, 1 });
    id_1831->setPaddingNd(DimsHW{ 0, 0 });


    dim.d[0] = NUM_CLASSES;
    dim.d[1] = INPUT_H;
    dim.d[2] = INPUT_W;
    auto id_1832 = netAddUpsampleBi(network, id_1831->getOutput(0), dim);
    auto id_1833 = network->addTopK(*id_1832->getOutput(0), TopKOperation::kMAX, 1, 0X01);
    // id_1833->getOutput(1) 1 is index
    id_1833->getOutput(1)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*id_1833->getOutput(1)); 

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize((1 << 30));  // 1G
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, int batchSize) {
    const ICudaEngine& engine = context.getEngine();
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = "hrnet_seg.engine";
    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./hrnetseg -s  // serialize model to plan file" << std::endl;
        std::cerr << "./hrnetseg -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    // prepare input data ---------------------------
    cudaSetDeviceFlags(cudaDeviceMapHost);
    float* data;
    int* prob;  // using int. output is index     
    CHECK(cudaHostAlloc((void **)&data, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&prob, BATCH_SIZE * OUTPUT_SIZE * sizeof(int), cudaHostAllocMapped));

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    for (int f = 0; f < (int)file_names.size(); f++) {
        cv::Mat pr_img;
        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f]); // BGR
        if (img.empty()) continue;  
        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
        img = pr_img.clone();  // for img show
        pr_img.convertTo(pr_img, CV_32FC3);
        if (!pr_img.isContinuous())
        {
            pr_img = pr_img.clone();
        }
        std::memcpy(data, pr_img.data, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float)); 
        
        cudaHostGetDevicePointer((void **)&buffers[inputIndex], (void *)data, 0); // buffers[inputIndex]-->data
        cudaHostGetDevicePointer((void **)&buffers[outputIndex], (void *)prob, 0); // buffers[outputIndex] --> prob

        // Run inference  
        auto start = std::chrono::high_resolution_clock::now();
        doInference(*context, stream, buffers, BATCH_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat outimg(INPUT_H, INPUT_W, CV_8UC1);
        for (int row = 0; row <INPUT_H; ++row)
        {
            uchar* uc_pixel = outimg.data + row * outimg.step;
            for (int col =0; col <INPUT_W; ++col)
            {
                uc_pixel[col] = (uchar)prob[row*INPUT_W + col];
            } 
        }
        cv::Mat im_color;
        cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
        cv::Mat lut = createLTU(NUM_CLASSES);
        cv::LUT(im_color, lut, im_color);
        // false color
        cv::cvtColor(im_color, im_color, cv::COLOR_RGB2GRAY);
        cv::applyColorMap(im_color, im_color, cv::COLORMAP_HOT);
        cv::imshow("False Color Map", im_color);
	    //fusion
        cv::Mat fusionImg;
        cv::addWeighted(img, 1, im_color, 0.5, 1, fusionImg);
        cv::imshow("Fusion Img", fusionImg);
        cv::waitKey(0);
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFreeHost(buffers[inputIndex]));
    CHECK(cudaFreeHost(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}