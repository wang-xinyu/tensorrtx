#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "common.hpp"
#include "logging.h"

static Logger gLogger;
#define USE_FP32
#define DEVICE 0     // GPU id
#define BATCH_SIZE 1 //

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "output";
static const int INPUT_H = 512;
static const int INPUT_W = 1024;
static const int NUM_CLASSES = 19;
static const int OUTPUT_SIZE = INPUT_H * INPUT_W;

// Creat the engine using only the API and not any parser.
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, std::string wtsPath, int width)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_H, INPUT_W, 3});
    assert(data);

    // hwc to chw
    auto ps = network->addShuffle(*data);
    ps->setFirstTranspose(nvinfer1::Permutation{2, 0, 1});
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    ITensor *preinput = MeanStd(network, ps->getOutput(0), mean, std, true);

    std::map<std::string, Weights> weightMap = loadWeights(wtsPath);
    auto relu_2 = convBnRelu(network, weightMap, *preinput, 64, 3, 2, 1, "conv1", "bn1");
    auto relu_5 = convBnRelu(network, weightMap, *relu_2->getOutput(0), 64, 3, 2, 1, "conv2", "bn2");
    auto relu_17 = ResBlock2Conv(network, weightMap, *relu_5->getOutput(0), 64, 256, 1, "layer1.0");
    auto relu_27 = ResBlock(network, weightMap, *relu_17->getOutput(0), 256, 64, 1, "layer1.1");
    auto relu_37 = ResBlock(network, weightMap, *relu_27->getOutput(0), 256, 64, 1, "layer1.2");
    auto relu_47 = ResBlock(network, weightMap, *relu_37->getOutput(0), 256, 64, 1, "layer1.3");

    auto relu_50 = convBnRelu(network, weightMap, *relu_47->getOutput(0), width, 3, 1, 1, "transition1.0.0", "transition1.0.1");
    auto relu_60 = liteResBlock(network, weightMap, *relu_50->getOutput(0), width, "stage2.0.branches.0.0");
    auto relu_67 = liteResBlock(network, weightMap, *relu_60->getOutput(0), width, "stage2.0.branches.0.1");
    auto relu_74 = liteResBlock(network, weightMap, *relu_67->getOutput(0), width, "stage2.0.branches.0.2");
    auto relu_81 = liteResBlock(network, weightMap, *relu_74->getOutput(0), width, "stage2.0.branches.0.3");

    auto relu_53 = convBnRelu(network, weightMap, *relu_47->getOutput(0), width * 2, 3, 2, 1, "transition1.1.0.0", "transition1.1.0.1");
    auto relu_88 = liteResBlock(network, weightMap, *relu_53->getOutput(0), width * 2, "stage2.0.branches.1.0");
    auto relu_95 = liteResBlock(network, weightMap, *relu_88->getOutput(0), width * 2, "stage2.0.branches.1.1");
    auto relu_102 = liteResBlock(network, weightMap, *relu_95->getOutput(0), width * 2, "stage2.0.branches.1.2");
    auto relu_109 = liteResBlock(network, weightMap, *relu_102->getOutput(0), width * 2, "stage2.0.branches.1.3");

    auto add_131 = convBnUpAdd(network, weightMap, *relu_109->getOutput(0), *relu_81->getOutput(0), width, 1, 1, 0, "stage2.0.fuse_layers.0.1.0", "stage2.0.fuse_layers.0.1.1", true);
    auto relu_132 = network->addActivation(*add_131->getOutput(0), ActivationType::kRELU);

    auto add_135 = convBnUpAdd(network, weightMap, *relu_81->getOutput(0), *relu_109->getOutput(0), width * 2, 3, 2, 1, "stage2.0.fuse_layers.1.0.0.0", "stage2.0.fuse_layers.1.0.0.1", false);
    auto relu_136 = network->addActivation(*add_135->getOutput(0), ActivationType::kRELU);

    auto relu_146 = liteResBlock(network, weightMap, *relu_132->getOutput(0), width, "stage3.0.branches.0.0");
    auto relu_153 = liteResBlock(network, weightMap, *relu_146->getOutput(0), width, "stage3.0.branches.0.1");
    auto relu_160 = liteResBlock(network, weightMap, *relu_153->getOutput(0), width, "stage3.0.branches.0.2");
    auto relu_167 = liteResBlock(network, weightMap, *relu_160->getOutput(0), width, "stage3.0.branches.0.3");

    auto relu_174 = liteResBlock(network, weightMap, *relu_136->getOutput(0), width * 2, "stage3.0.branches.1.0");
    auto relu_181 = liteResBlock(network, weightMap, *relu_174->getOutput(0), width * 2, "stage3.0.branches.1.1");
    auto relu_188 = liteResBlock(network, weightMap, *relu_181->getOutput(0), width * 2, "stage3.0.branches.1.2");
    auto relu_195 = liteResBlock(network, weightMap, *relu_188->getOutput(0), width * 2, "stage3.0.branches.1.3");

    auto relu_139 = convBnRelu(network, weightMap, *relu_136->getOutput(0), width * 4, 3, 2, 1, "transition2.2.0.0", "transition2.2.0.1");
    auto relu_202 = liteResBlock(network, weightMap, *relu_139->getOutput(0), width * 4, "stage3.0.branches.2.0");
    auto relu_209 = liteResBlock(network, weightMap, *relu_202->getOutput(0), width * 4, "stage3.0.branches.2.1");
    auto relu_216 = liteResBlock(network, weightMap, *relu_209->getOutput(0), width * 4, "stage3.0.branches.2.2");
    auto relu_223 = liteResBlock(network, weightMap, *relu_216->getOutput(0), width * 4, "stage3.0.branches.2.3");

    auto add_245 = convBnUpAdd(network, weightMap, *relu_195->getOutput(0), *relu_167->getOutput(0), width, 1, 1, 0, "stage3.0.fuse_layers.0.1.0", "stage3.0.fuse_layers.0.1.1", true);
    auto add_267 = convBnUpAdd(network, weightMap, *relu_223->getOutput(0), *add_245->getOutput(0), width, 1, 1, 0, "stage3.0.fuse_layers.0.2.0", "stage3.0.fuse_layers.0.2.1", true);
    auto relu_268 = network->addActivation(*add_267->getOutput(0), ActivationType::kRELU);

    auto add_271 = convBnUpAdd(network, weightMap, *relu_167->getOutput(0), *relu_195->getOutput(0), width * 2, 3, 2, 1, "stage3.0.fuse_layers.1.0.0.0", "stage3.0.fuse_layers.1.0.0.1", false);
    auto add_293 = convBnUpAdd(network, weightMap, *relu_223->getOutput(0), *add_271->getOutput(0), width * 2, 1, 1, 0, "stage3.0.fuse_layers.1.2.0", "stage3.0.fuse_layers.1.2.1", true);
    auto relu_294 = network->addActivation(*add_293->getOutput(0), ActivationType::kRELU);

    auto relu_297 = convBnRelu(network, weightMap, *relu_167->getOutput(0), width, 3, 2, 1, "stage3.0.fuse_layers.2.0.0.0", "stage3.0.fuse_layers.2.0.0.1");
    auto bn_299 = convBnRelu(network, weightMap, *relu_297->getOutput(0), width * 4, 3, 2, 1, "stage3.0.fuse_layers.2.0.1.0", "stage3.0.fuse_layers.2.0.1.1", false);
    auto add_302 = convBnUpAdd(network, weightMap, *relu_195->getOutput(0), *bn_299->getOutput(0), width * 4, 3, 2, 1, "stage3.0.fuse_layers.2.1.0.0", "stage3.0.fuse_layers.2.1.0.1", false);
    auto add_303 = network->addElementWise(*add_302->getOutput(0), *relu_223->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_304 = network->addActivation(*add_303->getOutput(0), ActivationType::kRELU);

    auto relu_311 = liteResBlock(network, weightMap, *relu_268->getOutput(0), width, "stage3.1.branches.0.0");
    auto relu_318 = liteResBlock(network, weightMap, *relu_311->getOutput(0), width, "stage3.1.branches.0.1");
    auto relu_325 = liteResBlock(network, weightMap, *relu_318->getOutput(0), width, "stage3.1.branches.0.2");
    auto relu_332 = liteResBlock(network, weightMap, *relu_325->getOutput(0), width, "stage3.1.branches.0.3");

    auto relu_339 = liteResBlock(network, weightMap, *relu_294->getOutput(0), width * 2, "stage3.1.branches.1.0");
    auto relu_346 = liteResBlock(network, weightMap, *relu_339->getOutput(0), width * 2, "stage3.1.branches.1.1");
    auto relu_353 = liteResBlock(network, weightMap, *relu_346->getOutput(0), width * 2, "stage3.1.branches.1.2");
    auto relu_360 = liteResBlock(network, weightMap, *relu_353->getOutput(0), width * 2, "stage3.1.branches.1.3");

    auto relu_367 = liteResBlock(network, weightMap, *relu_304->getOutput(0), width * 4, "stage3.1.branches.2.0");
    auto relu_374 = liteResBlock(network, weightMap, *relu_367->getOutput(0), width * 4, "stage3.1.branches.2.1");
    auto relu_381 = liteResBlock(network, weightMap, *relu_374->getOutput(0), width * 4, "stage3.1.branches.2.2");
    auto relu_388 = liteResBlock(network, weightMap, *relu_381->getOutput(0), width * 4, "stage3.1.branches.2.3");

    auto add_410 = convBnUpAdd(network, weightMap, *relu_360->getOutput(0), *relu_332->getOutput(0), width, 1, 1, 0, "stage3.1.fuse_layers.0.1.0", "stage3.1.fuse_layers.0.1.1", true);
    auto add_432 = convBnUpAdd(network, weightMap, *relu_388->getOutput(0), *add_410->getOutput(0), width, 1, 1, 0, "stage3.1.fuse_layers.0.2.0", "stage3.1.fuse_layers.0.2.1", true);
    auto relu_433 = network->addActivation(*add_432->getOutput(0), ActivationType::kRELU);

    auto add_436 = convBnUpAdd(network, weightMap, *relu_332->getOutput(0), *relu_360->getOutput(0), width * 2, 3, 2, 1, "stage3.1.fuse_layers.1.0.0.0", "stage3.1.fuse_layers.1.0.0.1", false);
    auto add_458 = convBnUpAdd(network, weightMap, *relu_388->getOutput(0), *add_436->getOutput(0), width * 2, 1, 1, 0, "stage3.1.fuse_layers.1.2.0", "stage3.1.fuse_layers.1.2.1", true);
    auto relu_459 = network->addActivation(*add_458->getOutput(0), ActivationType::kRELU);

    auto relu_462 = convBnRelu(network, weightMap, *relu_332->getOutput(0), width, 3, 2, 1, "stage3.1.fuse_layers.2.0.0.0", "stage3.1.fuse_layers.2.0.0.1");
    auto bn_464 = convBnRelu(network, weightMap, *relu_462->getOutput(0), width * 4, 3, 2, 1, "stage3.1.fuse_layers.2.0.1.0", "stage3.1.fuse_layers.2.0.1.1", false);
    auto add_467 = convBnUpAdd(network, weightMap, *relu_360->getOutput(0), *bn_464->getOutput(0), width * 4, 3, 2, 1, "stage3.1.fuse_layers.2.1.0.0", "stage3.1.fuse_layers.2.1.0.1", false);
    auto add_468 = network->addElementWise(*add_467->getOutput(0), *relu_388->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_469 = network->addActivation(*add_468->getOutput(0), ActivationType::kRELU);

    auto relu_476 = liteResBlock(network, weightMap, *relu_433->getOutput(0), width, "stage3.2.branches.0.0");
    auto relu_483 = liteResBlock(network, weightMap, *relu_476->getOutput(0), width, "stage3.2.branches.0.1");
    auto relu_490 = liteResBlock(network, weightMap, *relu_483->getOutput(0), width, "stage3.2.branches.0.2");
    auto relu_497 = liteResBlock(network, weightMap, *relu_490->getOutput(0), width, "stage3.2.branches.0.3");

    auto relu_504 = liteResBlock(network, weightMap, *relu_459->getOutput(0), width * 2, "stage3.2.branches.1.0");
    auto relu_511 = liteResBlock(network, weightMap, *relu_504->getOutput(0), width * 2, "stage3.2.branches.1.1");
    auto relu_518 = liteResBlock(network, weightMap, *relu_511->getOutput(0), width * 2, "stage3.2.branches.1.2");
    auto relu_525 = liteResBlock(network, weightMap, *relu_518->getOutput(0), width * 2, "stage3.2.branches.1.3");

    auto relu_532 = liteResBlock(network, weightMap, *relu_469->getOutput(0), width * 4, "stage3.2.branches.2.0");
    auto relu_539 = liteResBlock(network, weightMap, *relu_532->getOutput(0), width * 4, "stage3.2.branches.2.1");
    auto relu_546 = liteResBlock(network, weightMap, *relu_539->getOutput(0), width * 4, "stage3.2.branches.2.2");
    auto relu_553 = liteResBlock(network, weightMap, *relu_546->getOutput(0), width * 4, "stage3.2.branches.2.3");

    auto add_575 = convBnUpAdd(network, weightMap, *relu_525->getOutput(0), *relu_497->getOutput(0), width, 1, 1, 0, "stage3.2.fuse_layers.0.1.0", "stage3.2.fuse_layers.0.1.1", true);
    auto add_597 = convBnUpAdd(network, weightMap, *relu_553->getOutput(0), *add_575->getOutput(0), width, 1, 1, 0, "stage3.2.fuse_layers.0.2.0", "stage3.2.fuse_layers.0.2.1", true);

    auto relu_598 = network->addActivation(*add_597->getOutput(0), ActivationType::kRELU);

    auto add_601 = convBnUpAdd(network, weightMap, *relu_497->getOutput(0), *relu_525->getOutput(0), width * 2, 3, 2, 1, "stage3.2.fuse_layers.1.0.0.0", "stage3.2.fuse_layers.1.0.0.1", false);
    auto add_623 = convBnUpAdd(network, weightMap, *relu_553->getOutput(0), *add_601->getOutput(0), width * 2, 1, 1, 0, "stage3.2.fuse_layers.1.2.0", "stage3.2.fuse_layers.1.2.1", true);
    auto relu_624 = network->addActivation(*add_623->getOutput(0), ActivationType::kRELU);

    auto relu_627 = convBnRelu(network, weightMap, *relu_497->getOutput(0), width, 3, 2, 1, "stage3.2.fuse_layers.2.0.0.0", "stage3.2.fuse_layers.2.0.0.1");
    auto bn_629 = convBnRelu(network, weightMap, *relu_627->getOutput(0), width * 4, 3, 2, 1, "stage3.2.fuse_layers.2.0.1.0", "stage3.2.fuse_layers.2.0.1.1", false);
    auto add_632 = convBnUpAdd(network, weightMap, *relu_525->getOutput(0), *bn_629->getOutput(0), width * 4, 3, 2, 1, "stage3.2.fuse_layers.2.1.0.0", "stage3.2.fuse_layers.2.1.0.1", false);
    auto add_633 = network->addElementWise(*relu_553->getOutput(0), *add_632->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_634 = network->addActivation(*add_633->getOutput(0), ActivationType::kRELU);

    auto relu_641 = liteResBlock(network, weightMap, *relu_598->getOutput(0), width, "stage3.3.branches.0.0");
    auto relu_648 = liteResBlock(network, weightMap, *relu_641->getOutput(0), width, "stage3.3.branches.0.1");
    auto relu_655 = liteResBlock(network, weightMap, *relu_648->getOutput(0), width, "stage3.3.branches.0.2");
    auto relu_662 = liteResBlock(network, weightMap, *relu_655->getOutput(0), width, "stage3.3.branches.0.3");

    auto relu_669 = liteResBlock(network, weightMap, *relu_624->getOutput(0), width * 2, "stage3.3.branches.1.0");
    auto relu_676 = liteResBlock(network, weightMap, *relu_669->getOutput(0), width * 2, "stage3.3.branches.1.1");
    auto relu_683 = liteResBlock(network, weightMap, *relu_676->getOutput(0), width * 2, "stage3.3.branches.1.2");
    auto relu_690 = liteResBlock(network, weightMap, *relu_683->getOutput(0), width * 2, "stage3.3.branches.1.3");

    auto relu_697 = liteResBlock(network, weightMap, *relu_634->getOutput(0), width * 4, "stage3.3.branches.2.0");
    auto relu_704 = liteResBlock(network, weightMap, *relu_697->getOutput(0), width * 4, "stage3.3.branches.2.1");
    auto relu_711 = liteResBlock(network, weightMap, *relu_704->getOutput(0), width * 4, "stage3.3.branches.2.2");
    auto relu_718 = liteResBlock(network, weightMap, *relu_711->getOutput(0), width * 4, "stage3.3.branches.2.3");

    auto add_740 = convBnUpAdd(network, weightMap, *relu_690->getOutput(0), *relu_662->getOutput(0), width, 1, 1, 0, "stage3.3.fuse_layers.0.1.0", "stage3.3.fuse_layers.0.1.1", true);
    auto add_762 = convBnUpAdd(network, weightMap, *relu_718->getOutput(0), *add_740->getOutput(0), width, 1, 1, 0, "stage3.3.fuse_layers.0.2.0", "stage3.3.fuse_layers.0.2.1", true);
    auto relu_763 = network->addActivation(*add_762->getOutput(0), ActivationType::kRELU);

    auto add_766 = convBnUpAdd(network, weightMap, *relu_662->getOutput(0), *relu_690->getOutput(0), width * 2, 3, 2, 1, "stage3.3.fuse_layers.1.0.0.0", "stage3.3.fuse_layers.1.0.0.1", false);
    auto add_788 = convBnUpAdd(network, weightMap, *relu_718->getOutput(0), *add_766->getOutput(0), width * 2, 1, 1, 0, "stage3.3.fuse_layers.1.2.0", "stage3.3.fuse_layers.1.2.1", true);
    auto relu_789 = network->addActivation(*add_788->getOutput(0), ActivationType::kRELU);

    auto relu_792 = convBnRelu(network, weightMap, *relu_662->getOutput(0), width, 3, 2, 1, "stage3.3.fuse_layers.2.0.0.0", "stage3.3.fuse_layers.2.0.0.1");
    auto bn_794 = convBnRelu(network, weightMap, *relu_792->getOutput(0), width * 4, 3, 2, 1, "stage3.3.fuse_layers.2.0.1.0", "stage3.3.fuse_layers.2.0.1.1", false);
    auto add_797 = convBnUpAdd(network, weightMap, *relu_690->getOutput(0), *bn_794->getOutput(0), width * 4, 3, 2, 1, "stage3.3.fuse_layers.2.1.0.0", "stage3.3.fuse_layers.2.1.0.1", false);
    auto add_798 = network->addElementWise(*relu_718->getOutput(0), *add_797->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_799 = network->addActivation(*add_798->getOutput(0), ActivationType::kRELU);

    auto relu_809 = liteResBlock(network, weightMap, *relu_763->getOutput(0), width, "stage4.0.branches.0.0");
    auto relu_816 = liteResBlock(network, weightMap, *relu_809->getOutput(0), width, "stage4.0.branches.0.1");
    auto relu_823 = liteResBlock(network, weightMap, *relu_816->getOutput(0), width, "stage4.0.branches.0.2");
    auto relu_830 = liteResBlock(network, weightMap, *relu_823->getOutput(0), width, "stage4.0.branches.0.3");

    auto relu_837 = liteResBlock(network, weightMap, *relu_789->getOutput(0), width * 2, "stage4.0.branches.1.0");
    auto relu_844 = liteResBlock(network, weightMap, *relu_837->getOutput(0), width * 2, "stage4.0.branches.1.1");
    auto relu_851 = liteResBlock(network, weightMap, *relu_844->getOutput(0), width * 2, "stage4.0.branches.1.2");
    auto relu_858 = liteResBlock(network, weightMap, *relu_851->getOutput(0), width * 2, "stage4.0.branches.1.3");

    auto relu_865 = liteResBlock(network, weightMap, *relu_799->getOutput(0), width * 4, "stage4.0.branches.2.0");
    auto relu_872 = liteResBlock(network, weightMap, *relu_865->getOutput(0), width * 4, "stage4.0.branches.2.1");
    auto relu_879 = liteResBlock(network, weightMap, *relu_872->getOutput(0), width * 4, "stage4.0.branches.2.2");
    auto relu_886 = liteResBlock(network, weightMap, *relu_879->getOutput(0), width * 4, "stage4.0.branches.2.3"); //========

    auto relu_802 = convBnRelu(network, weightMap, *relu_799->getOutput(0), width * 8, 3, 2, 1, "transition3.3.0.0", "transition3.3.0.1");
    auto relu_893 = liteResBlock(network, weightMap, *relu_802->getOutput(0), width * 8, "stage4.0.branches.3.0");
    auto relu_900 = liteResBlock(network, weightMap, *relu_893->getOutput(0), width * 8, "stage4.0.branches.3.1");
    auto relu_907 = liteResBlock(network, weightMap, *relu_900->getOutput(0), width * 8, "stage4.0.branches.3.2");
    auto relu_914 = liteResBlock(network, weightMap, *relu_907->getOutput(0), width * 8, "stage4.0.branches.3.3");

    auto add_936 = convBnUpAdd(network, weightMap, *relu_858->getOutput(0), *relu_830->getOutput(0), width, 1, 1, 0, "stage4.0.fuse_layers.0.1.0", "stage4.0.fuse_layers.0.1.1", true);
    auto add_958 = convBnUpAdd(network, weightMap, *relu_886->getOutput(0), *add_936->getOutput(0), width, 1, 1, 0, "stage4.0.fuse_layers.0.2.0", "stage4.0.fuse_layers.0.2.1", true);
    auto add_980 = convBnUpAdd(network, weightMap, *relu_914->getOutput(0), *add_958->getOutput(0), width, 1, 1, 0, "stage4.0.fuse_layers.0.3.0", "stage4.0.fuse_layers.0.3.1", true);
    auto relu_981 = network->addActivation(*add_980->getOutput(0), ActivationType::kRELU);

    auto add_984 = convBnUpAdd(network, weightMap, *relu_830->getOutput(0), *relu_858->getOutput(0), width * 2, 3, 2, 1, "stage4.0.fuse_layers.1.0.0.0", "stage4.0.fuse_layers.1.0.0.1", false);
    auto add_1006 = convBnUpAdd(network, weightMap, *relu_886->getOutput(0), *add_984->getOutput(0), width * 2, 1, 1, 0, "stage4.0.fuse_layers.1.2.0", "stage4.0.fuse_layers.1.2.1", true);
    auto add_1028 = convBnUpAdd(network, weightMap, *relu_914->getOutput(0), *add_1006->getOutput(0), width * 2, 1, 1, 0, "stage4.0.fuse_layers.1.3.0", "stage4.0.fuse_layers.1.3.1", true);
    auto relu_1029 = network->addActivation(*add_1028->getOutput(0), ActivationType::kRELU);

    auto relu_1032 = convBnRelu(network, weightMap, *relu_830->getOutput(0), width, 3, 2, 1, "stage4.0.fuse_layers.2.0.0.0", "stage4.0.fuse_layers.2.0.0.1");
    auto bn_1034 = convBnRelu(network, weightMap, *relu_1032->getOutput(0), width * 4, 3, 2, 1, "stage4.0.fuse_layers.2.0.1.0", "stage4.0.fuse_layers.2.0.1.1", false);

    auto add_1037 = convBnUpAdd(network, weightMap, *relu_858->getOutput(0), *bn_1034->getOutput(0), width * 4, 3, 2, 1,
                                "stage4.0.fuse_layers.2.1.0.0", "stage4.0.fuse_layers.2.1.0.1", false);
    auto add_1038 = network->addElementWise(*relu_886->getOutput(0), *add_1037->getOutput(0), ElementWiseOperation::kSUM);
    auto add_1060 = convBnUpAdd(network, weightMap, *relu_914->getOutput(0), *add_1038->getOutput(0), width * 4, 1, 1, 0,
                                "stage4.0.fuse_layers.2.3.0", "stage4.0.fuse_layers.2.3.1", true);
    auto relu_1061 = network->addActivation(*add_1060->getOutput(0), ActivationType::kRELU);

    auto relu_1064 = convBnRelu(network, weightMap, *relu_830->getOutput(0), width, 3, 2, 1, "stage4.0.fuse_layers.3.0.0.0", "stage4.0.fuse_layers.3.0.0.1");
    auto relu_1067 = convBnRelu(network, weightMap, *relu_1064->getOutput(0), width, 3, 2, 1, "stage4.0.fuse_layers.3.0.1.0", "stage4.0.fuse_layers.3.0.1.1");
    auto bn_1069 = convBnRelu(network, weightMap, *relu_1067->getOutput(0), width * 8, 3, 2, 1, "stage4.0.fuse_layers.3.0.2.0", "stage4.0.fuse_layers.3.0.2.1", false);
    auto relu_1072 = convBnRelu(network, weightMap, *relu_858->getOutput(0), width * 2, 3, 2, 1, "stage4.0.fuse_layers.3.1.0.0", "stage4.0.fuse_layers.3.1.0.1");
    auto add_1075 = convBnUpAdd(network, weightMap, *relu_1072->getOutput(0), *bn_1069->getOutput(0), width * 8, 3, 2, 1,
                                "stage4.0.fuse_layers.3.1.1.0", "stage4.0.fuse_layers.3.1.1.1", false);
    auto add_1078 = convBnUpAdd(network, weightMap, *relu_886->getOutput(0), *add_1075->getOutput(0), width * 8, 3, 2, 1,
                                "stage4.0.fuse_layers.3.2.0.0", "stage4.0.fuse_layers.3.2.0.1", false);
    auto add_1079 = network->addElementWise(*relu_914->getOutput(0), *add_1078->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_1080 = network->addActivation(*add_1079->getOutput(0), ActivationType::kRELU);

    auto relu_1087 = liteResBlock(network, weightMap, *relu_981->getOutput(0), width, "stage4.1.branches.0.0");
    auto relu_1094 = liteResBlock(network, weightMap, *relu_1087->getOutput(0), width, "stage4.1.branches.0.1");
    auto relu_1101 = liteResBlock(network, weightMap, *relu_1094->getOutput(0), width, "stage4.1.branches.0.2");
    auto relu_1108 = liteResBlock(network, weightMap, *relu_1101->getOutput(0), width, "stage4.1.branches.0.3");

    auto relu_1115 = liteResBlock(network, weightMap, *relu_1029->getOutput(0), width * 2, "stage4.1.branches.1.0");
    auto relu_1122 = liteResBlock(network, weightMap, *relu_1115->getOutput(0), width * 2, "stage4.1.branches.1.1");
    auto relu_1129 = liteResBlock(network, weightMap, *relu_1122->getOutput(0), width * 2, "stage4.1.branches.1.2");
    auto relu_1136 = liteResBlock(network, weightMap, *relu_1129->getOutput(0), width * 2, "stage4.1.branches.1.3");

    auto relu_1143 = liteResBlock(network, weightMap, *relu_1061->getOutput(0), width * 4, "stage4.1.branches.2.0");
    auto relu_1150 = liteResBlock(network, weightMap, *relu_1143->getOutput(0), width * 4, "stage4.1.branches.2.1");
    auto relu_1157 = liteResBlock(network, weightMap, *relu_1150->getOutput(0), width * 4, "stage4.1.branches.2.2");
    auto relu_1164 = liteResBlock(network, weightMap, *relu_1157->getOutput(0), width * 4, "stage4.1.branches.2.3");

    auto relu_1171 = liteResBlock(network, weightMap, *relu_1080->getOutput(0), width * 8, "stage4.1.branches.3.0");
    auto relu_1178 = liteResBlock(network, weightMap, *relu_1171->getOutput(0), width * 8, "stage4.1.branches.3.1");
    auto relu_1185 = liteResBlock(network, weightMap, *relu_1178->getOutput(0), width * 8, "stage4.1.branches.3.2");
    auto relu_1192 = liteResBlock(network, weightMap, *relu_1185->getOutput(0), width * 8, "stage4.1.branches.3.3");

    auto add_1214 = convBnUpAdd(network, weightMap, *relu_1136->getOutput(0), *relu_1108->getOutput(0), width, 1, 1, 0,
                                "stage4.1.fuse_layers.0.1.0", "stage4.1.fuse_layers.0.1.1", true);
    auto add_1236 = convBnUpAdd(network, weightMap, *relu_1164->getOutput(0), *add_1214->getOutput(0), width, 1, 1, 0,
                                "stage4.1.fuse_layers.0.2.0", "stage4.1.fuse_layers.0.2.1", true);
    auto add_1258 = convBnUpAdd(network, weightMap, *relu_1192->getOutput(0), *add_1236->getOutput(0), width, 1, 1, 0,
                                "stage4.1.fuse_layers.0.3.0", "stage4.1.fuse_layers.0.3.1", true);
    auto relu_1259 = network->addActivation(*add_1258->getOutput(0), ActivationType::kRELU);

    auto add_1262 = convBnUpAdd(network, weightMap, *relu_1108->getOutput(0), *relu_1136->getOutput(0), width * 2, 3, 2, 1,
                                "stage4.1.fuse_layers.1.0.0.0", "stage4.1.fuse_layers.1.0.0.1", false);
    auto add_1284 = convBnUpAdd(network, weightMap, *relu_1164->getOutput(0), *add_1262->getOutput(0), width * 2, 1, 1, 0,
                                "stage4.1.fuse_layers.1.2.0", "stage4.1.fuse_layers.1.2.1", true);
    auto add_1306 = convBnUpAdd(network, weightMap, *relu_1192->getOutput(0), *add_1284->getOutput(0), width * 2, 1, 1, 0,
                                "stage4.1.fuse_layers.1.3.0", "stage4.1.fuse_layers.1.3.1", true);
    auto relu_1307 = network->addActivation(*add_1306->getOutput(0), ActivationType::kRELU);

    auto relu_1310 = convBnRelu(network, weightMap, *relu_1108->getOutput(0), width, 3, 2, 1, "stage4.1.fuse_layers.2.0.0.0", "stage4.1.fuse_layers.2.0.0.1");
    auto bn_1312 = convBnRelu(network, weightMap, *relu_1310->getOutput(0), width * 4, 3, 2, 1, "stage4.1.fuse_layers.2.0.1.0", "stage4.1.fuse_layers.2.0.1.1", false);
    auto add_1315 = convBnUpAdd(network, weightMap, *relu_1136->getOutput(0), *bn_1312->getOutput(0), width * 4, 3, 2, 1,
                                "stage4.1.fuse_layers.2.1.0.0", "stage4.1.fuse_layers.2.1.0.1", false);
    auto add_1316 = network->addElementWise(*relu_1164->getOutput(0), *add_1315->getOutput(0), ElementWiseOperation::kSUM);
    auto add_1338 = convBnUpAdd(network, weightMap, *relu_1192->getOutput(0), *add_1316->getOutput(0), width * 4, 1, 1, 0,
                                "stage4.1.fuse_layers.2.3.0", "stage4.1.fuse_layers.2.3.1", true);
    auto relu_1339 = network->addActivation(*add_1338->getOutput(0), ActivationType::kRELU);

    auto relu_1342 = convBnRelu(network, weightMap, *relu_1108->getOutput(0), width, 3, 2, 1, "stage4.1.fuse_layers.3.0.0.0", "stage4.1.fuse_layers.3.0.0.1");
    auto relu_1345 = convBnRelu(network, weightMap, *relu_1342->getOutput(0), width, 3, 2, 1, "stage4.1.fuse_layers.3.0.1.0", "stage4.1.fuse_layers.3.0.1.1");
    auto bn_1347 = convBnRelu(network, weightMap, *relu_1345->getOutput(0), width * 8, 3, 2, 1, "stage4.1.fuse_layers.3.0.2.0", "stage4.1.fuse_layers.3.0.2.1", false);
    auto relu_1350 = convBnRelu(network, weightMap, *relu_1136->getOutput(0), width * 2, 3, 2, 1, "stage4.1.fuse_layers.3.1.0.0", "stage4.1.fuse_layers.3.1.0.1");
    auto add_1353 = convBnUpAdd(network, weightMap, *relu_1350->getOutput(0), *bn_1347->getOutput(0), width * 8, 3, 2, 1,
                                "stage4.1.fuse_layers.3.1.1.0", "stage4.1.fuse_layers.3.1.1.1", false);
    auto add_1356 = convBnUpAdd(network, weightMap, *relu_1164->getOutput(0), *add_1353->getOutput(0), width * 8, 3, 2, 1,
                                "stage4.1.fuse_layers.3.2.0.0", "stage4.1.fuse_layers.3.2.0.1", false);
    auto add_1357 = network->addElementWise(*relu_1192->getOutput(0), *add_1356->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_1358 = network->addActivation(*add_1357->getOutput(0), ActivationType::kRELU);

    auto relu_1365 = liteResBlock(network, weightMap, *relu_1259->getOutput(0), width, "stage4.2.branches.0.0");
    auto relu_1372 = liteResBlock(network, weightMap, *relu_1365->getOutput(0), width, "stage4.2.branches.0.1");
    auto relu_1379 = liteResBlock(network, weightMap, *relu_1372->getOutput(0), width, "stage4.2.branches.0.2");
    auto relu_1386 = liteResBlock(network, weightMap, *relu_1379->getOutput(0), width, "stage4.2.branches.0.3");

    auto relu_1393 = liteResBlock(network, weightMap, *relu_1307->getOutput(0), width * 2, "stage4.2.branches.1.0");
    auto relu_1400 = liteResBlock(network, weightMap, *relu_1393->getOutput(0), width * 2, "stage4.2.branches.1.1");
    auto relu_1407 = liteResBlock(network, weightMap, *relu_1400->getOutput(0), width * 2, "stage4.2.branches.1.2");
    auto relu_1414 = liteResBlock(network, weightMap, *relu_1407->getOutput(0), width * 2, "stage4.2.branches.1.3");

    auto relu_1421 = liteResBlock(network, weightMap, *relu_1339->getOutput(0), width * 4, "stage4.2.branches.2.0");
    auto relu_1428 = liteResBlock(network, weightMap, *relu_1421->getOutput(0), width * 4, "stage4.2.branches.2.1");
    auto relu_1435 = liteResBlock(network, weightMap, *relu_1428->getOutput(0), width * 4, "stage4.2.branches.2.2");
    auto relu_1442 = liteResBlock(network, weightMap, *relu_1435->getOutput(0), width * 4, "stage4.2.branches.2.3");

    auto relu_1449 = liteResBlock(network, weightMap, *relu_1358->getOutput(0), width * 8, "stage4.2.branches.3.0");
    auto relu_1456 = liteResBlock(network, weightMap, *relu_1449->getOutput(0), width * 8, "stage4.2.branches.3.1");
    auto relu_1463 = liteResBlock(network, weightMap, *relu_1456->getOutput(0), width * 8, "stage4.2.branches.3.2");
    auto relu_1470 = liteResBlock(network, weightMap, *relu_1463->getOutput(0), width * 8, "stage4.2.branches.3.3");

    auto add_1492 = convBnUpAdd(network, weightMap, *relu_1414->getOutput(0), *relu_1386->getOutput(0), width, 1, 1, 0,
                                "stage4.2.fuse_layers.0.1.0", "stage4.2.fuse_layers.0.1.1", true);
    auto add_1514 = convBnUpAdd(network, weightMap, *relu_1442->getOutput(0), *add_1492->getOutput(0), width, 1, 1, 0,
                                "stage4.2.fuse_layers.0.2.0", "stage4.2.fuse_layers.0.2.1", true);

    auto add_1536 = convBnUpAdd(network, weightMap, *relu_1470->getOutput(0), *add_1514->getOutput(0), width, 1, 1, 0,
                                "stage4.2.fuse_layers.0.3.0", "stage4.2.fuse_layers.0.3.1", true);
    auto relu_1537 = network->addActivation(*add_1536->getOutput(0), ActivationType::kRELU);

    auto add_1540 = convBnUpAdd(network, weightMap, *relu_1386->getOutput(0), *relu_1414->getOutput(0),
                                width * 2, 3, 2, 1, "stage4.2.fuse_layers.1.0.0.0", "stage4.2.fuse_layers.1.0.0.1", false);
    auto add_1562 = convBnUpAdd(network, weightMap, *relu_1442->getOutput(0), *add_1540->getOutput(0),
                                width * 2, 1, 1, 0, "stage4.2.fuse_layers.1.2.0", "stage4.2.fuse_layers.1.2.1", true);
    auto add_1584 = convBnUpAdd(network, weightMap, *relu_1470->getOutput(0), *add_1562->getOutput(0),
                                width * 2, 1, 1, 0, "stage4.2.fuse_layers.1.3.0", "stage4.2.fuse_layers.1.3.1", true);
    auto relu_1585 = network->addActivation(*add_1584->getOutput(0), ActivationType::kRELU);

    auto relu_1588 = convBnRelu(network, weightMap, *relu_1386->getOutput(0), width, 3, 2, 1, "stage4.2.fuse_layers.2.0.0.0", "stage4.2.fuse_layers.2.0.0.1");
    auto bn_1590 = convBnRelu(network, weightMap, *relu_1588->getOutput(0), width * 4, 3, 2, 1, "stage4.2.fuse_layers.2.0.1.0", "stage4.2.fuse_layers.2.0.1.1", false);
    auto add_1593 = convBnUpAdd(network, weightMap, *relu_1414->getOutput(0), *bn_1590->getOutput(0), width * 4, 3, 2, 1,
                                "stage4.2.fuse_layers.2.1.0.0", "stage4.2.fuse_layers.2.1.0.1", false);
    auto add_1594 = network->addElementWise(*relu_1442->getOutput(0), *add_1593->getOutput(0), ElementWiseOperation::kSUM);
    auto add_1616 = convBnUpAdd(network, weightMap, *relu_1470->getOutput(0), *add_1594->getOutput(0), width * 4, 1, 1, 0,
                                "stage4.2.fuse_layers.2.3.0", "stage4.2.fuse_layers.2.3.1", true);
    auto relu_1617 = network->addActivation(*add_1616->getOutput(0), ActivationType::kRELU);

    auto relu_1620 = convBnRelu(network, weightMap, *relu_1386->getOutput(0), width, 3, 2, 1, "stage4.2.fuse_layers.3.0.0.0", "stage4.2.fuse_layers.3.0.0.1");
    auto relu_1623 = convBnRelu(network, weightMap, *relu_1620->getOutput(0), width, 3, 2, 1, "stage4.2.fuse_layers.3.0.1.0", "stage4.2.fuse_layers.3.0.1.1");
    auto bn_1625 = convBnRelu(network, weightMap, *relu_1623->getOutput(0), width * 8, 3, 2, 1, "stage4.2.fuse_layers.3.0.2.0", "stage4.2.fuse_layers.3.0.2.1", false);
    auto relu_1628 = convBnRelu(network, weightMap, *relu_1414->getOutput(0), width * 2, 3, 2, 1, "stage4.2.fuse_layers.3.1.0.0", "stage4.2.fuse_layers.3.1.0.1");
    auto add_1631 = convBnUpAdd(network, weightMap, *relu_1628->getOutput(0), *bn_1625->getOutput(0), width * 8, 3, 2, 1,
                                "stage4.2.fuse_layers.3.1.1.0", "stage4.2.fuse_layers.3.1.1.1", false);
    auto add_1634 = convBnUpAdd(network, weightMap, *relu_1442->getOutput(0), *add_1631->getOutput(0), width * 8, 3, 2, 1,
                                "stage4.2.fuse_layers.3.2.0.0", "stage4.2.fuse_layers.3.2.0.1", false);
    auto add_1635 = network->addElementWise(*relu_1470->getOutput(0), *add_1634->getOutput(0), ElementWiseOperation::kSUM);
    auto relu_1636 = network->addActivation(*add_1635->getOutput(0), ActivationType::kRELU);

    nvinfer1::Dims dim = relu_1537->getOutput(0)->getDimensions();
    dim.d[0] = relu_1585->getOutput(0)->getDimensions().d[0];
    auto resize_1655 = netAddUpsampleBi(network, relu_1585->getOutput(0), dim);
    dim.d[0] = relu_1617->getOutput(0)->getDimensions().d[0];
    auto resize_1668 = netAddUpsampleBi(network, relu_1617->getOutput(0), dim);
    dim.d[0] = relu_1636->getOutput(0)->getDimensions().d[0];
    auto resize_1681 = netAddUpsampleBi(network, relu_1636->getOutput(0), dim);

    ITensor *concatTensors[] = {relu_1537->getOutput(0), resize_1655->getOutput(0), resize_1668->getOutput(0), resize_1681->getOutput(0)};
    auto concat_1682 = network->addConcatenation(concatTensors, 4);
    concat_1682->setAxis(0);
    auto relu_1685 = convBnRelu(network, weightMap, *concat_1682->getOutput(0), width * 15, 1, 1, 0, "aux_head.0", "aux_head.1", true, true);
    auto conv_1686 = network->addConvolutionNd(*relu_1685->getOutput(0), NUM_CLASSES, DimsHW{1, 1}, weightMap["aux_head.3.weight"], weightMap["aux_head.3.bias"]);
    conv_1686->setStrideNd(DimsHW{1, 1});
    conv_1686->setPaddingNd(DimsHW{0, 0});
    auto reshape_1701 = network->addShuffle(*conv_1686->getOutput(0));
    nvinfer1::Dims reshape_dim;
    reshape_dim.nbDims = 2;
    reshape_dim.d[0] = NUM_CLASSES;
    reshape_dim.d[1] = -1;
    reshape_1701->setReshapeDimensions(reshape_dim);

    auto softmax_1714 = network->addSoftMax(*reshape_1701->getOutput(0));
    softmax_1714->setAxes(2);

    auto relu_1689 = convBnRelu(network, weightMap, *concat_1682->getOutput(0), 512, 3, 1, 1, "conv3x3_ocr.0", "conv3x3_ocr.1", true, true);

    auto reshape_1710 = network->addShuffle(*relu_1689->getOutput(0));
    nvinfer1::Dims reshape_dim1;
    reshape_dim1.nbDims = 2;
    reshape_dim1.d[0] = 512;
    reshape_dim1.d[1] = -1;
    reshape_1710->setReshapeDimensions(reshape_dim1);
    nvinfer1::Permutation permutation1;
    permutation1.order[0] = 1;
    permutation1.order[1] = 0;
    reshape_1710->setSecondTranspose(permutation1);

    auto matmul_1715 = network->addMatrixMultiply(*softmax_1714->getOutput(0), MatrixOperation::kNONE,
                                                  *reshape_1710->getOutput(0), MatrixOperation::kNONE);

    auto transpose_1716 = network->addShuffle(*matmul_1715->getOutput(0));
    nvinfer1::Permutation permutation2;
    permutation2.order[0] = 1;
    permutation2.order[1] = 0;
    transpose_1716->setFirstTranspose(permutation2);

    auto unsqueeze_1717 = network->addShuffle(*transpose_1716->getOutput(0));
    nvinfer1::Dims reshape_dim3;
    reshape_dim3.nbDims = 3;
    reshape_dim3.d[0] = 512;
    reshape_dim3.d[1] = NUM_CLASSES;
    reshape_dim3.d[2] = 1;
    unsqueeze_1717->setReshapeDimensions(reshape_dim3);

    auto relu_1737 = convBnRelu(network, weightMap, *unsqueeze_1717->getOutput(0), 256, 1, 1, 0, "ocr_distri_head.object_context_block.f_object.0", "ocr_distri_head.object_context_block.f_object.1.0", true, true);

    auto relu_1740 = convBnRelu(network, weightMap, *relu_1737->getOutput(0), 256, 1, 1, 0, "ocr_distri_head.object_context_block.f_object.2", "ocr_distri_head.object_context_block.f_object.3.0", true, true);

    auto reshape_1747 = network->addShuffle(*relu_1740->getOutput(0));
    nvinfer1::Dims reshape_dim4;
    reshape_dim4.nbDims = 2;
    reshape_dim4.d[0] = 256;
    reshape_dim4.d[1] = -1;
    reshape_1747->setReshapeDimensions(reshape_dim4);

    auto relu_1723 = convBnRelu(network, weightMap, *relu_1689->getOutput(0), 256, 1, 1, 0, "ocr_distri_head.object_context_block.f_pixel.0", "ocr_distri_head.object_context_block.f_pixel.1.0", true, true);
    auto relu_1726 = convBnRelu(network, weightMap, *relu_1723->getOutput(0), 256, 1, 1, 0, "ocr_distri_head.object_context_block.f_pixel.2", "ocr_distri_head.object_context_block.f_pixel.3.0", true, true);

    auto reshape_1733 = network->addShuffle(*relu_1726->getOutput(0));
    nvinfer1::Dims reshape_dim5;
    reshape_dim5.nbDims = 2;
    reshape_dim5.d[0] = 256;
    reshape_dim5.d[1] = -1;
    reshape_1733->setReshapeDimensions(reshape_dim5);
    nvinfer1::Permutation permutation3;
    permutation3.order[0] = 1;
    permutation3.order[1] = 0;
    reshape_1733->setSecondTranspose(permutation3);

    auto matmul_1759 = network->addMatrixMultiply(*reshape_1733->getOutput(0), MatrixOperation::kNONE, *reshape_1747->getOutput(0), MatrixOperation::kNONE);
    nvinfer1::Dims constant_dim;
    constant_dim.nbDims = 2;
    int allNum = INPUT_H * INPUT_W / 16;
    constant_dim.d[0] = INPUT_H * INPUT_W / 16;
    constant_dim.d[1] = 1;
    Weights wgt{DataType::kFLOAT, nullptr, allNum};
    float *w = new float[allNum];
    for (int i = 0; i < allNum; i++)
    {
        w[i] = 0.0625;
    }
    wgt.values = w;
    auto constant_1761 = network->addConstant(constant_dim, wgt);

    auto mul_1761 = network->addElementWise(*constant_1761->getOutput(0), *matmul_1759->getOutput(0), ElementWiseOperation::kPROD);

    auto softmax_1762 = network->addSoftMax(*mul_1761->getOutput(0));
    softmax_1762->setAxes(2);

    auto relu_1750 = convBnRelu(network, weightMap, *unsqueeze_1717->getOutput(0), 256, 1, 1, 0, "ocr_distri_head.object_context_block.f_down.0", "ocr_distri_head.object_context_block.f_down.1.0", true, true);

    auto reshape_1757 = network->addShuffle(*relu_1750->getOutput(0));
    nvinfer1::Dims reshape_dim6;
    reshape_dim6.nbDims = 2;
    reshape_dim6.d[0] = 256;
    reshape_dim6.d[1] = -1;
    reshape_1757->setReshapeDimensions(reshape_dim6);
    nvinfer1::Permutation permutation4;
    permutation4.order[0] = 1;
    permutation4.order[1] = 0;
    reshape_1757->setSecondTranspose(permutation4);

    auto matmul_1763 = network->addMatrixMultiply(*softmax_1762->getOutput(0), MatrixOperation::kNONE, *reshape_1757->getOutput(0), MatrixOperation::kNONE);

    auto reshape_1777 = network->addShuffle(*matmul_1763->getOutput(0));
    nvinfer1::Dims reshape_dim7;
    reshape_dim7.nbDims = 3;
    reshape_dim7.d[0] = 256;
    reshape_dim7.d[1] = INPUT_H / 4;
    reshape_dim7.d[2] = INPUT_W / 4;
    reshape_1777->setReshapeDimensions(reshape_dim7);
    nvinfer1::Permutation permutation5;
    permutation5.order[0] = 1;
    permutation5.order[1] = 0;
    reshape_1777->setFirstTranspose(permutation5);

    auto relu_1780 = convBnRelu(network, weightMap, *reshape_1777->getOutput(0), 512, 1, 1, 0, "ocr_distri_head.object_context_block.f_up.0", "ocr_distri_head.object_context_block.f_up.1.0", true, true);

    ITensor *concatTensors1[] = {relu_1780->getOutput(0), relu_1689->getOutput(0)};
    auto concat_1781 = network->addConcatenation(concatTensors1, 2);

    auto relu_1784 = convBnRelu(network, weightMap, *concat_1781->getOutput(0), 512, 1, 1, 0, "ocr_distri_head.conv_bn_dropout.0", "ocr_distri_head.conv_bn_dropout.1.0", true, true);

    auto conv_1785 = network->addConvolutionNd(*relu_1784->getOutput(0), NUM_CLASSES, DimsHW{1, 1}, weightMap["cls_head.weight"], weightMap["cls_head.bias"]);
    debug_print(conv_1785->getOutput(0), "cls_head");
    dim.nbDims = 3;
    dim.d[0] = NUM_CLASSES;
    dim.d[1] = INPUT_H;
    dim.d[2] = INPUT_W;
    auto feature_map = netAddUpsampleBi(network, conv_1785->getOutput(0), dim);
    debug_print(feature_map->getOutput(0), "upsample");
    auto topk = network->addTopK(*feature_map->getOutput(0), TopKOperation::kMAX, 1, 0X01);

    debug_print(topk->getOutput(0), "topk");

    std::cout << "set name out" << std::endl;
    topk->getOutput(1)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*topk->getOutput(1));
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize((1 << 30)); // 1G
#ifdef USE_FP16
    std::cout << "use fp16" << std::endl;
    config->setFlag(BuilderFlag::kFP16);
#endif
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build success!" << std::endl;
    network->destroy();
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}
void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, std::string wtsPath, int width)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wtsPath, width);
    assert(engine != nullptr);
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, int &width, std::string &img_dir)
{
    if (std::string(argv[1]) == "-s" && argc == 5)
    {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        width = std::stoi(argv[4]);
    }
    else if (std::string(argv[1]) == "-d" && argc == 4)
    {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    }
    else
    {
        return false;
    }
    return true;
}
void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, int batchSize)
{
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    cudaSetDevice(DEVICE);
    std::string wtsPath = "";
    std::string engine_name = "";
    int width;
    std::string img_dir;
    // parse args
    if (!parse_args(argc, argv, wtsPath, engine_name, width, img_dir))
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./hrnet_ocr -s [.wts] [.engine] [18 or 32 or 48]  // serialize model to plan file" << std::endl;
        std::cerr << "./hrnet_ocr -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    // create a model using the API directly and serialize it to a stream
    if (!wtsPath.empty())
    {
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, wtsPath, width);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cerr << "could not open plan file" << std::endl;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0)
    {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    // prepare input data ---------------------------
    cudaSetDeviceFlags(cudaDeviceMapHost);
    float *data;
    int *prob; // using int. output is index
    CHECK(cudaHostAlloc((void **)&data, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **)&prob, BATCH_SIZE * OUTPUT_SIZE * sizeof(int), cudaHostAllocMapped));

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    for (int f = 0; f < (int)file_names.size(); f++)
    {
        std::cout << file_names[f] << std::endl;
        cv::Mat pr_img;
        cv::Mat img_BGR = cv::imread(img_dir + "/" + file_names[f], 1); // BGR
        cv::Mat img;
        cv::cvtColor(img_BGR, img, cv::COLOR_BGR2RGB);
        if (img.empty())
            continue;
        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
        img = pr_img.clone(); // for img show
        pr_img.convertTo(pr_img, CV_32FC3);
        if (!pr_img.isContinuous())
        {
            pr_img = pr_img.clone();
        }
        std::memcpy(data, pr_img.data, BATCH_SIZE * 3 * INPUT_W * INPUT_H * sizeof(float));

        cudaHostGetDevicePointer((void **)&buffers[inputIndex], (void *)data, 0);  // buffers[inputIndex]-->data
        cudaHostGetDevicePointer((void **)&buffers[outputIndex], (void *)prob, 0); // buffers[outputIndex] --> prob

        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        doInference(*context, stream, buffers, BATCH_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat outimg(INPUT_H, INPUT_W, CV_8UC1);
        for (int row = 0; row < INPUT_H; ++row)
        {
            uchar *uc_pixel = outimg.data + row * outimg.step;
            for (int col = 0; col < INPUT_W; ++col)
            {
                uc_pixel[col] = (uchar)prob[row * INPUT_W + col];
            }
        }
        cv::Mat im_color;
        cv::cvtColor(outimg, im_color, cv::COLOR_GRAY2RGB);
        cv::Mat lut = createLTU(NUM_CLASSES);
        cv::LUT(im_color, lut, im_color);
        // false color
        cv::cvtColor(im_color, im_color, cv::COLOR_RGB2GRAY);
        cv::applyColorMap(im_color, im_color, cv::COLORMAP_HOT);
        // cv::imshow("False Color Map", im_color);
        cv::imwrite(std::to_string(f) + "_false_color_map.png", im_color);
        //fusion
        cv::Mat fusionImg;
        cv::addWeighted(img, 1, im_color, 0.8, 1, fusionImg);
        // cv::imshow("Fusion Img", fusionImg);
        // cv::waitKey(0);
        cv::imwrite(std::to_string(f) + "_fusion_img.png", fusionImg);
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
