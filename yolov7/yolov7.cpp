#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define CONF_THRESH 0.25
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;




static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}


ICudaEngine* build_engine_yolov7e6e(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path)
{
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    auto* conv0 = ReOrg(network, weightMap, *data, 3);


    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 80, 3, 1, 1, "model.1");
    auto conv2 = DownC(network, weightMap, *conv1->getOutput(0), 80, 160, "model.2");

    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.3");
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.4");

    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv4->getOutput(0), 64, 3, 1, 1, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");
    IElementWiseLayer* conv9 = convBnSilu(network, weightMap, *conv8->getOutput(0), 64, 3, 1, 1, "model.9");
    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *conv9->getOutput(0), 64, 3, 1, 1, "model.10");

    ITensor* input_tensor_11[] = { conv10->getOutput(0), conv8->getOutput(0),conv6->getOutput(0), conv4->getOutput(0),
        conv3->getOutput(0) };
    IConcatenationLayer* concat11 = network->addConcatenation(input_tensor_11, 5);

    IElementWiseLayer* conv12 = convBnSilu(network, weightMap, *concat11->getOutput(0), 160, 1, 1, 0, "model.12");


    IElementWiseLayer* conv13 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.13");
    IElementWiseLayer* conv14 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.14");

    IElementWiseLayer* conv15 = convBnSilu(network, weightMap, *conv14->getOutput(0), 64, 3, 1, 1, "model.15");
    IElementWiseLayer* conv16 = convBnSilu(network, weightMap, *conv15->getOutput(0), 64, 3, 1, 1, "model.16");
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *conv16->getOutput(0), 64, 3, 1, 1, "model.17");
    IElementWiseLayer* conv18 = convBnSilu(network, weightMap, *conv17->getOutput(0), 64, 3, 1, 1, "model.18");
    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *conv18->getOutput(0), 64, 3, 1, 1, "model.19");
    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *conv19->getOutput(0), 64, 3, 1, 1, "model.20");
    ITensor* input_tensor_21[] = { conv20->getOutput(0), conv18->getOutput(0),conv16->getOutput(0), conv14->getOutput(0),
        conv13->getOutput(0) };
    IConcatenationLayer* concat21 = network->addConcatenation(input_tensor_21, 5);
    
    IElementWiseLayer* conv22 = convBnSilu(network, weightMap, *concat21->getOutput(0), 160, 1, 1, 0, "model.22");
    auto conv23 = network->addElementWise(*conv22->getOutput(0), *conv12->getOutput(0), ElementWiseOperation::kSUM);

    auto conv24 = DownC(network, weightMap, *conv23->getOutput(0), 160, 320, "model.24");
    IElementWiseLayer* conv25 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 1, 1, 0, "model.25");
    IElementWiseLayer* conv26 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 1, 1, 0, "model.26");

    IElementWiseLayer* conv27 = convBnSilu(network, weightMap, *conv26->getOutput(0), 128, 3, 1, 1, "model.27");
    IElementWiseLayer* conv28 = convBnSilu(network, weightMap, *conv27->getOutput(0), 128, 3, 1, 1, "model.28");
    IElementWiseLayer* conv29 = convBnSilu(network, weightMap, *conv28->getOutput(0), 128, 3, 1, 1, "model.29");
    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *conv29->getOutput(0), 128, 3, 1, 1, "model.30");
    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *conv30->getOutput(0), 128, 3, 1, 1, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 128, 3, 1, 1, "model.32");

    ITensor* input_tensor_33[] = { conv32->getOutput(0), conv30->getOutput(0),conv28->getOutput(0), conv26->getOutput(0),
        conv25->getOutput(0)};
    IConcatenationLayer* concat33 = network->addConcatenation(input_tensor_33, 5);

    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *concat33->getOutput(0), 320, 1, 1, 0, "model.34");

    IElementWiseLayer* conv35 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 1, 1, 0, "model.35");
    IElementWiseLayer* conv36 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 1, 1, 0, "model.36");

    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *conv36->getOutput(0), 128, 3, 1, 1, "model.37");
    IElementWiseLayer* conv38 = convBnSilu(network, weightMap, *conv37->getOutput(0), 128, 3, 1, 1, "model.38");
    IElementWiseLayer* conv39 = convBnSilu(network, weightMap, *conv38->getOutput(0), 128, 3, 1, 1, "model.39");
    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *conv39->getOutput(0), 128, 3, 1, 1, "model.40");
    IElementWiseLayer* conv41 = convBnSilu(network, weightMap, *conv40->getOutput(0), 128, 3, 1, 1, "model.41");
    IElementWiseLayer* conv42 = convBnSilu(network, weightMap, *conv41->getOutput(0), 128, 3, 1, 1, "model.42");

    ITensor* input_tensor_43[] = { conv42->getOutput(0), conv40->getOutput(0),conv38->getOutput(0), conv36->getOutput(0),
        conv35->getOutput(0)};
    IConcatenationLayer* concat43 = network->addConcatenation(input_tensor_43, 5);
    IElementWiseLayer* conv44 = convBnSilu(network, weightMap, *concat43->getOutput(0), 320, 1, 1, 0, "model.44");

    auto conv45 = network->addElementWise(*conv44->getOutput(0), *conv34->getOutput(0), ElementWiseOperation::kSUM);

    auto conv46 = DownC(network, weightMap, *conv45->getOutput(0), 320, 640, "model.46");//=====


    IElementWiseLayer* conv47 = convBnSilu(network, weightMap, *conv46->getOutput(0), 256, 1, 1, 0, "model.47");
    IElementWiseLayer* conv48 = convBnSilu(network, weightMap, *conv46->getOutput(0), 256, 1, 1, 0, "model.48");

    IElementWiseLayer* conv49 = convBnSilu(network, weightMap, *conv48->getOutput(0), 256, 3, 1, 1, "model.49");
    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *conv49->getOutput(0), 256, 3, 1, 1, "model.50");
    IElementWiseLayer* conv51 = convBnSilu(network, weightMap, *conv50->getOutput(0), 256, 3, 1, 1, "model.51");
    IElementWiseLayer* conv52 = convBnSilu(network, weightMap, *conv51->getOutput(0), 256, 3, 1, 1, "model.52");
    IElementWiseLayer* conv53 = convBnSilu(network, weightMap, *conv52->getOutput(0), 256, 3, 1, 1, "model.53");
    IElementWiseLayer* conv54 = convBnSilu(network, weightMap, *conv53->getOutput(0), 256, 3, 1, 1, "model.54");
    
    ITensor* input_tensor_55[] = { conv54->getOutput(0), conv52->getOutput(0),conv50->getOutput(0), conv48->getOutput(0),
        conv47->getOutput(0) };
    IConcatenationLayer* concat55 = network->addConcatenation(input_tensor_55, 5);
    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *concat55->getOutput(0), 640, 1, 1, 0, "model.56");

    IElementWiseLayer* conv57 = convBnSilu(network, weightMap, *conv46->getOutput(0), 256, 1, 1, 0, "model.57");
    IElementWiseLayer* conv58 = convBnSilu(network, weightMap, *conv46->getOutput(0), 256, 1, 1, 0, "model.58");

    IElementWiseLayer* conv59 = convBnSilu(network, weightMap, *conv58->getOutput(0), 256, 3, 1, 1, "model.59");
    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv59->getOutput(0), 256, 3, 1, 1, "model.60");
    IElementWiseLayer* conv61 = convBnSilu(network, weightMap, *conv60->getOutput(0), 256, 3, 1, 1, "model.61");
    IElementWiseLayer* conv62 = convBnSilu(network, weightMap, *conv61->getOutput(0), 256, 3, 1, 1, "model.62");
    IElementWiseLayer* conv63 = convBnSilu(network, weightMap, *conv62->getOutput(0), 256, 3, 1, 1, "model.63");
    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *conv63->getOutput(0), 256, 3, 1, 1, "model.64");
    ITensor* input_tensor_65[] = { conv64->getOutput(0), conv62->getOutput(0),conv60->getOutput(0), conv58->getOutput(0),
        conv57->getOutput(0) };
    IConcatenationLayer* concat65 = network->addConcatenation(input_tensor_65, 5);
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *concat65->getOutput(0), 640, 1, 1, 0, "model.66");
    auto conv67 = network->addElementWise(*conv66->getOutput(0), *conv56->getOutput(0), ElementWiseOperation::kSUM);

    auto conv68 = DownC(network, weightMap, *conv67->getOutput(0), 640, 960, "model.68");//=====

    IElementWiseLayer* conv69 = convBnSilu(network, weightMap, *conv68->getOutput(0), 384, 1, 1, 0, "model.69");
    IElementWiseLayer* conv70 = convBnSilu(network, weightMap, *conv68->getOutput(0), 384, 1, 1, 0, "model.70");

    IElementWiseLayer* conv71 = convBnSilu(network, weightMap, *conv70->getOutput(0), 384, 3, 1, 1, "model.71");
    IElementWiseLayer* conv72 = convBnSilu(network, weightMap, *conv71->getOutput(0), 384, 3, 1, 1, "model.72");
    IElementWiseLayer* conv73 = convBnSilu(network, weightMap, *conv72->getOutput(0), 384, 3, 1, 1, "model.73");
    IElementWiseLayer* conv74 = convBnSilu(network, weightMap, *conv73->getOutput(0), 384, 3, 1, 1, "model.74");
    IElementWiseLayer* conv75 = convBnSilu(network, weightMap, *conv74->getOutput(0), 384, 3, 1, 1, "model.75");
    IElementWiseLayer* conv76 = convBnSilu(network, weightMap, *conv75->getOutput(0), 384, 3, 1, 1, "model.76");
    ITensor* input_tensor_77[] = { conv76->getOutput(0), conv74->getOutput(0),conv72->getOutput(0), conv70->getOutput(0),
        conv69->getOutput(0) };
    IConcatenationLayer* concat77 = network->addConcatenation(input_tensor_77, 5);
    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *concat77->getOutput(0), 960, 1, 1, 0, "model.78");

    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *conv68->getOutput(0), 384, 1, 1, 0, "model.79");
    IElementWiseLayer* conv80 = convBnSilu(network, weightMap, *conv68->getOutput(0), 384, 1, 1, 0, "model.80");

    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *conv80->getOutput(0), 384, 3, 1, 1, "model.81");
    IElementWiseLayer* conv82 = convBnSilu(network, weightMap, *conv81->getOutput(0), 384, 3, 1, 1, "model.82");
    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *conv82->getOutput(0), 384, 3, 1, 1, "model.83");
    IElementWiseLayer* conv84 = convBnSilu(network, weightMap, *conv83->getOutput(0), 384, 3, 1, 1, "model.84");
    IElementWiseLayer* conv85 = convBnSilu(network, weightMap, *conv84->getOutput(0), 384, 3, 1, 1, "model.85");
    IElementWiseLayer* conv86 = convBnSilu(network, weightMap, *conv85->getOutput(0), 384, 3, 1, 1, "model.86");
    ITensor* input_tensor_87[] = { conv86->getOutput(0), conv84->getOutput(0),conv82->getOutput(0), conv80->getOutput(0),
        conv79->getOutput(0) };
    IConcatenationLayer* concat87 = network->addConcatenation(input_tensor_87, 5);
    IElementWiseLayer* conv88 = convBnSilu(network, weightMap, *concat87->getOutput(0), 960, 1, 1, 0, "model.88");
    auto conv89 = network->addElementWise(*conv88->getOutput(0), *conv78->getOutput(0), ElementWiseOperation::kSUM);


    auto conv90 = DownC(network, weightMap, *conv89->getOutput(0), 960, 1280, "model.90");//=====

    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *conv90->getOutput(0), 512, 1, 1, 0, "model.91");
    IElementWiseLayer* conv92 = convBnSilu(network, weightMap, *conv90->getOutput(0), 512, 1, 1, 0, "model.92");

    IElementWiseLayer* conv93 = convBnSilu(network, weightMap, *conv92->getOutput(0), 512, 3, 1, 1, "model.93");
    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *conv93->getOutput(0), 512, 3, 1, 1, "model.94");
    IElementWiseLayer* conv95 = convBnSilu(network, weightMap, *conv94->getOutput(0), 512, 3, 1, 1, "model.95");
    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *conv95->getOutput(0), 512, 3, 1, 1, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *conv96->getOutput(0), 512, 3, 1, 1, "model.97");
    IElementWiseLayer* conv98 = convBnSilu(network, weightMap, *conv97->getOutput(0), 512, 3, 1, 1, "model.98");
    ITensor* input_tensor_99[] = { conv98->getOutput(0), conv96->getOutput(0),conv94->getOutput(0), conv92->getOutput(0),
      conv91->getOutput(0) };
    IConcatenationLayer* concat99 = network->addConcatenation(input_tensor_99, 5);
    IElementWiseLayer* conv100 = convBnSilu(network, weightMap, *concat99->getOutput(0), 1280, 1, 1, 0, "model.100");
    
    IElementWiseLayer* conv101 = convBnSilu(network, weightMap, *conv90->getOutput(0), 512, 1, 1, 0, "model.101");
    IElementWiseLayer* conv102 = convBnSilu(network, weightMap, *conv90->getOutput(0), 512, 1, 1, 0, "model.102");

    IElementWiseLayer* conv103 = convBnSilu(network, weightMap, *conv102->getOutput(0), 512, 3, 1, 1, "model.103");
    IElementWiseLayer* conv104 = convBnSilu(network, weightMap, *conv103->getOutput(0), 512, 3, 1, 1, "model.104");
    IElementWiseLayer* conv105 = convBnSilu(network, weightMap, *conv104->getOutput(0), 512, 3, 1, 1, "model.105");
    IElementWiseLayer* conv106 = convBnSilu(network, weightMap, *conv105->getOutput(0), 512, 3, 1, 1, "model.106");
    IElementWiseLayer* conv107 = convBnSilu(network, weightMap, *conv106->getOutput(0), 512, 3, 1, 1, "model.107");
    IElementWiseLayer* conv108 = convBnSilu(network, weightMap, *conv107->getOutput(0), 512, 3, 1, 1, "model.108");
    ITensor* input_tensor_109[] = { conv108->getOutput(0), conv106->getOutput(0),conv104->getOutput(0), conv102->getOutput(0),
      conv101->getOutput(0) };
    IConcatenationLayer* concat109 = network->addConcatenation(input_tensor_109, 5);
    IElementWiseLayer* conv110 = convBnSilu(network, weightMap, *concat109->getOutput(0), 1280, 1, 1, 0, "model.110");
    auto conv111 = network->addElementWise(*conv110->getOutput(0), *conv100->getOutput(0), ElementWiseOperation::kSUM);
    //---------------------------yolov7e6e head---------------------------------
    auto conv112 = SPPCSPC(network, weightMap, *conv111->getOutput(0), 640, "model.112");
    IElementWiseLayer* conv113 = convBnSilu(network, weightMap, *conv112->getOutput(0), 480, 1, 1, 0, "model.113");


    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* re114 = network->addResize(*conv113->getOutput(0));
    re114->setResizeMode(ResizeMode::kNEAREST);
    re114->setScales(scale, 3);

    IElementWiseLayer* conv115 = convBnSilu(network, weightMap, *conv89->getOutput(0), 480, 1, 1, 0, "model.115");
    ITensor* input_tensor_116[] = { conv115->getOutput(0), re114->getOutput(0) };
    IConcatenationLayer* concat116 = network->addConcatenation(input_tensor_116, 2);


    IElementWiseLayer* conv117 = convBnSilu(network, weightMap, *concat116->getOutput(0), 384, 1, 1, 0, "model.117");
    IElementWiseLayer* conv118 = convBnSilu(network, weightMap, *concat116->getOutput(0), 384, 1, 1, 0, "model.118");

    IElementWiseLayer* conv119 = convBnSilu(network, weightMap, *conv118->getOutput(0), 192, 3, 1, 1, "model.119");
    IElementWiseLayer* conv120 = convBnSilu(network, weightMap, *conv119->getOutput(0), 192, 3, 1, 1, "model.120");
    IElementWiseLayer* conv121 = convBnSilu(network, weightMap, *conv120->getOutput(0), 192, 3, 1, 1, "model.121");
    IElementWiseLayer* conv122 = convBnSilu(network, weightMap, *conv121->getOutput(0), 192, 3, 1, 1, "model.122");
    IElementWiseLayer* conv123 = convBnSilu(network, weightMap, *conv122->getOutput(0), 192, 3, 1, 1, "model.123");
    IElementWiseLayer* conv124 = convBnSilu(network, weightMap, *conv123->getOutput(0), 192, 3, 1, 1, "model.124");
    ITensor* input_tensor_125[] = { conv124->getOutput(0), conv123->getOutput(0),conv122->getOutput(0), conv121->getOutput(0),
        conv120->getOutput(0), conv119->getOutput(0), conv118->getOutput(0), conv117->getOutput(0) };
    IConcatenationLayer* concat125 = network->addConcatenation(input_tensor_125, 8);
    IElementWiseLayer* conv126 = convBnSilu(network, weightMap, *concat125->getOutput(0), 480, 1, 1, 0, "model.126");

    IElementWiseLayer* conv127 = convBnSilu(network, weightMap, *concat116->getOutput(0), 384, 1, 1, 0, "model.127");
    IElementWiseLayer* conv128 = convBnSilu(network, weightMap, *concat116->getOutput(0), 384, 1, 1, 0, "model.128");

    IElementWiseLayer* conv129 = convBnSilu(network, weightMap, *conv128->getOutput(0), 192, 3, 1, 1, "model.129");
    IElementWiseLayer* conv130 = convBnSilu(network, weightMap, *conv129->getOutput(0), 192, 3, 1, 1, "model.130");
    IElementWiseLayer* conv131 = convBnSilu(network, weightMap, *conv130->getOutput(0), 192, 3, 1, 1, "model.131");
    IElementWiseLayer* conv132 = convBnSilu(network, weightMap, *conv131->getOutput(0), 192, 3, 1, 1, "model.132");
    IElementWiseLayer* conv133 = convBnSilu(network, weightMap, *conv132->getOutput(0), 192, 3, 1, 1, "model.133");
    IElementWiseLayer* conv134 = convBnSilu(network, weightMap, *conv133->getOutput(0), 192, 3, 1, 1, "model.134");
    ITensor* input_tensor_135[] = { conv134->getOutput(0), conv133->getOutput(0),conv132->getOutput(0), conv131->getOutput(0),
        conv130->getOutput(0), conv129->getOutput(0), conv128->getOutput(0), conv127->getOutput(0) };
    IConcatenationLayer* concat135 = network->addConcatenation(input_tensor_135, 8);
    IElementWiseLayer* conv136 = convBnSilu(network, weightMap, *concat135->getOutput(0), 480, 1, 1, 0, "model.136");
    auto conv137 = network->addElementWise(*conv136->getOutput(0), *conv126->getOutput(0), ElementWiseOperation::kSUM);

    IElementWiseLayer* conv138 = convBnSilu(network, weightMap, *conv137->getOutput(0), 320, 1, 1, 0, "model.138");
    IResizeLayer* re139 = network->addResize(*conv138->getOutput(0));
    re139->setResizeMode(ResizeMode::kNEAREST);
    re139->setScales(scale, 3);
    IElementWiseLayer* conv140 = convBnSilu(network, weightMap, *conv67->getOutput(0), 320, 1, 1, 0, "model.140");
    ITensor* input_tensor_141[] = { conv140->getOutput(0), re139->getOutput(0) };
    IConcatenationLayer* concat141 = network->addConcatenation(input_tensor_141, 2);

    IElementWiseLayer* conv142 = convBnSilu(network, weightMap, *concat141->getOutput(0), 256, 1, 1, 0, "model.142");
    IElementWiseLayer* conv143 = convBnSilu(network, weightMap, *concat141->getOutput(0), 256, 1, 1, 0, "model.143");

    IElementWiseLayer* conv144 = convBnSilu(network, weightMap, *conv143->getOutput(0), 128, 3, 1, 1, "model.144");
    IElementWiseLayer* conv145 = convBnSilu(network, weightMap, *conv144->getOutput(0), 128, 3, 1, 1, "model.145");
    IElementWiseLayer* conv146 = convBnSilu(network, weightMap, *conv145->getOutput(0), 128, 3, 1, 1, "model.146");
    IElementWiseLayer* conv147 = convBnSilu(network, weightMap, *conv146->getOutput(0), 128, 3, 1, 1, "model.147");
    IElementWiseLayer* conv148 = convBnSilu(network, weightMap, *conv147->getOutput(0), 128, 3, 1, 1, "model.148");
    IElementWiseLayer* conv149 = convBnSilu(network, weightMap, *conv148->getOutput(0), 128, 3, 1, 1, "model.149");

    ITensor* input_tensor_150[] = { conv149->getOutput(0), conv148->getOutput(0),conv147->getOutput(0), conv146->getOutput(0),
        conv145->getOutput(0), conv144->getOutput(0), conv143->getOutput(0), conv142->getOutput(0) };
    IConcatenationLayer* concat150 = network->addConcatenation(input_tensor_150, 8);

    IElementWiseLayer* conv151 = convBnSilu(network, weightMap, *concat150->getOutput(0), 320, 1, 1, 0, "model.151");

    IElementWiseLayer* conv152 = convBnSilu(network, weightMap, *concat141->getOutput(0), 256, 1, 1, 0, "model.152");
    IElementWiseLayer* conv153 = convBnSilu(network, weightMap, *concat141->getOutput(0), 256, 1, 1, 0, "model.153");

    IElementWiseLayer* conv154 = convBnSilu(network, weightMap, *conv153->getOutput(0), 128, 3, 1, 1, "model.154");
    IElementWiseLayer* conv155 = convBnSilu(network, weightMap, *conv154->getOutput(0), 128, 3, 1, 1, "model.155");
    IElementWiseLayer* conv156 = convBnSilu(network, weightMap, *conv155->getOutput(0), 128, 3, 1, 1, "model.156");
    IElementWiseLayer* conv157 = convBnSilu(network, weightMap, *conv156->getOutput(0), 128, 3, 1, 1, "model.157");
    IElementWiseLayer* conv158 = convBnSilu(network, weightMap, *conv157->getOutput(0), 128, 3, 1, 1, "model.158");
    IElementWiseLayer* conv159 = convBnSilu(network, weightMap, *conv158->getOutput(0), 128, 3, 1, 1, "model.159");
    ITensor* input_tensor_160[] = { conv159->getOutput(0), conv158->getOutput(0),conv157->getOutput(0), conv156->getOutput(0),
        conv155->getOutput(0), conv154->getOutput(0), conv153->getOutput(0), conv152->getOutput(0) };
    IConcatenationLayer* concat160 = network->addConcatenation(input_tensor_160, 8);
    IElementWiseLayer* conv161 = convBnSilu(network, weightMap, *concat160->getOutput(0), 320, 1, 1, 0, "model.161");
    auto conv162 = network->addElementWise(*conv161->getOutput(0), *conv151->getOutput(0), ElementWiseOperation::kSUM);

    IElementWiseLayer* conv163 = convBnSilu(network, weightMap, *conv162->getOutput(0), 160, 1, 1, 0, "model.163");

    IResizeLayer* re164 = network->addResize(*conv163->getOutput(0));
    re164->setResizeMode(ResizeMode::kNEAREST);
    re164->setScales(scale, 3);

    IElementWiseLayer* conv165 = convBnSilu(network, weightMap, *conv45->getOutput(0), 160, 1, 1, 0, "model.165");
    ITensor* input_tensor_166[] = { conv165->getOutput(0), re164->getOutput(0) };
    IConcatenationLayer* concat166 = network->addConcatenation(input_tensor_166, 2);

    IElementWiseLayer* conv167 = convBnSilu(network, weightMap, *concat166->getOutput(0), 128, 1, 1, 0, "model.167");
    IElementWiseLayer* conv168 = convBnSilu(network, weightMap, *concat166->getOutput(0), 128, 1, 1, 0, "model.168");
    IElementWiseLayer* conv169 = convBnSilu(network, weightMap, *conv168->getOutput(0), 64, 3, 1, 1, "model.169");
    IElementWiseLayer* conv170 = convBnSilu(network, weightMap, *conv169->getOutput(0), 64, 3, 1, 1, "model.170");
    IElementWiseLayer* conv171 = convBnSilu(network, weightMap, *conv170->getOutput(0), 64, 3, 1, 1, "model.171");
    IElementWiseLayer* conv172 = convBnSilu(network, weightMap, *conv171->getOutput(0), 64, 3, 1, 1, "model.172");
    IElementWiseLayer* conv173 = convBnSilu(network, weightMap, *conv172->getOutput(0), 64, 3, 1, 1, "model.173");
    IElementWiseLayer* conv174 = convBnSilu(network, weightMap, *conv173->getOutput(0), 64, 3, 1, 1, "model.174");


    ITensor* input_tensor_175[] = { conv174->getOutput(0), conv173->getOutput(0),conv172->getOutput(0), conv171->getOutput(0),
       conv170->getOutput(0), conv169->getOutput(0), conv168->getOutput(0), conv167->getOutput(0) };
    IConcatenationLayer* concat175 = network->addConcatenation(input_tensor_175, 8); 
    IElementWiseLayer* conv176 = convBnSilu(network, weightMap, *concat175->getOutput(0), 160, 1, 1, 0, "model.176");
    IElementWiseLayer* conv177 = convBnSilu(network, weightMap, *concat166->getOutput(0), 128, 1, 1, 0, "model.177");
    IElementWiseLayer* conv178 = convBnSilu(network, weightMap, *concat166->getOutput(0), 128, 1, 1, 0, "model.178");
    
    IElementWiseLayer* conv179 = convBnSilu(network, weightMap, *conv178->getOutput(0), 64, 3, 1, 1, "model.179");
    IElementWiseLayer* conv180 = convBnSilu(network, weightMap, *conv179->getOutput(0), 64, 3, 1, 1, "model.180");
    IElementWiseLayer* conv181 = convBnSilu(network, weightMap, *conv180->getOutput(0), 64, 3, 1, 1, "model.181");
    IElementWiseLayer* conv182 = convBnSilu(network, weightMap, *conv181->getOutput(0), 64, 3, 1, 1, "model.182");
    IElementWiseLayer* conv183 = convBnSilu(network, weightMap, *conv182->getOutput(0), 64, 3, 1, 1, "model.183");
    IElementWiseLayer* conv184 = convBnSilu(network, weightMap, *conv183->getOutput(0), 64, 3, 1, 1, "model.184");
    ITensor* input_tensor_185[] = { conv184->getOutput(0), conv183->getOutput(0),conv182->getOutput(0), conv181->getOutput(0),
       conv180->getOutput(0), conv179->getOutput(0), conv178->getOutput(0), conv177->getOutput(0) };
    IConcatenationLayer* concat185 = network->addConcatenation(input_tensor_185, 8);
    IElementWiseLayer* conv186 = convBnSilu(network, weightMap, *concat185->getOutput(0), 160, 1, 1, 0, "model.186");
    auto conv187 = network->addElementWise(*conv186->getOutput(0), *conv176->getOutput(0), ElementWiseOperation::kSUM);

    auto conv188 = DownC(network, weightMap, *conv187->getOutput(0), 160, 320, "model.188");


    ITensor* input_tensor_189[] = { conv188->getOutput(0), conv162->getOutput(0) };
    IConcatenationLayer* concat189 = network->addConcatenation(input_tensor_189, 2);

    IElementWiseLayer* conv190 = convBnSilu(network, weightMap, *concat189->getOutput(0), 256, 1, 1, 0, "model.190");
    IElementWiseLayer* conv191 = convBnSilu(network, weightMap, *concat189->getOutput(0), 256, 1, 1, 0, "model.191");

    IElementWiseLayer* conv192 = convBnSilu(network, weightMap, *conv191->getOutput(0), 128, 3, 1, 1, "model.192");
    IElementWiseLayer* conv193 = convBnSilu(network, weightMap, *conv192->getOutput(0), 128, 3, 1, 1, "model.193");
    IElementWiseLayer* conv194 = convBnSilu(network, weightMap, *conv193->getOutput(0), 128, 3, 1, 1, "model.194");
    IElementWiseLayer* conv195 = convBnSilu(network, weightMap, *conv194->getOutput(0), 128, 3, 1, 1, "model.195");
    IElementWiseLayer* conv196 = convBnSilu(network, weightMap, *conv195->getOutput(0), 128, 3, 1, 1, "model.196");
    IElementWiseLayer* conv197 = convBnSilu(network, weightMap, *conv196->getOutput(0), 128, 3, 1, 1, "model.197");


    ITensor* input_tensor_198[] = { conv197->getOutput(0), conv196->getOutput(0),conv195->getOutput(0), conv194->getOutput(0),
       conv193->getOutput(0), conv192->getOutput(0), conv191->getOutput(0), conv190->getOutput(0) };
    IConcatenationLayer* concat198 = network->addConcatenation(input_tensor_198, 8);
    IElementWiseLayer* conv199 = convBnSilu(network, weightMap, *concat198->getOutput(0), 320, 1, 1, 0, "model.199");

    IElementWiseLayer* conv200 = convBnSilu(network, weightMap, *concat189->getOutput(0), 256, 1, 1, 0, "model.200");
    IElementWiseLayer* conv201 = convBnSilu(network, weightMap, *concat189->getOutput(0), 256, 1, 1, 0, "model.201");

    IElementWiseLayer* conv202 = convBnSilu(network, weightMap, *conv201->getOutput(0), 128, 3, 1, 1, "model.202");
    IElementWiseLayer* conv203 = convBnSilu(network, weightMap, *conv202->getOutput(0), 128, 3, 1, 1, "model.203");
    IElementWiseLayer* conv204 = convBnSilu(network, weightMap, *conv203->getOutput(0), 128, 3, 1, 1, "model.204");
    IElementWiseLayer* conv205 = convBnSilu(network, weightMap, *conv204->getOutput(0), 128, 3, 1, 1, "model.205");
    IElementWiseLayer* conv206 = convBnSilu(network, weightMap, *conv205->getOutput(0), 128, 3, 1, 1, "model.206");
    IElementWiseLayer* conv207 = convBnSilu(network, weightMap, *conv206->getOutput(0), 128, 3, 1, 1, "model.207");
    ITensor* input_tensor_208[] = { conv207->getOutput(0), conv206->getOutput(0),conv205->getOutput(0), conv204->getOutput(0),
      conv203->getOutput(0), conv202->getOutput(0), conv201->getOutput(0), conv200->getOutput(0) };
    IConcatenationLayer* concat208 = network->addConcatenation(input_tensor_208, 8);
    IElementWiseLayer* conv209 = convBnSilu(network, weightMap, *concat208->getOutput(0), 320, 1, 1, 0, "model.209");
    auto conv210 = network->addElementWise(*conv209->getOutput(0), *conv199->getOutput(0), ElementWiseOperation::kSUM);


    auto conv211 = DownC(network, weightMap, *conv210->getOutput(0), 320, 480, "model.211");
    ITensor* input_tensor_212[] = { conv211->getOutput(0), conv137->getOutput(0) };
    IConcatenationLayer* concat212 = network->addConcatenation(input_tensor_212, 2);

    IElementWiseLayer* conv213 = convBnSilu(network, weightMap, *concat212->getOutput(0), 384, 1, 1, 0, "model.213");
    IElementWiseLayer* conv214 = convBnSilu(network, weightMap, *concat212->getOutput(0), 384, 1, 1, 0, "model.214");

    IElementWiseLayer* conv215 = convBnSilu(network, weightMap, *conv214->getOutput(0), 192, 3, 1, 1, "model.215");
    IElementWiseLayer* conv216 = convBnSilu(network, weightMap, *conv215->getOutput(0), 192, 3, 1, 1, "model.216");
    IElementWiseLayer* conv217 = convBnSilu(network, weightMap, *conv216->getOutput(0), 192, 3, 1, 1, "model.217");
    IElementWiseLayer* conv218 = convBnSilu(network, weightMap, *conv217->getOutput(0), 192, 3, 1, 1, "model.218");
    IElementWiseLayer* conv219 = convBnSilu(network, weightMap, *conv218->getOutput(0), 192, 3, 1, 1, "model.219");
    IElementWiseLayer* conv220 = convBnSilu(network, weightMap, *conv219->getOutput(0), 192, 3, 1, 1, "model.220");

    ITensor* input_tensor_221[] = { conv220->getOutput(0), conv219->getOutput(0),conv218->getOutput(0), conv217->getOutput(0),
      conv216->getOutput(0), conv215->getOutput(0), conv214->getOutput(0), conv213->getOutput(0) };
    IConcatenationLayer* concat221 = network->addConcatenation(input_tensor_221, 8);
    IElementWiseLayer* conv222 = convBnSilu(network, weightMap, *concat221->getOutput(0), 480, 1, 1, 0, "model.222");

    IElementWiseLayer* conv223 = convBnSilu(network, weightMap, *concat212->getOutput(0), 384, 1, 1, 0, "model.223");
    IElementWiseLayer* conv224 = convBnSilu(network, weightMap, *concat212->getOutput(0), 384, 1, 1, 0, "model.224");

    IElementWiseLayer* conv225 = convBnSilu(network, weightMap, *conv224->getOutput(0), 192, 3, 1, 1, "model.225");
    IElementWiseLayer* conv226 = convBnSilu(network, weightMap, *conv225->getOutput(0), 192, 3, 1, 1, "model.226");
    IElementWiseLayer* conv227 = convBnSilu(network, weightMap, *conv226->getOutput(0), 192, 3, 1, 1, "model.227");
    IElementWiseLayer* conv228 = convBnSilu(network, weightMap, *conv227->getOutput(0), 192, 3, 1, 1, "model.228");
    IElementWiseLayer* conv229 = convBnSilu(network, weightMap, *conv228->getOutput(0), 192, 3, 1, 1, "model.229");
    IElementWiseLayer* conv230 = convBnSilu(network, weightMap, *conv229->getOutput(0), 192, 3, 1, 1, "model.230");
    ITensor* input_tensor_231[] = { conv230->getOutput(0), conv229->getOutput(0),conv228->getOutput(0), conv227->getOutput(0),
     conv226->getOutput(0), conv225->getOutput(0), conv224->getOutput(0), conv223->getOutput(0) };
    IConcatenationLayer* concat231 = network->addConcatenation(input_tensor_231, 8);
    IElementWiseLayer* conv232 = convBnSilu(network, weightMap, *concat231->getOutput(0), 480, 1, 1, 0, "model.232");

    auto conv233 = network->addElementWise(*conv232->getOutput(0), *conv222->getOutput(0), ElementWiseOperation::kSUM);


    auto conv234 = DownC(network, weightMap, *conv233->getOutput(0), 480, 640, "model.234");
    ITensor* input_tensor_235[] = { conv234->getOutput(0), conv112->getOutput(0) };
    IConcatenationLayer* concat235 = network->addConcatenation(input_tensor_235, 2);


    IElementWiseLayer* conv236 = convBnSilu(network, weightMap, *concat235->getOutput(0), 512, 1, 1, 0, "model.236");
    IElementWiseLayer* conv237 = convBnSilu(network, weightMap, *concat235->getOutput(0), 512, 1, 1, 0, "model.237");

    IElementWiseLayer* conv238 = convBnSilu(network, weightMap, *conv237->getOutput(0), 256, 3, 1, 1, "model.238");
    IElementWiseLayer* conv239 = convBnSilu(network, weightMap, *conv238->getOutput(0), 256, 3, 1, 1, "model.239");
    IElementWiseLayer* conv240 = convBnSilu(network, weightMap, *conv239->getOutput(0), 256, 3, 1, 1, "model.240");
    IElementWiseLayer* conv241 = convBnSilu(network, weightMap, *conv240->getOutput(0), 256, 3, 1, 1, "model.241");
    IElementWiseLayer* conv242 = convBnSilu(network, weightMap, *conv241->getOutput(0), 256, 3, 1, 1, "model.242");
    IElementWiseLayer* conv243 = convBnSilu(network, weightMap, *conv242->getOutput(0), 256, 3, 1, 1, "model.243");
  
    ITensor* input_tensor_244[] = { conv243->getOutput(0), conv242->getOutput(0),conv241->getOutput(0), conv240->getOutput(0),
     conv239->getOutput(0), conv238->getOutput(0), conv237->getOutput(0), conv236->getOutput(0) };
    IConcatenationLayer* concat244 = network->addConcatenation(input_tensor_244, 8);
    IElementWiseLayer* conv245 = convBnSilu(network, weightMap, *concat244->getOutput(0), 640, 1, 1, 0, "model.245");

    IElementWiseLayer* conv246 = convBnSilu(network, weightMap, *concat235->getOutput(0), 512, 1, 1, 0, "model.246");
    IElementWiseLayer* conv247 = convBnSilu(network, weightMap, *concat235->getOutput(0), 512, 1, 1, 0, "model.247");

    IElementWiseLayer* conv248 = convBnSilu(network, weightMap, *conv247->getOutput(0), 256, 3, 1, 1, "model.248");
    IElementWiseLayer* conv249 = convBnSilu(network, weightMap, *conv248->getOutput(0), 256, 3, 1, 1, "model.249");
    IElementWiseLayer* conv250 = convBnSilu(network, weightMap, *conv249->getOutput(0), 256, 3, 1, 1, "model.250");
    IElementWiseLayer* conv251 = convBnSilu(network, weightMap, *conv250->getOutput(0), 256, 3, 1, 1, "model.251");
    IElementWiseLayer* conv252 = convBnSilu(network, weightMap, *conv251->getOutput(0), 256, 3, 1, 1, "model.252");
    IElementWiseLayer* conv253 = convBnSilu(network, weightMap, *conv252->getOutput(0), 256, 3, 1, 1, "model.253");

    ITensor* input_tensor_254[] = { conv253->getOutput(0), conv252->getOutput(0),conv251->getOutput(0), conv250->getOutput(0),
    conv249->getOutput(0), conv248->getOutput(0), conv247->getOutput(0), conv246->getOutput(0) };
    IConcatenationLayer* concat254 = network->addConcatenation(input_tensor_254, 8);

    IElementWiseLayer* conv255= convBnSilu(network, weightMap, *concat254->getOutput(0), 640, 1, 1, 0, "model.255");
    auto conv256 = network->addElementWise(*conv255->getOutput(0), *conv245->getOutput(0), ElementWiseOperation::kSUM);

    IElementWiseLayer* conv257 = convBnSilu(network, weightMap, *conv187->getOutput(0), 320, 3, 1, 1, "model.257");
    IElementWiseLayer* conv258 = convBnSilu(network, weightMap, *conv210->getOutput(0), 640, 3, 1, 1, "model.258");
    IElementWiseLayer* conv259 = convBnSilu(network, weightMap, *conv233->getOutput(0), 960, 3, 1, 1, "model.259");
    IElementWiseLayer* conv260 = convBnSilu(network, weightMap, *conv256->getOutput(0), 1280, 3, 1, 1, "model.260");



    // out
    IConvolutionLayer* cv105_0 = network->addConvolutionNd(*conv257->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.261.m.0.weight"], weightMap["model.261.m.0.bias"]);
    assert(cv105_0);
    cv105_0->setName("cv105.0");
    IConvolutionLayer* cv105_1 = network->addConvolutionNd(*conv258->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.261.m.1.weight"], weightMap["model.261.m.1.bias"]);
    assert(cv105_1);
    cv105_1->setName("cv105.1");
    IConvolutionLayer* cv105_2 = network->addConvolutionNd(*conv259->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.261.m.2.weight"], weightMap["model.261.m.2.bias"]);
    assert(cv105_2);
    cv105_2->setName("cv105.2");
    IConvolutionLayer* cv105_3 = network->addConvolutionNd(*conv260->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.261.m.3.weight"], weightMap["model.261.m.3.bias"]);
    assert(cv105_3);
    cv105_3->setName("cv105.3");



    /*------------detect-----------*/
    auto yolo = addYoLoLayer(network, weightMap, "model.261", std::vector<IConvolutionLayer*>{cv105_0, cv105_1, cv105_2, cv105_3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_yolov7d6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path)
{
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    /*----------------------------------yolov7d6 backbone-----------------------------------------*/
    auto* conv0 = ReOrg(network, weightMap, *data, 3);


    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 96, 3, 1, 1, "model.1");
    auto conv2 = DownC(network, weightMap, *conv1->getOutput(0), 96, 192, "model.2");

    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.3");
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.4");

    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv4->getOutput(0), 64, 3, 1, 1, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");
    IElementWiseLayer* conv9 = convBnSilu(network, weightMap, *conv8->getOutput(0), 64, 3, 1, 1, "model.9");
    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *conv9->getOutput(0), 64, 3, 1, 1, "model.10");
    IElementWiseLayer* conv11 = convBnSilu(network, weightMap, *conv10->getOutput(0), 64, 3, 1, 1, "model.11");
    IElementWiseLayer* conv12 = convBnSilu(network, weightMap, *conv11->getOutput(0), 64, 3, 1, 1, "model.12");

    ITensor* input_tensor_13[] = { conv12->getOutput(0), conv10->getOutput(0),conv8->getOutput(0), conv6->getOutput(0),
        conv4->getOutput(0),conv3->getOutput(0) };
    IConcatenationLayer* concat13 = network->addConcatenation(input_tensor_13, 6);

    IElementWiseLayer* conv14 = convBnSilu(network, weightMap, *concat13->getOutput(0), 192, 1, 1, 0, "model.14");


    auto conv15 = DownC(network, weightMap, *conv14->getOutput(0), 192, 384, "model.15");
    IElementWiseLayer* conv16 = convBnSilu(network, weightMap, *conv15->getOutput(0), 128, 1, 1, 0, "model.16");
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *conv15->getOutput(0), 128, 1, 1, 0, "model.17");

    IElementWiseLayer* conv18 = convBnSilu(network, weightMap, *conv17->getOutput(0), 128, 3, 1, 1, "model.18");
    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *conv18->getOutput(0), 128, 3, 1, 1, "model.19");
    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *conv19->getOutput(0), 128, 3, 1, 1, "model.20");
    IElementWiseLayer* conv21 = convBnSilu(network, weightMap, *conv20->getOutput(0), 128, 3, 1, 1, "model.21");
    IElementWiseLayer* conv22 = convBnSilu(network, weightMap, *conv21->getOutput(0), 128, 3, 1, 1, "model.22");
    IElementWiseLayer* conv23 = convBnSilu(network, weightMap, *conv22->getOutput(0), 128, 3, 1, 1, "model.23");
    IElementWiseLayer* conv24 = convBnSilu(network, weightMap, *conv23->getOutput(0), 128, 3, 1, 1, "model.24");
    IElementWiseLayer* conv25 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 3, 1, 1, "model.25");
    ITensor* input_tensor_26[] = { conv25->getOutput(0), conv23->getOutput(0),conv21->getOutput(0), conv19->getOutput(0),
        conv17->getOutput(0),conv16->getOutput(0) };
    IConcatenationLayer* concat26 = network->addConcatenation(input_tensor_26, 6);

    IElementWiseLayer* conv27 = convBnSilu(network, weightMap, *concat26->getOutput(0), 384, 1, 1, 0, "model.27");


    auto conv28 = DownC(network, weightMap, *conv27->getOutput(0), 384, 768, "model.28");
    IElementWiseLayer* conv29 = convBnSilu(network, weightMap, *conv28->getOutput(0), 256, 1, 1, 0, "model.29");
    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *conv28->getOutput(0), 256, 1, 1, 0, "model.30");

    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *conv30->getOutput(0), 256, 3, 1, 1, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 256, 3, 1, 1, "model.32");
    IElementWiseLayer* conv33 = convBnSilu(network, weightMap, *conv32->getOutput(0), 256, 3, 1, 1, "model.33");
    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *conv33->getOutput(0), 256, 3, 1, 1, "model.34");
    IElementWiseLayer* conv35 = convBnSilu(network, weightMap, *conv34->getOutput(0), 256, 3, 1, 1, "model.35");
    IElementWiseLayer* conv36 = convBnSilu(network, weightMap, *conv35->getOutput(0), 256, 3, 1, 1, "model.36");
    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *conv36->getOutput(0), 256, 3, 1, 1, "model.37");
    IElementWiseLayer* conv38 = convBnSilu(network, weightMap, *conv37->getOutput(0), 256, 3, 1, 1, "model.38");
    ITensor* input_tensor_39[] = { conv38->getOutput(0), conv36->getOutput(0),conv34->getOutput(0), conv32->getOutput(0),
        conv30->getOutput(0), conv29 ->getOutput(0)};
    IConcatenationLayer* concat39 = network->addConcatenation(input_tensor_39, 6);

    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *concat39->getOutput(0), 768, 1, 1, 0, "model.40");
    auto conv41 = DownC(network, weightMap, *conv40->getOutput(0), 768, 1152, "model.41");
    IElementWiseLayer* conv42 = convBnSilu(network, weightMap, *conv41->getOutput(0), 384, 1, 1, 0, "model.42");
    IElementWiseLayer* conv43 = convBnSilu(network, weightMap, *conv41->getOutput(0), 384, 1, 1, 0, "model.43");

    IElementWiseLayer* conv44 = convBnSilu(network, weightMap, *conv43->getOutput(0), 384, 3, 1, 1, "model.44");
    IElementWiseLayer* conv45 = convBnSilu(network, weightMap, *conv44->getOutput(0), 384, 3, 1, 1, "model.45");
    IElementWiseLayer* conv46 = convBnSilu(network, weightMap, *conv45->getOutput(0), 384, 3, 1, 1, "model.46");
    IElementWiseLayer* conv47 = convBnSilu(network, weightMap, *conv46->getOutput(0), 384, 3, 1, 1, "model.47");
    IElementWiseLayer* conv48 = convBnSilu(network, weightMap, *conv47->getOutput(0), 384, 3, 1, 1, "model.48");
    IElementWiseLayer* conv49 = convBnSilu(network, weightMap, *conv48->getOutput(0), 384, 3, 1, 1, "model.49");
    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *conv49->getOutput(0), 384, 3, 1, 1, "model.50");
    IElementWiseLayer* conv51 = convBnSilu(network, weightMap, *conv50->getOutput(0), 384, 3, 1, 1, "model.51");

    ITensor* input_tensor_52[] = { conv51->getOutput(0), conv49->getOutput(0),conv47->getOutput(0), conv45->getOutput(0),
        conv43->getOutput(0),conv42->getOutput(0) };
    IConcatenationLayer* concat52 = network->addConcatenation(input_tensor_52, 6);
    IElementWiseLayer* conv53 = convBnSilu(network, weightMap, *concat52->getOutput(0), 1152, 1, 1, 0, "model.53");

    auto conv54 = DownC(network, weightMap, *conv53->getOutput(0), 1152, 1536, "model.54");//=====
    IElementWiseLayer* conv55 = convBnSilu(network, weightMap, *conv54->getOutput(0), 512, 1, 1, 0, "model.55");
    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *conv54->getOutput(0), 512, 1, 1, 0, "model.56");

    IElementWiseLayer* conv57 = convBnSilu(network, weightMap, *conv56->getOutput(0), 512, 3, 1, 1, "model.57");
    IElementWiseLayer* conv58 = convBnSilu(network, weightMap, *conv57->getOutput(0), 512, 3, 1, 1, "model.58");
    IElementWiseLayer* conv59 = convBnSilu(network, weightMap, *conv58->getOutput(0), 512, 3, 1, 1, "model.59");
    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv59->getOutput(0), 512, 3, 1, 1, "model.60");
    IElementWiseLayer* conv61 = convBnSilu(network, weightMap, *conv60->getOutput(0), 512, 3, 1, 1, "model.61");
    IElementWiseLayer* conv62 = convBnSilu(network, weightMap, *conv61->getOutput(0), 512, 3, 1, 1, "model.62");
    IElementWiseLayer* conv63 = convBnSilu(network, weightMap, *conv62->getOutput(0), 512, 3, 1, 1, "model.63");
    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *conv63->getOutput(0), 512, 3, 1, 1, "model.64");
    ITensor* input_tensor_65[] = { conv64->getOutput(0), conv62->getOutput(0),conv60->getOutput(0), conv58->getOutput(0),
        conv56->getOutput(0),conv55->getOutput(0) };
    IConcatenationLayer* concat65 = network->addConcatenation(input_tensor_65, 6);
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *concat65->getOutput(0), 1536, 1, 1, 0, "model.66");

    //------------------------yolov7e6 head-------------------------------
    auto conv67 = SPPCSPC(network, weightMap, *conv66->getOutput(0), 768, "model.67");
    IElementWiseLayer* conv68 = convBnSilu(network, weightMap, *conv67->getOutput(0), 576, 1, 1, 0, "model.68");


    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* re69 = network->addResize(*conv68->getOutput(0));
    re69->setResizeMode(ResizeMode::kNEAREST);
    re69->setScales(scale, 3);

    IElementWiseLayer* conv70 = convBnSilu(network, weightMap, *conv53->getOutput(0), 576, 1, 1, 0, "model.70");
    ITensor* input_tensor_71[] = { conv70->getOutput(0), re69->getOutput(0) };
    IConcatenationLayer* concat71 = network->addConcatenation(input_tensor_71, 2);
    IElementWiseLayer* conv72 = convBnSilu(network, weightMap, *concat71->getOutput(0), 384, 1, 1, 0, "model.72");
    IElementWiseLayer* conv73 = convBnSilu(network, weightMap, *concat71->getOutput(0), 384, 1, 1, 0, "model.73");

    IElementWiseLayer* conv74 = convBnSilu(network, weightMap, *conv73->getOutput(0), 192, 3, 1, 1, "model.74");
    IElementWiseLayer* conv75 = convBnSilu(network, weightMap, *conv74->getOutput(0), 192, 3, 1, 1, "model.75");
    IElementWiseLayer* conv76 = convBnSilu(network, weightMap, *conv75->getOutput(0), 192, 3, 1, 1, "model.76");
    IElementWiseLayer* conv77 = convBnSilu(network, weightMap, *conv76->getOutput(0), 192, 3, 1, 1, "model.77");
    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *conv77->getOutput(0), 192, 3, 1, 1, "model.78");
    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *conv78->getOutput(0), 192, 3, 1, 1, "model.79");
    IElementWiseLayer* conv80 = convBnSilu(network, weightMap, *conv79->getOutput(0), 192, 3, 1, 1, "model.80");
    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *conv80->getOutput(0), 192, 3, 1, 1, "model.81");

    ITensor* input_tensor_82[] = { conv81->getOutput(0), conv80->getOutput(0),conv79->getOutput(0), conv78->getOutput(0),
        conv77->getOutput(0), conv76->getOutput(0), conv75->getOutput(0), conv74->getOutput(0), conv73->getOutput(0),
        conv72->getOutput(0) };
    IConcatenationLayer* concat82 = network->addConcatenation(input_tensor_82, 10);
    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *concat82->getOutput(0), 576, 1, 1, 0, "model.83");

    IElementWiseLayer* conv84 = convBnSilu(network, weightMap, *conv83->getOutput(0), 384, 1, 1, 0, "model.84");
    IResizeLayer* re85 = network->addResize(*conv84->getOutput(0));
    re85->setResizeMode(ResizeMode::kNEAREST);
    re85->setScales(scale, 3);
    IElementWiseLayer* conv86 = convBnSilu(network, weightMap, *conv40->getOutput(0), 384, 1, 1, 0, "model.86");
    ITensor* input_tensor_87[] = { conv86->getOutput(0), re85->getOutput(0) };
    IConcatenationLayer* concat87 = network->addConcatenation(input_tensor_87, 2);

    IElementWiseLayer* conv88 = convBnSilu(network, weightMap, *concat87->getOutput(0), 256, 1, 1, 0, "model.88");
    IElementWiseLayer* conv89 = convBnSilu(network, weightMap, *concat87->getOutput(0), 256, 1, 1, 0, "model.89");

    IElementWiseLayer* conv90 = convBnSilu(network, weightMap, *conv89->getOutput(0), 128, 3, 1, 1, "model.90");
    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *conv90->getOutput(0), 128, 3, 1, 1, "model.91");
    IElementWiseLayer* conv92 = convBnSilu(network, weightMap, *conv91->getOutput(0), 128, 3, 1, 1, "model.92");
    IElementWiseLayer* conv93 = convBnSilu(network, weightMap, *conv92->getOutput(0), 128, 3, 1, 1, "model.93");
    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *conv93->getOutput(0), 128, 3, 1, 1, "model.94");
    IElementWiseLayer* conv95 = convBnSilu(network, weightMap, *conv94->getOutput(0), 128, 3, 1, 1, "model.95");
    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *conv95->getOutput(0), 128, 3, 1, 1, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *conv96->getOutput(0), 128, 3, 1, 1, "model.97");

    ITensor* input_tensor_98[] = { conv97->getOutput(0), conv96->getOutput(0),conv95->getOutput(0), conv94->getOutput(0),
        conv93->getOutput(0), conv92->getOutput(0), conv91->getOutput(0), conv90->getOutput(0),conv89->getOutput(0), 
        conv88->getOutput(0) };
    IConcatenationLayer* concat98 = network->addConcatenation(input_tensor_98, 10);

    IElementWiseLayer* conv99 = convBnSilu(network, weightMap, *concat98->getOutput(0), 384, 1, 1, 0, "model.99");

    IElementWiseLayer* conv100 = convBnSilu(network, weightMap, *conv99->getOutput(0), 192, 1, 1, 0, "model.100");
    IResizeLayer* re101 = network->addResize(*conv100->getOutput(0));
    re101->setResizeMode(ResizeMode::kNEAREST);
    re101->setScales(scale, 3);
    IElementWiseLayer* conv102 = convBnSilu(network, weightMap, *conv27->getOutput(0), 192, 1, 1, 0, "model.102");
    ITensor* input_tensor_103[] = { conv102->getOutput(0), re101->getOutput(0) };
    IConcatenationLayer* concat103 = network->addConcatenation(input_tensor_103, 2);

    IElementWiseLayer* conv104 = convBnSilu(network, weightMap, *concat103->getOutput(0), 128, 1, 1, 0, "model.104");
    IElementWiseLayer* conv105 = convBnSilu(network, weightMap, *concat103->getOutput(0), 128, 1, 1, 0, "model.105");
    IElementWiseLayer* conv106 = convBnSilu(network, weightMap, *conv105->getOutput(0), 64, 3, 1, 1, "model.106");
    IElementWiseLayer* conv107 = convBnSilu(network, weightMap, *conv106->getOutput(0), 64, 3, 1, 1, "model.107");
    IElementWiseLayer* conv108 = convBnSilu(network, weightMap, *conv107->getOutput(0), 64, 3, 1, 1, "model.108");
    IElementWiseLayer* conv109 = convBnSilu(network, weightMap, *conv108->getOutput(0), 64, 3, 1, 1, "model.109");
    IElementWiseLayer* conv110 = convBnSilu(network, weightMap, *conv109->getOutput(0), 64, 3, 1, 1, "model.110");
    IElementWiseLayer* conv111 = convBnSilu(network, weightMap, *conv110->getOutput(0), 64, 3, 1, 1, "model.111");
    IElementWiseLayer* conv112 = convBnSilu(network, weightMap, *conv111->getOutput(0), 64, 3, 1, 1, "model.112");
    IElementWiseLayer* conv113 = convBnSilu(network, weightMap, *conv112->getOutput(0), 64, 3, 1, 1, "model.113");

    ITensor* input_tensor_114[] = { conv113->getOutput(0), conv112->getOutput(0),conv111->getOutput(0), conv110->getOutput(0),
       conv109->getOutput(0), conv108->getOutput(0), conv107->getOutput(0), conv106->getOutput(0), conv105->getOutput(0),
        conv104->getOutput(0) };
    IConcatenationLayer* concat114 = network->addConcatenation(input_tensor_114, 10);

    IElementWiseLayer* conv115 = convBnSilu(network, weightMap, *concat114->getOutput(0), 192, 1, 1, 0, "model.115");

    auto conv116 = DownC(network, weightMap, *conv115->getOutput(0), 192, 384, "model.116");
    ITensor* input_tensor_117[] = { conv116->getOutput(0), conv99->getOutput(0) };
    IConcatenationLayer* concat117 = network->addConcatenation(input_tensor_117, 2);

    IElementWiseLayer* conv118 = convBnSilu(network, weightMap, *concat117->getOutput(0), 256, 1, 1, 0, "model.118");
    IElementWiseLayer* conv119 = convBnSilu(network, weightMap, *concat117->getOutput(0), 256, 1, 1, 0, "model.119");

    IElementWiseLayer* conv120 = convBnSilu(network, weightMap, *conv119->getOutput(0), 128, 3, 1, 1, "model.120");
    IElementWiseLayer* conv121 = convBnSilu(network, weightMap, *conv120->getOutput(0), 128, 3, 1, 1, "model.121");
    IElementWiseLayer* conv122 = convBnSilu(network, weightMap, *conv121->getOutput(0), 128, 3, 1, 1, "model.122");
    IElementWiseLayer* conv123 = convBnSilu(network, weightMap, *conv122->getOutput(0), 128, 3, 1, 1, "model.123");
    IElementWiseLayer* conv124 = convBnSilu(network, weightMap, *conv123->getOutput(0), 128, 3, 1, 1, "model.124");
    IElementWiseLayer* conv125 = convBnSilu(network, weightMap, *conv124->getOutput(0), 128, 3, 1, 1, "model.125");
    IElementWiseLayer* conv126 = convBnSilu(network, weightMap, *conv125->getOutput(0), 128, 3, 1, 1, "model.126");
    IElementWiseLayer* conv127 = convBnSilu(network, weightMap, *conv126->getOutput(0), 128, 3, 1, 1, "model.127");

    ITensor* input_tensor_128[] = { conv127->getOutput(0), conv126->getOutput(0),conv125->getOutput(0), conv124->getOutput(0),
       conv123->getOutput(0), conv122->getOutput(0), conv121->getOutput(0), conv120->getOutput(0), conv119->getOutput(0),
       conv118->getOutput(0) };
    IConcatenationLayer* concat128 = network->addConcatenation(input_tensor_128, 10);
    IElementWiseLayer* conv129 = convBnSilu(network, weightMap, *concat128->getOutput(0), 384, 1, 1, 0, "model.129");

    auto conv130 = DownC(network, weightMap, *conv129->getOutput(0), 384, 576, "model.130");
    ITensor* input_tensor_131[] = { conv130->getOutput(0), conv83->getOutput(0) };
    IConcatenationLayer* concat131 = network->addConcatenation(input_tensor_131, 2);

    IElementWiseLayer* conv132 = convBnSilu(network, weightMap, *concat131->getOutput(0), 384, 1, 1, 0, "model.132");
    IElementWiseLayer* conv133 = convBnSilu(network, weightMap, *concat131->getOutput(0), 384, 1, 1, 0, "model.133");

    IElementWiseLayer* conv134 = convBnSilu(network, weightMap, *conv133->getOutput(0), 192, 3, 1, 1, "model.134");
    IElementWiseLayer* conv135 = convBnSilu(network, weightMap, *conv134->getOutput(0), 192, 3, 1, 1, "model.135");
    IElementWiseLayer* conv136 = convBnSilu(network, weightMap, *conv135->getOutput(0), 192, 3, 1, 1, "model.136");
    IElementWiseLayer* conv137 = convBnSilu(network, weightMap, *conv136->getOutput(0), 192, 3, 1, 1, "model.137");
    IElementWiseLayer* conv138 = convBnSilu(network, weightMap, *conv137->getOutput(0), 192, 3, 1, 1, "model.138");
    IElementWiseLayer* conv139 = convBnSilu(network, weightMap, *conv138->getOutput(0), 192, 3, 1, 1, "model.139");
    IElementWiseLayer* conv140 = convBnSilu(network, weightMap, *conv139->getOutput(0), 192, 3, 1, 1, "model.140");
    IElementWiseLayer* conv141 = convBnSilu(network, weightMap, *conv140->getOutput(0), 192, 3, 1, 1, "model.141");
    ITensor* input_tensor_142[] = { conv141->getOutput(0), conv140->getOutput(0),conv139->getOutput(0), conv138->getOutput(0),
      conv137->getOutput(0), conv136->getOutput(0), conv135->getOutput(0), conv134->getOutput(0), conv133->getOutput(0), 
        conv132->getOutput(0) };
    IConcatenationLayer* concat142 = network->addConcatenation(input_tensor_142, 10);
    IElementWiseLayer* conv143 = convBnSilu(network, weightMap, *concat142->getOutput(0), 576, 1, 1, 0, "model.143");

    auto conv144 = DownC(network, weightMap, *conv143->getOutput(0), 576, 768, "model.144");
    ITensor* input_tensor_145[] = { conv144->getOutput(0), conv67->getOutput(0) };
    IConcatenationLayer* concat145 = network->addConcatenation(input_tensor_145, 2);

    IElementWiseLayer* conv146 = convBnSilu(network, weightMap, *concat145->getOutput(0), 512, 1, 1, 0, "model.146");
    IElementWiseLayer* conv147 = convBnSilu(network, weightMap, *concat145->getOutput(0), 512, 1, 1, 0, "model.147");

    IElementWiseLayer* conv148 = convBnSilu(network, weightMap, *conv147->getOutput(0), 256, 3, 1, 1, "model.148");
    IElementWiseLayer* conv149 = convBnSilu(network, weightMap, *conv148->getOutput(0), 256, 3, 1, 1, "model.149");
    IElementWiseLayer* conv150 = convBnSilu(network, weightMap, *conv149->getOutput(0), 256, 3, 1, 1, "model.150");
    IElementWiseLayer* conv151 = convBnSilu(network, weightMap, *conv150->getOutput(0), 256, 3, 1, 1, "model.151");
    IElementWiseLayer* conv152 = convBnSilu(network, weightMap, *conv151->getOutput(0), 256, 3, 1, 1, "model.152");
    IElementWiseLayer* conv153 = convBnSilu(network, weightMap, *conv152->getOutput(0), 256, 3, 1, 1, "model.153");
    IElementWiseLayer* conv154 = convBnSilu(network, weightMap, *conv153->getOutput(0), 256, 3, 1, 1, "model.154");
    IElementWiseLayer* conv155 = convBnSilu(network, weightMap, *conv154->getOutput(0), 256, 3, 1, 1, "model.155");
    ITensor* input_tensor_156[] = { conv155->getOutput(0), conv154->getOutput(0),conv153->getOutput(0), conv152->getOutput(0),
     conv151->getOutput(0), conv150->getOutput(0), conv149->getOutput(0), conv148->getOutput(0),conv147->getOutput(0),
        conv146->getOutput(0) };
    IConcatenationLayer* concat156 = network->addConcatenation(input_tensor_156, 10);
    IElementWiseLayer* conv157 = convBnSilu(network, weightMap, *concat156->getOutput(0), 768, 1, 1, 0, "model.157");

    IElementWiseLayer* conv158= convBnSilu(network, weightMap, *conv115->getOutput(0), 384, 3, 1, 1, "model.158");
    IElementWiseLayer* conv159 = convBnSilu(network, weightMap, *conv129->getOutput(0), 768, 3, 1, 1, "model.159");
    IElementWiseLayer* conv160 = convBnSilu(network, weightMap, *conv143->getOutput(0), 1152, 3, 1, 1, "model.160");
    IElementWiseLayer* conv161 = convBnSilu(network, weightMap, *conv157->getOutput(0), 1536, 3, 1, 1, "model.161");



    // out
    IConvolutionLayer* cv105_0 = network->addConvolutionNd(*conv158->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.162.m.0.weight"], weightMap["model.162.m.0.bias"]);
    assert(cv105_0);
    cv105_0->setName("cv105.0");
    IConvolutionLayer* cv105_1 = network->addConvolutionNd(*conv159->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.162.m.1.weight"], weightMap["model.162.m.1.bias"]);
    assert(cv105_1);
    cv105_1->setName("cv105.1");
    IConvolutionLayer* cv105_2 = network->addConvolutionNd(*conv160->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.162.m.2.weight"], weightMap["model.162.m.2.bias"]);
    assert(cv105_2);
    cv105_2->setName("cv105.2");
    IConvolutionLayer* cv105_3 = network->addConvolutionNd(*conv161->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.162.m.3.weight"], weightMap["model.162.m.3.bias"]);
    assert(cv105_3);
    cv105_3->setName("cv105.3");




    /*------------detect-----------*/
    auto yolo = addYoLoLayer(network, weightMap, "model.162", std::vector<IConvolutionLayer*>{cv105_0, cv105_1, cv105_2, cv105_3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;


    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}


ICudaEngine* build_engine_yolov7e6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path)
{
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    /*----------------------------------yolov7e6 backbone-----------------------------------------*/
    auto* conv0 = ReOrg(network, weightMap, *data, 3);


    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 80, 3, 1, 1, "model.1");
    auto conv2 = DownC(network, weightMap, *conv1->getOutput(0), 80, 160, "model.2");

    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.3");
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.4");

    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv4->getOutput(0), 64, 3, 1, 1, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");
    IElementWiseLayer* conv9 = convBnSilu(network, weightMap, *conv8->getOutput(0), 64, 3, 1, 1, "model.9");
    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *conv9->getOutput(0), 64, 3, 1, 1, "model.10");

    ITensor* input_tensor_11[] = { conv10->getOutput(0), conv8->getOutput(0),conv6->getOutput(0), conv4->getOutput(0),conv3->getOutput(0) };
    IConcatenationLayer* concat11 = network->addConcatenation(input_tensor_11, 5);

    IElementWiseLayer* conv12 = convBnSilu(network, weightMap, *concat11->getOutput(0), 160, 1, 1, 0, "model.12");


    auto conv13 = DownC(network, weightMap, *conv12->getOutput(0), 160, 320, "model.13");
    IElementWiseLayer* conv14 = convBnSilu(network, weightMap, *conv13->getOutput(0), 128, 1, 1, 0, "model.14");
    IElementWiseLayer* conv15 = convBnSilu(network, weightMap, *conv13->getOutput(0), 128, 1, 1, 0, "model.15");
    
    IElementWiseLayer* conv16 = convBnSilu(network, weightMap, *conv15->getOutput(0), 128, 3, 1, 1, "model.16");
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *conv16->getOutput(0), 128, 3, 1, 1, "model.17");
    IElementWiseLayer* conv18 = convBnSilu(network, weightMap, *conv17->getOutput(0), 128, 3, 1, 1, "model.18");
    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *conv18->getOutput(0), 128, 3, 1, 1, "model.19");
    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *conv19->getOutput(0), 128, 3, 1, 1, "model.20");
    IElementWiseLayer* conv21 = convBnSilu(network, weightMap, *conv20->getOutput(0), 128, 3, 1, 1, "model.21");
    ITensor* input_tensor_22[] = { conv21->getOutput(0), conv19->getOutput(0),conv17->getOutput(0), conv15->getOutput(0),conv14->getOutput(0) };
    IConcatenationLayer* concat22 = network->addConcatenation(input_tensor_22, 5);

    IElementWiseLayer* conv23 = convBnSilu(network, weightMap, *concat22->getOutput(0), 320, 1, 1, 0, "model.23");


    auto conv24 = DownC(network, weightMap, *conv23->getOutput(0), 320, 640, "model.24");
    IElementWiseLayer* conv25 = convBnSilu(network, weightMap, *conv24->getOutput(0), 256, 1, 1, 0, "model.25");
    IElementWiseLayer* conv26 = convBnSilu(network, weightMap, *conv24->getOutput(0), 256, 1, 1, 0, "model.26");

    IElementWiseLayer* conv27 = convBnSilu(network, weightMap, *conv26->getOutput(0), 256, 3, 1, 1, "model.27");
    IElementWiseLayer* conv28 = convBnSilu(network, weightMap, *conv27->getOutput(0), 256, 3, 1, 1, "model.28");
    IElementWiseLayer* conv29 = convBnSilu(network, weightMap, *conv28->getOutput(0), 256, 3, 1, 1, "model.29");
    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *conv29->getOutput(0), 256, 3, 1, 1, "model.30");
    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *conv30->getOutput(0), 256, 3, 1, 1, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 256, 3, 1, 1, "model.32");
    ITensor* input_tensor_33[] = { conv32->getOutput(0), conv30->getOutput(0),conv28->getOutput(0), conv26->getOutput(0),conv25->getOutput(0) };
    IConcatenationLayer* concat33 = network->addConcatenation(input_tensor_33, 5);

    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *concat33->getOutput(0), 640, 1, 1, 0, "model.34");
    auto conv35 = DownC(network, weightMap, *conv34->getOutput(0), 640, 960, "model.35");
    IElementWiseLayer* conv36 = convBnSilu(network, weightMap, *conv35->getOutput(0), 384, 1, 1, 0, "model.36");
    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *conv35->getOutput(0), 384, 1, 1, 0, "model.37");

    IElementWiseLayer* conv38 = convBnSilu(network, weightMap, *conv37->getOutput(0), 384, 3, 1, 1, "model.38");
    IElementWiseLayer* conv39 = convBnSilu(network, weightMap, *conv38->getOutput(0), 384, 3, 1, 1, "model.39");
    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *conv39->getOutput(0), 384, 3, 1, 1, "model.40");
    IElementWiseLayer* conv41 = convBnSilu(network, weightMap, *conv40->getOutput(0), 384, 3, 1, 1, "model.41");
    IElementWiseLayer* conv42 = convBnSilu(network, weightMap, *conv41->getOutput(0), 384, 3, 1, 1, "model.42");
    IElementWiseLayer* conv43 = convBnSilu(network, weightMap, *conv42->getOutput(0), 384, 3, 1, 1, "model.43");

    ITensor* input_tensor_44[] = { conv43->getOutput(0), conv41->getOutput(0),conv39->getOutput(0), conv37->getOutput(0),conv36->getOutput(0) };
    IConcatenationLayer* concat44 = network->addConcatenation(input_tensor_44, 5);
    IElementWiseLayer* conv45 = convBnSilu(network, weightMap, *concat44->getOutput(0), 960, 1, 1, 0, "model.45");

    auto conv46 = DownC(network, weightMap, *conv45->getOutput(0), 960, 1280, "model.46");
    IElementWiseLayer* conv47 = convBnSilu(network, weightMap, *conv46->getOutput(0), 512, 1, 1, 0, "model.47");
    IElementWiseLayer* conv48 = convBnSilu(network, weightMap, *conv46->getOutput(0), 512, 1, 1, 0, "model.48");

    IElementWiseLayer* conv49 = convBnSilu(network, weightMap, *conv48->getOutput(0), 512, 3, 1, 1, "model.49");
    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *conv49->getOutput(0), 512, 3, 1, 1, "model.50");
    IElementWiseLayer* conv51 = convBnSilu(network, weightMap, *conv50->getOutput(0), 512, 3, 1, 1, "model.51");
    IElementWiseLayer* conv52 = convBnSilu(network, weightMap, *conv51->getOutput(0), 512, 3, 1, 1, "model.52");
    IElementWiseLayer* conv53 = convBnSilu(network, weightMap, *conv52->getOutput(0), 512, 3, 1, 1, "model.53");
    IElementWiseLayer* conv54 = convBnSilu(network, weightMap, *conv53->getOutput(0), 512, 3, 1, 1, "model.54");
    ITensor* input_tensor_55[] = { conv54->getOutput(0), conv52->getOutput(0),conv50->getOutput(0), conv48->getOutput(0),conv47->getOutput(0) };
    IConcatenationLayer* concat55 = network->addConcatenation(input_tensor_55, 5);
    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *concat55->getOutput(0), 1280, 1, 1, 0, "model.56");

    //------------------------yolov7e6 head-------------------------------
    auto conv57 = SPPCSPC(network, weightMap, *conv56->getOutput(0), 640, "model.57");
    IElementWiseLayer* conv58 = convBnSilu(network, weightMap, *conv57->getOutput(0), 480, 1, 1, 0, "model.58");


    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* re59 = network->addResize(*conv58->getOutput(0));
    re59->setResizeMode(ResizeMode::kNEAREST);
    re59->setScales(scale, 3);

    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv45->getOutput(0), 480, 1, 1, 0, "model.60");
    ITensor* input_tensor_61[] = { conv60->getOutput(0), re59->getOutput(0) };
    IConcatenationLayer* concat61 = network->addConcatenation(input_tensor_61, 2);
    IElementWiseLayer* conv62 = convBnSilu(network, weightMap, *concat61->getOutput(0), 384, 1, 1, 0, "model.62");
    IElementWiseLayer* conv63 = convBnSilu(network, weightMap, *concat61->getOutput(0), 384, 1, 1, 0, "model.63");

    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *conv63->getOutput(0), 192, 3, 1, 1, "model.64");
    IElementWiseLayer* conv65 = convBnSilu(network, weightMap, *conv64->getOutput(0), 192, 3, 1, 1, "model.65");
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *conv65->getOutput(0), 192, 3, 1, 1, "model.66");
    IElementWiseLayer* conv67 = convBnSilu(network, weightMap, *conv66->getOutput(0), 192, 3, 1, 1, "model.67");
    IElementWiseLayer* conv68 = convBnSilu(network, weightMap, *conv67->getOutput(0), 192, 3, 1, 1, "model.68");
    IElementWiseLayer* conv69 = convBnSilu(network, weightMap, *conv68->getOutput(0), 192, 3, 1, 1, "model.69");

    ITensor* input_tensor_70[] = { conv69->getOutput(0), conv68->getOutput(0),conv67->getOutput(0), conv66->getOutput(0),
        conv65->getOutput(0), conv64->getOutput(0), conv63->getOutput(0), conv62->getOutput(0) };
    IConcatenationLayer* concat70 = network->addConcatenation(input_tensor_70, 8);
    IElementWiseLayer* conv71 = convBnSilu(network, weightMap, *concat70->getOutput(0), 480, 1, 1, 0, "model.71");

    IElementWiseLayer* conv72 = convBnSilu(network, weightMap, *conv71->getOutput(0), 320, 1, 1, 0, "model.72");
    IResizeLayer* re73 = network->addResize(*conv72->getOutput(0));
    re73->setResizeMode(ResizeMode::kNEAREST);
    re73->setScales(scale, 3);
    IElementWiseLayer* conv74 = convBnSilu(network, weightMap, *conv34->getOutput(0), 320, 1, 1, 0, "model.74");
    ITensor* input_tensor_75[] = { conv74->getOutput(0), re73->getOutput(0) };
    IConcatenationLayer* concat75 = network->addConcatenation(input_tensor_75, 2);

    IElementWiseLayer* conv76 = convBnSilu(network, weightMap, *concat75->getOutput(0), 256, 1, 1, 0, "model.76");
    IElementWiseLayer* conv77 = convBnSilu(network, weightMap, *concat75->getOutput(0), 256, 1, 1, 0, "model.77");

    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *conv77->getOutput(0), 128, 3, 1, 1, "model.78");
    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *conv78->getOutput(0), 128, 3, 1, 1, "model.79");
    IElementWiseLayer* conv80 = convBnSilu(network, weightMap, *conv79->getOutput(0), 128, 3, 1, 1, "model.80");
    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *conv80->getOutput(0), 128, 3, 1, 1, "model.81");
    IElementWiseLayer* conv82 = convBnSilu(network, weightMap, *conv81->getOutput(0), 128, 3, 1, 1, "model.82");
    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *conv82->getOutput(0), 128, 3, 1, 1, "model.83");

    ITensor* input_tensor_84[] = { conv83->getOutput(0), conv82->getOutput(0),conv81->getOutput(0), conv80->getOutput(0),
        conv79->getOutput(0), conv78->getOutput(0), conv77->getOutput(0), conv76->getOutput(0) };
    IConcatenationLayer* concat84 = network->addConcatenation(input_tensor_84, 8);

    IElementWiseLayer* conv85 = convBnSilu(network, weightMap, *concat84->getOutput(0), 320, 1, 1, 0, "model.85");

    IElementWiseLayer* conv86 = convBnSilu(network, weightMap, *conv85->getOutput(0), 160, 1, 1, 0, "model.86");
    IResizeLayer* re87 = network->addResize(*conv86->getOutput(0));
    re87->setResizeMode(ResizeMode::kNEAREST);
    re87->setScales(scale, 3);
    IElementWiseLayer* conv88 = convBnSilu(network, weightMap, *conv23->getOutput(0), 160, 1, 1, 0, "model.88");
    ITensor* input_tensor_89[] = { conv88->getOutput(0), re87->getOutput(0) };
    IConcatenationLayer* concat89 = network->addConcatenation(input_tensor_89, 2);

    IElementWiseLayer* conv90 = convBnSilu(network, weightMap, *concat89->getOutput(0), 128, 1, 1, 0, "model.90");
    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *concat89->getOutput(0), 128, 1, 1, 0, "model.91");
    IElementWiseLayer* conv92 = convBnSilu(network, weightMap, *conv91->getOutput(0), 64, 3, 1, 1, "model.92");
    IElementWiseLayer* conv93 = convBnSilu(network, weightMap, *conv92->getOutput(0), 64, 3, 1, 1, "model.93");
    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *conv93->getOutput(0), 64, 3, 1, 1, "model.94");
    IElementWiseLayer* conv95 = convBnSilu(network, weightMap, *conv94->getOutput(0), 64, 3, 1, 1, "model.95");
    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *conv95->getOutput(0), 64, 3, 1, 1, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *conv96->getOutput(0), 64, 3, 1, 1, "model.97");

    ITensor* input_tensor_98[] = { conv97->getOutput(0), conv96->getOutput(0),conv95->getOutput(0), conv94->getOutput(0),
       conv93->getOutput(0), conv92->getOutput(0), conv91->getOutput(0), conv90->getOutput(0) };
    IConcatenationLayer* concat98 = network->addConcatenation(input_tensor_98, 8);

    IElementWiseLayer* conv99 = convBnSilu(network, weightMap, *concat98->getOutput(0), 160, 1, 1, 0, "model.99");

    auto conv100 = DownC(network, weightMap, *conv99->getOutput(0), 160, 320, "model.100");
    ITensor* input_tensor_101[] = { conv100->getOutput(0), conv85->getOutput(0) };
    IConcatenationLayer* concat101 = network->addConcatenation(input_tensor_101, 2);

    IElementWiseLayer* conv102 = convBnSilu(network, weightMap, *concat101->getOutput(0), 256, 1, 1, 0, "model.102");
    IElementWiseLayer* conv103 = convBnSilu(network, weightMap, *concat101->getOutput(0), 256, 1, 1, 0, "model.103");

    IElementWiseLayer* conv104 = convBnSilu(network, weightMap, *conv103->getOutput(0), 128, 3, 1, 1, "model.104");
    IElementWiseLayer* conv105 = convBnSilu(network, weightMap, *conv104->getOutput(0), 128, 3, 1, 1, "model.105");
    IElementWiseLayer* conv106 = convBnSilu(network, weightMap, *conv105->getOutput(0), 128, 3, 1, 1, "model.106");
    IElementWiseLayer* conv107 = convBnSilu(network, weightMap, *conv106->getOutput(0), 128, 3, 1, 1, "model.107");
    IElementWiseLayer* conv108 = convBnSilu(network, weightMap, *conv107->getOutput(0), 128, 3, 1, 1, "model.108");
    IElementWiseLayer* conv109 = convBnSilu(network, weightMap, *conv108->getOutput(0), 128, 3, 1, 1, "model.109");

    ITensor* input_tensor_110[] = { conv109->getOutput(0), conv108->getOutput(0),conv107->getOutput(0), conv106->getOutput(0),
       conv105->getOutput(0), conv104->getOutput(0), conv103->getOutput(0), conv102->getOutput(0) };
    IConcatenationLayer* concat110 = network->addConcatenation(input_tensor_110, 8);
    IElementWiseLayer* conv111 = convBnSilu(network, weightMap, *concat110->getOutput(0), 320, 1, 1, 0, "model.111");

    auto conv112 = DownC(network, weightMap, *conv111->getOutput(0), 320, 480, "model.112");
    ITensor* input_tensor_113[] = { conv112->getOutput(0), conv71->getOutput(0) };
    IConcatenationLayer* concat113 = network->addConcatenation(input_tensor_113, 2);

    IElementWiseLayer* conv114 = convBnSilu(network, weightMap, *concat113->getOutput(0), 384, 1, 1, 0, "model.114");
    IElementWiseLayer* conv115 = convBnSilu(network, weightMap, *concat113->getOutput(0), 384, 1, 1, 0, "model.115");

    IElementWiseLayer* conv116 = convBnSilu(network, weightMap, *conv115->getOutput(0), 192, 3, 1, 1, "model.116");
    IElementWiseLayer* conv117 = convBnSilu(network, weightMap, *conv116->getOutput(0), 192, 3, 1, 1, "model.117");
    IElementWiseLayer* conv118 = convBnSilu(network, weightMap, *conv117->getOutput(0), 192, 3, 1, 1, "model.118");
    IElementWiseLayer* conv119 = convBnSilu(network, weightMap, *conv118->getOutput(0), 192, 3, 1, 1, "model.119");
    IElementWiseLayer* conv120 = convBnSilu(network, weightMap, *conv119->getOutput(0), 192, 3, 1, 1, "model.120");
    IElementWiseLayer* conv121 = convBnSilu(network, weightMap, *conv120->getOutput(0), 192, 3, 1, 1, "model.121");
    ITensor* input_tensor_122[] = { conv121->getOutput(0), conv120->getOutput(0),conv119->getOutput(0), conv118->getOutput(0),
      conv117->getOutput(0), conv116->getOutput(0), conv115->getOutput(0), conv114->getOutput(0) };
    IConcatenationLayer* concat122 = network->addConcatenation(input_tensor_122, 8);
    IElementWiseLayer* conv123 = convBnSilu(network, weightMap, *concat122->getOutput(0), 480, 1, 1, 0, "model.123");

    auto conv124 = DownC(network, weightMap, *conv123->getOutput(0), 480, 640, "model.124");
    ITensor* input_tensor_125[] = { conv124->getOutput(0), conv57->getOutput(0) };
    IConcatenationLayer* concat125 = network->addConcatenation(input_tensor_125, 2);

    IElementWiseLayer* conv126 = convBnSilu(network, weightMap, *concat125->getOutput(0), 512, 1, 1, 0, "model.126");
    IElementWiseLayer* conv127 = convBnSilu(network, weightMap, *concat125->getOutput(0), 512, 1, 1, 0, "model.127");

    IElementWiseLayer* conv128 = convBnSilu(network, weightMap, *conv127->getOutput(0), 256, 3, 1, 1, "model.128");
    IElementWiseLayer* conv129 = convBnSilu(network, weightMap, *conv128->getOutput(0), 256, 3, 1, 1, "model.129");
    IElementWiseLayer* conv130 = convBnSilu(network, weightMap, *conv129->getOutput(0), 256, 3, 1, 1, "model.130");
    IElementWiseLayer* conv131 = convBnSilu(network, weightMap, *conv130->getOutput(0), 256, 3, 1, 1, "model.131");
    IElementWiseLayer* conv132 = convBnSilu(network, weightMap, *conv131->getOutput(0), 256, 3, 1, 1, "model.132");
    IElementWiseLayer* conv133 = convBnSilu(network, weightMap, *conv132->getOutput(0), 256, 3, 1, 1, "model.133");
    ITensor* input_tensor_134[] = { conv133->getOutput(0), conv132->getOutput(0),conv131->getOutput(0), conv130->getOutput(0),
     conv129->getOutput(0), conv128->getOutput(0), conv127->getOutput(0), conv126->getOutput(0) };
    IConcatenationLayer* concat134 = network->addConcatenation(input_tensor_134, 8);
    IElementWiseLayer* conv135 = convBnSilu(network, weightMap, *concat134->getOutput(0), 640, 1, 1, 0, "model.135");

    IElementWiseLayer* conv136 = convBnSilu(network, weightMap, *conv99->getOutput(0), 320, 3, 1, 1, "model.136");
    IElementWiseLayer* conv137 = convBnSilu(network, weightMap, *conv111->getOutput(0), 640, 3, 1, 1, "model.137");
    IElementWiseLayer* conv138 = convBnSilu(network, weightMap, *conv123->getOutput(0), 960, 3, 1, 1, "model.138");
    IElementWiseLayer* conv139 = convBnSilu(network, weightMap, *conv135->getOutput(0), 1280, 3, 1, 1, "model.139");



     // out
    IConvolutionLayer* cv105_0 = network->addConvolutionNd(*conv136->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.140.m.0.weight"], weightMap["model.140.m.0.bias"]);
    assert(cv105_0);
    cv105_0->setName("cv105.0");
    IConvolutionLayer* cv105_1 = network->addConvolutionNd(*conv137->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.140.m.1.weight"], weightMap["model.140.m.1.bias"]);
    assert(cv105_1);
    cv105_1->setName("cv105.1");
    IConvolutionLayer* cv105_2 = network->addConvolutionNd(*conv138->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.140.m.2.weight"], weightMap["model.140.m.2.bias"]);
    assert(cv105_2);
    cv105_2->setName("cv105.2");
    IConvolutionLayer* cv105_3 = network->addConvolutionNd(*conv139->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.140.m.3.weight"], weightMap["model.140.m.3.bias"]);
    assert(cv105_3);
    cv105_3->setName("cv105.3");




    /*------------detect-----------*/
    auto yolo = addYoLoLayer(network, weightMap, "model.140", std::vector<IConvolutionLayer*>{cv105_0, cv105_1, cv105_2, cv105_3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;


    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}




ICudaEngine* build_engine_yolov7w6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path)
{
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    /*----------------------------------yolov7w6 backbone-----------------------------------------*/
    auto* conv0 = ReOrg(network, weightMap, *data, 3);


    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 64, 3, 1, 1, "model.1");

    IElementWiseLayer* conv2 = convBnSilu(network, weightMap, *conv1->getOutput(0), 128, 3, 2, 1, "model.2");

    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.3");
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv2->getOutput(0), 64, 1, 1, 0, "model.4");

    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv4->getOutput(0), 64, 3, 1, 1, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");



    ITensor* input_tensor_9[] = { conv8->getOutput(0), conv6->getOutput(0), conv4->getOutput(0), conv3->getOutput(0) };
    IConcatenationLayer* concat9 = network->addConcatenation(input_tensor_9, 4);
    concat9->setAxis(0);
    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *concat9->getOutput(0), 128, 1, 1, 0, "model.10");

    IElementWiseLayer* conv11 = convBnSilu(network, weightMap, *conv10->getOutput(0), 256, 3, 2, 1, "model.11");

    IElementWiseLayer* conv12 = convBnSilu(network, weightMap, *conv11->getOutput(0), 128, 1, 1, 0, "model.12");
    IElementWiseLayer* conv13 = convBnSilu(network, weightMap, *conv11->getOutput(0), 128, 1, 1, 0, "model.13");
    IElementWiseLayer* conv14 = convBnSilu(network, weightMap, *conv13->getOutput(0), 128, 3, 1, 1, "model.14");
    IElementWiseLayer* conv15 = convBnSilu(network, weightMap, *conv14->getOutput(0), 128, 3, 1, 1, "model.15");
    IElementWiseLayer* conv16 = convBnSilu(network, weightMap, *conv15->getOutput(0), 128, 3, 1, 1, "model.16");
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *conv16->getOutput(0), 128, 3, 1, 1, "model.17");
    ITensor* input_tensor_18[] = { conv17->getOutput(0), conv15->getOutput(0), conv13->getOutput(0), conv12->getOutput(0) };
    IConcatenationLayer* concat18 = network->addConcatenation(input_tensor_18, 4);
    concat18->setAxis(0);
    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *concat18->getOutput(0), 256, 1, 1, 0, "model.19");

    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *conv19->getOutput(0), 512, 3, 2, 1, "model.20");

    IElementWiseLayer* conv21 = convBnSilu(network, weightMap, *conv20->getOutput(0), 256, 1, 1, 0, "model.21");
    IElementWiseLayer* conv22 = convBnSilu(network, weightMap, *conv20->getOutput(0), 256, 1, 1, 0, "model.22");
    IElementWiseLayer* conv23 = convBnSilu(network, weightMap, *conv22->getOutput(0), 256, 3, 1, 1, "model.23");
    IElementWiseLayer* conv24 = convBnSilu(network, weightMap, *conv23->getOutput(0), 256, 3, 1, 1, "model.24");
    IElementWiseLayer* conv25 = convBnSilu(network, weightMap, *conv24->getOutput(0), 256, 3, 1, 1, "model.25");
    IElementWiseLayer* conv26 = convBnSilu(network, weightMap, *conv25->getOutput(0), 256, 3, 1, 1, "model.26");
    ITensor* input_tensor_27[] = { conv26->getOutput(0), conv24->getOutput(0), conv22->getOutput(0), conv21->getOutput(0) };
    IConcatenationLayer* concat27 = network->addConcatenation(input_tensor_27, 4);
    concat27->setAxis(0);



    IElementWiseLayer* conv28 = convBnSilu(network, weightMap, *concat27->getOutput(0), 512, 1, 1, 0, "model.28");

    IElementWiseLayer* conv29 = convBnSilu(network, weightMap, *conv28->getOutput(0), 768, 3, 2, 1, "model.29");

    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *conv29->getOutput(0), 384, 1, 1, 0, "model.30");
    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *conv29->getOutput(0), 384, 1, 1, 0, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 384, 3, 1, 1, "model.32");
    IElementWiseLayer* conv33 = convBnSilu(network, weightMap, *conv32->getOutput(0), 384, 3, 1, 1, "model.33");
    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *conv33->getOutput(0), 384, 3, 1, 1, "model.34");
    IElementWiseLayer* conv35 = convBnSilu(network, weightMap, *conv34->getOutput(0), 384, 3, 1, 1, "model.35");
    ITensor* input_tensor_36[] = { conv35->getOutput(0), conv33->getOutput(0), conv31->getOutput(0), conv30->getOutput(0) };
    IConcatenationLayer* concat36 = network->addConcatenation(input_tensor_36, 4);
    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *concat36->getOutput(0), 768, 1, 1, 0, "model.37");

    IElementWiseLayer* conv38 = convBnSilu(network, weightMap, *conv37->getOutput(0), 1024, 3, 2, 1, "model.38");

    IElementWiseLayer* conv39 = convBnSilu(network, weightMap, *conv38->getOutput(0), 512, 1, 1, 0, "model.39");
    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *conv38->getOutput(0), 512, 1, 1, 0, "model.40");
    IElementWiseLayer* conv41 = convBnSilu(network, weightMap, *conv40->getOutput(0), 512, 3, 1, 1, "model.41");
    IElementWiseLayer* conv42 = convBnSilu(network, weightMap, *conv41->getOutput(0), 512, 3, 1, 1, "model.42");
    IElementWiseLayer* conv43 = convBnSilu(network, weightMap, *conv42->getOutput(0), 512, 3, 1, 1, "model.43");
    IElementWiseLayer* conv44 = convBnSilu(network, weightMap, *conv43->getOutput(0), 512, 3, 1, 1, "model.44");
    ITensor* input_tensor_45[] = { conv44->getOutput(0), conv42->getOutput(0), conv40->getOutput(0), conv39->getOutput(0) };
    IConcatenationLayer* concat45 = network->addConcatenation(input_tensor_45, 4);
    IElementWiseLayer* conv46 = convBnSilu(network, weightMap, *concat45->getOutput(0), 1024, 1, 1, 0, "model.46");

    //----------------head============================
    auto conv47 = SPPCSPC(network, weightMap, *conv46->getOutput(0), 512, "model.47");
    IElementWiseLayer* conv48 = convBnSilu(network, weightMap, *conv47->getOutput(0), 384, 1, 1, 0, "model.48");


    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* re49 = network->addResize(*conv48->getOutput(0));
    re49->setResizeMode(ResizeMode::kNEAREST);
    re49->setScales(scale, 3);

    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *conv37->getOutput(0), 384, 1, 1, 0, "model.50");
    ITensor* input_tensor_51[] = { conv50->getOutput(0), re49->getOutput(0) };
    IConcatenationLayer* concat51 = network->addConcatenation(input_tensor_51, 2);

    IElementWiseLayer* conv52 = convBnSilu(network, weightMap, *concat51->getOutput(0), 384, 1, 1, 0, "model.52");
    IElementWiseLayer* conv53 = convBnSilu(network, weightMap, *concat51->getOutput(0), 384, 1, 1, 0, "model.53");
    IElementWiseLayer* conv54 = convBnSilu(network, weightMap, *conv53->getOutput(0), 192, 3, 1, 1, "model.54");
    IElementWiseLayer* conv55 = convBnSilu(network, weightMap, *conv54->getOutput(0), 192, 3, 1, 1, "model.55");
    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *conv55->getOutput(0), 192, 3, 1, 1, "model.56");
    IElementWiseLayer* conv57 = convBnSilu(network, weightMap, *conv56->getOutput(0), 192, 3, 1, 1, "model.57");


    ITensor* input_tensor_58[] = { conv57->getOutput(0), conv56->getOutput(0), conv55->getOutput(0), conv54->getOutput(0), conv53->getOutput(0), conv52->getOutput(0) };
    IConcatenationLayer* concat58 = network->addConcatenation(input_tensor_58, 6);

    IElementWiseLayer* conv59 = convBnSilu(network, weightMap, *concat58->getOutput(0), 384, 1, 1, 0, "model.59");

    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv59->getOutput(0), 256, 1, 1, 0, "model.60");
    IResizeLayer* re61 = network->addResize(*conv60->getOutput(0));
    re61->setResizeMode(ResizeMode::kNEAREST);
    re61->setScales(scale, 3);
    IElementWiseLayer* conv62 = convBnSilu(network, weightMap, *conv28->getOutput(0), 256, 1, 1, 0, "model.62");
    ITensor* input_tensor_63[] = { conv62->getOutput(0), re61->getOutput(0) };
    IConcatenationLayer* concat63 = network->addConcatenation(input_tensor_63, 2);


    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *concat63->getOutput(0), 256, 1, 1, 0, "model.64");
    IElementWiseLayer* conv65 = convBnSilu(network, weightMap, *concat63->getOutput(0), 256, 1, 1, 0, "model.65");
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *conv65->getOutput(0), 128, 3, 1, 1, "model.66");
    IElementWiseLayer* conv67 = convBnSilu(network, weightMap, *conv66->getOutput(0), 128, 3, 1, 1, "model.67");
    IElementWiseLayer* conv68 = convBnSilu(network, weightMap, *conv67->getOutput(0), 128, 3, 1, 1, "model.68");
    IElementWiseLayer* conv69 = convBnSilu(network, weightMap, *conv68->getOutput(0), 128, 3, 1, 1, "model.69");

    ITensor* input_tensor_70[] = { conv69->getOutput(0), conv68->getOutput(0), conv67->getOutput(0), conv66->getOutput(0), conv65->getOutput(0), conv64->getOutput(0) };
    IConcatenationLayer* concat70 = network->addConcatenation(input_tensor_70, 6);

    IElementWiseLayer* conv71 = convBnSilu(network, weightMap, *concat70->getOutput(0), 256, 1, 1, 0, "model.71");
    IElementWiseLayer* conv72 = convBnSilu(network, weightMap, *conv71->getOutput(0), 128, 1, 1, 0, "model.72");
    IResizeLayer* re73 = network->addResize(*conv72->getOutput(0));
    re73->setResizeMode(ResizeMode::kNEAREST);
    re73->setScales(scale, 3);

    IElementWiseLayer* conv74 = convBnSilu(network, weightMap, *conv19->getOutput(0), 128, 1, 1, 0, "model.74");
    ITensor* input_tensor_75[] = { conv74->getOutput(0), re73->getOutput(0) };
    IConcatenationLayer* concat75 = network->addConcatenation(input_tensor_75, 2);
    IElementWiseLayer* conv76 = convBnSilu(network, weightMap, *concat75->getOutput(0), 128, 1, 1, 0, "model.76");
    IElementWiseLayer* conv77 = convBnSilu(network, weightMap, *concat75->getOutput(0), 128, 1, 1, 0, "model.77");

    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *conv77->getOutput(0), 64, 3, 1, 1, "model.78");
    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *conv78->getOutput(0), 64, 3, 1, 1, "model.79");
    IElementWiseLayer* conv80 = convBnSilu(network, weightMap, *conv79->getOutput(0), 64, 3, 1, 1, "model.80");
    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *conv80->getOutput(0), 64, 3, 1, 1, "model.81");
    ITensor* input_tensor_82[] = { conv81->getOutput(0), conv80->getOutput(0), conv79->getOutput(0), conv78->getOutput(0), conv77->getOutput(0), conv76->getOutput(0) };
    IConcatenationLayer* concat82 = network->addConcatenation(input_tensor_82, 6);

    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *concat82->getOutput(0), 128, 1, 1, 0, "model.83");

    IElementWiseLayer* conv84 = convBnSilu(network, weightMap, *conv83->getOutput(0), 256, 3, 2, 1, "model.84");
    ITensor* input_tensor_85[] = { conv84->getOutput(0), conv71->getOutput(0) };
    IConcatenationLayer* concat85 = network->addConcatenation(input_tensor_85, 2);

    IElementWiseLayer* conv86 = convBnSilu(network, weightMap, *concat85->getOutput(0), 256, 1, 1, 0, "model.86");
    IElementWiseLayer* conv87 = convBnSilu(network, weightMap, *concat85->getOutput(0), 256, 1, 1, 0, "model.87");
    IElementWiseLayer* conv88 = convBnSilu(network, weightMap, *conv87->getOutput(0), 128, 3, 1, 1, "model.88");
    IElementWiseLayer* conv89 = convBnSilu(network, weightMap, *conv88->getOutput(0), 128, 3, 1, 1, "model.89");
    IElementWiseLayer* conv90 = convBnSilu(network, weightMap, *conv89->getOutput(0), 128, 3, 1, 1, "model.90");
    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *conv90->getOutput(0), 128, 3, 1, 1, "model.91");

    ITensor* input_tensor_92[] = { conv91->getOutput(0), conv90->getOutput(0), conv89->getOutput(0), conv88->getOutput(0), conv87->getOutput(0), conv86->getOutput(0) };
    IConcatenationLayer* concat92 = network->addConcatenation(input_tensor_92, 6);

    IElementWiseLayer* conv93 = convBnSilu(network, weightMap, *concat92->getOutput(0), 256, 1, 1, 0, "model.93");

    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *conv93->getOutput(0), 384, 3, 2, 1, "model.94");
    ITensor* input_tensor_95[] = { conv94->getOutput(0), conv59->getOutput(0) };
    IConcatenationLayer* concat95 = network->addConcatenation(input_tensor_95, 2);

    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *concat95->getOutput(0), 384, 1, 1, 0, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *concat95->getOutput(0), 384, 1, 1, 0, "model.97");

    IElementWiseLayer* conv98 = convBnSilu(network, weightMap, *conv97->getOutput(0), 192, 3, 1, 1, "model.98");
    IElementWiseLayer* conv99 = convBnSilu(network, weightMap, *conv98->getOutput(0), 192, 3, 1, 1, "model.99");
    IElementWiseLayer* conv100 = convBnSilu(network, weightMap, *conv99->getOutput(0), 192, 3, 1, 1, "model.100");
    IElementWiseLayer* conv101 = convBnSilu(network, weightMap, *conv100->getOutput(0), 192, 3, 1, 1, "model.101");
    ITensor* input_tensor_102[] = { conv101->getOutput(0), conv100->getOutput(0), conv99->getOutput(0), conv98->getOutput(0), conv97->getOutput(0), conv96->getOutput(0) };
    IConcatenationLayer* concat102 = network->addConcatenation(input_tensor_102, 6);
    IElementWiseLayer* conv103 = convBnSilu(network, weightMap, *concat102->getOutput(0), 384, 1, 1, 0, "model.103");

    IElementWiseLayer* conv104 = convBnSilu(network, weightMap, *conv103->getOutput(0), 512, 3, 2, 1, "model.104");


    ITensor* input_tensor_105[] = { conv104->getOutput(0), conv47->getOutput(0) };
    IConcatenationLayer* concat105 = network->addConcatenation(input_tensor_105, 2);

    IElementWiseLayer* conv106 = convBnSilu(network, weightMap, *concat105->getOutput(0), 512, 1, 1, 0, "model.106");
    IElementWiseLayer* conv107 = convBnSilu(network, weightMap, *concat105->getOutput(0), 512, 1, 1, 0, "model.107");

    IElementWiseLayer* conv108 = convBnSilu(network, weightMap, *conv107->getOutput(0), 256, 3, 1, 1, "model.108");
    IElementWiseLayer* conv109 = convBnSilu(network, weightMap, *conv108->getOutput(0), 256, 3, 1, 1, "model.109");
    IElementWiseLayer* conv110 = convBnSilu(network, weightMap, *conv109->getOutput(0), 256, 3, 1, 1, "model.110");
    IElementWiseLayer* conv111 = convBnSilu(network, weightMap, *conv110->getOutput(0), 256, 3, 1, 1, "model.111");
    ITensor* input_tensor_112[] = { conv111->getOutput(0), conv110->getOutput(0), conv109->getOutput(0), conv108->getOutput(0), conv107->getOutput(0), conv106->getOutput(0) };
    IConcatenationLayer* concat112 = network->addConcatenation(input_tensor_112, 6);

    IElementWiseLayer* conv113 = convBnSilu(network, weightMap, *concat112->getOutput(0), 512, 1, 1, 0, "model.113");

    IElementWiseLayer* conv114 = convBnSilu(network, weightMap, *conv83->getOutput(0), 256, 3, 1, 1, "model.114");
    IElementWiseLayer* conv115 = convBnSilu(network, weightMap, *conv93->getOutput(0), 512, 3, 1, 1, "model.115");
    IElementWiseLayer* conv116 = convBnSilu(network, weightMap, *conv103->getOutput(0), 768, 3, 1, 1, "model.116");
    IElementWiseLayer* conv117 = convBnSilu(network, weightMap, *conv113->getOutput(0), 1024, 3, 1, 1, "model.117");



    // out
    IConvolutionLayer* cv105_0 = network->addConvolutionNd(*conv114->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.118.m.0.weight"], weightMap["model.118.m.0.bias"]);
    assert(cv105_0);
    cv105_0->setName("cv105.0");
    IConvolutionLayer* cv105_1 = network->addConvolutionNd(*conv115->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.118.m.1.weight"], weightMap["model.118.m.1.bias"]);
    assert(cv105_1);
    cv105_1->setName("cv105.1");
    IConvolutionLayer* cv105_2 = network->addConvolutionNd(*conv116->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.118.m.2.weight"], weightMap["model.118.m.2.bias"]);
    assert(cv105_2);
    cv105_2->setName("cv105.2");
    IConvolutionLayer* cv105_3 = network->addConvolutionNd(*conv117->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.118.m.3.weight"], weightMap["model.118.m.3.bias"]);
    assert(cv105_3);
    cv105_3->setName("cv105.3");



    /*------------detect-----------*/
    auto yolo = addYoLoLayer(network, weightMap, "model.118", std::vector<IConvolutionLayer*>{cv105_0, cv105_1, cv105_2, cv105_3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif
   
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;


    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}



//----------------------------------yolov7x---------------------------------------------------
ICudaEngine* build_engine_yolov7x(unsigned int maxBatchSize,IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path) {
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    /*----------------------------------yolov7x backbone-----------------------------------------*/
    IElementWiseLayer* conv0 = convBnSilu(network, weightMap, *data, 40, 3, 1, 1, "model.0");

    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 80, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = convBnSilu(network, weightMap, *conv1->getOutput(0), 80, 3, 1, 1, "model.2");
    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 160, 3, 2, 1, "model.3");


    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv3->getOutput(0), 64, 1, 1, 0, "model.4");

    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv3->getOutput(0), 64, 1, 1, 0, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");
    IElementWiseLayer* conv9 = convBnSilu(network, weightMap, *conv8->getOutput(0), 64, 3, 1, 1, "model.9");
    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *conv9->getOutput(0), 64, 3, 1, 1, "model.10");
    IElementWiseLayer* conv11 = convBnSilu(network, weightMap, *conv10->getOutput(0), 64, 3, 1, 1, "model.11");

    ITensor* input_tensor_12[] = { conv11->getOutput(0), conv9->getOutput(0), conv7->getOutput(0), conv5->getOutput(0),conv4->getOutput(0) };
    IConcatenationLayer* concat12 = network->addConcatenation(input_tensor_12, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv13 = convBnSilu(network, weightMap, *concat12->getOutput(0), 320, 1, 1, 0, "model.13");



    IPoolingLayer* mp1 = network->addPoolingNd(*conv13->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp1->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv15 = convBnSilu(network, weightMap, *mp1->getOutput(0), 160, 1, 1, 0, "model.15");

    IElementWiseLayer* conv16 = convBnSilu(network, weightMap, *conv13->getOutput(0), 160, 1, 1, 0, "model.16");
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *conv16->getOutput(0), 160, 3, 2, 1, "model.17");
    ITensor* input_tensor_18[] = { conv17->getOutput(0), conv15->getOutput(0) };
    IConcatenationLayer* concat18 = network->addConcatenation(input_tensor_18, 2);

    //IConcatenationLayer* mp1 = MPC3(network, weightMap, *conv13->getOutput(0), 160, "model.15", "model.16", "model.17");


    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *concat18->getOutput(0), 128, 1, 1, 0, "model.19");

    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *concat18->getOutput(0), 128, 1, 1, 0, "model.20");
    IElementWiseLayer* conv21 = convBnSilu(network, weightMap, *conv20->getOutput(0), 128, 3, 1, 1, "model.21");
    IElementWiseLayer* conv22 = convBnSilu(network, weightMap, *conv21->getOutput(0), 128, 3, 1, 1, "model.22");
    IElementWiseLayer* conv23 = convBnSilu(network, weightMap, *conv22->getOutput(0), 128, 3, 1, 1, "model.23");
    IElementWiseLayer* conv24 = convBnSilu(network, weightMap, *conv23->getOutput(0), 128, 3, 1, 1, "model.24");
    IElementWiseLayer* conv25 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 3, 1, 1, "model.25");
    IElementWiseLayer* conv26 = convBnSilu(network, weightMap, *conv25->getOutput(0), 128, 3, 1, 1, "model.26");

    ITensor* input_tensor_27[] = { conv26->getOutput(0), conv24->getOutput(0), conv22->getOutput(0), conv20->getOutput(0),conv19->getOutput(0) };
    IConcatenationLayer* concat27 = network->addConcatenation(input_tensor_27, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv28 = convBnSilu(network, weightMap, *concat27->getOutput(0), 640, 1, 1, 0, "model.28");


    IPoolingLayer* mp2 = network->addPoolingNd(*conv28->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp1->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *mp2->getOutput(0), 320, 1, 1, 0, "model.30");

    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *conv28->getOutput(0), 320, 1, 1, 0, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 320, 3, 2, 1, "model.32");
    ITensor* input_tensor_33[] = { conv32->getOutput(0), conv30->getOutput(0) };
    IConcatenationLayer* concat33 = network->addConcatenation(input_tensor_33, 2);
    //IConcatenationLayer* mp2 = MPC3(network, weightMap, *conv28->getOutput(0), 320, "model.30", "model.31", "model.32");


    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *concat33->getOutput(0), 256, 1, 1, 0, "model.34");

    IElementWiseLayer* conv35 = convBnSilu(network, weightMap, *concat33->getOutput(0), 256, 1, 1, 0, "model.35");
    IElementWiseLayer* conv36 = convBnSilu(network, weightMap, *conv35->getOutput(0), 256, 3, 1, 1, "model.36");
    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *conv36->getOutput(0), 256, 3, 1, 1, "model.37");
    IElementWiseLayer* conv38 = convBnSilu(network, weightMap, *conv37->getOutput(0), 256, 3, 1, 1, "model.38");
    IElementWiseLayer* conv39 = convBnSilu(network, weightMap, *conv38->getOutput(0), 256, 3, 1, 1, "model.39");
    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *conv39->getOutput(0), 256, 3, 1, 1, "model.40");
    IElementWiseLayer* conv41 = convBnSilu(network, weightMap, *conv40->getOutput(0), 256, 3, 1, 1, "model.41");

    ITensor* input_tensor_42[] = { conv41->getOutput(0), conv39->getOutput(0), conv37->getOutput(0), conv35->getOutput(0),conv34->getOutput(0) };
    IConcatenationLayer* concat42 = network->addConcatenation(input_tensor_42, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv43 = convBnSilu(network, weightMap, *concat42->getOutput(0), 1280, 1, 1, 0, "model.43");


    IPoolingLayer* mp3 = network->addPoolingNd(*conv43->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp1->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv45 = convBnSilu(network, weightMap, *mp3->getOutput(0), 640, 1, 1, 0, "model.45");

    IElementWiseLayer* conv46 = convBnSilu(network, weightMap, *conv43->getOutput(0), 640, 1, 1, 0, "model.46");
    IElementWiseLayer* conv47 = convBnSilu(network, weightMap, *conv46->getOutput(0), 640, 3, 2, 1, "model.47");
    ITensor* input_tensor_48[] = { conv47->getOutput(0), conv45->getOutput(0) };
    IConcatenationLayer* concat48 = network->addConcatenation(input_tensor_48, 2);

    //IConcatenationLayer* mp3 = MPC3(network, weightMap, *conv43->getOutput(0), 640, "model.45", "model.46", "model.47");


    IElementWiseLayer* conv49 = convBnSilu(network, weightMap, *concat48->getOutput(0), 256, 1, 1, 0, "model.49");

    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *concat48->getOutput(0), 256, 1, 1, 0, "model.50");
    IElementWiseLayer* conv51 = convBnSilu(network, weightMap, *conv50->getOutput(0), 256, 3, 1, 1, "model.51");
    IElementWiseLayer* conv52 = convBnSilu(network, weightMap, *conv51->getOutput(0), 256, 3, 1, 1, "model.52");
    IElementWiseLayer* conv53 = convBnSilu(network, weightMap, *conv52->getOutput(0), 256, 3, 1, 1, "model.53");
    IElementWiseLayer* conv54 = convBnSilu(network, weightMap, *conv53->getOutput(0), 256, 3, 1, 1, "model.54");
    IElementWiseLayer* conv55 = convBnSilu(network, weightMap, *conv54->getOutput(0), 256, 3, 1, 1, "model.55");
    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *conv55->getOutput(0), 256, 3, 1, 1, "model.56");

    ITensor* input_tensor_57[] = { conv56->getOutput(0), conv54->getOutput(0), conv52->getOutput(0), conv50->getOutput(0),conv49->getOutput(0) };
    IConcatenationLayer* concat57 = network->addConcatenation(input_tensor_57, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv58 = convBnSilu(network, weightMap, *concat57->getOutput(0), 1280, 1, 1, 0, "model.58");


    //-----------------------yolov7 head---------------------------
    //-----SPPCSPC-----------
    IElementWiseLayer* conv59 = SPPCSPC(network, weightMap, *conv58->getOutput(0), 640, "model.59");

    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv59->getOutput(0), 320, 1, 1, 0, "model.60");


    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* re61 = network->addResize(*conv60->getOutput(0));
    re61->setResizeMode(ResizeMode::kNEAREST);
    re61->setScales(scale, 3);

    IElementWiseLayer* conv62 = convBnSilu(network, weightMap, *conv43->getOutput(0), 320, 1, 1, 0, "model.62");


    ITensor* input_tensor_63[] = { conv62->getOutput(0), re61->getOutput(0) };
    IConcatenationLayer* concat63 = network->addConcatenation(input_tensor_63, 2);
    //concat63->setAxis(0);


    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *concat63->getOutput(0), 256, 1, 1, 0, "model.64");

    IElementWiseLayer* conv65 = convBnSilu(network, weightMap, *concat63->getOutput(0), 256, 1, 1, 0, "model.65");
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *conv65->getOutput(0), 256, 3, 1, 1, "model.66");
    IElementWiseLayer* conv67 = convBnSilu(network, weightMap, *conv66->getOutput(0), 256, 3, 1, 1, "model.67");
    IElementWiseLayer* conv68 = convBnSilu(network, weightMap, *conv67->getOutput(0), 256, 3, 1, 1, "model.68");
    IElementWiseLayer* conv69 = convBnSilu(network, weightMap, *conv68->getOutput(0), 256, 3, 1, 1, "model.69");
    IElementWiseLayer* conv70 = convBnSilu(network, weightMap, *conv69->getOutput(0), 256, 3, 1, 1, "model.70");
    IElementWiseLayer* conv71 = convBnSilu(network, weightMap, *conv70->getOutput(0), 256, 3, 1, 1, "model.71");

    ITensor* input_tensor_72[] = { conv71->getOutput(0), conv69->getOutput(0), conv67->getOutput(0), conv65->getOutput(0),conv64->getOutput(0) };
    IConcatenationLayer* concat72 = network->addConcatenation(input_tensor_72, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv73 = convBnSilu(network, weightMap, *concat72->getOutput(0), 320, 1, 1, 0, "model.73");

    IElementWiseLayer* conv74 = convBnSilu(network, weightMap, *conv73->getOutput(0), 160, 1, 1, 0, "model.74");

    IResizeLayer* re75 = network->addResize(*conv74->getOutput(0));
    re75->setResizeMode(ResizeMode::kNEAREST);
    re75->setScales(scale, 3);


    IElementWiseLayer* conv76 = convBnSilu(network, weightMap, *conv28->getOutput(0), 160, 1, 1, 0, "model.76");


    ITensor* input_tensor_77[] = { conv76->getOutput(0), re75->getOutput(0) };
    IConcatenationLayer* concat77 = network->addConcatenation(input_tensor_77, 2);



    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *concat77->getOutput(0), 128, 1, 1, 0, "model.78");

    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *concat77->getOutput(0), 128, 1, 1, 0, "model.79");
    IElementWiseLayer* conv80 = convBnSilu(network, weightMap, *conv79->getOutput(0), 128, 3, 1, 1, "model.80");
    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *conv80->getOutput(0), 128, 3, 1, 1, "model.81");
    IElementWiseLayer* conv82 = convBnSilu(network, weightMap, *conv81->getOutput(0), 128, 3, 1, 1, "model.82");
    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *conv82->getOutput(0), 128, 3, 1, 1, "model.83");
    IElementWiseLayer* conv84 = convBnSilu(network, weightMap, *conv83->getOutput(0), 128, 3, 1, 1, "model.84");
    IElementWiseLayer* conv85 = convBnSilu(network, weightMap, *conv84->getOutput(0), 128, 3, 1, 1, "model.85");


    ITensor* input_tensor_86[] = { conv85->getOutput(0), conv83->getOutput(0), conv81->getOutput(0), conv79->getOutput(0),conv78->getOutput(0) };
    IConcatenationLayer* concat86 = network->addConcatenation(input_tensor_86, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv87 = convBnSilu(network, weightMap, *concat86->getOutput(0), 160, 1, 1, 0, "model.87");


    IPoolingLayer* mp88 = network->addPoolingNd(*conv87->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp88->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv89 = convBnSilu(network, weightMap, *mp88->getOutput(0), 160, 1, 1, 0, "model.89");

    IElementWiseLayer* conv90 = convBnSilu(network, weightMap, *conv87->getOutput(0), 160, 1, 1, 0, "model.90");
    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *conv90->getOutput(0), 160, 3, 2, 1, "model.91");


    ITensor* input_tensor_92[] = { conv91->getOutput(0), conv89->getOutput(0),conv73->getOutput(0) };
    IConcatenationLayer* concat92 = network->addConcatenation(input_tensor_92, 3);


    IElementWiseLayer* conv93 = convBnSilu(network, weightMap, *concat92->getOutput(0), 256, 1, 1, 0, "model.93");

    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *concat92->getOutput(0), 256, 1, 1, 0, "model.94");
    IElementWiseLayer* conv95 = convBnSilu(network, weightMap, *conv94->getOutput(0), 256, 3, 1, 1, "model.95");
    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *conv95->getOutput(0), 256, 3, 1, 1, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *conv96->getOutput(0), 256, 3, 1, 1, "model.97");
    IElementWiseLayer* conv98 = convBnSilu(network, weightMap, *conv97->getOutput(0), 256, 3, 1, 1, "model.98");
    IElementWiseLayer* conv99 = convBnSilu(network, weightMap, *conv98->getOutput(0), 256, 3, 1, 1, "model.99");
    IElementWiseLayer* conv100 = convBnSilu(network, weightMap, *conv99->getOutput(0), 256, 3, 1, 1, "model.100");


    ITensor* input_tensor_101[] = { conv100->getOutput(0), conv98->getOutput(0), conv96->getOutput(0), conv94->getOutput(0),conv93->getOutput(0) };
    IConcatenationLayer* concat101 = network->addConcatenation(input_tensor_101, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv102 = convBnSilu(network, weightMap, *concat101->getOutput(0), 320, 1, 1, 0, "model.102");

    IPoolingLayer* mp103 = network->addPoolingNd(*conv102->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp103->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv104 = convBnSilu(network, weightMap, *mp103->getOutput(0), 320, 1, 1, 0, "model.104");

    IElementWiseLayer* conv105 = convBnSilu(network, weightMap, *conv102->getOutput(0), 320, 1, 1, 0, "model.105");
    IElementWiseLayer* conv106 = convBnSilu(network, weightMap, *conv105->getOutput(0), 320, 3, 2, 1, "model.106");


    ITensor* input_tensor_107[] = { conv106->getOutput(0), conv104->getOutput(0),conv59->getOutput(0) };
    IConcatenationLayer* concat107 = network->addConcatenation(input_tensor_107, 3);



    IElementWiseLayer* conv108 = convBnSilu(network, weightMap, *concat107->getOutput(0), 512, 1, 1, 0, "model.108");

    IElementWiseLayer* conv109 = convBnSilu(network, weightMap, *concat107->getOutput(0), 512, 1, 1, 0, "model.109");
    IElementWiseLayer* conv110 = convBnSilu(network, weightMap, *conv109->getOutput(0), 512, 3, 1, 1, "model.110");
    IElementWiseLayer* conv111 = convBnSilu(network, weightMap, *conv110->getOutput(0), 512, 3, 1, 1, "model.111");
    IElementWiseLayer* conv112 = convBnSilu(network, weightMap, *conv111->getOutput(0), 512, 3, 1, 1, "model.112");
    IElementWiseLayer* conv113 = convBnSilu(network, weightMap, *conv112->getOutput(0), 512, 3, 1, 1, "model.113");
    IElementWiseLayer* conv114 = convBnSilu(network, weightMap, *conv113->getOutput(0), 512, 3, 1, 1, "model.114");
    IElementWiseLayer* conv115 = convBnSilu(network, weightMap, *conv114->getOutput(0), 512, 3, 1, 1, "model.115");

    ITensor* input_tensor_116[] = { conv115->getOutput(0), conv113->getOutput(0), conv111->getOutput(0), conv109->getOutput(0),conv108->getOutput(0) };
    IConcatenationLayer* concat116 = network->addConcatenation(input_tensor_116, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv117 = convBnSilu(network, weightMap, *concat116->getOutput(0), 640, 1, 1, 0, "model.117");


    IElementWiseLayer* con_0 = convBnSilu(network, weightMap, *conv87->getOutput(0), 320, 3, 1, 1, "model.118");
    IElementWiseLayer* con_1 = convBnSilu(network, weightMap, *conv102->getOutput(0), 640, 3, 1, 1, "model.119");
    IElementWiseLayer* con_2 = convBnSilu(network, weightMap, *conv117->getOutput(0), 1280, 3, 1, 1, "model.120");


    /*----------------------------------yolov7 out-----------------------------------------*/
    IConvolutionLayer* det0 = network->addConvolutionNd(*con_0->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.121.m.0.weight"], weightMap["model.121.m.0.bias"]);
    assert(det0);
    det0->setName("det0");
    IConvolutionLayer* det1 = network->addConvolutionNd(*con_1->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.121.m.1.weight"], weightMap["model.121.m.1.bias"]);
    assert(det1);
    det1->setName("det1");
    IConvolutionLayer* det2 = network->addConvolutionNd(*con_2->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.121.m.2.weight"], weightMap["model.121.m.2.bias"]);
    assert(det2);
    det2->setName("det2");


    auto yolo = addYoLoLayer(network, weightMap, "model.121", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
    config->setFlag(BuilderFlag::kFP16);

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



ICudaEngine* build_engine_yolov7(unsigned int maxBatchSize,IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string& wts_path) {
    std::map<std::string, Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    /*----------------------------------yolov7 backbone-----------------------------------------*/
    IElementWiseLayer* conv0 = convBnSilu(network, weightMap, *data, 32, 3, 1, 1, "model.0");

    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, *conv0->getOutput(0), 64, 3, 2, 1, "model.1");
    IElementWiseLayer* conv2 = convBnSilu(network, weightMap, *conv1->getOutput(0), 64, 3, 1, 1, "model.2");

    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), 128, 3, 2, 1, "model.3");
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv3->getOutput(0), 64, 1, 1, 0, "model.4");
    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv3->getOutput(0), 64, 1, 1, 0, "model.5");
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "model.6");
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), 64, 3, 1, 1, "model.7");
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), 64, 3, 1, 1, "model.8");
    IElementWiseLayer* conv9 = convBnSilu(network, weightMap, *conv8->getOutput(0), 64, 3, 1, 1, "model.9");
    ITensor* input_tensor_10[] = { conv9->getOutput(0), conv7->getOutput(0), conv5->getOutput(0), conv4->getOutput(0) };
    IConcatenationLayer* concat10 = network->addConcatenation(input_tensor_10, 4);
    concat10->setAxis(0);
    IElementWiseLayer* conv11 = convBnSilu(network, weightMap, *concat10->getOutput(0), 256, 1, 1, 0, "model.11");

    IPoolingLayer* mp12 = network->addPoolingNd(*conv11->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp12->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv13 = convBnSilu(network, weightMap, *mp12->getOutput(0), 128, 1, 1, 0, "model.13");
    IElementWiseLayer* conv14 = convBnSilu(network, weightMap, *conv11->getOutput(0), 128, 1, 1, 0, "model.14");
    IElementWiseLayer* conv15 = convBnSilu(network, weightMap, *conv14->getOutput(0), 128, 3, 2, 1, "model.15");
    ITensor* input_tensor_16[] = { conv15->getOutput(0), conv13->getOutput(0) };
    IConcatenationLayer* concat16 = network->addConcatenation(input_tensor_16, 2);
    IElementWiseLayer* conv17 = convBnSilu(network, weightMap, *concat16->getOutput(0), 128, 1, 1, 0, "model.17");
    IElementWiseLayer* conv18 = convBnSilu(network, weightMap, *concat16->getOutput(0), 128, 1, 1, 0, "model.18");
    IElementWiseLayer* conv19 = convBnSilu(network, weightMap, *conv18->getOutput(0), 128, 3, 1, 1, "model.19");
    IElementWiseLayer* conv20 = convBnSilu(network, weightMap, *conv19->getOutput(0), 128, 3, 1, 1, "model.20");
    IElementWiseLayer* conv21 = convBnSilu(network, weightMap, *conv20->getOutput(0), 128, 3, 1, 1, "model.21");
    IElementWiseLayer* conv22 = convBnSilu(network, weightMap, *conv21->getOutput(0), 128, 3, 1, 1, "model.22");
    ITensor* input_tensor_23[] = { conv22->getOutput(0), conv20->getOutput(0), conv18->getOutput(0), conv17->getOutput(0) };
    IConcatenationLayer* concat23 = network->addConcatenation(input_tensor_23, 4);
    concat23->setAxis(0);
    IElementWiseLayer* conv24 = convBnSilu(network, weightMap, *concat23->getOutput(0), 512, 1, 1, 0, "model.24");

    IPoolingLayer* mp25 = network->addPoolingNd(*conv24->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp25->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv26 = convBnSilu(network, weightMap, *mp25->getOutput(0), 256, 1, 1, 0, "model.26");
    IElementWiseLayer* conv27 = convBnSilu(network, weightMap, *conv24->getOutput(0), 256, 1, 1, 0, "model.27");
    IElementWiseLayer* conv28 = convBnSilu(network, weightMap, *conv27->getOutput(0), 256, 3, 2, 1, "model.28");
    ITensor* input_tensor_29[] = { conv28->getOutput(0), conv26->getOutput(0) };
    IConcatenationLayer* concat29 = network->addConcatenation(input_tensor_29, 2);
    IElementWiseLayer* conv30 = convBnSilu(network, weightMap, *concat29->getOutput(0), 256, 1, 1, 0, "model.30");
    IElementWiseLayer* conv31 = convBnSilu(network, weightMap, *concat29->getOutput(0), 256, 1, 1, 0, "model.31");
    IElementWiseLayer* conv32 = convBnSilu(network, weightMap, *conv31->getOutput(0), 256, 3, 1, 1, "model.32");
    IElementWiseLayer* conv33 = convBnSilu(network, weightMap, *conv32->getOutput(0), 256, 3, 1, 1, "model.33");
    IElementWiseLayer* conv34 = convBnSilu(network, weightMap, *conv33->getOutput(0), 256, 3, 1, 1, "model.34");
    IElementWiseLayer* conv35 = convBnSilu(network, weightMap, *conv34->getOutput(0), 256, 3, 1, 1, "model.35");
    ITensor* input_tensor_36[] = { conv35->getOutput(0), conv33->getOutput(0), conv31->getOutput(0), conv30->getOutput(0) };
    IConcatenationLayer* concat36 = network->addConcatenation(input_tensor_36, 4);
    concat36->setAxis(0);
    IElementWiseLayer* conv37 = convBnSilu(network, weightMap, *concat36->getOutput(0), 1024, 1, 1, 0, "model.37");

    IPoolingLayer* mp38 = network->addPoolingNd(*conv37->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp38->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv39 = convBnSilu(network, weightMap, *mp38->getOutput(0), 512, 1, 1, 0, "model.39");
    IElementWiseLayer* conv40 = convBnSilu(network, weightMap, *conv37->getOutput(0), 512, 1, 1, 0, "model.40");
    IElementWiseLayer* conv41 = convBnSilu(network, weightMap, *conv40->getOutput(0), 512, 3, 2, 1, "model.41");
    ITensor* input_tensor_42[] = { conv41->getOutput(0), conv39->getOutput(0) };
    IConcatenationLayer* concat42 = network->addConcatenation(input_tensor_42, 2);
    concat42->setAxis(0);
    IElementWiseLayer* conv43 = convBnSilu(network, weightMap, *concat42->getOutput(0), 256, 1, 1, 0, "model.43");
    IElementWiseLayer* conv44 = convBnSilu(network, weightMap, *concat42->getOutput(0), 256, 1, 1, 0, "model.44");
    IElementWiseLayer* conv45 = convBnSilu(network, weightMap, *conv44->getOutput(0), 256, 3, 1, 1, "model.45");
    IElementWiseLayer* conv46 = convBnSilu(network, weightMap, *conv45->getOutput(0), 256, 3, 1, 1, "model.46");
    IElementWiseLayer* conv47 = convBnSilu(network, weightMap, *conv46->getOutput(0), 256, 3, 1, 1, "model.47");
    IElementWiseLayer* conv48 = convBnSilu(network, weightMap, *conv47->getOutput(0), 256, 3, 1, 1, "model.48");
    ITensor* input_tensor_49[] = { conv48->getOutput(0), conv46->getOutput(0), conv44->getOutput(0), conv43->getOutput(0) };
    IConcatenationLayer* concat49 = network->addConcatenation(input_tensor_49, 4);
    concat49->setAxis(0);
    IElementWiseLayer* conv50 = convBnSilu(network, weightMap, *concat49->getOutput(0), 1024, 1, 1, 0, "model.50");

    /*----------------------------------yolov7 head-----------------------------------------*/
    IElementWiseLayer* conv51 = SPPCSPC(network, weightMap, *conv50->getOutput(0), 512, "model.51");

    IElementWiseLayer* conv52 = convBnSilu(network, weightMap, *conv51->getOutput(0), 256, 1, 1, 0, "model.52");
    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* re53 = network->addResize(*conv52->getOutput(0));
    re53->setResizeMode(ResizeMode::kNEAREST);
    re53->setScales(scale, 3);
    IElementWiseLayer* conv54 = convBnSilu(network, weightMap, *conv37->getOutput(0), 256, 1, 1, 0, "model.54");
    ITensor* input_tensor_55[] = { conv54->getOutput(0), re53->getOutput(0) };
    IConcatenationLayer* concat55 = network->addConcatenation(input_tensor_55, 2);
    concat55->setAxis(0);

    IElementWiseLayer* conv56 = convBnSilu(network, weightMap, *concat55->getOutput(0), 256, 1, 1, 0, "model.56");
    IElementWiseLayer* conv57 = convBnSilu(network, weightMap, *concat55->getOutput(0), 256, 1, 1, 0, "model.57");
    IElementWiseLayer* conv58 = convBnSilu(network, weightMap, *conv57->getOutput(0), 128, 3, 1, 1, "model.58");
    IElementWiseLayer* conv59 = convBnSilu(network, weightMap, *conv58->getOutput(0), 128, 3, 1, 1, "model.59");
    IElementWiseLayer* conv60 = convBnSilu(network, weightMap, *conv59->getOutput(0), 128, 3, 1, 1, "model.60");
    IElementWiseLayer* conv61 = convBnSilu(network, weightMap, *conv60->getOutput(0), 128, 3, 1, 1, "model.61");
    ITensor* input_tensor_62[] = { conv61->getOutput(0), conv60->getOutput(0), conv59->getOutput(0), conv58->getOutput(0), conv57->getOutput(0), conv56->getOutput(0) };
    IConcatenationLayer* concat62 = network->addConcatenation(input_tensor_62, 6);
    concat62->setAxis(0);
    IElementWiseLayer* conv63 = convBnSilu(network, weightMap, *concat62->getOutput(0), 256, 1, 1, 0, "model.63");

    IElementWiseLayer* conv64 = convBnSilu(network, weightMap, *conv63->getOutput(0), 128, 1, 1, 0, "model.64");
    IResizeLayer* re65 = network->addResize(*conv64->getOutput(0));
    re65->setResizeMode(ResizeMode::kNEAREST);
    re65->setScales(scale, 3);
    IElementWiseLayer* conv66 = convBnSilu(network, weightMap, *conv24->getOutput(0), 128, 1, 1, 0, "model.66");
    ITensor* input_tensor_67[] = { conv66->getOutput(0), re65->getOutput(0) };
    IConcatenationLayer* concat67 = network->addConcatenation(input_tensor_67, 2);
    concat67->setAxis(0);

    IElementWiseLayer* conv68 = convBnSilu(network, weightMap, *concat67->getOutput(0), 128, 1, 1, 0, "model.68");
    IElementWiseLayer* conv69 = convBnSilu(network, weightMap, *concat67->getOutput(0), 128, 1, 1, 0, "model.69");
    IElementWiseLayer* conv70 = convBnSilu(network, weightMap, *conv69->getOutput(0), 64, 3, 1, 1, "model.70");
    IElementWiseLayer* conv71 = convBnSilu(network, weightMap, *conv70->getOutput(0), 64, 3, 1, 1, "model.71");
    IElementWiseLayer* conv72 = convBnSilu(network, weightMap, *conv71->getOutput(0), 64, 3, 1, 1, "model.72");
    IElementWiseLayer* conv73 = convBnSilu(network, weightMap, *conv72->getOutput(0), 64, 3, 1, 1, "model.73");
    ITensor* input_tensor_74[] = { conv73->getOutput(0), conv72->getOutput(0), conv71->getOutput(0), conv70->getOutput(0), conv69->getOutput(0), conv68->getOutput(0) };
    IConcatenationLayer* concat74 = network->addConcatenation(input_tensor_74, 6);
    concat74->setAxis(0);
    IElementWiseLayer* conv75 = convBnSilu(network, weightMap, *concat74->getOutput(0), 128, 1, 1, 0, "model.75");

    IPoolingLayer* mp76 = network->addPoolingNd(*conv75->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp76->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv77 = convBnSilu(network, weightMap, *mp76->getOutput(0), 128, 1, 1, 0, "model.77");
    IElementWiseLayer* conv78 = convBnSilu(network, weightMap, *conv75->getOutput(0), 128, 1, 1, 0, "model.78");
    IElementWiseLayer* conv79 = convBnSilu(network, weightMap, *conv78->getOutput(0), 128, 3, 2, 1, "model.79");
    ITensor* input_tensor_80[] = { conv79->getOutput(0), conv77->getOutput(0), conv63->getOutput(0) };
    IConcatenationLayer* concat80 = network->addConcatenation(input_tensor_80, 3);
    concat80->setAxis(0);

    IElementWiseLayer* conv81 = convBnSilu(network, weightMap, *concat80->getOutput(0), 256, 1, 1, 0, "model.81");
    IElementWiseLayer* conv82 = convBnSilu(network, weightMap, *concat80->getOutput(0), 256, 1, 1, 0, "model.82");
    IElementWiseLayer* conv83 = convBnSilu(network, weightMap, *conv82->getOutput(0), 128, 3, 1, 1, "model.83");
    IElementWiseLayer* conv84 = convBnSilu(network, weightMap, *conv83->getOutput(0), 128, 3, 1, 1, "model.84");
    IElementWiseLayer* conv85 = convBnSilu(network, weightMap, *conv84->getOutput(0), 128, 3, 1, 1, "model.85");
    IElementWiseLayer* conv86 = convBnSilu(network, weightMap, *conv85->getOutput(0), 128, 3, 1, 1, "model.86");
    ITensor* input_tensor_87[] = { conv86->getOutput(0), conv85->getOutput(0), conv84->getOutput(0), conv83->getOutput(0), conv82->getOutput(0), conv81->getOutput(0) };
    IConcatenationLayer* concat87 = network->addConcatenation(input_tensor_87, 6);
    concat87->setAxis(0);
    IElementWiseLayer* conv88 = convBnSilu(network, weightMap, *concat87->getOutput(0), 256, 1, 1, 0, "model.88");

    IPoolingLayer* mp89 = network->addPoolingNd(*conv88->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    mp89->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv90 = convBnSilu(network, weightMap, *mp89->getOutput(0), 256, 1, 1, 0, "model.90");
    IElementWiseLayer* conv91 = convBnSilu(network, weightMap, *conv88->getOutput(0), 256, 1, 1, 0, "model.91");
    IElementWiseLayer* conv92 = convBnSilu(network, weightMap, *conv91->getOutput(0), 256, 3, 2, 1, "model.92");
    ITensor* input_tensor_93[] = { conv92->getOutput(0), conv90->getOutput(0), conv51->getOutput(0) };
    IConcatenationLayer* concat93 = network->addConcatenation(input_tensor_93, 3);
    concat93->setAxis(0);

    IElementWiseLayer* conv94 = convBnSilu(network, weightMap, *concat93->getOutput(0), 512, 1, 1, 0, "model.94");
    IElementWiseLayer* conv95 = convBnSilu(network, weightMap, *concat93->getOutput(0), 512, 1, 1, 0, "model.95");
    IElementWiseLayer* conv96 = convBnSilu(network, weightMap, *conv95->getOutput(0), 256, 3, 1, 1, "model.96");
    IElementWiseLayer* conv97 = convBnSilu(network, weightMap, *conv96->getOutput(0), 256, 3, 1, 1, "model.97");
    IElementWiseLayer* conv98 = convBnSilu(network, weightMap, *conv97->getOutput(0), 256, 3, 1, 1, "model.98");
    IElementWiseLayer* conv99 = convBnSilu(network, weightMap, *conv98->getOutput(0), 256, 3, 1, 1, "model.99");
    ITensor* input_tensor_100[] = { conv99->getOutput(0), conv98->getOutput(0), conv97->getOutput(0), conv96->getOutput(0), conv95->getOutput(0), conv94->getOutput(0) };
    IConcatenationLayer* concat100 = network->addConcatenation(input_tensor_100, 6);
    concat100->setAxis(0);
    IElementWiseLayer* conv101 = convBnSilu(network, weightMap, *concat100->getOutput(0), 512, 1, 1, 0, "model.101");

    IElementWiseLayer* conv102 = RepConv(network, weightMap, *conv75->getOutput(0), 256, 3, 1, "model.102");
    IElementWiseLayer* conv103 = RepConv(network, weightMap, *conv88->getOutput(0), 512, 3, 1, "model.103");
    IElementWiseLayer* conv104 = RepConv(network, weightMap, *conv101->getOutput(0), 1024, 3, 1, "model.104");

    /*----------------------------------yolov7 out-----------------------------------------*/
    IConvolutionLayer* cv105_0 = network->addConvolutionNd(*conv102->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.105.m.0.weight"], weightMap["model.105.m.0.bias"]);
    assert(cv105_0);
    cv105_0->setName("cv105.0");
    IConvolutionLayer* cv105_1 = network->addConvolutionNd(*conv103->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.105.m.1.weight"], weightMap["model.105.m.1.bias"]);
    assert(cv105_1);
    cv105_1->setName("cv105.1");
    IConvolutionLayer* cv105_2 = network->addConvolutionNd(*conv104->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.105.m.2.weight"], weightMap["model.105.m.2.bias"]);
    assert(cv105_2);
    cv105_2->setName("cv105.2");

    auto yolo = addYoLoLayer(network, weightMap, "model.105", std::vector<IConvolutionLayer*>{cv105_0, cv105_1, cv105_2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
    config->setFlag(BuilderFlag::kFP16);

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



ICudaEngine* build_engine_yolov7_tiny(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);


    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    
    
    /* ------ yolov7-tiny backbone------ */
    // [32, 3, 2, None, 1, nn.LeakyReLU(0.1)]]---> outch、ksize、stride、padding、groups------
    auto conv0 = convBlockLeakRelu(network, weightMap, *data, 32, 3, 2, 1, "model.0");
    assert(conv0);

    // [-1, 1, Conv, [64, 3, 2, None, 1, nn.LeakyReLU(0.1)]],  # 1-P2/4    
    auto conv1 = convBlockLeakRelu(network, weightMap, *conv0->getOutput(0), 64, 3, 2, 1, "model.1");
    assert(conv1);

    //  [-1, 1, Conv, [32, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv2 = convBlockLeakRelu(network, weightMap, *conv1->getOutput(0), 32, 1, 1, 0, "model.2");
    assert(conv2);

    // [-2, 1, Conv, [32, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv3 = convBlockLeakRelu(network, weightMap, *conv1->getOutput(0), 32, 1, 1, 0, "model.3");
    assert(conv3);

    // [-1, 1, Conv, [32, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv4 = convBlockLeakRelu(network, weightMap, *conv3->getOutput(0), 32, 3, 1, 1, "model.4");
    assert(conv4);

    // [-1, 1, Conv, [32, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv5 = convBlockLeakRelu(network, weightMap, *conv4->getOutput(0), 32, 3, 1, 1, "model.5");
    assert(conv5);


    ITensor* input_tensor_6[] = { conv5->getOutput(0), conv4->getOutput(0), conv3->getOutput(0), conv2->getOutput(0) };
    auto cat6 = network->addConcatenation(input_tensor_6, 4);
    //cat6->setAxis(0);

    // [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 7
    auto conv7 = convBlockLeakRelu(network, weightMap, *cat6->getOutput(0), 64, 1, 1, 0, "model.7");
    assert(conv7);


    auto* pool8 = network->addPoolingNd(*conv7->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    assert(pool8);
    pool8->setStrideNd(DimsHW{ 2, 2 });


    //[-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]] ,
    auto conv9 = convBlockLeakRelu(network, weightMap, *pool8->getOutput(0), 64, 1, 1, 0, "model.9");
    assert(conv9);

    // [-2, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv10 = convBlockLeakRelu(network, weightMap, *pool8->getOutput(0), 64, 1, 1, 0, "model.10");
    assert(conv10);
    //[-1, 1, Conv, [64, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv11 = convBlockLeakRelu(network, weightMap, *conv10->getOutput(0), 64, 3, 1, 1, "model.11");
    assert(conv11);
    //[-1, 1, Conv, [64, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv12 = convBlockLeakRelu(network, weightMap, *conv11->getOutput(0), 64, 3, 1, 1, "model.12");
    assert(conv12);

    ITensor* input_tensor_13[] = { conv12->getOutput(0), conv11->getOutput(0), conv10->getOutput(0), conv9->getOutput(0) };
    auto cat13 = network->addConcatenation(input_tensor_13, 4);
    //cat2->setAxis(0);
    
    // [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 14
    auto conv14 = convBlockLeakRelu(network, weightMap, *cat13->getOutput(0), 128, 1, 1, 0, "model.14");
    assert(conv14);


    auto* pool15 = network->addPoolingNd(*conv14->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    assert(pool15);
    pool15->setStrideNd(DimsHW{ 2, 2 });





    // [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv16 = convBlockLeakRelu(network, weightMap, *pool15->getOutput(0), 128, 1, 1, 0, "model.16");
    assert(conv16);
    //[-2, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv17 = convBlockLeakRelu(network, weightMap, *pool15->getOutput(0), 128, 1, 1, 0, "model.17");
    assert(conv17);
    //[-1, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv18 = convBlockLeakRelu(network, weightMap, *conv17->getOutput(0), 128, 3, 1, 1, "model.18");
    assert(conv18);
    // [-1, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv19 = convBlockLeakRelu(network, weightMap, *conv18->getOutput(0), 128, 3, 1, 1, "model.19");
    assert(conv19);

    ITensor* input_tensor_20[] = { conv19->getOutput(0), conv18->getOutput(0), conv17->getOutput(0), conv16->getOutput(0) };
    auto cat20 = network->addConcatenation(input_tensor_20, 4);
    //cat20->setAxis(0);
    //[-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 21
    auto conv21 = convBlockLeakRelu(network, weightMap, *cat20->getOutput(0), 256, 1, 1, 0, "model.21");
    assert(conv21);


    auto* pool22 = network->addPoolingNd(*conv21->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
    assert(pool22);
    pool22->setStrideNd(DimsHW{ 2, 2 });



    // [-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv23 = convBlockLeakRelu(network, weightMap, *pool22->getOutput(0), 256, 1, 1, 0, "model.23");
    assert(conv23);

    // [-2, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv24 = convBlockLeakRelu(network, weightMap, *pool22->getOutput(0), 256, 1, 1, 0, "model.24");
    assert(conv24);

    // [-1, 1, Conv, [256, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv25 = convBlockLeakRelu(network, weightMap, *conv24->getOutput(0), 256, 3, 1, 1, "model.25");
    assert(conv25);

    // [-1, 1, Conv, [256, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv26 = convBlockLeakRelu(network, weightMap, *conv25->getOutput(0), 256, 3, 1, 1, "model.26");
    assert(conv26);


    ITensor* input_tensor_27[] = { conv26->getOutput(0), conv25->getOutput(0), conv24->getOutput(0), conv23->getOutput(0) };
    auto cat27 = network->addConcatenation(input_tensor_27, 4);
    //cat27->setAxis(0);

    // [-1, 1, Conv, [512, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 28
    auto conv28 = convBlockLeakRelu(network, weightMap, *cat27->getOutput(0), 512, 1, 1, 0, "model.28");
    assert(conv28);

    /*===============================yolov7-tiny head======================================*/

    // [-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]]
    auto conv29 = convBlockLeakRelu(network, weightMap, *conv28->getOutput(0), 256, 1, 1, 0, "model.29");
    assert(conv29);

    // [-2, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv30 = convBlockLeakRelu(network, weightMap, *conv28->getOutput(0), 256, 1, 1, 0, "model.30");
    assert(conv30);

    //[-1, 1, SP, [5]],
    auto* pool31 = network->addPoolingNd(*conv30->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
    assert(pool31);
    pool31->setStrideNd(DimsHW{ 1, 1 });
    pool31->setPaddingNd(DimsHW{2,2});
    // [-2, 1, SP, [9]],
    auto* pool32 = network->addPoolingNd(*conv30->getOutput(0), PoolingType::kMAX, DimsHW{ 9, 9 });
    assert(pool32);
    pool32->setStrideNd(DimsHW{ 1, 1 });
    pool32->setPaddingNd(DimsHW{ 4, 4 });

    // [-3, 1, SP, [13]],
    auto* pool33 = network->addPoolingNd(*conv30->getOutput(0), PoolingType::kMAX, DimsHW{ 13, 13 });
    assert(pool33);
    pool33->setStrideNd(DimsHW{ 1, 1 });
    pool33->setPaddingNd(DimsHW{ 6, 6 });



    ITensor* input_tensor_34[] = { pool33->getOutput(0), pool32->getOutput(0), pool31->getOutput(0), conv30->getOutput(0) };
    auto cat34 = network->addConcatenation(input_tensor_34, 4);
    //cat34->setAxis(0);

    // [-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv35 = convBlockLeakRelu(network, weightMap, *cat34->getOutput(0), 256, 1, 1, 0, "model.35");
    assert(conv35);





    ITensor* input_tensor_36[] = { conv35->getOutput(0), conv29->getOutput(0) };
    auto cat36 = network->addConcatenation(input_tensor_36, 2);
    //cat36->setAxis(0);

    // [-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 37
    auto conv37 = convBlockLeakRelu(network, weightMap, *cat36->getOutput(0), 256, 1, 1, 0, "model.37");
    assert(conv37);

    // [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv38 = convBlockLeakRelu(network, weightMap, *conv37->getOutput(0), 128, 1, 1, 0, "model.38");
    assert(conv38);


    float scale[] = { 1.0, 2.0, 2.0 };
    IResizeLayer* resize39 = network->addResize(*conv38->getOutput(0));
    resize39->setResizeMode(ResizeMode::kNEAREST);
    resize39->setScales(scale, 3);

    //    [21, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # route backbone P4 ---->conv16
    auto conv40 = convBlockLeakRelu(network, weightMap, *conv21->getOutput(0), 128, 1, 1, 0, "model.40");
    assert(conv40);



    ITensor* input_tensor_41[] = { conv40->getOutput(0), resize39->getOutput(0) };
    auto cat41 = network->addConcatenation(input_tensor_41, 2);
    //cat41->setAxis(0);



    //   [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv42 = convBlockLeakRelu(network, weightMap, *cat41->getOutput(0), 64, 1, 1, 0, "model.42");
    assert(conv42);

    //[-2, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv43 = convBlockLeakRelu(network, weightMap, *cat41->getOutput(0), 64, 1, 1, 0, "model.43");
    assert(conv43);

    // [-1, 1, Conv, [64, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv44 = convBlockLeakRelu(network, weightMap, *conv43->getOutput(0), 64, 3, 1, 1, "model.44");
    assert(conv44);

    // [-1, 1, Conv, [64, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv45 = convBlockLeakRelu(network, weightMap, *conv44->getOutput(0), 64, 3, 1, 1, "model.45");
    assert(conv45);


    ITensor* input_tensor_46[] = { conv45->getOutput(0), conv44->getOutput(0), conv43->getOutput(0), conv42->getOutput(0) };
    auto cat46 = network->addConcatenation(input_tensor_46, 4);
    //cat46->setAxis(0);

    //  [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 47
    auto conv47 = convBlockLeakRelu(network, weightMap, *cat46->getOutput(0), 128, 1, 1, 0, "model.47");
    assert(conv47);

    //    [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv48 = convBlockLeakRelu(network, weightMap, *conv47->getOutput(0), 64, 1, 1, 0, "model.48");
    assert(conv48);



    IResizeLayer* resize49 = network->addResize(*conv48->getOutput(0));
    resize49->setResizeMode(ResizeMode::kNEAREST);
    resize49->setScales(scale, 3);

    //   [14, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]], # route backbone P3        conv11
    auto conv50 = convBlockLeakRelu(network, weightMap, *conv14->getOutput(0), 64, 1, 1, 0, "model.50");
    assert(conv50);

    

    ITensor* input_tensor_51[] = { conv50->getOutput(0), resize49->getOutput(0) };
    IConcatenationLayer* cat51 = network->addConcatenation(input_tensor_51, 2);
    //cat51->setAxis(0);


    //    [-1, 1, Conv, [32, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv52 = convBlockLeakRelu(network, weightMap, *cat51->getOutput(0), 32, 1, 1, 0, "model.52");
    assert(conv52);
    //   [-2, 1, Conv, [32, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv53 = convBlockLeakRelu(network, weightMap, *cat51->getOutput(0), 32, 1, 1, 0, "model.53");
    assert(conv53);

    //  [-1, 1, Conv, [32, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv54 = convBlockLeakRelu(network, weightMap, *conv53->getOutput(0), 32, 3, 1, 1, "model.54");
    assert(conv54);
    //   [-1, 1, Conv, [32, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv55 = convBlockLeakRelu(network, weightMap, *conv54->getOutput(0), 32, 3, 1, 1, "model.55");
    assert(conv55);


    ITensor* input_tensor_56[] = { conv55->getOutput(0), conv54->getOutput(0), conv53->getOutput(0),conv52->getOutput(0) };
    IConcatenationLayer* cat56 = network->addConcatenation(input_tensor_56, 4);
    //cat56->setAxis(0);

    //    [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 57
    auto conv57 = convBlockLeakRelu(network, weightMap, *cat56->getOutput(0), 64, 1, 1, 0, "model.57");
    assert(conv57);

    //   [-1, 1, Conv, [128, 3, 2, None, 1, nn.LeakyReLU(0.1)]],
    auto conv58 = convBlockLeakRelu(network, weightMap, *conv57->getOutput(0), 128, 3, 2, 1, "model.58");
    assert(conv58);

    // conv32   [[-1, 47], 1, Concat, [1]],


    ITensor* input_tensor_59[] = { conv58->getOutput(0), conv47->getOutput(0) };
    IConcatenationLayer* cat59 = network->addConcatenation(input_tensor_59, 2);
    //cat59->setAxis(0);


    //    [-1, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv60 = convBlockLeakRelu(network, weightMap, *cat59->getOutput(0), 64, 1, 1, 0, "model.60");
    assert(conv60);
    //    [-2, 1, Conv, [64, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv61 = convBlockLeakRelu(network, weightMap, *cat59->getOutput(0), 64, 1, 1, 0, "model.61");
    assert(conv61);

    //   [-1, 1, Conv, [64, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv62 = convBlockLeakRelu(network, weightMap, *conv61->getOutput(0), 64, 3, 1, 1, "model.62");
    assert(conv62);
    //   [-1, 1, Conv, [64, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv63 = convBlockLeakRelu(network, weightMap, *conv62->getOutput(0), 64, 3, 1, 1, "model.63");
    assert(conv63);


    ITensor* input_tensor_64[] = { conv63->getOutput(0), conv62->getOutput(0), conv61->getOutput(0), conv60->getOutput(0) };
    IConcatenationLayer* cat64 = network->addConcatenation(input_tensor_64, 4);
    //cat64->setAxis(0);

    // [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]] , # 65
    auto conv65 = convBlockLeakRelu(network, weightMap, *cat64->getOutput(0), 128, 1, 1, 0, "model.65");
    assert(conv65);


    //[-1, 1, Conv, [256, 3, 2, None, 1, nn.LeakyReLU(0.1)]] ,
    auto conv66 = convBlockLeakRelu(network, weightMap, *conv65->getOutput(0), 256, 3, 2, 1, "model.66");
    assert(conv66);


    ITensor* input_tensor_67[] = { conv66->getOutput(0), conv37->getOutput(0) };
    IConcatenationLayer* cat67 = network->addConcatenation(input_tensor_67, 2);
    //cat67->setAxis(0);

    // [-1, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv68 = convBlockLeakRelu(network, weightMap, *cat67->getOutput(0), 128, 1, 1, 0, "model.68");
    assert(conv68);
    //   [-2, 1, Conv, [128, 1, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv69 = convBlockLeakRelu(network, weightMap, *cat67->getOutput(0), 128, 1, 1, 0, "model.69");
    assert(conv69);

    //   [-1, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv70 = convBlockLeakRelu(network, weightMap, *conv69->getOutput(0), 128, 3, 1, 1, "model.70");
    assert(conv70);
    //   [-1, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv71 = convBlockLeakRelu(network, weightMap, *conv70->getOutput(0), 128, 3, 1, 1, "model.71");
    assert(conv71);


    ITensor* input_tensor_72[] = { conv71->getOutput(0), conv70->getOutput(0), conv69->getOutput(0), conv68->getOutput(0) };
    IConcatenationLayer* cat72 = network->addConcatenation(input_tensor_72, 4);
    //cat72->setAxis(0);

    //    [-1, 1, Conv, [256, 1, 1, None, 1, nn.LeakyReLU(0.1)]],  # 73
    auto conv73 = convBlockLeakRelu(network, weightMap, *cat72->getOutput(0), 256, 1, 1, 0, "model.73");
    assert(conv73);


    // [57, 1, Conv, [128, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv74 = convBlockLeakRelu(network, weightMap, *conv57->getOutput(0), 128, 3, 1, 1, "model.74");
    assert(conv74);
    //    [65, 1, Conv, [256, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv75 = convBlockLeakRelu(network, weightMap, *conv65->getOutput(0), 256, 3, 1, 1, "model.75");
    assert(conv75);
    //    [73, 1, Conv, [512, 3, 1, None, 1, nn.LeakyReLU(0.1)]],
    auto conv76 = convBlockLeakRelu(network, weightMap, *conv73->getOutput(0), 512, 3, 1, 1, "model.76");
    assert(conv76);

    /*--------------------detect--------------*/
    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*conv74->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.77.m.0.weight"], weightMap["model.77.m.0.bias"]);
   
    IConvolutionLayer* det1 = network->addConvolutionNd(*conv75->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.77.m.1.weight"], weightMap["model.77.m.1.bias"]);

    IConvolutionLayer* det2 = network->addConvolutionNd(*conv76->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.77.m.2.weight"], weightMap["model.77.m.2.bias"]);
 


    auto yolo = addYoLoLayer(network, weightMap, "model.77", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    return engine;
}


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream,std::string& wts_name,std::string &model_check) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = nullptr;


    if (model_check == "yolov7-tiny")
    {
        engine = build_engine_yolov7_tiny(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }else if (model_check == "yolov7")
    {
        engine = build_engine_yolov7(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }
    else if (model_check == "yolov7x")
    {
        engine = build_engine_yolov7x(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);

    }
    else if (model_check == "yolov7w6")
    {
        engine = build_engine_yolov7w6(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }
    else if (model_check == "yolov7e6")
    {
        engine = build_engine_yolov7e6(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }
    else if (model_check == "yolov7d6")
    {
        engine = build_engine_yolov7d6(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }
    else if (model_check == "yolov7e6e")
    {
        engine = build_engine_yolov7e6e(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir,std::string& model_check) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);


        if (net.size() == 1 && net[0] == 't' ) {
            model_check = "yolov7-tiny";
            gd = 1.0;
            gw = 1.0;
        }


        if (net.size() == 2 && net[0] == 'v' && net[1]=='7') {
            model_check ="yolov7";
            gd = 5.0;
            gw = 1.0;
        }


        if (net.size() == 1 && net[0] == 'x' ) {
            model_check = "yolov7x";
            gd = 1.0;
            gw = 1.0;
        }

        if (net.size() == 2 && net[0] == 'w' && net[1]=='6') {
            model_check = "yolov7w6";
            gd = 1.0;
            gw = 1.0;
        }
        if (net.size() == 2 && net[0] == 'e' && net[1]=='6') {
            model_check ="yolov7e6";
            gd = 1.0;
            gw = 1.0;
        }
        if (net.size() == 2 && net[0] == 'd' && net[1]=='6') {
            model_check ="yolov7d6";
            gd = 1.0;
            gw = 1.0;
        }
        if (net.size() == 3 && net[0] == 'e' && net[1]=='6' && net[2]=='e' ) {
            model_check = "yolov7e6e";
            gd = 1.0;
            gw = 1.0;

        }
        }else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);

    } else {
        return false;
    }
    return true;
}




int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    std::string model_check="";

    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir,model_check)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov7 -s [.wts] [.engine] [t/v7/x/w6/e6/d6/e6e gd gw]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov7 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
    return -1;
}




    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };

        APIToModel(BATCH_SIZE, &modelStream, wts_name, model_check);
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


    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char* trtModelStream = nullptr;
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

    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    int fcount = 0;
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        //auto start = std::chrono::system_clock::now();
        float* buffer_idx = (float*)buffers[inputIndex];
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            imgs_buffer[b] = img;
            size_t  size_image = img.cols * img.rows * 3;
            size_t  size_image_dst = INPUT_H * INPUT_W * 3;
            //copy data to pinned memory
            memcpy(img_host, img.data, size_image);
            //copy data to device memory
            CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
            buffer_idx += size_image_dst;
        }
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            cv::Mat img = imgs_buffer[b];
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("__" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
