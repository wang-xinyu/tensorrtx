#include <iostream>
#include <chrono>
#include <dirent.h>

#include "logging.h"
#include "utils.h"
#include "cuda_runtime_api.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

static Logger gLogger;


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder -> createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network -> addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov4_csp.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // define yolov4 csp layers
    auto l0 = convBnMish(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto l1 = convBnMish(network, weightMap, *l0 -> getOutput(0), 64, 3, 2, 1, 1);
    auto l2 = convBnMish(network, weightMap, *l1 -> getOutput(0), 32, 1, 1, 0, 2);
    auto l3 = convBnMish(network, weightMap, *l2 -> getOutput(0), 64, 3, 1, 1, 3);
    auto ew4 = network -> addElementWise(*l3 -> getOutput(0), *l1 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l5 = convBnMish(network, weightMap, *ew4 -> getOutput(0), 128, 3, 2, 1, 5);
    auto l6 = convBnMish(network, weightMap, *l5 -> getOutput(0), 64, 1, 1, 0, 6);
    auto l7 = l5;
    auto l8 = convBnMish(network, weightMap, *l7 -> getOutput(0), 64, 1, 1, 0, 8);
    auto l9 = convBnMish(network, weightMap, *l8 -> getOutput(0), 64, 1, 1, 0, 9);
    auto l10 = convBnMish(network, weightMap, *l9 -> getOutput(0), 64, 3, 1, 1, 10);
    auto ew11 = network -> addElementWise(*l10 -> getOutput(0), *l8 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l12 = convBnMish(network, weightMap, *ew11 -> getOutput(0), 64, 1, 1, 0, 12);
    auto l13 = convBnMish(network, weightMap, *l12 -> getOutput(0), 64, 3, 1, 1, 13);
    auto ew14 = network -> addElementWise(*l13 -> getOutput(0), *ew11 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l15 = convBnMish(network, weightMap, *ew14 -> getOutput(0), 64, 1, 1, 0, 15);

    ITensor* inputTensors16[] = {l15 -> getOutput(0), l6 -> getOutput(0)};
    auto cat16 = network -> addConcatenation(inputTensors16, 2);

    auto l17 = convBnMish(network, weightMap, *cat16 -> getOutput(0), 128, 1, 1, 0, 17);
    auto l18 = convBnMish(network, weightMap, *l17 -> getOutput(0), 256, 3, 2, 1, 18);
    auto l19 = convBnMish(network, weightMap, *l18 -> getOutput(0), 128, 1, 1, 0, 19);
    auto l20 = l18;
    auto l21 = convBnMish(network, weightMap, *l20 -> getOutput(0), 128, 1, 1, 0, 21);
    auto l22 = convBnMish(network, weightMap, *l21 -> getOutput(0), 128, 1, 1, 0, 22);
    auto l23 = convBnMish(network, weightMap, *l22 -> getOutput(0), 128, 3, 1, 1, 23);
    auto ew24 = network -> addElementWise(*l23 -> getOutput(0), *l21 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l25 = convBnMish(network, weightMap, *ew24 -> getOutput(0), 128, 1, 1, 0, 25);
    auto l26 = convBnMish(network, weightMap, *l25 -> getOutput(0), 128, 3, 1, 1, 26);
    auto ew27 = network -> addElementWise(*l26 -> getOutput(0), *ew24 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l28 = convBnMish(network, weightMap, *ew27 -> getOutput(0), 128, 1, 1, 0, 28);
    auto l29 = convBnMish(network, weightMap, *l28 -> getOutput(0), 128, 3, 1, 1, 29);
    auto ew30 = network -> addElementWise(*l29 -> getOutput(0), *ew27 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l31 = convBnMish(network, weightMap, *ew30 -> getOutput(0), 128, 1, 1, 0, 31);
    auto l32 = convBnMish(network, weightMap, *l31 -> getOutput(0), 128, 3, 1, 1, 32);
    auto ew33 = network -> addElementWise(*l32 -> getOutput(0), *ew30 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l34 = convBnMish(network, weightMap, *ew33 -> getOutput(0), 128, 1, 1, 0, 34);
    auto l35 = convBnMish(network, weightMap, *l34 -> getOutput(0), 128, 3, 1, 1, 35);
    auto ew36 = network -> addElementWise(*l35 -> getOutput(0), *ew33 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l37 = convBnMish(network, weightMap, *ew36 -> getOutput(0), 128, 1, 1, 0, 37);
    auto l38 = convBnMish(network, weightMap, *l37 -> getOutput(0), 128, 3, 1, 1, 38);
    auto ew39 = network -> addElementWise(*l38 -> getOutput(0), *ew36 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l40 = convBnMish(network, weightMap, *ew39 -> getOutput(0), 128, 1, 1, 0, 40);
    auto l41 = convBnMish(network, weightMap, *l40 -> getOutput(0), 128, 3, 1, 1, 41);
    auto ew42 = network -> addElementWise(*l41 -> getOutput(0), *ew39 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l43 = convBnMish(network, weightMap, *ew42 -> getOutput(0), 128, 1, 1, 0, 43);
    auto l44 = convBnMish(network, weightMap, *l43 -> getOutput(0), 128, 3, 1, 1, 44);
    auto ew45 = network -> addElementWise(*l44 -> getOutput(0), *ew42 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l46 = convBnMish(network, weightMap, *ew45 -> getOutput(0), 128, 1, 1, 0, 46);

    ITensor* inputTensors47[] = {l46 -> getOutput(0), l19 -> getOutput(0)};
    auto cat47 = network -> addConcatenation(inputTensors47, 2);

    auto l48 = convBnMish(network, weightMap, *cat47 -> getOutput(0), 256, 1, 1, 0, 48);
    auto l49 = convBnMish(network, weightMap, *l48 -> getOutput(0), 512, 3, 2, 1, 49);
    auto l50 = convBnMish(network, weightMap, *l49 -> getOutput(0), 256, 1, 1, 0, 50);
    auto l51 = l49;
    auto l52 = convBnMish(network, weightMap, *l51 -> getOutput(0), 256, 1, 1, 0, 52);
    auto l53 = convBnMish(network, weightMap, *l52 -> getOutput(0), 256, 1, 1, 0, 53);
    auto l54 = convBnMish(network, weightMap, *l53 -> getOutput(0), 256, 3, 1, 1, 54);
    auto ew55 = network -> addElementWise(*l54 -> getOutput(0), *l52 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l56 = convBnMish(network, weightMap, *ew55 -> getOutput(0), 256, 1, 1, 0, 56);
    auto l57 = convBnMish(network, weightMap, *l56 -> getOutput(0), 256, 3, 1, 1, 57);
    auto ew58 = network -> addElementWise(*l57 -> getOutput(0), *ew55 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l59 = convBnMish(network, weightMap, *ew58 -> getOutput(0), 256, 1, 1, 0, 59);
    auto l60 = convBnMish(network, weightMap, *l59 -> getOutput(0), 256, 3, 1, 1, 60);
    auto ew61 = network -> addElementWise(*l60 -> getOutput(0), *ew58 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l62 = convBnMish(network, weightMap, *ew61 -> getOutput(0), 256, 1, 1, 0, 62);
    auto l63 = convBnMish(network, weightMap, *l62 -> getOutput(0), 256, 3, 1, 1, 63);
    auto ew64 = network -> addElementWise(*l63 -> getOutput(0), *ew61 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l65 = convBnMish(network, weightMap, *ew64 -> getOutput(0), 256, 1, 1, 0, 65);
    auto l66 = convBnMish(network, weightMap, *l65 -> getOutput(0), 256, 3, 1, 1, 66);
    auto ew67 = network -> addElementWise(*l66 -> getOutput(0), *ew64 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l68 = convBnMish(network, weightMap, *ew67 -> getOutput(0), 256, 1, 1, 0, 68);
    auto l69 = convBnMish(network, weightMap, *l68 -> getOutput(0), 256, 3, 1, 1, 69);
    auto ew70 = network -> addElementWise(*l69 -> getOutput(0), *ew67 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l71 = convBnMish(network, weightMap, *ew70 -> getOutput(0), 256, 1, 1, 0, 71);
    auto l72 = convBnMish(network, weightMap, *l71 -> getOutput(0), 256, 3, 1, 1, 72);
    auto ew73 = network -> addElementWise(*l72 -> getOutput(0), *ew70 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l74 = convBnMish(network, weightMap, *ew73 -> getOutput(0), 256, 1, 1, 0, 74);
    auto l75 = convBnMish(network, weightMap, *l74 -> getOutput(0), 256, 3, 1, 1, 75);
    auto ew76 = network -> addElementWise(*l75 -> getOutput(0), *ew73 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l77 = convBnMish(network, weightMap, *ew76 -> getOutput(0), 256, 1, 1, 0, 77);

    ITensor* inputTensors78[] = {l77 -> getOutput(0), l50 -> getOutput(0)};
    auto cat78 = network -> addConcatenation(inputTensors78, 2);

    auto l79 = convBnMish(network, weightMap, *cat78 -> getOutput(0), 512, 1, 1, 0, 79);
    auto l80 = convBnMish(network, weightMap, *l79 -> getOutput(0), 1024, 3, 2, 1, 80);
    auto l81 = convBnMish(network, weightMap, *l80 -> getOutput(0), 512, 1, 1, 0, 81);
    auto l82 = l80;
    auto l83 = convBnMish(network, weightMap, *l82 -> getOutput(0), 512, 1, 1, 0, 83);
    auto l84 = convBnMish(network, weightMap, *l83 -> getOutput(0), 512, 1, 1, 0, 84);
    auto l85 = convBnMish(network, weightMap, *l84 -> getOutput(0), 512, 3, 1, 1, 85);
    auto ew86 = network -> addElementWise(*l85 -> getOutput(0), *l83 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l87 = convBnMish(network, weightMap, *ew86 -> getOutput(0), 512, 1, 1, 0, 87);
    auto l88 = convBnMish(network, weightMap, *l87 -> getOutput(0), 512, 3, 1, 1, 88);
    auto ew89 = network -> addElementWise(*l88 -> getOutput(0), *ew86 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l90 = convBnMish(network, weightMap, *ew89 -> getOutput(0), 512, 1, 1, 0, 90);
    auto l91 = convBnMish(network, weightMap, *l90 -> getOutput(0), 512, 3, 1, 1, 91);
    auto ew92 = network -> addElementWise(*l91 -> getOutput(0), *ew89 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l93 = convBnMish(network, weightMap, *ew92 -> getOutput(0), 512, 1, 1, 0, 93);
    auto l94 = convBnMish(network, weightMap, *l93 -> getOutput(0), 512, 3, 1, 1, 94);
    auto ew95 = network -> addElementWise(*l94 -> getOutput(0), *ew92 -> getOutput(0), ElementWiseOperation::kSUM);
    auto l96 = convBnMish(network, weightMap, *ew95 -> getOutput(0), 512, 1, 1, 0, 96);

    ITensor* inputTensors97[] = {l96 -> getOutput(0), l81 -> getOutput(0)};
    
    auto cat97 = network -> addConcatenation(inputTensors97, 2);

    auto l98 = convBnMish(network, weightMap, *cat97 -> getOutput(0), 1024, 1, 1, 0, 98);

    // ----
    auto l99 = convBnMish(network, weightMap, *l98 -> getOutput(0), 512, 1, 1, 0, 99);
    auto l100 = l98;
    auto l101 = convBnMish(network, weightMap, *l100 -> getOutput(0), 512, 1, 1, 0, 101);
    auto l102 = convBnMish(network, weightMap, *l101 -> getOutput(0), 512, 3, 1, 1, 102);
    auto l103 = convBnMish(network, weightMap, *l102 -> getOutput(0), 512, 1, 1, 0, 103);

    auto pool104 = network -> addPoolingNd(*l103 -> getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
    pool104 -> setPaddingNd(DimsHW{2, 2});
    pool104 -> setStrideNd(DimsHW{1, 1});

    auto l105 = l103;

    auto pool106 = network -> addPoolingNd(*l105 -> getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
    pool106 -> setPaddingNd(DimsHW{4, 4});
    pool106 -> setStrideNd(DimsHW{1, 1});

    auto l107 = l103;

    auto pool108 = network -> addPoolingNd(*l107 -> getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
    pool108 -> setPaddingNd(DimsHW{6, 6});
    pool108 -> setStrideNd(DimsHW{1, 1});

    ITensor* inputTensors109[] = {pool108 -> getOutput(0), pool106 -> getOutput(0), pool104 -> getOutput(0), l103 -> getOutput(0)};
    auto cat109 = network -> addConcatenation(inputTensors109, 4);

    // ---- end spp

    auto l110 = convBnMish(network, weightMap, *cat109 -> getOutput(0), 512, 1, 1, 0, 110);
    auto l111 = convBnMish(network, weightMap, *l110 -> getOutput(0), 512, 3, 1, 1, 111);

    ITensor* inputTensors112[] =  { l111 -> getOutput(0), l99 -> getOutput(0) };
    auto cat112 = network -> addConcatenation(inputTensors112, 2);

    auto l113 = convBnMish(network, weightMap, *cat112 -> getOutput(0), 512, 1, 1, 0, 113);
    auto l114 = convBnMish(network, weightMap, *l113 -> getOutput(0), 256, 1, 1, 0, 114);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights upsamplewts115{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* upsample115 = network -> addDeconvolutionNd(*l114 -> getOutput(0), 256, DimsHW{2, 2}, upsamplewts115, emptywts);
    assert(upsample115);
    upsample115 -> setStrideNd(DimsHW{2, 2});
    upsample115 -> setNbGroups(256);
    weightMap["upsample115"] = upsamplewts115;

    auto l116 = l79;
    auto l117 = convBnMish(network, weightMap, *l116 -> getOutput(0), 256, 1, 1, 0, 117);

    ITensor* inputTensors118[] = {l117 -> getOutput(0), upsample115 -> getOutput(0)};
    auto cat118 = network -> addConcatenation(inputTensors118, 2);

    auto l119 = convBnMish(network, weightMap, *cat118 -> getOutput(0), 256, 1, 1, 0, 119);
    auto l120 = convBnMish(network, weightMap, *l119 -> getOutput(0), 256, 1, 1, 0, 120);
    auto l121 = l119;
    auto l122 = convBnMish(network, weightMap, *l121 -> getOutput(0), 256, 1, 1, 0, 122);
    auto l123 = convBnMish(network, weightMap, *l122 -> getOutput(0), 256, 3, 1, 1, 123);
    auto l124 = convBnMish(network, weightMap, *l123 -> getOutput(0), 256, 1, 1, 0, 124);
    auto l125 = convBnMish(network, weightMap, *l124 -> getOutput(0), 256, 3, 1, 1, 125);
    
    ITensor* inputTensors126[] = {l125 -> getOutput(0), l120 -> getOutput(0)};
    auto cat126 = network -> addConcatenation(inputTensors126, 2);

    auto l127 = convBnMish(network, weightMap, *cat126 -> getOutput(0), 256, 1, 1, 0, 127);
    auto l128 = convBnMish(network, weightMap, *l127 -> getOutput(0), 128, 1, 1, 0, 128);
    
    Weights upsamplewts129{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* upsample129 = network -> addDeconvolutionNd(*l128 -> getOutput(0), 128, DimsHW{2, 2}, upsamplewts129, emptywts);
    assert(upsample129);
    upsample129 -> setStrideNd(DimsHW{2, 2});
    upsample129 -> setNbGroups(128);

    auto l130 = l48;
    auto l131 = convBnMish(network, weightMap, *l130 -> getOutput(0), 128, 1, 1, 0, 131);

    ITensor* inputTensors132[] = {l131 -> getOutput(0), upsample129 -> getOutput(0)};
    auto cat132 = network -> addConcatenation(inputTensors132, 2);

    auto l133 = convBnMish(network, weightMap, *cat132 -> getOutput(0), 128, 1, 1, 0, 133);
    auto l134 = convBnMish(network, weightMap, *l133 -> getOutput(0), 128, 1, 1, 0, 134);
    auto l135 = l133;
    auto l136 = convBnMish(network, weightMap, *l135 -> getOutput(0), 128, 1, 1, 0, 136);
    auto l137 = convBnMish(network, weightMap, *l136 -> getOutput(0), 128, 3, 1, 1, 137);
    auto l138 = convBnMish(network, weightMap, *l137 -> getOutput(0), 128, 1, 1, 0, 138);
    auto l139 = convBnMish(network, weightMap, *l138 -> getOutput(0), 128, 3, 1, 1, 139);

    ITensor* inputTensors140[] = {l139 -> getOutput(0), l134 -> getOutput(0)};
    auto cat140 = network -> addConcatenation(inputTensors140, 2);

    auto l141 = convBnMish(network, weightMap, *cat140 -> getOutput(0), 128, 1, 1, 0, 141);

    // ---
    auto l142 = convBnMish(network, weightMap, *l141 -> getOutput(0), 256, 3, 1, 1, 142);
    IConvolutionLayer* conv143 = network -> addConvolutionNd(*l142 -> getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.143.Conv2d.weight"], weightMap["module_list.143.Conv2d.bias"]);
    assert(conv143);

    // 144 is yolo layer
    auto l145 = l141;
    auto l146 = convBnMish(network, weightMap, *l145 -> getOutput(0), 256, 3, 2, 1, 146);

    ITensor* inputTensors147[] = {l146 -> getOutput(0), l127 -> getOutput(0)};
    auto cat147 = network -> addConcatenation(inputTensors147, 2);

    auto l148 = convBnMish(network, weightMap, *cat147 -> getOutput(0), 256, 1, 1, 0, 148);
    auto l149 = convBnMish(network, weightMap, *l148 -> getOutput(0), 256, 1, 1, 0, 149);
    auto l150 = l148;
    auto l151 = convBnMish(network, weightMap, *l150 -> getOutput(0), 256, 1, 1, 0, 151);
    auto l152 = convBnMish(network, weightMap, *l151 -> getOutput(0), 256, 3, 1, 1, 152);
    auto l153 = convBnMish(network, weightMap, *l152 -> getOutput(0), 256, 1, 1, 0, 153);
    auto l154 = convBnMish(network, weightMap, *l153 -> getOutput(0), 256, 3, 1, 1, 154);

    ITensor* inputTensors155[] = {l154 -> getOutput(0), l149 -> getOutput(0)};
    auto cat155 = network -> addConcatenation(inputTensors155, 2);

    auto l156 = convBnMish(network, weightMap, *cat155 -> getOutput(0), 256, 1, 1, 0, 156);
    auto l157 = convBnMish(network, weightMap, *l156 -> getOutput(0), 512, 3, 1, 1, 157);   
    IConvolutionLayer* conv158 = network -> addConvolutionNd(*l157 -> getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.158.Conv2d.weight"], weightMap["module_list.158.Conv2d.bias"]);
    assert(conv158);
    // 159 is yolo layer

    auto l160 = l156;
    auto l161 = convBnMish(network, weightMap, *l160 -> getOutput(0), 512, 3, 2, 1, 161);

    ITensor* inputTensors162[] = {l161 -> getOutput(0), l113 -> getOutput(0)};
    auto cat162 = network -> addConcatenation(inputTensors162, 2);

    auto l163 = convBnMish(network, weightMap, *cat162 -> getOutput(0), 512, 1, 1, 0, 163); 
    auto l164 = convBnMish(network, weightMap, *l163 -> getOutput(0), 512, 1, 1, 0, 164); 
    auto l165 = l163;
    auto l166 = convBnMish(network, weightMap, *l165 -> getOutput(0), 512, 1, 1, 0, 166); 
    auto l167 = convBnMish(network, weightMap, *l166 -> getOutput(0), 512, 3, 1, 1, 167);
    auto l168 = convBnMish(network, weightMap, *l167 -> getOutput(0), 512, 1, 1, 0, 168);
    auto l169 = convBnMish(network, weightMap, *l168 -> getOutput(0), 512, 3, 1, 1, 169);

    ITensor* inputTensors170[] = {l169 -> getOutput(0), l164 -> getOutput(0)};
    auto cat170 = network -> addConcatenation(inputTensors170, 2);

    auto l171 = convBnMish(network, weightMap, *cat170 -> getOutput(0), 512, 1, 1, 0, 171);
    auto l172 = convBnMish(network, weightMap, *l171 -> getOutput(0), 1024, 3, 1, 1, 172);

    IConvolutionLayer* conv173 = network -> addConvolutionNd(*l172 -> getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.173.Conv2d.weight"], weightMap["module_list.173.Conv2d.bias"]);
    assert(conv173);
    // 174 is yolo layer

    // add yolo plugin
    auto creator = getPluginRegistry() -> getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator -> getFieldNames();
    IPluginV2* pluginObj = creator -> createPlugin("yololayer", pluginData);
    ITensor* inputTensorsYolo[] = {conv143 -> getOutput(0), conv158 -> getOutput(0), conv173 -> getOutput(0)};
    auto yolo = network -> addPluginV2(inputTensorsYolo, 3, *pluginObj);

    yolo -> getOutput(0) -> setName(OUTPUT_BLOB_NAME);
    network -> markOutput(*yolo -> getOutput(0));

    // Build engine
    builder -> setMaxBatchSize(maxBatchSize);
    config -> setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config -> setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building tensorrt engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder -> buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network -> destroy();

    
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // create builder config
    IBuilderConfig* config = builder -> createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the trt engine
    (*modelStream) = engine -> serialize();
    
    // Close everything down
    engine -> destroy();
    builder -> destroy();
    config -> destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char* p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file -> d_name, ".") != 0 &&
            strcmp(p_file -> d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

int main(int argc, char** argv){
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("yolov4csp.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file("yolov4csp.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov4 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img);
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[b * 3 * INPUT_H * INPUT_W + i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], BBOX_CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                float *p = (float*)&res[j];
                for (size_t k = 0; k < 7; k++) {
                   std::cout << p[k] << ", ";
                }
                std::cout << std::endl;
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    //Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << i / 10 << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}