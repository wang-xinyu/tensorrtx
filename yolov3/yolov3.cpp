#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "logging.h"
#include "yololayer.h"
#include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov3.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Yeah I am stupid, I just want to expand the complete arch of darknet..
    auto lr0 = convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto lr1 = convBnLeaky(network, weightMap, *lr0->getOutput(0), 64, 3, 2, 1, 1);
    auto lr2 = convBnLeaky(network, weightMap, *lr1->getOutput(0), 32, 1, 1, 0, 2);
    auto lr3 = convBnLeaky(network, weightMap, *lr2->getOutput(0), 64, 3, 1, 1, 3);
    auto ew4 = network->addElementWise(*lr3->getOutput(0), *lr1->getOutput(0), ElementWiseOperation::kSUM);
    auto lr5 = convBnLeaky(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
    auto lr6 = convBnLeaky(network, weightMap, *lr5->getOutput(0), 64, 1, 1, 0, 6);
    auto lr7 = convBnLeaky(network, weightMap, *lr6->getOutput(0), 128, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*lr7->getOutput(0), *lr5->getOutput(0), ElementWiseOperation::kSUM);
    auto lr9 = convBnLeaky(network, weightMap, *ew8->getOutput(0), 64, 1, 1, 0, 9);
    auto lr10 = convBnLeaky(network, weightMap, *lr9->getOutput(0), 128, 3, 1, 1, 10);
    auto ew11 = network->addElementWise(*lr10->getOutput(0), *ew8->getOutput(0), ElementWiseOperation::kSUM);
    auto lr12 = convBnLeaky(network, weightMap, *ew11->getOutput(0), 256, 3, 2, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 128, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 256, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*lr14->getOutput(0), *lr12->getOutput(0), ElementWiseOperation::kSUM);
    auto lr16 = convBnLeaky(network, weightMap, *ew15->getOutput(0), 128, 1, 1, 0, 16);
    auto lr17 = convBnLeaky(network, weightMap, *lr16->getOutput(0), 256, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*lr17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto lr19 = convBnLeaky(network, weightMap, *ew18->getOutput(0), 128, 1, 1, 0, 19);
    auto lr20 = convBnLeaky(network, weightMap, *lr19->getOutput(0), 256, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*lr20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    auto lr22 = convBnLeaky(network, weightMap, *ew21->getOutput(0), 128, 1, 1, 0, 22);
    auto lr23 = convBnLeaky(network, weightMap, *lr22->getOutput(0), 256, 3, 1, 1, 23);
    auto ew24 = network->addElementWise(*lr23->getOutput(0), *ew21->getOutput(0), ElementWiseOperation::kSUM);
    auto lr25 = convBnLeaky(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
    auto lr26 = convBnLeaky(network, weightMap, *lr25->getOutput(0), 256, 3, 1, 1, 26);
    auto ew27 = network->addElementWise(*lr26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
    auto lr28 = convBnLeaky(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
    auto lr29 = convBnLeaky(network, weightMap, *lr28->getOutput(0), 256, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*lr29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
    auto lr31 = convBnLeaky(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto lr32 = convBnLeaky(network, weightMap, *lr31->getOutput(0), 256, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*lr32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto lr34 = convBnLeaky(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto lr35 = convBnLeaky(network, weightMap, *lr34->getOutput(0), 256, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*lr35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto lr37 = convBnLeaky(network, weightMap, *ew36->getOutput(0), 512, 3, 2, 1, 37);
    auto lr38 = convBnLeaky(network, weightMap, *lr37->getOutput(0), 256, 1, 1, 0, 38);
    auto lr39 = convBnLeaky(network, weightMap, *lr38->getOutput(0), 512, 3, 1, 1, 39);
    auto ew40 = network->addElementWise(*lr39->getOutput(0), *lr37->getOutput(0), ElementWiseOperation::kSUM);
    auto lr41 = convBnLeaky(network, weightMap, *ew40->getOutput(0), 256, 1, 1, 0, 41);
    auto lr42 = convBnLeaky(network, weightMap, *lr41->getOutput(0), 512, 3, 1, 1, 42);
    auto ew43 = network->addElementWise(*lr42->getOutput(0), *ew40->getOutput(0), ElementWiseOperation::kSUM);
    auto lr44 = convBnLeaky(network, weightMap, *ew43->getOutput(0), 256, 1, 1, 0, 44);
    auto lr45 = convBnLeaky(network, weightMap, *lr44->getOutput(0), 512, 3, 1, 1, 45);
    auto ew46 = network->addElementWise(*lr45->getOutput(0), *ew43->getOutput(0), ElementWiseOperation::kSUM);
    auto lr47 = convBnLeaky(network, weightMap, *ew46->getOutput(0), 256, 1, 1, 0, 47);
    auto lr48 = convBnLeaky(network, weightMap, *lr47->getOutput(0), 512, 3, 1, 1, 48);
    auto ew49 = network->addElementWise(*lr48->getOutput(0), *ew46->getOutput(0), ElementWiseOperation::kSUM);
    auto lr50 = convBnLeaky(network, weightMap, *ew49->getOutput(0), 256, 1, 1, 0, 50);
    auto lr51 = convBnLeaky(network, weightMap, *lr50->getOutput(0), 512, 3, 1, 1, 51);
    auto ew52 = network->addElementWise(*lr51->getOutput(0), *ew49->getOutput(0), ElementWiseOperation::kSUM);
    auto lr53 = convBnLeaky(network, weightMap, *ew52->getOutput(0), 256, 1, 1, 0, 53);
    auto lr54 = convBnLeaky(network, weightMap, *lr53->getOutput(0), 512, 3, 1, 1, 54);
    auto ew55 = network->addElementWise(*lr54->getOutput(0), *ew52->getOutput(0), ElementWiseOperation::kSUM);
    auto lr56 = convBnLeaky(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
    auto lr57 = convBnLeaky(network, weightMap, *lr56->getOutput(0), 512, 3, 1, 1, 57);
    auto ew58 = network->addElementWise(*lr57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);
    auto lr59 = convBnLeaky(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
    auto lr60 = convBnLeaky(network, weightMap, *lr59->getOutput(0), 512, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*lr60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
    auto lr62 = convBnLeaky(network, weightMap, *ew61->getOutput(0), 1024, 3, 2, 1, 62);
    auto lr63 = convBnLeaky(network, weightMap, *lr62->getOutput(0), 512, 1, 1, 0, 63);
    auto lr64 = convBnLeaky(network, weightMap, *lr63->getOutput(0), 1024, 3, 1, 1, 64);
    auto ew65 = network->addElementWise(*lr64->getOutput(0), *lr62->getOutput(0), ElementWiseOperation::kSUM);
    auto lr66 = convBnLeaky(network, weightMap, *ew65->getOutput(0), 512, 1, 1, 0, 66);
    auto lr67 = convBnLeaky(network, weightMap, *lr66->getOutput(0), 1024, 3, 1, 1, 67);
    auto ew68 = network->addElementWise(*lr67->getOutput(0), *ew65->getOutput(0), ElementWiseOperation::kSUM);
    auto lr69 = convBnLeaky(network, weightMap, *ew68->getOutput(0), 512, 1, 1, 0, 69);
    auto lr70 = convBnLeaky(network, weightMap, *lr69->getOutput(0), 1024, 3, 1, 1, 70);
    auto ew71 = network->addElementWise(*lr70->getOutput(0), *ew68->getOutput(0), ElementWiseOperation::kSUM);
    auto lr72 = convBnLeaky(network, weightMap, *ew71->getOutput(0), 512, 1, 1, 0, 72);
    auto lr73 = convBnLeaky(network, weightMap, *lr72->getOutput(0), 1024, 3, 1, 1, 73);
    auto ew74 = network->addElementWise(*lr73->getOutput(0), *ew71->getOutput(0), ElementWiseOperation::kSUM);
    auto lr75 = convBnLeaky(network, weightMap, *ew74->getOutput(0), 512, 1, 1, 0, 75);
    auto lr76 = convBnLeaky(network, weightMap, *lr75->getOutput(0), 1024, 3, 1, 1, 76);
    auto lr77 = convBnLeaky(network, weightMap, *lr76->getOutput(0), 512, 1, 1, 0, 77);
    auto lr78 = convBnLeaky(network, weightMap, *lr77->getOutput(0), 1024, 3, 1, 1, 78);
    auto lr79 = convBnLeaky(network, weightMap, *lr78->getOutput(0), 512, 1, 1, 0, 79);
    auto lr80 = convBnLeaky(network, weightMap, *lr79->getOutput(0), 1024, 3, 1, 1, 80);
    IConvolutionLayer* conv81 = network->addConvolutionNd(*lr80->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.81.Conv2d.weight"], weightMap["module_list.81.Conv2d.bias"]);
    assert(conv81);
    // 82 is yolo
    auto l83 = lr79;
    auto lr84 = convBnLeaky(network, weightMap, *l83->getOutput(0), 256, 1, 1, 0, 84);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts85{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv85 = network->addDeconvolutionNd(*lr84->getOutput(0), 256, DimsHW{2, 2}, deconvwts85, emptywts);
    assert(deconv85);
    deconv85->setStrideNd(DimsHW{2, 2});
    deconv85->setNbGroups(256);
    weightMap["deconv85"] = deconvwts85;

    ITensor* inputTensors[] = {deconv85->getOutput(0), ew61->getOutput(0)};
    auto cat86 = network->addConcatenation(inputTensors, 2);
    auto lr87 = convBnLeaky(network, weightMap, *cat86->getOutput(0), 256, 1, 1, 0, 87);
    auto lr88 = convBnLeaky(network, weightMap, *lr87->getOutput(0), 512, 3, 1, 1, 88);
    auto lr89 = convBnLeaky(network, weightMap, *lr88->getOutput(0), 256, 1, 1, 0, 89);
    auto lr90 = convBnLeaky(network, weightMap, *lr89->getOutput(0), 512, 3, 1, 1, 90);
    auto lr91 = convBnLeaky(network, weightMap, *lr90->getOutput(0), 256, 1, 1, 0, 91);
    auto lr92 = convBnLeaky(network, weightMap, *lr91->getOutput(0), 512, 3, 1, 1, 92);
    IConvolutionLayer* conv93 = network->addConvolutionNd(*lr92->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.93.Conv2d.weight"], weightMap["module_list.93.Conv2d.bias"]);
    assert(conv93);
    // 94 is yolo
    auto l95 = lr91;
    auto lr96 = convBnLeaky(network, weightMap, *l95->getOutput(0), 128, 1, 1, 0, 96);
    Weights deconvwts97{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv97 = network->addDeconvolutionNd(*lr96->getOutput(0), 128, DimsHW{2, 2}, deconvwts97, emptywts);
    assert(deconv97);
    deconv97->setStrideNd(DimsHW{2, 2});
    deconv97->setNbGroups(128);
    ITensor* inputTensors1[] = {deconv97->getOutput(0), ew36->getOutput(0)};
    auto cat98 = network->addConcatenation(inputTensors1, 2);
    auto lr99 = convBnLeaky(network, weightMap, *cat98->getOutput(0), 128, 1, 1, 0, 99);
    auto lr100 = convBnLeaky(network, weightMap, *lr99->getOutput(0), 256, 3, 1, 1, 100);
    auto lr101 = convBnLeaky(network, weightMap, *lr100->getOutput(0), 128, 1, 1, 0, 101);
    auto lr102 = convBnLeaky(network, weightMap, *lr101->getOutput(0), 256, 3, 1, 1, 102);
    auto lr103 = convBnLeaky(network, weightMap, *lr102->getOutput(0), 128, 1, 1, 0, 103);
    auto lr104 = convBnLeaky(network, weightMap, *lr103->getOutput(0), 256, 3, 1, 1, 104);
    IConvolutionLayer* conv105 = network->addConvolutionNd(*lr104->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.105.Conv2d.weight"], weightMap["module_list.105.Conv2d.bias"]);
    assert(conv105);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

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
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
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
        free((void*) (mem.second.values));
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

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("yolov3.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file("yolov3.engine", std::ios::binary);
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
        std::cerr << "./yolov3 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov3 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    int fcount = 0;
    for (auto f: file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + f);
        if (img.empty()) continue;
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<Yolo::Detection> res;
        nms(res, prob);
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite("_" + f, img);
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
