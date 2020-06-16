#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "decode.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1  // currently, only support BATCH=1

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = decodeplugin::INPUT_H;
static const int INPUT_W = decodeplugin::INPUT_W;
static const int DETECTION_SIZE = sizeof(decodeplugin::Detection) / sizeof(float);
static const int OUTPUT_SIZE = (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2  * DETECTION_SIZE + 1;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect_adapt_landmark(cv::Mat& img, float bbox[4], float lmk[10]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (INPUT_H - r_w * img.rows) / 2) / r_w;
        }
    } else {
        l = (bbox[0] - (INPUT_W - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (INPUT_W - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < 10; i += 2) {
            lmk[i] = (lmk[i] - (INPUT_W - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
        }
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0], rbox[0]), //left
        std::min(lbox[2], rbox[2]), //right
        std::max(lbox[1], rbox[1]), //top
        std::min(lbox[3], rbox[3]), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) -interBoxS + 0.000001f);
}

bool cmp(decodeplugin::Detection& a, decodeplugin::Detection& b) {
    return a.class_confidence > b.class_confidence;
}

void nms(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.4) {
    std::vector<decodeplugin::Detection> dets;
    for (int i = 0; i < output[0]; i++) {
        if (output[DETECTION_SIZE * i + 1 + 4] <= 0.1) continue;
        decodeplugin::Detection det;
        memcpy(&det, &output[DETECTION_SIZE * i + 1], sizeof(decodeplugin::Detection));
        dets.push_back(det);
    }
    std::sort(dets.begin(), dets.end(), cmp);
    if (dets.size() > 5000) dets.erase(dets.begin() + 5000, dets.end());
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
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
    float *gamma = (float*)weightMap[lname + "_gamma"].values;
    float *beta = (float*)weightMap[lname + "_beta"].values;
    float *mean = (float*)weightMap[lname + "_moving_mean"].values;
    float *var = (float*)weightMap[lname + "_moving_var"].values;
    int len = weightMap[lname + "_moving_var"].count;

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

ILayer* convBnRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int k, int s, int p, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = network->addConvolutionNd(input, num_filters, DimsHW{k, k}, weightMap[lname + "_conv2d_weight"], emptywts);
    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    conv->setNbGroups(g);
    auto bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + "_batchnorm", 1e-3);
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

ILayer* convBiasBnRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int k, int s, int p, std::string lname) {
    IConvolutionLayer* conv = network->addConvolutionNd(input, num_filters, DimsHW{k, k}, weightMap[lname + "_weight"], weightMap[lname + "_bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    auto bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + "_bn", 2e-5);
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

ILayer* head(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    auto conv1 = network->addConvolutionNd(input, 32, DimsHW{3, 3}, weightMap[lname + "_conv1_weight"], weightMap[lname + "_conv1_bias"]);
    assert(conv1);
    conv1->setPaddingNd(DimsHW{1, 1});
    auto conv1bn = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_conv1_bn", 2e-5);

    auto ctxconv1 = convBiasBnRelu(network, weightMap, input, 16, 3, 1, 1, lname + "_context_conv1");

    auto ctxconv2 = network->addConvolutionNd(*ctxconv1->getOutput(0), 16, DimsHW{3, 3}, weightMap[lname + "_context_conv2_weight"], weightMap[lname + "_context_conv2_bias"]);
    assert(ctxconv2);
    ctxconv2->setPaddingNd(DimsHW{1, 1});
    auto ctxconv2bn = addBatchNorm2d(network, weightMap, *ctxconv2->getOutput(0), lname + "_context_conv2_bn", 2e-5);

    auto ctxconv3_1 = convBiasBnRelu(network, weightMap, *ctxconv1->getOutput(0), 16, 3, 1, 1, lname + "_context_conv3_1");
    auto ctxconv3_2 = network->addConvolutionNd(*ctxconv3_1->getOutput(0), 16, DimsHW{3, 3}, weightMap[lname + "_context_conv3_2_weight"], weightMap[lname + "_context_conv3_2_bias"]);
    assert(ctxconv3_2);
    ctxconv3_2->setPaddingNd(DimsHW{1, 1});
    auto ctxconv3_2bn = addBatchNorm2d(network, weightMap, *ctxconv3_2->getOutput(0), lname + "_context_conv3_2_bn", 2e-5);

    ITensor* inputTensors[] = {conv1bn->getOutput(0), ctxconv2bn->getOutput(0), ctxconv3_2bn->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 3);
    assert(cat);

    IActivationLayer* relu = network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

ILayer* reshapeSoftmax(INetworkDefinition *network, ITensor& input, int c) {
    auto re1 = network->addShuffle(input);
    assert(re1);
    re1->setReshapeDimensions(Dims3(c / 2, -1, 0));

    auto sm = network->addSoftMax(*re1->getOutput(0));
    assert(sm);

    auto re2 = network->addShuffle(*sm->getOutput(0));
    assert(re2);
    re2->setReshapeDimensions(Dims3(c, -1, 0));

    return re2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../retinafaceAntiCov.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto conv1 = convBnRelu(network, weightMap, *data, 16, 3, 2, 1, 1, "conv_1");
    auto conv2 = convBnRelu(network, weightMap, *conv1->getOutput(0), 32, 1, 1, 0, 1, "conv_2");
    auto conv3dw = convBnRelu(network, weightMap, *conv2->getOutput(0), 32, 3, 2, 1, 32, "conv_3_dw");
    auto conv3 = convBnRelu(network, weightMap, *conv3dw->getOutput(0), 32, 1, 1, 0, 1, "conv_3");
    auto conv4dw = convBnRelu(network, weightMap, *conv3->getOutput(0), 32, 3, 1, 1, 32, "conv_4_dw");
    auto conv4 = convBnRelu(network, weightMap, *conv4dw->getOutput(0), 32, 1, 1, 0, 1, "conv_4");
    auto conv5dw = convBnRelu(network, weightMap, *conv4->getOutput(0), 32, 3, 2, 1, 32, "conv_5_dw");
    auto conv5 = convBnRelu(network, weightMap, *conv5dw->getOutput(0), 64, 1, 1, 0, 1, "conv_5");
    auto conv6dw = convBnRelu(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, 64, "conv_6_dw");
    auto conv6 = convBnRelu(network, weightMap, *conv6dw->getOutput(0), 64, 1, 1, 0, 1, "conv_6");
    // conv6 to c1
    auto conv7dw = convBnRelu(network, weightMap, *conv6->getOutput(0), 64, 3, 2, 1, 64, "conv_7_dw");
    auto conv7 = convBnRelu(network, weightMap, *conv7dw->getOutput(0), 128, 1, 1, 0, 1, "conv_7");
    auto conv8dw = convBnRelu(network, weightMap, *conv7->getOutput(0), 128, 3, 1, 1, 128, "conv_8_dw");
    auto conv8 = convBnRelu(network, weightMap, *conv8dw->getOutput(0), 128, 1, 1, 0, 1, "conv_8");
    auto conv9dw = convBnRelu(network, weightMap, *conv8->getOutput(0), 128, 3, 1, 1, 128, "conv_9_dw");
    auto conv9 = convBnRelu(network, weightMap, *conv9dw->getOutput(0), 128, 1, 1, 0, 1, "conv_9");
    auto conv10dw = convBnRelu(network, weightMap, *conv9->getOutput(0), 128, 3, 1, 1, 128, "conv_10_dw");
    auto conv10 = convBnRelu(network, weightMap, *conv10dw->getOutput(0), 128, 1, 1, 0, 1, "conv_10");
    auto conv11dw = convBnRelu(network, weightMap, *conv10->getOutput(0), 128, 3, 1, 1, 128, "conv_11_dw");
    auto conv11 = convBnRelu(network, weightMap, *conv11dw->getOutput(0), 128, 1, 1, 0, 1, "conv_11");
    auto conv12dw = convBnRelu(network, weightMap, *conv11->getOutput(0), 128, 3, 1, 1, 128, "conv_12_dw");
    auto conv12 = convBnRelu(network, weightMap, *conv12dw->getOutput(0), 128, 1, 1, 0, 1, "conv_12");
    // conv12 to c2
    auto conv13dw = convBnRelu(network, weightMap, *conv12->getOutput(0), 128, 3, 2, 1, 128, "conv_13_dw");
    auto conv13 = convBnRelu(network, weightMap, *conv13dw->getOutput(0), 256, 1, 1, 0, 1, "conv_13");
    auto conv14dw = convBnRelu(network, weightMap, *conv13->getOutput(0), 256, 3, 1, 1, 256, "conv_14_dw");
    auto conv14 = convBnRelu(network, weightMap, *conv14dw->getOutput(0), 256, 1, 1, 0, 1, "conv_14");
    auto conv_final = convBnRelu(network, weightMap, *conv14->getOutput(0), 256, 1, 1, 0, 1, "conv_final");
    // convfinal to c3

    auto rf_c3_lateral = convBiasBnRelu(network, weightMap, *conv_final->getOutput(0), 64, 1, 1, 0, "rf_c3_lateral");
    auto rf_head_s32 = head(network, weightMap, *rf_c3_lateral->getOutput(0), "rf_head_stride32");
    ILayer *cls_score_s32 = network->addConvolutionNd(*rf_head_s32->getOutput(0), 4, DimsHW{1, 1}, weightMap["face_rpn_cls_score_stride32_weight"], weightMap["face_rpn_cls_score_stride32_bias"]);
    cls_score_s32 = reshapeSoftmax(network, *cls_score_s32->getOutput(0), 4);
    auto bbox_s32 = network->addConvolutionNd(*rf_head_s32->getOutput(0), 8, DimsHW{1, 1}, weightMap["face_rpn_bbox_pred_stride32_weight"], weightMap["face_rpn_bbox_pred_stride32_bias"]);
    auto landmark_s32 = network->addConvolutionNd(*rf_head_s32->getOutput(0), 20, DimsHW{1, 1}, weightMap["face_rpn_landmark_pred_stride32_weight"], weightMap["face_rpn_landmark_pred_stride32_bias"]);
    auto rf_head2_s32 = head(network, weightMap, *rf_c3_lateral->getOutput(0), "rf_head2_stride32");
    ILayer *type_score_s32 = network->addConvolutionNd(*rf_head2_s32->getOutput(0), 6, DimsHW{1, 1}, weightMap["face_rpn_type_score_stride32_weight"], weightMap["face_rpn_type_score_stride32_bias"]);
    type_score_s32 = reshapeSoftmax(network, *type_score_s32->getOutput(0), 6);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 64 * 2 * 2));
    for (int i = 0; i < 64 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts{DataType::kFLOAT, deval, 64 * 2 * 2};
    IDeconvolutionLayer* c3_deconv = network->addDeconvolutionNd(*rf_c3_lateral->getOutput(0), 64, DimsHW{2, 2}, deconvwts, emptywts);
    assert(c3_deconv);
    c3_deconv->setStrideNd(DimsHW{2, 2});
    c3_deconv->setNbGroups(64);
    weightMap["c3_deconv"] = deconvwts;
    auto rf_c2_lateral = convBiasBnRelu(network, weightMap, *conv12->getOutput(0), 64, 1, 1, 0, "rf_c2_lateral");
    auto plus0 = network->addElementWise(*c3_deconv->getOutput(0), *rf_c2_lateral->getOutput(0), ElementWiseOperation::kSUM);
    auto rf_c2_aggr = convBiasBnRelu(network, weightMap, *plus0->getOutput(0), 64, 3, 1, 1, "rf_c2_aggr");
    auto rf_head_s16 = head(network, weightMap, *rf_c2_aggr->getOutput(0), "rf_head_stride16");
    ILayer *cls_score_s16 = network->addConvolutionNd(*rf_head_s16->getOutput(0), 4, DimsHW{1, 1}, weightMap["face_rpn_cls_score_stride16_weight"], weightMap["face_rpn_cls_score_stride16_bias"]);
    cls_score_s16 = reshapeSoftmax(network, *cls_score_s16->getOutput(0), 4);
    auto bbox_s16 = network->addConvolutionNd(*rf_head_s16->getOutput(0), 8, DimsHW{1, 1}, weightMap["face_rpn_bbox_pred_stride16_weight"], weightMap["face_rpn_bbox_pred_stride16_bias"]);
    auto landmark_s16 = network->addConvolutionNd(*rf_head_s16->getOutput(0), 20, DimsHW{1, 1}, weightMap["face_rpn_landmark_pred_stride16_weight"], weightMap["face_rpn_landmark_pred_stride16_bias"]);
    auto rf_head2_s16 = head(network, weightMap, *rf_c2_aggr->getOutput(0), "rf_head2_stride16");
    ILayer *type_score_s16 = network->addConvolutionNd(*rf_head2_s16->getOutput(0), 6, DimsHW{1, 1}, weightMap["face_rpn_type_score_stride16_weight"], weightMap["face_rpn_type_score_stride16_bias"]);
    type_score_s16 = reshapeSoftmax(network, *type_score_s16->getOutput(0), 6);

    IDeconvolutionLayer* c2_deconv = network->addDeconvolutionNd(*rf_c2_aggr->getOutput(0), 64, DimsHW{2, 2}, deconvwts, emptywts);
    assert(c2_deconv);
    c2_deconv->setStrideNd(DimsHW{2, 2});
    c2_deconv->setNbGroups(64);
    auto rf_c1_red = convBiasBnRelu(network, weightMap, *conv6->getOutput(0), 64, 1, 1, 0, "rf_c1_red_conv");
    auto plus1 = network->addElementWise(*c2_deconv->getOutput(0), *rf_c1_red->getOutput(0), ElementWiseOperation::kSUM);
    auto rf_c1_aggr = convBiasBnRelu(network, weightMap, *plus1->getOutput(0), 64, 3, 1, 1, "rf_c1_aggr");
    auto rf_head_s8 = head(network, weightMap, *rf_c1_aggr->getOutput(0), "rf_head_stride8");
    ILayer *cls_score_s8 = network->addConvolutionNd(*rf_head_s8->getOutput(0), 4, DimsHW{1, 1}, weightMap["face_rpn_cls_score_stride8_weight"], weightMap["face_rpn_cls_score_stride8_bias"]);
    cls_score_s8 = reshapeSoftmax(network, *cls_score_s8->getOutput(0), 4);
    auto bbox_s8 = network->addConvolutionNd(*rf_head_s8->getOutput(0), 8, DimsHW{1, 1}, weightMap["face_rpn_bbox_pred_stride8_weight"], weightMap["face_rpn_bbox_pred_stride8_bias"]);
    auto landmark_s8 = network->addConvolutionNd(*rf_head_s8->getOutput(0), 20, DimsHW{1, 1}, weightMap["face_rpn_landmark_pred_stride8_weight"], weightMap["face_rpn_landmark_pred_stride8_bias"]);
    auto rf_head2_s8 = head(network, weightMap, *rf_c1_aggr->getOutput(0), "rf_head2_stride8");
    ILayer *type_score_s8 = network->addConvolutionNd(*rf_head2_s8->getOutput(0), 6, DimsHW{1, 1}, weightMap["face_rpn_type_score_stride8_weight"], weightMap["face_rpn_type_score_stride8_bias"]);
    type_score_s8 = reshapeSoftmax(network, *type_score_s8->getOutput(0), 6);

    ITensor* inputTensors_s32[] = {cls_score_s32->getOutput(0), bbox_s32->getOutput(0), landmark_s32->getOutput(0), type_score_s32->getOutput(0)};
    auto cat_s32 = network->addConcatenation(inputTensors_s32, 4);
    assert(cat_s32);

    ITensor* inputTensors_s16[] = {cls_score_s16->getOutput(0), bbox_s16->getOutput(0), landmark_s16->getOutput(0), type_score_s16->getOutput(0)};
    auto cat_s16 = network->addConcatenation(inputTensors_s16, 4);
    assert(cat_s16);

    ITensor* inputTensors_s8[] = {cls_score_s8->getOutput(0), bbox_s8->getOutput(0), landmark_s8->getOutput(0), type_score_s8->getOutput(0)};
    auto cat_s8 = network->addConcatenation(inputTensors_s8, 4);
    assert(cat_s8);

    auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
    PluginFieldCollection pfc;
    IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
    ITensor* inputTensors[] = {cat_s8->getOutput(0), cat_s16->getOutput(0), cat_s32->getOutput(0)};
    auto decodelayer = network->addPluginV2(inputTensors, 3, *pluginObj);
    assert(decodelayer);

    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
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
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("retinafaceAntiCov.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("retinafaceAntiCov.engine", std::ios::binary);
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
        std::cerr << "./retinafaceAntiCov -s  // serialize model to plan file" << std::endl;
        std::cerr << "./retinafaceAntiCov -d  // deserialize plan file and run inference" << std::endl;
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

    cv::Mat img = cv::imread("test.jpg");
    cv::Mat pr_img = preprocess_img(img);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)pr_img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)pr_img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)pr_img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<decodeplugin::Detection> res;
    nms(res, prob);

    for (size_t j = 0; j < res.size(); j++) {
        //if (res[j].class_confidence < 0.1) continue;
        cv::Rect r = get_rect_adapt_landmark(img, res[j].bbox, res[j].landmark);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, "face: " + std::to_string((int)(res[j].class_confidence * 100)) + "%", cv::Point(r.x, r.y + 20), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 1);
        for (int k = 0; k < 10; k += 2) {
            cv::circle(img, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1, cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
        }
        cv::putText(img, "mask: " + std::to_string((int)(res[j].mask_confidence * 100)) + "%", cv::Point(r.x, r.y + 40), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x00, 0x00, 0xFF), 1);
    }
    cv::imwrite("out.jpg", img);

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
