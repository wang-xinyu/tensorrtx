#ifndef YOLOV7_COMMON_H_
#define YOLOV7_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"
using namespace nvinfer1;



cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}
//=================IOU===============
float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);//
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

float giou(float lbox[4], float rbox[4]) {

    float interBoxMax[] = {
        (std::min)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::max)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::min)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::max)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    float interBoxMaxS = (interBoxMax[1] - interBoxMax[0]) * (interBoxMax[3] - interBoxMax[2]);


    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);//
    float iou1 = interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);//正常计算IOU

    float iou2 = (interBoxMaxS - (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS)) / interBoxMaxS;

    return iou1 - iou2;
}

float diou(float lbox[4], float rbox[4]) {

    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);//
    float iou = interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);

    float interBoxMax[] = {
        (std::min)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::max)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::min)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::max)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };


    float d_center1 = (lbox[0] - rbox[0]) * (lbox[0] - rbox[0]) + (lbox[1] - rbox[1]) * (lbox[1] - rbox[1]);

    float d_center2 = (interBoxMax[0] - interBoxMax[2]) * (interBoxMax[0] - interBoxMax[2])
        + (interBoxMax[1] - interBoxMax[3]) * (interBoxMax[1] - interBoxMax[3]);

    return iou - d_center1 / d_center2;
}

float ciou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);//
    float iou = interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);

    float interBoxMax[] = {
        (std::min)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::max)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::min)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::max)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };


    float d_center1 = (lbox[0] - rbox[0]) * (lbox[0] - rbox[0]) + (lbox[1] - rbox[1]) * (lbox[1] - rbox[1]);

    float d_center2 = (interBoxMax[0] - interBoxMax[2]) * (interBoxMax[0] - interBoxMax[2])
        + (interBoxMax[1] - interBoxMax[3]) * (interBoxMax[1] - interBoxMax[3]);

    float v = 4 * (atan(lbox[2] / lbox[3]) * atan(rbox[2] / rbox[3])) * (atan(lbox[2] / lbox[3]) * atan(rbox[2] / rbox[3])) / (acos(-1) * acos(-1));
    float alpha = v / (1 - iou + v);
    float ciou = iou - d_center1 / d_center2 - alpha * v;
    return ciou;
}


bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {

        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
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
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
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



IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


IElementWiseLayer * convBnSilu(INetworkDefinition * network, std::map<std::string, Weights>&weightMap, ITensor & input, int c2, int k, int s, int p, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, c2, DimsHW{ k, k }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname + ".conv").c_str());
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });


    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);


    // silu = x * sigmoid(x)
    IActivationLayer* sig1 = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig1);
    IElementWiseLayer* ew1 = network->addElementWise(*bn1->getOutput(0), *sig1->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew1);
    return ew1;
}
ILayer* ReOrg(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch) {
    ISliceLayer* s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer* s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    return cat;
}
ILayer* DownC(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, const std::string& lname)
{
    int c_ = int(c2 * 0.5);
    IElementWiseLayer* cv1 = convBnSilu(network, weightMap, input, c1, 1, 1, 0, lname + ".cv1");
    IElementWiseLayer* cv2 = convBnSilu(network, weightMap, *cv1->getOutput(0), c_, 3, 2, 1, lname + ".cv2");

    IPoolingLayer* m1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
    m1->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* cv3 = convBnSilu(network, weightMap, *m1->getOutput(0), c_, 1, 1, 0, lname + ".cv3");

    ITensor* input_tensors[] = { cv2->getOutput(0),  cv3->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(input_tensors, 2);

    return concat;

}

// SPPCSPC
IElementWiseLayer* SPPCSPC(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, const std::string& lname) {
    int c_ = int(2 * c2 * 0.5);
    IElementWiseLayer* cv1 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, lname + ".cv1");
    IElementWiseLayer* cv2 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, lname + ".cv2");

    IElementWiseLayer* cv3 = convBnSilu(network, weightMap, *cv1->getOutput(0), c_, 3, 1, 1, lname + ".cv3");
    IElementWiseLayer* cv4 = convBnSilu(network, weightMap, *cv3->getOutput(0), c_, 1, 1, 0, lname + ".cv4");

    IPoolingLayer* m1 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{ 5, 5 });
    m1->setStrideNd(DimsHW{ 1, 1 });
    m1->setPaddingNd(DimsHW{ 2, 2 });
    IPoolingLayer* m2 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{ 9, 9 });
    m2->setStrideNd(DimsHW{ 1, 1 });
    m2->setPaddingNd(DimsHW{ 4, 4 });
    IPoolingLayer* m3 = network->addPoolingNd(*cv4->getOutput(0), PoolingType::kMAX, DimsHW{ 13, 13 });
    m3->setStrideNd(DimsHW{ 1, 1 });
    m3->setPaddingNd(DimsHW{ 6, 6 });

    ITensor* input_tensors[] = { cv4->getOutput(0), m1->getOutput(0), m2->getOutput(0), m3->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(input_tensors, 4);
    // 0U
    concat->setAxis(0);

    IElementWiseLayer* cv5 = convBnSilu(network, weightMap, *concat->getOutput(0), c_, 1, 1, 0, lname + ".cv5");
    IElementWiseLayer* cv6 = convBnSilu(network, weightMap, *cv5->getOutput(0), c_, 3, 1, 1, lname + ".cv6");

    ITensor* input_tensors2[] = { cv6->getOutput(0), cv2->getOutput(0) };
    IConcatenationLayer* concat1 = network->addConcatenation(input_tensors2, 2);
    // 0U
    concat1->setAxis(0);


    IElementWiseLayer* cv7 = convBnSilu(network, weightMap, *concat1->getOutput(0), c2, 1, 1, 0, lname + ".cv7");
    return cv7;
}

// RepConv
IElementWiseLayer* RepConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2, int k, int s, const std::string& lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // 256 * 128 * 3 *3
    IConvolutionLayer* rbr_dense_conv = network->addConvolutionNd(input, c2, DimsHW{ k, k }, weightMap[lname + ".rbr_dense.0.weight"], emptywts);
    assert(rbr_dense_conv);
    rbr_dense_conv->setPaddingNd(DimsHW{ k / 2, k / 2 });
    rbr_dense_conv->setStrideNd(DimsHW{ s, s });
    rbr_dense_conv->setName((lname + ".rbr_dense.0").c_str());
    IScaleLayer* rbr_dense_bn = addBatchNorm2d(network, weightMap, *rbr_dense_conv->getOutput(0), lname + ".rbr_dense.1", 1e-3);

    IConvolutionLayer* rbr_1x1_conv = network->addConvolutionNd(input, c2, DimsHW{ 1, 1 }, weightMap[lname + ".rbr_1x1.0.weight"], emptywts);
    assert(rbr_1x1_conv);
    rbr_1x1_conv->setStrideNd(DimsHW{ s, s });
    rbr_1x1_conv->setName((lname + ".rbr_1x1.0").c_str());
    IScaleLayer* rbr_1x1_bn = addBatchNorm2d(network, weightMap, *rbr_1x1_conv->getOutput(0), lname + ".rbr_1x1.1", 1e-3);

    IElementWiseLayer* ew1 = network->addElementWise(*rbr_dense_bn->getOutput(0), *rbr_1x1_bn->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew1);
    // silu
    IActivationLayer* sigmoid = network->addActivation(*ew1->getOutput(0), ActivationType::kSIGMOID);
    IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), *sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    return ew2;
}




IConcatenationLayer* MPC3(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname1, std::string lname2, std::string lname3) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IPoolingLayer* mp1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
    mp1->setStrideNd(DimsHW{ 2, 2 });
    IElementWiseLayer* conv2 = convBnSilu(network, weightMap, *mp1->getOutput(0), outch, 1, 1, 0, lname1); // 左侧分支

    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, input, outch, 1, 1, 0, lname2);
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv3->getOutput(0), outch, 3, 2, 1, lname3);
    ITensor* input_tensor_1[] = { conv4->getOutput(0), conv2->getOutput(0)};
    IConcatenationLayer* concat1 = network->addConcatenation(input_tensor_1, 2);
    concat1->setAxis(0);
    
    return concat1;
}

IElementWiseLayer* C9_1(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch,int outch2, std::string lname1, std::string lname2, std::string lname3,
    std::string lname4, std::string lname5, std::string lname6, std::string lname7, std::string lname8, std::string lname9)
{
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IElementWiseLayer* conv1 = convBnSilu(network, weightMap, input, outch, 1, 1, 0, lname1);//单独得分支


    IElementWiseLayer* conv2 = convBnSilu(network, weightMap, input, outch, 1, 1, 0, lname2);
    IElementWiseLayer* conv3 = convBnSilu(network, weightMap, *conv2->getOutput(0), outch, 3, 1, 1, lname3);
    IElementWiseLayer* conv4 = convBnSilu(network, weightMap, *conv3->getOutput(0), outch, 3, 1, 1, lname4);
    IElementWiseLayer* conv5 = convBnSilu(network, weightMap, *conv4->getOutput(0), outch, 3, 1, 1, lname5);
    IElementWiseLayer* conv6 = convBnSilu(network, weightMap, *conv5->getOutput(0), outch, 3, 1, 1, lname6);
    IElementWiseLayer* conv7 = convBnSilu(network, weightMap, *conv6->getOutput(0), outch, 3, 1, 1, lname7);
    IElementWiseLayer* conv8 = convBnSilu(network, weightMap, *conv7->getOutput(0), outch, 3, 1, 1, lname8);
    ITensor* input_tensor_9[] = { conv8->getOutput(0), conv6->getOutput(0), conv4->getOutput(0), conv2->getOutput(0),conv1->getOutput(0) };
    IConcatenationLayer* concat9 = network->addConcatenation(input_tensor_9, 5);
    //concat9->setAxis(0);
    IElementWiseLayer* conv10 = convBnSilu(network, weightMap, *concat9->getOutput(0), outch2, 1, 1, 0, lname9);


    return conv10;
}






ILayer* convBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 3;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

IActivationLayer* convBlockLeakRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname + ".conv").c_str());
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    //conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);

    auto ew1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    ew1->setAlpha(0.1);
    return ew1;
}



ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname) {
    ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor* inputTensors[] = { cv3->getOutput(0), cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}



ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor* inputTensors[] = { y1, cv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k1, k1 });
    pool1->setPaddingNd(DimsHW{ k1 / 2, k1 / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k2, k2 });
    pool2->setPaddingNd(DimsHW{ k2 / 2, k2 / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k3, k3 });
    pool3->setPaddingNd(DimsHW{ k3 / 2, k3 / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });

    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

ILayer* SPPF(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k, std::string lname) {
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool1->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool1->setStrideNd(DimsHW{ 1, 1 });
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool2->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool2->setStrideNd(DimsHW{ 1, 1 });
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{ k, k });
    pool3->setPaddingNd(DimsHW{ k / 2, k / 2 });
    pool3->setStrideNd(DimsHW{ 1, 1 });
    ITensor* inputTensors[] = { cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0) };
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = Yolo::CHECK_COUNT * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    float a[3][6] = { {12.0,16.0,19.0,36.0,40.0,28.0},{36.0,75.0,76.0,55.0,72.0,146.0},{142.0,110.0,192.0,243.0,459.0,401.0} };
    //std::vector<std::vector<float>> anchors = { {12.0,16.0,19.0,36.0,40.0,28.0},{36.0,75.0,76.0,55.0,72.0,146.0},{142.0,110.0,192.0,243.0,459.0,401.0} };
    /*for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            anchors[i][j] = a[i][j];

        }
    }*/
//    for (int i = 0; i < anchors.size(); i++)
//    {
//        for (int j = 0; j < anchors[i].size(); j++)
//        {
//            std::cout << anchors[i][j] << "  ";
//
//        }
//        std::cout << std::endl;
//    }

    PluginField plugin_fields[2];
    int netinfo[4] = {Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, Yolo::MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;

    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = Yolo::INPUT_W / scale;
        kernel.height = Yolo::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}


IPluginV2Layer* addYoLoLayer2(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IShuffleLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    float a[3][6] = { {12.0,16.0,19.0,36.0,40.0,28.0},{36.0,75.0,76.0,55.0,72.0,146.0},{142.0,110.0,192.0,243.0,459.0,401.0} };
    //std::vector<std::vector<float>> anchors = { {12.0,16.0,19.0,36.0,40.0,28.0},{36.0,75.0,76.0,55.0,72.0,146.0},{142.0,110.0,192.0,243.0,459.0,401.0} };
    /*for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            anchors[i][j] = a[i][j];

        }
    }*/
//    for (int i = 0; i < anchors.size(); i++)
//    {
//        for (int j = 0; j < anchors[i].size(); j++)
//        {
//            std::cout << anchors[i][j] << "  ";
//
//        }
//        std::cout << std::endl;
//    }

    PluginField plugin_fields[2];
    int netinfo[4] = { Yolo::CLASS_NUM, Yolo::INPUT_W, Yolo::INPUT_H, Yolo::MAX_OUTPUT_BBOX_COUNT };
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;

    std::vector<Yolo::YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        Yolo::YoloKernel kernel;
        kernel.width = Yolo::INPUT_W / scale;
        kernel.height = Yolo::INPUT_H / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2* plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor*> input_tensors;
    for (auto det : dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}
#endif

