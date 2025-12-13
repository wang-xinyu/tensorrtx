#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yololayer.h"

using namespace nvinfer1;

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) 
    {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } 
    else
    {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float iou(float lbox[4], float rbox[4]) 
{
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    {
        std::cout << "The data is questionable!" << std::endl;
        return 0.0f;
    }

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) 
{
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) 
{
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
        //std::cout << it->second[0].class_id << " --- " << std::endl;
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
std::map<std::string, Weights> loadWeights(const std::string file) 
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        /*
        class Weights
        {
        public:
            DataType type;      //!< The type of the weights.
            void const* values; //!< The weight values, in a contiguous array.
            int64_t count;      //!< The number of weights in the array.
        };
        */
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size; 
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    //  for (auto it = weightMap.begin(); it != weightMap.end(); it++) {
    //     std::cout << "========= keys: " << it -> first << " =================" <<  std::endl;
    // }

    return weightMap;
}

nvinfer1::IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
 {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    // gamma / sqrt(running_var + eps)
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) 
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) 
    {
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

nvinfer1::IPoolingLayer *conv_bn_relu_maxpool(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> & weightMap, nvinfer1::ITensor &input, int outch, std::string lname){
  nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
  nvinfer1::IConvolutionLayer *conv0 = network->addConvolutionNd(input, outch, nvinfer1::DimsHW{3, 3}, weightMap[lname + "conv.0.weight"], emptywts);
  conv0->setStrideNd(nvinfer1::DimsHW{2, 2});
  conv0->setPaddingNd(nvinfer1::DimsHW{1, 1});

  nvinfer1::IScaleLayer * bn1 = addBatchNorm2d(network, weightMap, *conv0->getOutput(0), lname + "conv.1", 1e-3);
  
  auto Relu = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
  assert(Relu);
  IPoolingLayer *pool = network->addPoolingNd(*Relu->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{3, 3});
  pool->setStrideNd(nvinfer1::DimsHW{2, 2});
  pool->setPaddingNd(nvinfer1::DimsHW{1, 1});
  assert(pool);
  return pool;
}




nvinfer1::IElementWiseLayer *HardSwish(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor &input){
    auto hsig = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    auto ew = network->addElementWise(input, *hsig->getOutput(0), ElementWiseOperation::kPROD);
    return ew;
    
}



nvinfer1::IElementWiseLayer *CBH(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::ITensor &input, 
        int num_filters, int filter_size, int stride, std::string lname, int num_groups=1){
    
    int pad = (filter_size - 1) / 2;
    nvinfer1::Weights emptywts {nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::IConvolutionLayer *conv = network->addConvolutionNd(input, num_filters, nvinfer1::DimsHW{filter_size, filter_size}, 
                 weightMap[lname + ".conv.weight"], emptywts);
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{pad, pad});
    conv->setNbGroups(num_groups);

    nvinfer1::IScaleLayer *bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    nvinfer1::IElementWiseLayer *hash = HardSwish(network, *bn->getOutput(0));
    
    nvinfer1::Dims dims = hash->getOutput(0)->getDimensions();
   
    return hash;
}




nvinfer1::IElementWiseLayer *SiLU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input)
{
    // Create Sigmoid activation layer
    nvinfer1::IActivationLayer *sig = network->addActivation(input, ActivationType::kSIGMOID);

    nvinfer1::IElementWiseLayer *mul = network->addElementWise(input, *sig->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    return mul;
}


nvinfer1::IElementWiseLayer *LC_SEModule(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::ITensor &input,
       int in_channels, std::string lname, int reduction=4){

    nvinfer1::IIdentityLayer *identity = network->addIdentity(input);
    nvinfer1::IReduceLayer *avg_pool = network->addReduce(input, nvinfer1::ReduceOperation::kAVG, (1 << 1) | (1 << 2), true);
    nvinfer1::IConvolutionLayer *conv1 = network->addConvolutionNd(*avg_pool->getOutput(0), in_channels / reduction, nvinfer1::DimsHW{1, 1},
             weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
    nvinfer1::IActivationLayer *relu = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU);
    nvinfer1::IConvolutionLayer *conv2 = network->addConvolutionNd(*relu->getOutput(0), in_channels, nvinfer1::DimsHW{1, 1},
             weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
    nvinfer1::IElementWiseLayer *silu = SiLU(network, *conv2->getOutput(0));

    nvinfer1::IElementWiseLayer *out = network->addElementWise(*silu->getOutput(0), *identity->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    return out;
}

nvinfer1::IElementWiseLayer *LC_Block(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::ITensor &input,
     int num_channels, int num_filters, int stride, int dw_size, std::string lname, bool use_se=false){
    // num_channels : in_channel
    // num_filters : out_channel
    // stride:dw_conv's stride
    // dw_size: dw_conv's filter-size
    nvinfer1::IElementWiseLayer *dw_conv = CBH(network, weightMap, input, num_channels, dw_size, stride, lname + ".dw_conv", num_channels);
    if(use_se){
        nvinfer1::IElementWiseLayer *se = LC_SEModule(network, weightMap, *dw_conv->getOutput(0), num_channels, lname + ".se");
        nvinfer1::IElementWiseLayer *pw_conv = CBH(network, weightMap, *se->getOutput(0), num_filters, 1, 1, lname + ".pw_conv");

        return pw_conv;
    }
    nvinfer1::IElementWiseLayer *pw_conv = CBH(network, weightMap, *dw_conv->getOutput(0), num_filters, 1, 1, lname + ".pw_conv");
    
    return pw_conv;
}


nvinfer1::IElementWiseLayer *Dense(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::ITensor &input, 
      int num_filters, int filter_size, std::string lname){
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer *dense_conv = network->addConvolutionNd(input, num_filters, nvinfer1::DimsHW{filter_size, filter_size}
     , weightMap[lname + ".dense_conv.weight"], emptywts);
    
    nvinfer1::IElementWiseLayer *hash = HardSwish(network, *dense_conv->getOutput(0));
    nvinfer1::Dims dims_o = hash->getOutput(0)->getDimensions();
    return hash;
}


nvinfer1::IElementWiseLayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
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

nvinfer1::IShuffleLayer* shuffle_block(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int branch_features = outch / 2;
    ITensor *x1, *x2i, *x2o;
    if (s > 1) {
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3}, weightMap[lname + "branch1.0.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{1, 1});
        conv1->setNbGroups(inch);
        IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "branch1.1", 1e-5);
        IConvolutionLayer* conv2 = network->addConvolutionNd(*bn1->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch1.2.weight"], emptywts);
        assert(conv2);
        IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "branch1.3", 1e-5);
        IActivationLayer* relu1 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        x1 = relu1->getOutput(0);
        x2i = &input;
    } else {
        Dims d = input.getDimensions();
        ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        ISliceLayer *s2 = network->addSlice(input, Dims3{ d.d[0] / 2, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        x1 = s1->getOutput(0);
        x2i = s2->getOutput(0);
    }

    IConvolutionLayer* conv3 = network->addConvolutionNd(*x2i, branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.0.weight"], emptywts);
    assert(conv3);
    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "branch2.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu2->getOutput(0), branch_features, DimsHW{3, 3}, weightMap[lname + "branch2.3.weight"], emptywts);
    assert(conv4);
    conv4->setStrideNd(DimsHW{s, s});
    conv4->setPaddingNd(DimsHW{1, 1});
    conv4->setNbGroups(branch_features);
    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "branch2.4", 1e-5);
    IConvolutionLayer* conv5 = network->addConvolutionNd(*bn4->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.5.weight"], emptywts);
    assert(conv5);
    IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), lname + "branch2.6", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn5->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    ITensor* inputTensors1[] = {x1, relu3->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);

    Dims dims = cat1->getOutput(0)->getDimensions();
    std::cout << cat1->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << ", ";
    }
    std::cout << std::endl;

    IShuffleLayer *sf1 = network->addShuffle(*cat1->getOutput(0));
    assert(sf1);
    sf1->setReshapeDimensions(Dims4(2, dims.d[0] / 2, dims.d[1], dims.d[2]));
    sf1->setSecondTranspose(Permutation{1, 0, 2, 3});

    Dims dims1 = sf1->getOutput(0)->getDimensions();
    std::cout << sf1->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims1.nbDims; i++) {
        std::cout << dims1.d[i] << ", ";
    }
    std::cout << std::endl;

    IShuffleLayer *sf2 = network->addShuffle(*sf1->getOutput(0));
    assert(sf2);
    sf2->setReshapeDimensions(Dims3(dims.d[0], dims.d[1], dims.d[2]));

    Dims dims2 = sf2->getOutput(0)->getDimensions();
    std::cout << sf2->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims2.nbDims; i++) {
        std::cout << dims2.d[i] << ", ";
    }
    std::cout << std::endl;

    return sf2;
}

nvinfer1::IElementWiseLayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
  int c1, int c2, int k1, int k2, int k3, std::string lname) {
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

nvinfer1::IElementWiseLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname) {
  auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, lname + ".cv1");
  auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
  if (shortcut && c1 == c2) {
    auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
    return ew;
  }
  return cv2;
}

nvinfer1::IElementWiseLayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
  Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
  int c_ = (int)((float)c2 * e);
  auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
  auto cv2 = network->addConvolutionNd(input, c_, DimsHW{ 1, 1 }, weightMap[lname + ".cv2.weight"], emptywts);
  ITensor* y1 = cv1->getOutput(0);
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

nvinfer1::IElementWiseLayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,
           int c1, int c2, int n, bool shortcut, int g, float e, std::string lname) {
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


nvinfer1::IScaleLayer *conv_bn(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
           std::string lname, int out_channels, int kernel_size, int stride, int padding, int groups=1){
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer *conv = network->addConvolutionNd(input, out_channels, nvinfer1::DimsHW{kernel_size, kernel_size},
         weightMap[lname + ".conv.weight"], emptywts);
    conv->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv->setPaddingNd(nvinfer1::DimsHW{padding, padding});
    conv->setNbGroups(groups);

    nvinfer1::IScaleLayer *bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-5);
    return bn;
   }

nvinfer1::IActivationLayer *RepVGGBlock(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
        std::string lname, int out_channels, int kernel_size = 3, int stride = 1, int padding = 1, int groups=1){

    nvinfer1::IScaleLayer *rbr_dense = conv_bn(network, weightMap, input, lname + ".rbr_dense", out_channels, kernel_size, stride, padding, groups);
    int padding_11 = padding - kernel_size / 2;
    nvinfer1::IScaleLayer *rbr_1x1 = conv_bn(network, weightMap, input, lname + ".rbr_1x1", out_channels, 1, stride, padding_11, groups);
    nvinfer1::IElementWiseLayer *add = network->addElementWise(*rbr_dense->getOutput(0), *rbr_1x1->getOutput(0),  nvinfer1::ElementWiseOperation::kSUM);

    nvinfer1::IActivationLayer *silu = network->addActivation(*add->getOutput(0), nvinfer1::ActivationType::kRELU);
    return silu;
}

nvinfer1::IActivationLayer *DWConvblock(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
     std::string lname, int in_channels, int out_channels, int kernel_size, int stride){
    nvinfer1::Weights emptywts {nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IConvolutionLayer *conv1 = network->addConvolutionNd(input, in_channels, nvinfer1::DimsHW{kernel_size, kernel_size},
      weightMap[lname + ".conv1.weight"], emptywts);
    conv1->setStrideNd(nvinfer1::DimsHW{stride, stride});
    std::cout << (kernel_size / 2) << std::endl;
    conv1->setPaddingNd(nvinfer1::DimsHW{kernel_size / 2, kernel_size / 2});
    conv1->setNbGroups(in_channels);
    nvinfer1::IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    nvinfer1::IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kRELU);
    nvinfer1::IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), out_channels, nvinfer1::DimsHW{1, 1}, weightMap[lname + ".conv2.weight"], emptywts);
    conv2->setStrideNd(nvinfer1::DimsHW{1, 1});
    conv2->setPaddingNd(nvinfer1::DimsHW{0, 0});
    nvinfer1::IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);
    nvinfer1::IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), nvinfer1::ActivationType::kRELU);

    return relu2;    
    }

std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) 
{
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"];
    int anchor_len = Yolo::CHECK_COUNT * 2; // 6
    for (int i = 0; i < wts.count / anchor_len; i++) 
    {
        auto *p = (const float*)wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

nvinfer1::IElementWiseLayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch,
           int outch, int ksize, std::string lname) {
  ISliceLayer* s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
  ISliceLayer* s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
  ISliceLayer* s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
  ISliceLayer* s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2 }, Dims3{ 1, 2, 2 });
  ITensor* inputTensors[] = { s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0) };
  auto cat = network->addConcatenation(inputTensors, 4);
  auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
  return conv;
}

nvinfer1::IElementWiseLayer *ADD(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor& x1,nvinfer1::ITensor& x2, float alpha) {
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, 0}; 
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, &alpha, 1};  
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0}; 

    nvinfer1::IScaleLayer* scaleLayer = network->addScale(x2, nvinfer1::ScaleMode::kUNIFORM, shift, scale, power);

    nvinfer1::IElementWiseLayer* addLayer = network->addElementWise(x1, *scaleLayer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    return addLayer; 
}

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
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



#endif

