#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

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

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    //std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
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

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, std::string convname, std::string bnname, bool bias = false) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1;
    //Dims dim;
    if (!bias)
    {
        conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[convname + ".weight"], emptywts);
    }
    else
    {
        conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[convname + ".weight"], weightMap[convname + ".bias"]);
    }
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), bnname, 1e-4);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    return lr;
}

IActivationLayer* ResBlock2Conv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    ///
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), inch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ stride, stride });
    conv2->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    //////
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
    assert(conv3);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv3->setPaddingNd(DimsHW{ 0, 0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + ".downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{ stride, stride });
        conv4->setPaddingNd(DimsHW{ 0, 0 });
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

IActivationLayer* ResBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // in 256 out 64
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    ///
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ stride, stride });
    conv2->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    //////
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), inch, DimsHW{ 1, 1 }, weightMap[lname + ".conv3.weight"], emptywts);
    assert(conv3);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 0, 0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);

    IElementWiseLayer* ew1;
    ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

IActivationLayer* liteResBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    // in 256 out 64
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    ///
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    conv2->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IElementWiseLayer* ew1;
    ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);

    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

ILayer* netAddUpsample(INetworkDefinition* network, ITensor* input, int inputChannels, int stride){
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3); // chw
    assert(inpDims.d[1] == inpDims.d[2]);
    int h = inpDims.d[1];
    int w = inpDims.d[2];
    // add pre multiply matrix as a constant
    /*
    kSPATIA Elements correspond to different spatial data.

    kCHANNEL Elements correspond to different channels.
    */
    nvinfer1::Dims preDims{ 3,
                           {1, stride * h, w},
                           {nvinfer1::DimensionType::kCHANNEL,
                            nvinfer1::DimensionType::kSPATIAL,
                            nvinfer1::DimensionType::kSPATIAL} };
    int size = stride * h * w;
    nvinfer1::Weights preMul{ nvinfer1::DataType::kFLOAT, nullptr, size };
    float* preWt = new float[size];
    /* (2*h * w)
    [ [1, 0, ..., 0],
      [1, 0, ..., 0],
      [0, 1, ..., 0],
      [0, 1, ..., 0],
      ...,
      ...,
      [0, 0, ..., 1],
      [0, 0, ..., 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int s = 0; s < stride; ++s)
        {
            for (int j = 0; j < w; ++j, ++idx)
            {
                preWt[idx] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    preMul.values = preWt;
    nvinfer1::IConstantLayer* preM = network->addConstant(preDims, preMul);
    assert(preM != nullptr);
    //std::string preLayerName = "preMul_" + std::to_string(layerIdx);
    //preM->setName(preLayerName.c_str());
    // add post multiply matrix as a constant
    nvinfer1::Dims postDims{ 3,
                            {1, h, stride * w},
                            {nvinfer1::DimensionType::kCHANNEL,
                             nvinfer1::DimensionType::kSPATIAL,
                             nvinfer1::DimensionType::kSPATIAL} };
    size = stride * h * w;
    nvinfer1::Weights postMul{ nvinfer1::DataType::kFLOAT, nullptr, size };
    float* postWt = new float[size];
    /* (h * 2*w)
    [ [1, 1, 0, 0, ..., 0, 0],
      [0, 0, 1, 1, ..., 0, 0],
      ...,
      ...,
      [0, 0, 0, 0, ..., 1, 1] ]
    */
    for (int i = 0, idx = 0; i < h; ++i)
    {
        for (int j = 0; j < stride * w; ++j, ++idx)
        {
            postWt[idx] = (j / stride == i) ? 1.0 : 0.0;
        }
    }
    postMul.values = postWt;
    nvinfer1::IConstantLayer* post_m = network->addConstant(postDims, postMul);
    assert(post_m != nullptr);
    // add matrix multiply layers for upsampling
    nvinfer1::IMatrixMultiplyLayer* mm1
        = network->addMatrixMultiply(*preM->getOutput(0),
            nvinfer1::MatrixOperation::kNONE, *input,
            nvinfer1::MatrixOperation::kNONE);
    assert(mm1 != nullptr);
    nvinfer1::IMatrixMultiplyLayer* mm2
        = network->addMatrixMultiply(*mm1->getOutput(0),
            nvinfer1::MatrixOperation::kNONE,
            *post_m->getOutput(0),
            nvinfer1::MatrixOperation::kNONE);
    assert(mm2 != nullptr);
    return mm2;
}

