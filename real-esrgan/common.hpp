#ifndef REAL_ESRGAN_COMMON_H_
#define REAL_ESRGAN_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

using namespace nvinfer1;

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

ITensor* residualDenseBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname)
{
    IConvolutionLayer* conv_1 = network->addConvolutionNd(*x, 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
    conv_1->setStrideNd(DimsHW{ 1, 1 });
    conv_1->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_1 = network->addActivation(*conv_1->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_1->setAlpha(0.2);
    ITensor* x1 = leaky_relu_1->getOutput(0);

    ITensor* concat_input2[] = { x, x1 };
    IConcatenationLayer* concat2 = network->addConcatenation(concat_input2, 2);
    concat2->setAxis(0);
    IConvolutionLayer* conv_2 = network->addConvolutionNd(*concat2->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
    conv_2->setStrideNd(DimsHW{ 1, 1 });
    conv_2->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_2 = network->addActivation(*conv_2->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_2->setAlpha(0.2);
    ITensor* x2 = leaky_relu_2->getOutput(0);

    ITensor* concat_input3[] = { x, x1, x2 };
    IConcatenationLayer* concat3 = network->addConcatenation(concat_input3, 3);
    concat3->setAxis(0);
    IConvolutionLayer* conv_3 = network->addConvolutionNd(*concat3->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv3.weight"], weightMap[lname + ".conv3.bias"]);
    conv_3->setStrideNd(DimsHW{ 1, 1 });
    conv_3->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_3 = network->addActivation(*conv_3->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_3->setAlpha(0.2);
    ITensor* x3 = leaky_relu_3->getOutput(0);

    ITensor* concat_input4[] = { x, x1, x2, x3 };
    IConcatenationLayer* concat4 = network->addConcatenation(concat_input4, 4);
    concat4->setAxis(0);
    IConvolutionLayer* conv_4 = network->addConvolutionNd(*concat4->getOutput(0), 32, DimsHW{ 3, 3 }, weightMap[lname + ".conv4.weight"], weightMap[lname + ".conv4.bias"]);
    conv_4->setStrideNd(DimsHW{ 1, 1 });
    conv_4->setPaddingNd(DimsHW{ 1, 1 });
    IActivationLayer* leaky_relu_4 = network->addActivation(*conv_4->getOutput(0), ActivationType::kLEAKY_RELU);
    leaky_relu_4->setAlpha(0.2);
    ITensor* x4 = leaky_relu_4->getOutput(0);

    ITensor* concat_input5[] = { x, x1, x2, x3, x4 };
    IConcatenationLayer* concat5 = network->addConcatenation(concat_input5, 5);
    concat5->setAxis(0);
    IConvolutionLayer* conv_5 = network->addConvolutionNd(*concat5->getOutput(0), 64, DimsHW{ 3, 3 }, weightMap[lname + ".conv5.weight"], weightMap[lname + ".conv5.bias"]);
    conv_5->setStrideNd(DimsHW{ 1, 1 });
    conv_5->setPaddingNd(DimsHW{ 1, 1 });
    ITensor* x5 = conv_5->getOutput(0);

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scval = 0.2;
    Weights scale{ DataType::kFLOAT, scval, 1 };
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *shval = 0.0;
    Weights shift{ DataType::kFLOAT, shval, 1 };
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *pval = 1.0;
    Weights power{ DataType::kFLOAT, pval, 1 };

    IScaleLayer* scaled = network->addScale(*x5, ScaleMode::kUNIFORM, shift, scale, power);
    IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
    return ew1->getOutput(0);
}

ITensor* RRDB(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor* x, std::string lname)
{
    ITensor* out = residualDenseBlock(network, weightMap, x, lname + ".rdb1");
    out = residualDenseBlock(network, weightMap, out, lname + ".rdb2");
    out = residualDenseBlock(network, weightMap, out, lname + ".rdb3");

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *scval = 0.2;
    Weights scale{ DataType::kFLOAT, scval, 1 };
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *shval = 0.0;
    Weights shift{ DataType::kFLOAT, shval, 1 };
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float)));
    *pval = 1.0;
    Weights power{ DataType::kFLOAT, pval, 1 };

    IScaleLayer* scaled = network->addScale(*out, ScaleMode::kUNIFORM, shift, scale, power);
    IElementWiseLayer* ew1 = network->addElementWise(*scaled->getOutput(0), *x, ElementWiseOperation::kSUM);
    return ew1->getOutput(0);
}


#endif