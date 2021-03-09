#ifndef TENSORRTX_LAYERS_H
#define TENSORRTX_LAYERS_H

#include <map>
#include <math.h>
#include <assert.h>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
using namespace nvinfer1;

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps);

IActivationLayer *bottleneck(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int ch, int stride, std::string lname, int branch_type);

IActivationLayer *addConvRelu(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int kernel, int stride, std::string lname);

#endif
