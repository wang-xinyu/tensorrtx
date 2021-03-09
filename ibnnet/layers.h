#pragma once

#include <map>
#include <math.h>
#include <assert.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
using namespace nvinfer1;

namespace trtxapi {

    ITensor* MeanStd(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor* input, 
        const std::string lname,
        const float* mean, 
        const float* std, 
        const bool div255);

    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname, 
        const float eps);

    IScaleLayer* addInstanceNorm2d(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname, 
        const float eps);

    IConcatenationLayer* addIBN(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname);

    IActivationLayer* bottleneck_ibn(INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const int inch, 
        const int outch,
        const int stride, 
        const std::string lname, 
        const std::string ibn);

}