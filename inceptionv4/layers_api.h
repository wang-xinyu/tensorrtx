#ifndef TRTX_LAYERS_API_H
#define TRTX_LAYERS_API_H

#include <map>
#include <math.h>
#include <assert.h>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

namespace trtxlayers {

    // Declare your layers here
    IActivationLayer* basicConv2d(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        int outch, 
        DimsHW ksize, 
        int s, 
        DimsHW p, 
        std::string lname)
}

#endif  // TRTX_LAYERS_API_H