#ifndef TRTX_LAYERS_API_H
#define TRTX_LAYERS_API_H

#include <map>
#include <math.h>
#include <assert.h>
#include <iostream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;

namespace trtxlayers {

    // Declare your layers here
    IScaleLayer* addBatchNorm2d(
        INetworkDefinition *network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        std::string lname, 
        float eps
    );

    IActivationLayer* basicConv2d(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        int outch, 
        DimsHW ksize, 
        int s, 
        DimsHW p, 
        std::string lname
    );

    IConcatenationLayer* mixed_3a(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );

    IConcatenationLayer* mixed_4a(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );

    IConcatenationLayer* mixed_5a(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );

    IConcatenationLayer* inceptionA(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );

    IConcatenationLayer* reductionA(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );
    
    IConcatenationLayer* inceptionB(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );

    IConcatenationLayer* reductionB(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );

    IConcatenationLayer* inceptionC(
        INetworkDefinition *network,
        std::map<std::string, Weights>& weightMap, 
        ITensor& input,  
        std::string lname
    );
}

#endif  // TRTX_LAYERS_API_H