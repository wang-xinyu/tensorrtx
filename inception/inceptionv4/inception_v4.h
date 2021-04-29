#ifndef TRTX_INCEPTION_NETWORK_H
#define TRTX_INCEPTION_NETWORK_H


#include <memory>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "logging.h"
#include "utils.h"
#include "layers_api.h"


static Logger gLogger;
using namespace trtxlayers;

namespace trtx {
    struct InceptionV4Params
    {
        /* data */
        int32_t batchSize{1};              // Number of inputs in a batch
        bool int8{false};                  // Allow runnning the network in Int8 mode.
        bool fp16{false};                  // Allow running the network in FP16 mode.
        const char* inputTensorName = "data";
        const char* outputTensorName = "prob";

        int inputW;                // The input width of the network.
        int inputH;                // The input height of the the network.
        int outputSize;           // THe output size of the network.
        std::string weightsFile;   // Weights file filename.
        std::string trtEngineFile; // trt engine file name
    };
    
    class InceptionV4 {
    public:
        InceptionV4(const InceptionV4Params &enginecfg);
        ~InceptionV4() {};

        bool serializeEngine();                  // create & serialize netowrk Engine 
        bool deserializeCudaEngine();

        void doInference(float* input, float* output, int batchSize);
        bool cleanUp();
    private:
        bool buildEngine(IBuilder *builder, IBuilderConfig *config);
        // Runs the Tensorrt network inference engine on a sample.
    private:
        InceptionV4Params mParams;
        ICudaEngine* mEngine;  // The tensorrt engine used to run the network.
        std::map<std::string, Weights> weightMap; // The weight value map.
        IExecutionContext* mContext; // The TensorRT execution context to run inference.
        std::string inception;
        DataType dt{DataType::kFLOAT};
    };
}

#endif