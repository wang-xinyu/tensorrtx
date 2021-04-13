#ifndef TRTX_INCEPTION_NETWORK_H
#define TRTX_INCEPTION_NETWORK_H


#include <memory>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "layers_api.h"
#include "engine.h"

using namespace trtxlayers;

namespace trtx {
    class InceptionV4 {
    public:
        InceptionV4(trt::InferenceEngineConfig &enginecfg);
        ~InceptionV4() {};

        bool serializeEngine();                  // create & serialize netowrk Engine 
        bool deserializeCudaEngine();
        bool infer(std::vector<cv::Mat> &input); // batch inference 

        float* getOutput(); 
        int getDeviceID(); /* cuda deviceid */ 

    private:
        ICudaEngine *buildEngine(IBuilder *builder, IBuilderConfig *config);

    private:
        trtx::InferenceEngineConfig engineConfig;
        std::unique_ptr<trtx::InceptionInferEngine> inferEngine{nullptr};
        std::string inception;
        DataType dt{DataType::kFLOAT};
    };
}

#endif