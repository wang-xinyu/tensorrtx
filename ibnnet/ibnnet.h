#pragma once

#include "utils.h"
#include "holder.h"
#include "layers.h"
#include "InferenceEngine.h"
#include <memory>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
extern Logger gLogger;
using namespace trtxapi;

namespace trt {

    enum IBN {
        A, // resnet50-ibna,
        B, // resnet50-ibnb,
        NONE // resnet50
    };

    class IBNNet {
    public:
        IBNNet(trt::EngineConfig &enginecfg, const IBN ibn);
        ~IBNNet() {};

        bool serializeEngine(); /* create & serializeEngine */ 
        bool deserializeEngine();
        bool inference(std::vector<cv::Mat> &input); /* support batch inference */

        float* getOutput(); 
        int getDeviceID(); /* cuda deviceid */ 

    private:
        ICudaEngine *createEngine(IBuilder *builder, IBuilderConfig *config);
        void preprocessing(const cv::Mat& img, float* const data, const std::size_t stride);

    private:
        trt::EngineConfig _engineCfg;
        std::unique_ptr<trt::InferenceEngine> _inferEngine{nullptr};
        std::string _ibn;
        DataType _dt{DataType::kFLOAT};
    };

}