#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static YoloKernel yolo1 = {
        INPUT_W / 8,
        INPUT_H / 8,
        {12,16, 19,36, 40,28}
    };
    static YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {36,75, 76,55, 72,146}
    };
    static YoloKernel yolo3 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {142,110, 192,243, 459,401}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}


namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginExt
    {
    public:
        explicit YoloLayerPlugin(const int cudaThread = 256);
        YoloLayerPlugin(const void* data, size_t length);

        ~YoloLayerPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override { 
            return type == DataType::kFLOAT && format == PluginFormat::kNCHW; 
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override {};

        int initialize() override;

        virtual void terminate() override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override;

        virtual void serialize(void* buffer) override;

        void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);

    private:
        int mClassCount;
        int mKernelCount;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        int mThreadCount;
        //int mDetNum;
    };
};

#endif 
