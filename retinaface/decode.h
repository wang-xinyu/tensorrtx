#ifndef _DECODE_CU_H
#define _DECODE_CU_H

#include "NvInfer.h"

namespace decodeplugin
{
    struct alignas(float) Detection{
        float bbox[4];  //x1 y1 x2 y2
        float class_confidence;
        float landmark[10];
    };
    static const int INPUT_H = 928;
    static const int INPUT_W = 1600;
}


namespace nvinfer1
{
    class DecodePlugin: public IPluginExt
    {
    public:
        explicit DecodePlugin(const int cudaThread = 256);
        DecodePlugin(const void* data, size_t length);

        ~DecodePlugin();

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
        int thread_count_ = 256;
    };
};

#endif 
