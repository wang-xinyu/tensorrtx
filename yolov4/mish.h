#ifndef _MISH_PLUGIN_H
#define _MISH_PLUGIN_H

#include "NvInfer.h"

namespace nvinfer1
{
    class MishPlugin: public IPluginExt
    {
    public:
        explicit MishPlugin(const int cudaThread = 256);
        MishPlugin(const void* data, size_t length);

        ~MishPlugin();

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

        void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);

    private:
        int thread_count_ = 256;
        int input_size_;
    };
};

#endif 
