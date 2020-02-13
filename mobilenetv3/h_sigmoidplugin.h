#ifndef HSIGMOID_PLUGIN_H
#define HSIGMOID_PLUGIN_H
#include <NvInfer.h>

namespace nvinfer1 {
class HSigmoidPlugin : public IPlugin {
    public:
        HSigmoidPlugin();
        HSigmoidPlugin(const void* buffer, size_t size);
        ~HSigmoidPlugin() override = default;
        int getNbOutputs() const override;
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
        void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;
        int initialize() override;
        void terminate() override;
        size_t getWorkspaceSize(int maxBatchSize) const override;
        int enqueue(
                int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
        size_t getSerializationSize() override;
        void serialize(void* buffer) override;

    private:
        int input_size_;
};

class PluginFactory : public IPluginFactory {
    public:
        IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
};

}
#endif
