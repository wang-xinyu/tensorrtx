#ifndef CHUNK_PLUGIN_H
#define CHUNK_PLUGIN_H
#include <NvInfer.h>

namespace nvinfer1 {
class ChunkPlugin : public IPlugin {
    public:
        ChunkPlugin();
        ChunkPlugin(const void* buffer, size_t size);
        ~ChunkPlugin() override = default;
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
        int count_;
};

class PluginFactory : public IPluginFactory {
    public:
        IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
};

}
#endif
