#include "common.h"
#include "chunk.h"

using namespace nvinfer1;
using nvinfer1::ChunkPlugin;
using nvinfer1::PluginFactory;

ChunkPlugin::ChunkPlugin() {
}

ChunkPlugin::ChunkPlugin(const void* buffer, size_t size) {
    assert(size == sizeof(count_));
    count_ = *reinterpret_cast<const int*>(buffer);
}

int ChunkPlugin::getNbOutputs() const {
    // Plugin layer has 2 outputs
    return 2;
}

Dims ChunkPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    assert(index == 0 || index == 1);
    // Output dimensions
    return DimsCHW(inputs[0].d[0] / 2, inputs[0].d[1], inputs[0].d[2]);
}

void ChunkPlugin::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) {
    count_ = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2] / 2 * sizeof(float);
}

int ChunkPlugin::initialize() {
    return 0;
}

void ChunkPlugin::terminate() {}

size_t ChunkPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int ChunkPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
    CHECK(cudaMemcpy(outputs[0], inputs[0], count_, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(outputs[1], (void*)((char*)inputs[0] + count_), count_, cudaMemcpyDeviceToDevice));
    return 0;
}

size_t ChunkPlugin::getSerializationSize() {
    return sizeof(count_);
}

void ChunkPlugin::serialize(void* buffer) {
    *reinterpret_cast<int*>(buffer) = count_;
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    IPlugin *plugin = nullptr;
    if (strstr(layerName, "chunk") != NULL) {
        plugin = new ChunkPlugin(serialData, serialLength);
    }
    return plugin;
}
