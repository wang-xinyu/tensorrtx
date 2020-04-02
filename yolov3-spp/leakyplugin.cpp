#include "common.h"
#include "leaky.cuh"
#include "leakyplugin.h"

using namespace nvinfer1;
using nvinfer1::LeakyPlugin;
using nvinfer1::PluginFactory;

LeakyPlugin::LeakyPlugin() {
}

LeakyPlugin::LeakyPlugin(const void* buffer, size_t size) {
    assert(size == sizeof(input_size_));
    input_size_ = *reinterpret_cast<const int*>(buffer);
}

int LeakyPlugin::getNbOutputs() const {
    return 1;
}

Dims LeakyPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    assert(index == 0);
    // Output dimensions
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

void LeakyPlugin::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) {
    input_size_ = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
}

int LeakyPlugin::initialize() {
    return 0;
}

void LeakyPlugin::terminate() {}

size_t LeakyPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int LeakyPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
    culeaky(reinterpret_cast<float const*>(inputs[0]), reinterpret_cast<float*>(outputs[0]), input_size_);
    return 0;
}

size_t LeakyPlugin::getSerializationSize() {
    return sizeof(input_size_);
}

void LeakyPlugin::serialize(void* buffer) {
    *reinterpret_cast<int*>(buffer) = input_size_;
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    IPlugin *plugin = nullptr;
    if (strstr(layerName, "leaky") != NULL) {
        plugin = new LeakyPlugin(serialData, serialLength);
    }
    return plugin;
}

