#include "common.h"
#include "h_sigmoid.cuh"
#include "h_sigmoidplugin.h"

using namespace nvinfer1;
using nvinfer1::HSigmoidPlugin;
using nvinfer1::PluginFactory;

HSigmoidPlugin::HSigmoidPlugin() {
}

HSigmoidPlugin::HSigmoidPlugin(const void* buffer, size_t size) {
    assert(size == sizeof(input_size_));
    input_size_ = *reinterpret_cast<const int*>(buffer);
}

int HSigmoidPlugin::getNbOutputs() const {
    return 1;
}

Dims HSigmoidPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    assert(index == 0);
    // Output dimensions
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

void HSigmoidPlugin::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) {
    input_size_ = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
}

int HSigmoidPlugin::initialize() {
    return 0;
}

void HSigmoidPlugin::terminate() {}

size_t HSigmoidPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int HSigmoidPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
    cuh_sigmoid(reinterpret_cast<float const*>(inputs[0]), reinterpret_cast<float*>(outputs[0]), input_size_);
    return 0;
}

size_t HSigmoidPlugin::getSerializationSize() {
    return sizeof(input_size_);
}

void HSigmoidPlugin::serialize(void* buffer) {
    *reinterpret_cast<int*>(buffer) = input_size_;
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    IPlugin *plugin = nullptr;
    if (strstr(layerName, "h_sigmoid") != NULL) {
        plugin = new HSigmoidPlugin(serialData, serialLength);
    }
    return plugin;
}

