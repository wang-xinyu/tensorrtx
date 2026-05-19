#include "ppocrv5_db_layer.h"

#include <cassert>
#include <cstring>
#include <iostream>

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

namespace {

__global__ void dbSigmoidKernel(const float* input, float* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = 1.0f / (1.0f + expf(-input[index]));
    }
}

int volume(const nvinfer1::Dims& dims) {
    int count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        count *= dims.d[i];
    }
    return count;
}

}  // namespace

void ppocrv5CudaDbSigmoid(const float* input, float* output, int count, cudaStream_t stream) {
    if (!input || !output || count <= 0) {
        return;
    }
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    dbSigmoidKernel<<<blocks, threads, 0, stream>>>(input, output, count);
}

void ppocrv5EnsureDbPlugin() {}

namespace nvinfer1 {

static const char* kPluginName = "Ppocrv5DbPlugin";
static const char* kPluginVersion = "1";

PluginFieldCollection Ppocrv5DbPluginCreator::mFC{};
std::vector<PluginField> Ppocrv5DbPluginCreator::mPluginAttributes;

Ppocrv5DbPlugin::Ppocrv5DbPlugin(const void* data, size_t length) {
    assert(length == 0);
}

DimsExprs Ppocrv5DbPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                               IExprBuilder& exprBuilder) noexcept {
    assert(outputIndex == 0);
    assert(nbInputs == 1);
    return inputs[0];
}

bool Ppocrv5DbPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                                int nbOutputs) noexcept {
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    assert(pos >= 0 && pos < 2);

    const PluginTensorDesc& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR || desc.type != DataType::kFLOAT) {
        return false;
    }
    return pos == 0 || desc.type == inOut[0].type;
}

void Ppocrv5DbPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                                      const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

size_t Ppocrv5DbPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
                                         int nbOutputs) const noexcept {
    return 0;
}

int Ppocrv5DbPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                             const void* const* inputs, void* const* outputs, void* workspace,
                             cudaStream_t stream) noexcept {
    int count = volume(inputDesc[0].dims);
    ppocrv5CudaDbSigmoid(reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<float*>(outputs[0]), count,
                         stream);
    CHECK(cudaPeekAtLastError());
    return 0;
}

const char* Ppocrv5DbPlugin::getPluginType() const noexcept {
    return kPluginName;
}

const char* Ppocrv5DbPlugin::getPluginVersion() const noexcept {
    return kPluginVersion;
}

IPluginV2DynamicExt* Ppocrv5DbPlugin::clone() const noexcept {
    Ppocrv5DbPlugin* plugin = new Ppocrv5DbPlugin();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void Ppocrv5DbPlugin::destroy() noexcept {
    delete this;
}

DataType Ppocrv5DbPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
    assert(index == 0);
    assert(nbInputs == 1);
    return inputTypes[0];
}

void Ppocrv5DbPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* Ppocrv5DbPlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace;
}

Ppocrv5DbPluginCreator::Ppocrv5DbPluginCreator() {
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

const char* Ppocrv5DbPluginCreator::getPluginName() const noexcept {
    return kPluginName;
}

const char* Ppocrv5DbPluginCreator::getPluginVersion() const noexcept {
    return kPluginVersion;
}

const PluginFieldCollection* Ppocrv5DbPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* Ppocrv5DbPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    return new Ppocrv5DbPlugin();
}

IPluginV2* Ppocrv5DbPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                     size_t serialLength) noexcept {
    return new Ppocrv5DbPlugin(serialData, serialLength);
}

void Ppocrv5DbPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* Ppocrv5DbPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(Ppocrv5DbPluginCreator);

}  // namespace nvinfer1
