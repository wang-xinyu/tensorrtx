#include "apply_rotary_pos_emb_plugin.h"

#include <cassert>
#include <string>

#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

void launchApplyRotaryPosEmb(const void* q, const void* k, const void* cos, const void* sin, void* qEmbed, void* kEmbed,
                             int batch, int qHeads, int kHeads, int seq, int headDim, nvinfer1::DataType type,
                             cudaStream_t stream);

namespace {

constexpr char kPluginName[] = "ApplyRotaryPosEmbPlugin";
constexpr char kPluginVersion[] = "1";

class ApplyRotaryPosEmbPlugin final : public nvinfer1::IPluginV2DynamicExt {
   public:
    ApplyRotaryPosEmbPlugin() = default;

    ApplyRotaryPosEmbPlugin(const void*, size_t) {}

    int getNbOutputs() const noexcept override { return 2; }

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                            nvinfer1::IExprBuilder&) noexcept override {
        assert(nbInputs == 4);
        assert(outputIndex == 0 || outputIndex == 1);
        return inputs[outputIndex];
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int, const nvinfer1::PluginTensorDesc*,
                            int) const noexcept override {
        return 0;
    }

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        const auto& qDims = inputDesc[0].dims;
        const auto& kDims = inputDesc[1].dims;
        const auto& cosDims = inputDesc[2].dims;
        const auto& sinDims = inputDesc[3].dims;
        assert(qDims.nbDims == 4);
        assert(kDims.nbDims == 4);
        assert(cosDims.nbDims == 3);
        assert(sinDims.nbDims == 3);

        launchApplyRotaryPosEmb(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0], outputs[1], qDims.d[0],
                                qDims.d[1], kDims.d[1], qDims.d[2], qDims.d[3], outputDesc[0].type, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                   int nbOutputs) noexcept override {
        assert(nbInputs == 4);
        assert(nbOutputs == 2);
        const auto& desc = inOut[pos];
        if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
        if (pos == 0) {
            return desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF ||
                   desc.type == nvinfer1::DataType::kBF16;
        }
        return desc.type == inOut[0].type;
    }

    const char* getPluginType() const noexcept override { return kPluginName; }
    const char* getPluginVersion() const noexcept override { return kPluginVersion; }
    void destroy() noexcept override { delete this; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* plugin = new ApplyRotaryPosEmbPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const noexcept override {
        return inputTypes[0];
    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept override {
        assert(nbInputs == 4);
        assert(nbOutputs == 2);
        assert(inputs[0].desc.dims.nbDims == 4);
        assert(inputs[1].desc.dims.nbDims == 4);
        assert(inputs[2].desc.dims.nbDims == 3);
        assert(inputs[3].desc.dims.nbDims == 3);
        assert(outputs[0].desc.dims.nbDims == 4);
        assert(outputs[1].desc.dims.nbDims == 4);
    }

    void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {}
    void detachFromContext() noexcept override {}

   private:
    std::string mNamespace;
};

class ApplyRotaryPosEmbPluginCreator final : public nvinfer1::IPluginCreator {
   public:
    ApplyRotaryPosEmbPluginCreator() {
        mFieldCollection.nbFields = 0;
        mFieldCollection.fields = nullptr;
    }

    const char* getPluginName() const noexcept override { return kPluginName; }
    const char* getPluginVersion() const noexcept override { return kPluginVersion; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFieldCollection; }

    nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection*) noexcept override {
        return new ApplyRotaryPosEmbPlugin();
    }

    nvinfer1::IPluginV2* deserializePlugin(const char*, const void* serialData, size_t serialLength) noexcept override {
        return new ApplyRotaryPosEmbPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

   private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(ApplyRotaryPosEmbPluginCreator);

nvinfer1::IPluginV2* createApplyRotaryPosEmbPlugin() {
    auto* registry = getPluginRegistry();
    assert(registry != nullptr);
    auto* creator = registry->getPluginCreator(kPluginName, kPluginVersion);
    assert(creator != nullptr);

    nvinfer1::PluginFieldCollection fieldCollection{};
    fieldCollection.nbFields = 0;
    fieldCollection.fields = nullptr;
    auto* plugin = creator->createPlugin(kPluginName, &fieldCollection);
    assert(plugin != nullptr);
    return plugin;
}
