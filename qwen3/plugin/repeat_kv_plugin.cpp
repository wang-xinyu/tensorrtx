#include "repeat_kv_plugin.h"

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

void launchRepeatKV(const void* input, void* output, int batch, int heads, int seq, int headDim, int groups,
                    nvinfer1::DataType type, cudaStream_t stream);

namespace {

constexpr char kPluginName[] = "RepeatKVPlugin";
constexpr char kPluginVersion[] = "1";

class RepeatKVPlugin final : public nvinfer1::IPluginV2DynamicExt {
   public:
    explicit RepeatKVPlugin(int groups) : mGroups(groups) { assert(groups > 0); }

    RepeatKVPlugin(const void* data, size_t length) {
        assert(length == sizeof(mGroups));
        std::memcpy(&mGroups, data, sizeof(mGroups));
        assert(mGroups > 0);
    }

    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        assert(outputIndex == 0);
        assert(nbInputs == 1);
        nvinfer1::DimsExprs output = inputs[0];
        output.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[0].d[1],
                                            *exprBuilder.constant(mGroups));
        return output;
    }

    int initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int, const nvinfer1::PluginTensorDesc*,
                            int) const noexcept override {
        return 0;
    }

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        const auto& inDims = inputDesc[0].dims;
        assert(inDims.nbDims == 4);
        launchRepeatKV(inputs[0], outputs[0], inDims.d[0], inDims.d[1], inDims.d[2], inDims.d[3], mGroups,
                       outputDesc[0].type, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return sizeof(mGroups); }

    void serialize(void* buffer) const noexcept override { std::memcpy(buffer, &mGroups, sizeof(mGroups)); }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                   int nbOutputs) noexcept override {
        assert(nbInputs == 1);
        assert(nbOutputs == 1);
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
        auto* plugin = new RepeatKVPlugin(mGroups);
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
        assert(nbInputs == 1);
        assert(nbOutputs == 1);
        assert(inputs[0].desc.dims.nbDims == 4);
        assert(outputs[0].desc.dims.nbDims == 4);
    }

    void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {}

    void detachFromContext() noexcept override {}

   private:
    int mGroups{};
    std::string mNamespace;
};

class RepeatKVPluginCreator final : public nvinfer1::IPluginCreator {
   public:
    RepeatKVPluginCreator() {
        mFields.emplace_back(nvinfer1::PluginField{"groups", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
        mFieldCollection.nbFields = static_cast<int>(mFields.size());
        mFieldCollection.fields = mFields.data();
    }

    const char* getPluginName() const noexcept override { return kPluginName; }

    const char* getPluginVersion() const noexcept override { return kPluginVersion; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFieldCollection; }

    nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        int groups = 1;
        for (int i = 0; i < fc->nbFields; ++i) {
            const auto& field = fc->fields[i];
            if (std::strcmp(field.name, "groups") == 0) {
                groups = *static_cast<const int*>(field.data);
            }
        }
        return new RepeatKVPlugin(groups);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char*, const void* serialData, size_t serialLength) noexcept override {
        return new RepeatKVPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

   private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(RepeatKVPluginCreator);

nvinfer1::IPluginV2* createRepeatKVPlugin(int groups) {
    auto* registry = getPluginRegistry();
    assert(registry != nullptr);

    auto* creator = registry->getPluginCreator(kPluginName, kPluginVersion);
    assert(creator != nullptr);

    nvinfer1::PluginField field{"groups", &groups, nvinfer1::PluginFieldType::kINT32, 1};
    nvinfer1::PluginFieldCollection fieldCollection{};
    fieldCollection.nbFields = 1;
    fieldCollection.fields = &field;

    auto* plugin = creator->createPlugin(kPluginName, &fieldCollection);
    assert(plugin != nullptr);
    return plugin;
}
