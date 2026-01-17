#ifndef LAYER_NORM_PLUGIN_H
#define LAYER_NORM_PLUGIN_H

#include <NvInfer.h>
#include <vector>
#include <string>

using namespace nvinfer1;

class LayerNormPlugin : public IPluginV2DynamicExt {
public:
    LayerNormPlugin(const std::string& name, float epsilon, int hidden_size);
    LayerNormPlugin(const std::string& name, const void* data, size_t length);
    LayerNormPlugin() = delete;
    ~LayerNormPlugin() override;

    // IPluginV2DynamicExt Methods
    IPluginV2DynamicExt* clone() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    DataType getOutputDataType(int32_t index, const DataType* inputTypes, int32_t nbInputs) const noexcept override;
    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    void destroy() noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mName;
    std::string mNamespace;
    float mEpsilon;
    int mHiddenSize; // Number of channels
};

class LayerNormPluginCreator : public IPluginCreator {
public:
    LayerNormPluginCreator();
    ~LayerNormPluginCreator() override;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // LAYER_NORM_PLUGIN_H
