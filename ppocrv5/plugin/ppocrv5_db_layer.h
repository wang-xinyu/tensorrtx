#pragma once

#include "NvInfer.h"

#include <cuda_runtime_api.h>
#include <string>
#include <vector>

void ppocrv5CudaDbSigmoid(const float* input, float* output, int count, cudaStream_t stream);
void ppocrv5EnsureDbPlugin();

namespace nvinfer1 {

class Ppocrv5DbPlugin : public IPluginV2DynamicExt {
   public:
    Ppocrv5DbPlugin() = default;
    Ppocrv5DbPlugin(const void* data, size_t length);
    ~Ppocrv5DbPlugin() override = default;

    int getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                   int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
                         int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
                            int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
                void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {}
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    void destroy() noexcept override;
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

   private:
    const char* mPluginNamespace{""};
};

class Ppocrv5DbPluginCreator : public IPluginCreator {
   public:
    Ppocrv5DbPluginCreator();
    ~Ppocrv5DbPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

   private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

}  // namespace nvinfer1
