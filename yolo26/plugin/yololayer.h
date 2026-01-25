#pragma once
#include <string>
#include <vector>
#include "NvInfer.h"
#include "macros.h"

namespace nvinfer1 {
class API YoloLayerPlugin : public IPluginV2IOExt {
   public:
    YoloLayerPlugin(int classCount, int numberOfPoints, float confThresholdKeypoints, int inputWidth, int inputHeight,
                    int maxDetections, bool isSegmentation, bool isPose, bool isObb, int anchor_count);
    YoloLayerPlugin(const void* data, size_t length);

    ~YoloLayerPlugin();

    int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

    int initialize() TRT_NOEXCEPT override;

    virtual void terminate() TRT_NOEXCEPT override {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

    virtual int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
                        cudaStream_t stream) TRT_NOEXCEPT override;

    virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

    virtual void serialize(void* buffer) const TRT_NOEXCEPT override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                   int nbOutputs) const TRT_NOEXCEPT override {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    const char* getPluginType() const TRT_NOEXCEPT override;

    const char* getPluginVersion() const TRT_NOEXCEPT override;

    void destroy() TRT_NOEXCEPT override;

    IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

    const char* getPluginNamespace() const TRT_NOEXCEPT override;

    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const TRT_NOEXCEPT override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                      int nbInputs) const TRT_NOEXCEPT override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                         IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

    void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out,
                         int32_t nbOutput) TRT_NOEXCEPT override;

    void detachFromContext() TRT_NOEXCEPT override;

   private:
    void gatherKernelLauncher(const float* const* inputs, float* outputs, cudaStream_t stream, int modelInputWidth,
                              int modelInputHeight, int batchSize);
    int mThreadCount = 256;
    const char* mPluginNamespace = "";
    int mClassCount;
    int mNumberOfPoints;
    float mConfThresholdKeypoints;
    int mInputWidth;
    int mInputHeight;
    int mMaxDetections;
    bool mIsSegmentation;
    bool mIsPose;
    bool mIsObb;
    int mAnchorCount;
};

class API YoloLayerPluginCreator : public IPluginCreator {
   public:
    YoloLayerPluginCreator();

    const char* getPluginName() const TRT_NOEXCEPT override;

    const char* getPluginVersion() const TRT_NOEXCEPT override;

    const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData,
                                      size_t serialLength) TRT_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override { mNamespace = pluginNamespace; }

    const char* getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

   private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
}  // namespace nvinfer1