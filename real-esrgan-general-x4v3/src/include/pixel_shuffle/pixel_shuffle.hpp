#ifndef REAL_ESRGAN_TRT_PIXEL_SHUFFLE_HPP
#define REAL_ESRGAN_TRT_PIXEL_SHUFFLE_HPP

#include <string>
#include <vector>
#include "NvInfer.h"

class PixelShufflePlugin : public nvinfer1::IPluginV2DynamicExt {
   public:
    PixelShufflePlugin(int upscaleFactor) : mUpscaleFactor(upscaleFactor) {}

    PixelShufflePlugin(const void* data, size_t length) { memcpy(&mUpscaleFactor, data, sizeof(mUpscaleFactor)); }

    const char* getPluginType() const noexcept override { return "PixelShufflePlugin"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    int getNbOutputs() const noexcept override { return 1; }

    // nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override
    // {
    //     assert(outputIndex == 0);
    //     auto* in = &inputs[0];
    //     nvinfer1::DimsExprs outputDims = *in;
    //     int channels = in->d[0];
    //     int height = in->d[1];
    //     int width = in->d[2];
    //     int upscaleFactor = mUpscaleFactor;
    //     outputDims.d[0] = exprBuilder.constant(channels / (upscaleFactor * upscaleFactor));
    //     outputDims.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, {height, exprBuilder.constant(upscaleFactor)});
    //     outputDims.d[2] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, {width, exprBuilder.constant(upscaleFactor)});
    //     return outputDims;
    // }
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        // assert(nbInputs == 1);
        auto inDims = inputs[0];
        // assert(inDims.nbDims == 4);
        int c = inDims.d[1]->getConstantValue() / (mUpscaleFactor * mUpscaleFactor);
        int h = inDims.d[2]->getConstantValue() * mUpscaleFactor;
        int w = inDims.d[3]->getConstantValue() * mUpscaleFactor;
        nvinfer1::DimsExprs outDims;
        outDims.nbDims = 4;
        outDims.d[0] = inDims.d[0];
        outDims.d[1] = exprBuilder.constant(c);
        outDims.d[2] = exprBuilder.constant(h);
        outDims.d[3] = exprBuilder.constant(w);
        return outDims;
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                   int nbOutputs) noexcept override {
        return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                         int nbInputs) const noexcept override {

        return inputTypes[0];
    }

    // bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override
    // {
    //     return false;
    // }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept override {}

    // void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override
    // {
    //     // Optionally configure plugin if necessary
    // }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return sizeof(mUpscaleFactor); }

    void serialize(void* buffer) const noexcept override { memcpy(buffer, &mUpscaleFactor, sizeof(mUpscaleFactor)); }

    void destroy() noexcept override {
        // delete this;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return new PixelShufflePlugin(mUpscaleFactor); }

    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    int initialize() noexcept override { return 0; }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

    void terminate() noexcept override {}

   private:
    int mUpscaleFactor;
    std::string mNamespace;
};

class PixelShufflePluginCreator : public nvinfer1::IPluginCreator {
   public:
    PixelShufflePluginCreator() {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(
                nvinfer1::PluginField("upscaleFactor", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~PixelShufflePluginCreator() override = default;

    const char* getPluginName() const noexcept override { return "PixelShufflePlugin"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        int upscaleFactor = 0;
        for (int i = 0; i < fc->nbFields; ++i) {
            if (strcmp(fc->fields[i].name, "upscaleFactor") == 0) {
                upscaleFactor = *static_cast<const int*>(fc->fields[i].data);
            }
        }
        return new PixelShufflePlugin(upscaleFactor);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                           size_t serialLength) noexcept override {
        return new PixelShufflePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

   private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

nvinfer1::PluginFieldCollection PixelShufflePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> PixelShufflePluginCreator::mPluginAttributes{
        nvinfer1::PluginField{"upscaleFactor", nullptr, nvinfer1::PluginFieldType::kINT32, 1}};

REGISTER_TENSORRT_PLUGIN(PixelShufflePluginCreator);

#endif  //REAL_ESRGAN_TRT_PIXEL_SHUFFLE_HPP
