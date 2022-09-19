#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include "macros.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.5f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 640;  // yolov7's input height and width must be divisible by 32.
    static constexpr int INPUT_W = 640;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 8,
        INPUT_H / 8,
        {12,16, 19,36, 40,28}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {36,75, 76,55, 72,146}
    };
    static constexpr YoloKernel yolo3 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {142,110, 192,243, 459,401}
    };


}

namespace nvinfer1
{
    class API YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const TRT_NOEXCEPT override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

        int initialize() TRT_NOEXCEPT override;

        virtual void terminate() TRT_NOEXCEPT override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

        virtual int enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

        virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

        virtual void serialize(void* buffer) const TRT_NOEXCEPT override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override;

        IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

        void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

        const char* getPluginNamespace() const TRT_NOEXCEPT override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

        void detachFromContext() TRT_NOEXCEPT override;

    private:
        void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        int mThreadCount2 = (Yolo::CLASS_NUM+5)*3+1;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class API YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

        void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const TRT_NOEXCEPT override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif  // _YOLO_LAYER_H
