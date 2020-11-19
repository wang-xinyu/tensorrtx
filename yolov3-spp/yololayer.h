#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;

    struct YoloKernel
    {
        int width;
        int height;
        int stride;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        -1,  // dynamic width and height
        -1,
        32,
        {116,90,  156,198,  373,326}
    };
    static constexpr YoloKernel yolo2 = {
        -1,
        -1,
        16,
        {30,61,  62,45,  59,119}
    };
    static constexpr YoloKernel yolo3 = {
        -1,
        -1,
        8,
        {10,13,  16,30,  33,23}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginV2DynamicExt
    {
        public:
            explicit YoloLayerPlugin();
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            //virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) final;
            virtual DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;

            int initialize() override;

            virtual void terminate() override {};

            //virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}
            size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override { return 0; }

            //virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
            int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() const override;

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            IPluginV2DynamicExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override;

            const char* getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;

            void detachFromContext() override;

        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<Yolo::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            void** mAnchor;
            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
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

#endif 
