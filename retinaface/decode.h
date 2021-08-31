#ifndef _DECODE_CU_H
#define _DECODE_CU_H

#include <string>
#include <vector>
#include "NvInfer.h"
#include "macros.h"

namespace decodeplugin
{
    struct alignas(float) Detection{
        float bbox[4];  //x1 y1 x2 y2
        float class_confidence;
        float landmark[10];
    };
    static const int INPUT_H = 480;
    static const int INPUT_W = 640;
}

namespace nvinfer1
{
    class DecodePlugin: public IPluginV2IOExt
    {
        public:
            DecodePlugin();
            DecodePlugin(const void* data, size_t length);

            ~DecodePlugin();

            int getNbOutputs() const TRT_NOEXCEPT override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

            int initialize() TRT_NOEXCEPT override;

            virtual void terminate() TRT_NOEXCEPT override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

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

            int input_size_;
        private:
            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
            int thread_count_ = 256;
            const char* mPluginNamespace;
    };

    class DecodePluginCreator : public IPluginCreator
    {
        public:
            DecodePluginCreator();

            ~DecodePluginCreator() override = default;

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
    REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
};

#endif 
