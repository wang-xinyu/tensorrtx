#ifndef GELU_H
#define GELU_H

#include <vector>
#include <string>
#include "NvInfer.h"
#include "myhpp.h"
#include <assert.h>
#include "utilsn.h"
#define M_PI       3.14159265358979323846   // pi
namespace nvinfer1
{
    class gelu:public IPluginV2IOExt
    {
    public:
        explicit gelu();
        gelu(const void* data, size_t length);
        ~gelu();
        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
        int initialize() override;
        virtual void terminate() override {};
        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}
        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
        virtual size_t getSerializationSize() const override;
        virtual void serialize(void* buffer) const override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const override;
        const char* getPluginVersion() const override;
        void destroy() override;
        IPluginV2IOExt* clone() const override;
        void setPluginNamespace(const char* pluginNamespace) override;
        const char* getPluginNamespace() const override;
        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
        bool canBroadcastInputAcrossBatch(int inputIndex) const override;
        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;
        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
        void detachFromContext() override;

        void setInputSize(int s) {
            mInputSize = s;
        }

    private:
        void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
        int mThreadCount = 256;
        int mInputSize;
        const char* mPluginNamespace;
    };

    class geluCreator : public IPluginCreator
    {
        public:
            geluCreator();
            ~geluCreator() override = default;
            const char* getPluginName() const override;
            const char* getPluginVersion() const override;
            const PluginFieldCollection* getFieldNames() override;
            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

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
    REGISTER_TENSORRT_PLUGIN(geluCreator);
};
#endif // GELU_H
