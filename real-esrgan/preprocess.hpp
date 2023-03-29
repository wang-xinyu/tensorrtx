#pragma once
#include <NvInfer.h>
#include <fstream>
#include "macros.h"
#include <assert.h>

struct Preprocess {
    int N;
    int C;
    int H;
    int W;
};

namespace nvinfer1
{
    class PreprocessPluginV2 : public IPluginV2IOExt
    {
    public:
        PreprocessPluginV2(const Preprocess& arg)
        {
            mPreprocess = arg;
        }

        PreprocessPluginV2(const void* data, size_t length)
        {
            const char* d = static_cast<const char*>(data);
            const char* const a = d;
            mPreprocess = read<Preprocess>(d);
            assert(d == a + length);
        }
        PreprocessPluginV2() = delete;

        virtual ~PreprocessPluginV2() {}

    public:
        int getNbOutputs() const noexcept override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override
        {
            return Dims3(mPreprocess.C, mPreprocess.H, mPreprocess.W);
        }

        int initialize() noexcept override
        {
            return 0;
        }

        void terminate() noexcept override
        {
        }

        size_t getWorkspaceSize(int maxBatchSize) const noexcept override
        {
            return 0;
        }

        int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

        size_t getSerializationSize() const noexcept override
        {
            size_t serializationSize = 0;
            serializationSize += sizeof(mPreprocess);
            return serializationSize;
        }

        void serialize(void* buffer) const noexcept override
        {
            char* d = static_cast<char*>(buffer);
            const char* const a = d;
            write(d, mPreprocess);
            assert(d == a + getSerializationSize());
        }

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept override
        {
        }

        //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override
        {
            assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
            bool condition = inOut[pos].format == TensorFormat::kLINEAR;
            condition &= inOut[pos].type != DataType::kINT32;
            condition &= inOut[pos].type == inOut[0].type;
            return condition;
        }

        DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
        {
            assert(inputTypes && nbInputs == 1);
            return DataType::kFLOAT; //
        }

        const char* getPluginType() const noexcept override
        {
            return "preprocess";
        }

        const char* getPluginVersion() const noexcept override
        {
            return "1";
        }

        void destroy() noexcept override
        {
            delete this;
        }

        IPluginV2Ext* clone() const noexcept override
        {
            PreprocessPluginV2* plugin = new PreprocessPluginV2(*this);
            return plugin;
        }

        void setPluginNamespace(const char* libNamespace) noexcept override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const noexcept override
        {
            return mNamespace.data();
        }

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override
        {
            return false;
        }

        bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override
        {
            return false;
        }

    private:
        template <typename T>
        void write(char*& buffer, const T& val) const
        {
            *reinterpret_cast<T*>(buffer) = val;
            buffer += sizeof(T);
        }

        template <typename T>
        T read(const char*& buffer) const
        {
            T val = *reinterpret_cast<const T*>(buffer);
            buffer += sizeof(T);
            return val;
        }

    private:
        Preprocess mPreprocess;
        std::string mNamespace;
    };

    class PreprocessPluginV2Creator : public IPluginCreator
    {
    public:
        const char* getPluginName() const noexcept override
        {
            return "preprocess";
        }

        const char* getPluginVersion() const noexcept override
        {
            return "1";
        }

        const PluginFieldCollection* getFieldNames() noexcept override
        {
            return nullptr;
        }

        IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
        {
            PreprocessPluginV2* plugin = new PreprocessPluginV2(*(Preprocess*)fc);
            mPluginName = name;
            return plugin;
        }

        IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
        {
            auto plugin = new PreprocessPluginV2(serialData, serialLength);
            mPluginName = name;
            return plugin;
        }

        void setPluginNamespace(const char* libNamespace) noexcept override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const noexcept override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        std::string mPluginName;
    };
    REGISTER_TENSORRT_PLUGIN(PreprocessPluginV2Creator);
};
