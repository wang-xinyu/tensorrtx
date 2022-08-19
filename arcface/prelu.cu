#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "prelu.h"

namespace nvinfer1
{
    PReluPlugin::PReluPlugin(const std::vector<float>& gamma) : gamma_(gamma)
    {
    }

    PReluPlugin::~PReluPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    PReluPlugin::PReluPlugin(const void* data, size_t length)
    {
        char *p = (char*)data;
        input_size_ = reinterpret_cast<const int*>(p)[0];
        p += sizeof(int);
        gamma_.assign((float*)p, (float*)p + (length - sizeof(int)) / sizeof(float));
    }

    void PReluPlugin::serialize(void* buffer) const TRT_NOEXCEPT 
    {
        *reinterpret_cast<int*>(buffer) = input_size_;
        char *p = reinterpret_cast<char*>(buffer);
        p += sizeof(int);
        memcpy(p, gamma_.data(), gamma_.size() * sizeof(float));
    }

    size_t PReluPlugin::getSerializationSize() const TRT_NOEXCEPT
    {  
        return sizeof(input_size_) + gamma_.size() * sizeof(float);
    }

    int PReluPlugin::initialize() TRT_NOEXCEPT
    { 
        return 0;
    }

    Dims PReluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // Output dimensions
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    // Set plugin namespace
    void PReluPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* PReluPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType PReluPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool PReluPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool PReluPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void PReluPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void PReluPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    // Detach the plugin object from its execution context.
    void PReluPlugin::detachFromContext() TRT_NOEXCEPT {}

    const char* PReluPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "PRelu_TRT";
    }

    const char* PReluPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void PReluPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* PReluPlugin::clone() const TRT_NOEXCEPT
    {
        PReluPlugin *p = new PReluPlugin(gamma_);
        p->input_size_ = input_size_;
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __global__ void prelu_kernel(const float *input, float *output, int num_elem, int input_size, int fm_size, const float* gamma) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        if (input[idx] >= 0.0f) {
            output[idx] = input[idx];
            return;
        }
        int c = (idx % input_size) / fm_size;
        output[idx] = input[idx] * gamma[c];
    }

    void PReluPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        int block_size = thread_count_;
        int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
        void *dev_gamma;
        assert(cudaMalloc(&dev_gamma, sizeof(float) * gamma_.size()) == cudaSuccess);
        assert(cudaMemcpy(dev_gamma, gamma_.data(), sizeof(float) * gamma_.size(), cudaMemcpyHostToDevice)  == cudaSuccess);
        prelu_kernel<<<grid_size, block_size>>>(inputs[0], output, input_size_ * batchSize, input_size_, input_size_ / gamma_.size(), (const float*)dev_gamma);
        assert(cudaFree(dev_gamma) == cudaSuccess);
    }

    int PReluPlugin::enqueue(int batchSize, const void*const * inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection PReluPluginCreator::mFC{};
    std::vector<PluginField> PReluPluginCreator::mPluginAttributes;

    PReluPluginCreator::PReluPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kFLOAT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* PReluPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
            return "PRelu_TRT";
    }

    const char* PReluPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
            return "1";
    }

    const PluginFieldCollection* PReluPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
            return &mFC;
    }

    IPluginV2IOExt* PReluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
    {
        std::vector<float> gamma;
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i) {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "gamma")) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int size = fields[i].length;
                gamma.reserve(size);
                const auto* w = static_cast<const float*>(fields[i].data);
                for (int j = 0; j < size; j++)
                {
                    gamma.push_back(*w);
                    w++;
                }
            }
        }

        PReluPlugin* obj = new PReluPlugin(gamma);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* PReluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
    {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        PReluPlugin* obj = new PReluPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}

