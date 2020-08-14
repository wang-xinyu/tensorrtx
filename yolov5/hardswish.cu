#include <assert.h>
#include "hardswish.h"
#include "utils.h"

namespace nvinfer1
{
    HardSwishPlugin::HardSwishPlugin()
    {
    }

    HardSwishPlugin::~HardSwishPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    HardSwishPlugin::HardSwishPlugin(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        Tn::read(d, mInputSize);
        assert(d == a + length);
    }

    void HardSwishPlugin::serialize(void* buffer) const
    {
        char* d = static_cast<char*>(buffer), *a = d;
        Tn::write(d, mInputSize);
        assert(d == a + getSerializationSize());
    }

    size_t HardSwishPlugin::getSerializationSize() const
    {
        return sizeof(mInputSize);
    }

    int HardSwishPlugin::initialize()
    {
        return 0;
    }

    Dims HardSwishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    // Set plugin namespace
    void HardSwishPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* HardSwishPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType HardSwishPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool HardSwishPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool HardSwishPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void HardSwishPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
        mInputSize = in[0].dims.d[0] * in[0].dims.d[1] * in[0].dims.d[2];
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void HardSwishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void HardSwishPlugin::detachFromContext() {}

    const char* HardSwishPlugin::getPluginType() const
    {
        return "HardSwishLayer_TRT";
    }

    const char* HardSwishPlugin::getPluginVersion() const
    {
        return "1";
    }

    void HardSwishPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* HardSwishPlugin::clone() const
    {
        HardSwishPlugin *p = new HardSwishPlugin();
        p->setPluginNamespace(mPluginNamespace);
        p->setInputSize(mInputSize);
        return p;
    }


    __global__ void HardSwishKer(const float *in, float *out, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= size)
            return;

        if (in[idx] >= 3.0f)
            out[idx] = in[idx];
        else if (in[idx] < -3.0f)
            out[idx] = 0.0f;
        else
            out[idx] = in[idx] * (in[idx] + 3.0f) / 6.0f;
    }

    void HardSwishPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        int numElem = batchSize * mInputSize;
        HardSwishKer<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount>>>
            (inputs[0], output, numElem);
    }


    int HardSwishPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection HardSwishPluginCreator::mFC{};
    std::vector<PluginField> HardSwishPluginCreator::mPluginAttributes;

    HardSwishPluginCreator::HardSwishPluginCreator()
    {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* HardSwishPluginCreator::getPluginName() const
    {
            return "HardSwishLayer_TRT";
    }

    const char* HardSwishPluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* HardSwishPluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* HardSwishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        HardSwishPlugin* obj = new HardSwishPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* HardSwishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        HardSwishPlugin* obj = new HardSwishPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
