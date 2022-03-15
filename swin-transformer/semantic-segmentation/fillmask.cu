#include "fillmask.h"
#include <math.h>
namespace nvinfer1
{
    fillmask::fillmask()
    {
    }

    fillmask::~fillmask()
    {
    }
    // create the plugin at runtime from a byte stream
    fillmask::fillmask(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        Tn::read(d, mInputSize);
        assert(d == a + length);
    }

    void fillmask::serialize(void* buffer) const
    {
        char* d = static_cast<char*>(buffer), *a = d;
        Tn::write(d, mInputSize);
        assert(d == a + getSerializationSize());
    }

    size_t fillmask::getSerializationSize() const
    {
        return sizeof(mInputSize);
    }

    int fillmask::initialize()
    {
        return 0;
    }

    Dims fillmask::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        assert(nbInputDims == 1);
        Dims outputDims;
        outputDims.nbDims = inputs[0].nbDims;
        for (int i = 0; i < inputs[0].nbDims; i++) {
            outputDims.d[i] = inputs[0].d[i];
        }
        return outputDims;
    }

    // Set plugin namespace
    void fillmask::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* fillmask::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType fillmask::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool fillmask::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool fillmask::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void fillmask::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {

        mInputSize = 1;
        for (int i = 0; i < in[0].dims.nbDims; i++) {
            mInputSize *= in[0].dims.d[i];
        }
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void fillmask::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void fillmask::detachFromContext() {}

    const char* fillmask::getPluginType() const
    {
        return "fillmaskLayer_TRT";
    }

    const char* fillmask::getPluginVersion() const
    {
        return "1";
    }

    void fillmask::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* fillmask::clone() const
    {
        fillmask *p = new fillmask();
        p->setPluginNamespace(mPluginNamespace);
        p->setInputSize(mInputSize);
        return p;
    }

    __global__ void fillmaskKer(const float *in, float *out, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= size)
            return;
        if (in[idx] != 0.0)
            out[idx] = -100.0;
        else
            out[idx] = 0.0;
    }
    void fillmask::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        int numElem = batchSize * mInputSize;
        fillmaskKer<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount>>>
            (inputs[0], output, numElem);
    }

    int fillmask::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection fillmaskCreator::mFC{};
    std::vector<PluginField> fillmaskCreator::mPluginAttributes;

    fillmaskCreator::fillmaskCreator()
    {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* fillmaskCreator::getPluginName() const
    {
            return "fillmaskLayer_TRT";
    }

    const char* fillmaskCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* fillmaskCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* fillmaskCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        fillmask* obj = new fillmask();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* fillmaskCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        fillmask* obj = new fillmask(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }


}

