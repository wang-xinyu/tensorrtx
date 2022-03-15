#include <assert.h>
#include "layerNorm.h"
#include "utilsn.h"
#include <assert.h>
#include <vector>





namespace nvinfer1
{

layernorm::layernorm()
{
}
layernorm::~layernorm()
{

}
layernorm::layernorm(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    Tn::read(d, mInputSize);
    Tn::read(d,Length);

    assert(d == a + length);
}
int layernorm::initialize()
{
    return 0;
}
void layernorm::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer), *a = d;
    Tn::write(d, mInputSize);
    Tn::write(d,Length);
    assert(d == a + getSerializationSize());
}
size_t layernorm::getSerializationSize() const
{
    return sizeof(mInputSize) + sizeof(Length);
}
Dims layernorm::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
//    outputDims.nbDims  = inputs[0].nbDims;
//    outputDims.d[0] = inputs[0].d[0];
//    for (int var = 1; var < inputs[0].nbDims; ++var) {
//        outputDims.d[var] = 1;
//    }
    return Dims2{inputs[0].d[0],1};
}
void layernorm::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}
const char* layernorm::getPluginNamespace() const
{
    return mPluginNamespace;
}
const char* layernorm::getPluginType() const
{
    return "layerNorm_trt";
}
const char* layernorm::getPluginVersion() const
{
    return "1";
}
DataType layernorm::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0] ;//== nvinfer1::DataType::kFLOAT ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
}
void layernorm::destroy()
{
    delete this;
}
IPluginV2IOExt* layernorm::clone() const
{
    layernorm *ln = new layernorm();
    ln->setPluginNamespace(mPluginNamespace);
    ln->setInputSize(mInputSize,Length);
    return ln;
}
bool layernorm::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}
bool layernorm::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}
void layernorm::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{}
void layernorm::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
{

    int size = 1;
    for(int i = 0 ; i < in[0].dims.nbDims ; i++)
    {
        size *= in[0].dims.d[i];
    }
    mInputSize = size;
    Length = in[0].dims.d[in[0].dims.nbDims - 1];
}
void layernorm::detachFromContext()
{}

__device__ welford welford_update(welford a, const float *currValue, int length)
{
    #pragma unroll
    for(int i = 0; i < length; i++){
        a.count += 1;
        float delta = currValue[i] - a.mean;
        a.mean += delta / a.count;
        float delta2 = currValue[i] - a.mean;
        a.M2 += delta * delta2;
    }
    return a;
}
__device__ void mean_std(float* mean, float *std, const float *currValue,int l,int count = 0, float m = 0.0, float s = 0.0)
{
    #pragma unroll
    for(int i = 0; i < l; i++){
        count += 1;
        float delta = currValue[i] - m;
        m += delta / count;
        float delta2 = currValue[i] - m;
        s += delta * delta2;
    }
    *mean = m;
    *std = sqrt((s / count) + 1e-5);
}
__global__ void lnCudaKer(const float *in, float *mean, float *std, int size,int l)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size)
        return;
    mean_std(&mean[idx],&std[idx],in+idx*l,l);
    //printf("idx = %d,mean = %f, std = %f\n",idx,mean[idx],std[idx]);
}
void layernorm::forwardGpu(const float *const *inputs, float *mean, float *std, cudaStream_t stream, int batchSize)
{
    int numElem = batchSize * mInputSize/Length;

    lnCudaKer<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount>>>
        (inputs[0], mean,std, numElem,Length);
}
int layernorm::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    forwardGpu((const float *const *)inputs, (float*)outputs[0], (float*)outputs[1], stream, batchSize);
    return 0;
}

PluginFieldCollection layernormCreator::mFC{};
std::vector<PluginField> layernormCreator::mPluginAttributes;
layernormCreator::layernormCreator()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* layernormCreator::getPluginName() const
{
    return "layerNorm_trt";
}
const char* layernormCreator::getPluginVersion() const
{
    return "1";
}
const PluginFieldCollection* layernormCreator::getFieldNames()
{
    return &mFC;
}
IPluginV2IOExt* layernormCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    layernorm* obj = new layernorm();
    obj->setPluginNamespace(mNamespace.c_str());

    return obj;
}
IPluginV2IOExt* layernormCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    layernorm* obj = new layernorm(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}






}
