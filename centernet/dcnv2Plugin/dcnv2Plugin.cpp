#include "dcnv2Plugin.h"
#include <iostream>

using namespace nvinfer1;
using nvinfer1::plugin::DeformableConvolutionalLayer;
using nvinfer1::plugin::DCNv2PluginCreator;

namespace
{
const char* DCNv2_PLUGIN_VERSION{"1"};
const char* DCNv2_PLUGIN_NAME{"DCNv2_TRT"};
} // namespace

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

PluginFieldCollection DCNv2PluginCreator::mFC{};
std::vector<PluginField> DCNv2PluginCreator::mPluginAttributes;

// Parameterized constructor
DeformableConvolutionalLayer::DeformableConvolutionalLayer(
                         int out_channels,
                         int kernel,
                         int deformable_group,
                         int dilation,
                         int padding,
                         int stride,
                         const Weights* weight, const Weights* bias):
                         out_channels(out_channels),kernel_size(kernel),deformable_group(deformable_group),
                         dilation(dilation),padding(padding),stride(stride){
        mWeight = copyToDevice(weight[0].values, weight[0].count);
        mBias = copyToDevice(bias[0].values, bias[0].count);
}

DeformableConvolutionalLayer::DeformableConvolutionalLayer(const void* buffer, size_t length)
{
    const char* d = static_cast<const char*>(buffer);
    const char* a = d;
    in_channels = read<int>(d);
    height = read<int>(d);
    width = read<int>(d);
    height_out = read<int>(d);
    width_out = read<int>(d);

    out_channels = read<int>(d);
    kernel_size = read<int>(d);
    deformable_group = read<int>(d);
    dilation = read<int>(d);
    padding = read<int>(d);
    stride = read<int>(d);

    int count = read<int>(d);
    mWeight = deserializeToDevice(d, count);
    count = read<int>(d);
    mBias = deserializeToDevice(d, count);

    ASSERT(d == a + length);
}

int DeformableConvolutionalLayer::getNbOutputs() const
{
    // Plugin layer has 2 outputs
    return 1;
}

int DeformableConvolutionalLayer::initialize()
{
    size_t oneSize = height_out * width_out * sizeof(float);
    std::vector<float> one_((int)oneSize, 1.0f);
    CHECK_CUDA(cudaMalloc((void**)&mOne, oneSize));
    CHECK_CUDA(cudaMalloc((void**)&mColumn, in_channels * kernel_size * kernel_size * oneSize));
    CHECK_CUDA(cudaMemcpy(mOne, one_.data(), oneSize, cudaMemcpyHostToDevice));
    return STATUS_SUCCESS; 
}

Dims DeformableConvolutionalLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputs)
{
    ASSERT(index == 0);
    ASSERT(nbInputs == 3);

    in_channels = inputs[0].d[0];
    height = inputs[0].d[1];
    width = inputs[0].d[2];
    height_out = (inputs[0].d[1] + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    width_out = (inputs[0].d[2] + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;

    return Dims3(out_channels, height_out, width_out);
}

size_t DeformableConvolutionalLayer::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int DeformableConvolutionalLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const float* input = static_cast<const float *>(inputs[0]);
    const float* offset = static_cast<const float *>(inputs[1]);
    const float* offset_mask = static_cast<const float *>(inputs[2]);
    const float* mask = offset_mask + deformable_group * 2 * kernel_size * kernel_size * height * width;
    float * output = static_cast<float *>(outputs[0]);

    float alpha{1}, beta{0};

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    // (N x 1) (1 x M)
    int m_ = out_channels;
    int n_ = height_out * width_out;
    int k_ = 1;
    cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, n_, m_, k_, &alpha,
                mOne, k_,
                static_cast<const float *>(mBias.values), k_, &beta,
                output, n_);

    modulated_deformable_im2col_cuda(stream, input, offset, mask,
                                    1, in_channels, height, width,
                                    height_out, width_out, kernel_size, kernel_size,
                                    padding, padding, stride, stride, dilation, dilation,
                                    deformable_group, mColumn); 

    //(k * m)  x  (m * n)
    // Y = WC
    int m = out_channels;
    int n = height_out * width_out;
    int k = in_channels * kernel_size * kernel_size;
    cublasSgemm(mCublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                mColumn, n,
                static_cast<const float *>(mWeight.values), k, &alpha,
                output, n);
    
    return 0;
}

size_t DeformableConvolutionalLayer::getSerializationSize() const
{
    return sizeof(int) * 13 + (mWeight.count + mBias.count) * sizeof(float);
}

void DeformableConvolutionalLayer::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, in_channels);
    write(d, height);
    write(d, width);
    write(d, height_out);
    write(d, width_out);

    write(d, out_channels);    
    write(d, kernel_size);
    write(d, deformable_group);
    write(d, dilation);
    write(d, padding);
    write(d, stride);

    write(d, (int) mWeight.count);
    serializeFromDevice(d, mWeight);
    write(d, (int) mBias.count);
    serializeFromDevice(d, mBias);

    ASSERT(d == a + getSerializationSize());
}

bool DeformableConvolutionalLayer::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

Weights DeformableConvolutionalLayer::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void DeformableConvolutionalLayer::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    CUASSERT(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights DeformableConvolutionalLayer::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

const char* DeformableConvolutionalLayer::getPluginType() const
{
    return DCNv2_PLUGIN_NAME;
}

const char* DeformableConvolutionalLayer::getPluginVersion() const
{
    return DCNv2_PLUGIN_VERSION;
}

void DeformableConvolutionalLayer::terminate() {
        if (mOne)
        {
            cudaFree(mOne);
            mOne = nullptr;
        }
        if (mColumn)
        {
            cudaFree(mColumn);
            mColumn = nullptr;
        }
}

void DeformableConvolutionalLayer::destroy()
{
    delete this;
}

IPluginV2Ext* DeformableConvolutionalLayer::clone() const
{
    IPluginV2Ext* plugin = new DeformableConvolutionalLayer(*this);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

// Set plugin namespace
void DeformableConvolutionalLayer::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* DeformableConvolutionalLayer::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType DeformableConvolutionalLayer::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}
// Return true if output tensor is broadcast across a batch.
bool DeformableConvolutionalLayer::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool DeformableConvolutionalLayer::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
// inutDims: input Dimensions for the plugin layer
// nInputs : Number of inputs to the plugin layer
// outputDims: output Dimensions from the plugin layer
// nOutputs: number of outputs from the plugin layer
// type: DataType configuration for the plugin layer
// format: format NCHW, NHWC etc
// maxbatchSize: maximum batch size for the plugin layer
void DeformableConvolutionalLayer::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    ASSERT(*inputTypes == DataType::kFLOAT && floatFormat == PluginFormat::kNCHW);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void DeformableConvolutionalLayer::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
    mCublas = cublasContext;
}

// Detach the plugin object from its execution context.
void DeformableConvolutionalLayer::detachFromContext() {}

DCNv2PluginCreator::DCNv2PluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("out_channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("deformable_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DCNv2PluginCreator::getPluginName() const
{
    return DCNv2_PLUGIN_NAME;
}

const char* DCNv2PluginCreator::getPluginVersion() const
{
    return DCNv2_PLUGIN_VERSION;
}

const PluginFieldCollection* DCNv2PluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* DCNv2PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::vector<float> weight;
    std::vector<float> bias;
    int out_channels, kernel, deformable_group, padding, stride, dilation;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "out_channels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            out_channels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "deformable_group"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            deformable_group = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "dilation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            dilation = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            stride = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            padding = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "weight"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weight.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weight.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            bias.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                bias.push_back(*w);
                w++;
            }
        }
    }

    Weights mWeight{DataType::kFLOAT, weight.data(), (int64_t) weight.size()};
    Weights mBias{DataType::kFLOAT, bias.data(), (int64_t) bias.size()};

    DeformableConvolutionalLayer* obj = new DeformableConvolutionalLayer(out_channels,
                         kernel,
                         deformable_group,
                         dilation,
                         padding,
                         stride,
                         &mWeight, &mBias);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* DCNv2PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call Normalize::destroy()
    DeformableConvolutionalLayer* obj = new DeformableConvolutionalLayer(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}