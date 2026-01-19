#include <cuda_fp16.h>
#include <cassert>
#include <cstring>
#include <cub/cub.cuh>
#include <iostream>
#include "LayerNormPlugin.h"

using namespace nvinfer1;

static const char* PLUGIN_NAME = "LayerNorm";
static const char* PLUGIN_VERSION = "1";

PluginFieldCollection LayerNormPluginCreator::mFC{};
std::vector<PluginField> LayerNormPluginCreator::mPluginAttributes;

// Helper to check CUDA errors
#define CHECK(status)                                                                     \
    do {                                                                                  \
        auto ret = (status);                                                              \
        if (ret != 0) {                                                                   \
            std::cerr << "Cuda failure: " << ret << " at line " << __LINE__ << std::endl; \
            abort();                                                                      \
        }                                                                                 \
    } while (0)

template <typename T>
__device__ inline T epsilon();

template <>
__device__ inline float epsilon<float>() {
    return 1e-6f;
}

template <>
__device__ inline half epsilon<half>() {
    return (half)1e-6f;
}

// --- Kernel ---
// Supports hidden_size up to 1024 with TPB=256, VPT=4
template <typename T, int VPT>
__global__ void layerNormKernel(const T* __restrict__ input, const T* __restrict__ gamma, const T* __restrict__ beta,
                                T* __restrict__ output, int hidden_size, float eps) {
    // blockIdx.x corresponds to one instance (one row of hidden_size elements)

    int row_offset = blockIdx.x * hidden_size;

    // Load data
    float vals[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        int col = threadIdx.x * VPT + i;
        if (col < hidden_size) {
            vals[i] = (float)input[row_offset + col];
        } else {
            vals[i] = 0.0f;
        }
    }

    // Compute mean
    float thread_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        if (threadIdx.x * VPT + i < hidden_size)
            thread_sum += vals[i];
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(thread_sum);
    __shared__ float mean;
    if (threadIdx.x == 0)
        mean = sum / hidden_size;
    __syncthreads();

    // Compute variance
    float thread_sq_diff = 0.0f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        if (threadIdx.x * VPT + i < hidden_size) {
            float diff = vals[i] - mean;
            thread_sq_diff += diff * diff;
        }
    }
    float sq_diff_sum = BlockReduce(temp_storage).Sum(thread_sq_diff);
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf((sq_diff_sum / hidden_size) + eps);
    }
    __syncthreads();

// Normalize and scale
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        int col = threadIdx.x * VPT + i;
        if (col < hidden_size) {
            float val = (vals[i] - mean) * inv_std;
            float g = (float)gamma[col];
            float b = (float)beta[col];
            output[row_offset + col] = (T)(val * g + b);
        }
    }
}

// --- Plugin Implementation ---

LayerNormPlugin::LayerNormPlugin(const std::string& name, float epsilon, int hidden_size)
    : mName(name), mEpsilon(epsilon), mHiddenSize(hidden_size) {}

LayerNormPlugin::LayerNormPlugin(const std::string& name, const void* data, size_t length) : mName(name) {
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    mEpsilon = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    mHiddenSize = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    assert(d == a + length);
}

LayerNormPlugin::~LayerNormPlugin() {}

IPluginV2DynamicExt* LayerNormPlugin::clone() const noexcept {
    auto p = new LayerNormPlugin(mName, mEpsilon, mHiddenSize);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

int32_t LayerNormPlugin::getNbOutputs() const noexcept {
    return 1;
}

DataType LayerNormPlugin::getOutputDataType(int32_t index, const DataType* inputTypes,
                                            int32_t nbInputs) const noexcept {
    return inputTypes[0];
}

DimsExprs LayerNormPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs,
                                               IExprBuilder& exprBuilder) noexcept {
    return inputs[0];
}

bool LayerNormPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs,
                                                int32_t nbOutputs) noexcept {
    if (pos == 0) {  // Input
        return (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF) &&
               inOut[0].format == TensorFormat::kLINEAR;
    }
    if (pos == 1 || pos == 2) {  // Gamma, Beta
        return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
    }
    if (pos == 3) {  // Output
        return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return false;
}

void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
                                      const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {
    // Validate inputs
    mHiddenSize = in[0].desc.dims.d[in[0].desc.dims.nbDims - 1];
}

size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
                                         const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                                 const void* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {

    int total = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
        total *= inputDesc[0].dims.d[i];
    int rows = total / mHiddenSize;

    if (inputDesc[0].type == DataType::kFLOAT) {
        layerNormKernel<float, 4><<<rows, 256, 0, stream>>>((const float*)inputs[0], (const float*)inputs[1],
                                                            (const float*)inputs[2], (float*)outputs[0], mHiddenSize,
                                                            mEpsilon);
    } else {
        layerNormKernel<half, 4><<<rows, 256, 0, stream>>>((const half*)inputs[0], (const half*)inputs[1],
                                                           (const half*)inputs[2], (half*)outputs[0], mHiddenSize,
                                                           mEpsilon);
    }
    return 0;
}

const char* LayerNormPlugin::getPluginType() const noexcept {
    return PLUGIN_NAME;
}
const char* LayerNormPlugin::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

void LayerNormPlugin::destroy() noexcept {
    delete this;
}

int32_t LayerNormPlugin::initialize() noexcept {
    return 0;
}
void LayerNormPlugin::terminate() noexcept {}

size_t LayerNormPlugin::getSerializationSize() const noexcept {
    return sizeof(float) + sizeof(int);
}

void LayerNormPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    *reinterpret_cast<float*>(d) = mEpsilon;
    d += sizeof(float);
    *reinterpret_cast<int*>(d) = mHiddenSize;
    d += sizeof(int);
}

void LayerNormPlugin::setPluginNamespace(const char* libNamespace) noexcept {
    mNamespace = libNamespace;
}
const char* LayerNormPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// --- Creator Implementation ---

LayerNormPluginCreator::LayerNormPluginCreator() {
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

LayerNormPluginCreator::~LayerNormPluginCreator() {}

const char* LayerNormPluginCreator::getPluginName() const noexcept {
    return PLUGIN_NAME;
}
const char* LayerNormPluginCreator::getPluginVersion() const noexcept {
    return PLUGIN_VERSION;
}

const PluginFieldCollection* LayerNormPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    float epsilon = 1e-6f;
    for (int i = 0; i < fc->nbFields; ++i) {
        if (strcmp(fc->fields[i].name, "epsilon") == 0) {
            epsilon = *static_cast<const float*>(fc->fields[i].data);
        }
    }
    return new LayerNormPlugin(name, epsilon, 0);  // hidden_size will be set in configure
}

IPluginV2* LayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                     size_t serialLength) noexcept {
    return new LayerNormPlugin(name, serialData, serialLength);
}

void LayerNormPluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    mNamespace = libNamespace;
}
const char* LayerNormPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
