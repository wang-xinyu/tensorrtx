#include "ppocrv5_rtdetr_layer.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

namespace {

static constexpr int kNumHeads = 8;
static constexpr int kHeadDim = 32;
static constexpr int kNumLevels = 3;
static constexpr int kNumPoints = 4;

__device__ float sampleValue(const float* value, int batchIndex, int levelStart, int height, int width, int head,
                             int channel, float x, float y, int totalLength) {
    float px = x * static_cast<float>(width) - 0.5f;
    float py = y * static_cast<float>(height) - 0.5f;
    int x0 = static_cast<int>(floorf(px));
    int y0 = static_cast<int>(floorf(py));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float wx1 = px - static_cast<float>(x0);
    float wy1 = py - static_cast<float>(y0);
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    float result = 0.0f;
    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
        int offset =
                (((batchIndex * totalLength + levelStart + y0 * width + x0) * kNumHeads + head) * kHeadDim + channel);
        result += value[offset] * wx0 * wy0;
    }
    if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
        int offset =
                (((batchIndex * totalLength + levelStart + y0 * width + x1) * kNumHeads + head) * kHeadDim + channel);
        result += value[offset] * wx1 * wy0;
    }
    if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
        int offset =
                (((batchIndex * totalLength + levelStart + y1 * width + x0) * kNumHeads + head) * kHeadDim + channel);
        result += value[offset] * wx0 * wy1;
    }
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
        int offset =
                (((batchIndex * totalLength + levelStart + y1 * width + x1) * kNumHeads + head) * kHeadDim + channel);
        result += value[offset] * wx1 * wy1;
    }
    return result;
}

__global__ void deformableAttentionKernel(const float* value, const float* reference, const float* offsets,
                                          const float* weights, float* output, int batch, int query, int totalLength,
                                          int p3Size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * query * kNumHeads * kHeadDim;
    if (index >= total) {
        return;
    }

    int channel = index % kHeadDim;
    int head = (index / kHeadDim) % kNumHeads;
    int q = (index / (kHeadDim * kNumHeads)) % query;
    int b = index / (kHeadDim * kNumHeads * query);

    int refBase = (b * query + q) * 4;
    float cx = reference[refBase + 0];
    float cy = reference[refBase + 1];
    float bw = reference[refBase + 2];
    float bh = reference[refBase + 3];

    int p4Size = p3Size / 2;
    int p5Size = p3Size / 4;
    const int heights[kNumLevels] = {p3Size, p4Size, p5Size};
    const int widths[kNumLevels] = {p3Size, p4Size, p5Size};
    const int starts[kNumLevels] = {0, p3Size * p3Size, p3Size * p3Size + p4Size * p4Size};

    float sum = 0.0f;
    for (int level = 0; level < kNumLevels; ++level) {
        for (int point = 0; point < kNumPoints; ++point) {
            int offIndex = (((((b * query + q) * kNumHeads + head) * kNumLevels + level) * kNumPoints + point) * 2);
            float ox = offsets[offIndex + 0];
            float oy = offsets[offIndex + 1];
            float sampleX = cx + ox * bw * 0.5f / static_cast<float>(kNumPoints);
            float sampleY = cy + oy * bh * 0.5f / static_cast<float>(kNumPoints);
            int weightIndex = ((((b * query + q) * kNumHeads + head) * kNumLevels + level) * kNumPoints + point);
            float weight = weights[weightIndex];
            sum += weight * sampleValue(value, b, starts[level], heights[level], widths[level], head, channel, sampleX,
                                        sampleY, totalLength);
        }
    }

    int outIndex = (b * query + q) * (kNumHeads * kHeadDim) + head * kHeadDim + channel;
    output[outIndex] = sum;
}

}  // namespace

void ppocrv5CudaRtDetrDeformableAttention(const float* value, const float* reference, const float* offsets,
                                          const float* weights, float* output, int batch, int query, int totalLength,
                                          int p3Size, cudaStream_t stream) {
    int count = batch * query * kNumHeads * kHeadDim;
    if (count <= 0) {
        return;
    }
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    deformableAttentionKernel<<<blocks, threads, 0, stream>>>(value, reference, offsets, weights, output, batch, query,
                                                              totalLength, p3Size);
}

void ppocrv5EnsureRtDetrPlugin() {}

namespace nvinfer1 {

static const char* kPluginName = "Ppocrv5RtDetrPlugin";
static const char* kPluginVersion = "1";

PluginFieldCollection Ppocrv5RtDetrPluginCreator::mFC{};
std::vector<PluginField> Ppocrv5RtDetrPluginCreator::mPluginAttributes;

Ppocrv5RtDetrPlugin::Ppocrv5RtDetrPlugin(const void* data, size_t length) {
    assert(length == 0);
}

DimsExprs Ppocrv5RtDetrPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                                   IExprBuilder& exprBuilder) noexcept {
    assert(outputIndex == 0);
    assert(nbInputs == 4);
    DimsExprs out{};
    out.nbDims = 3;
    out.d[0] = inputs[1].d[0];
    out.d[1] = inputs[1].d[1];
    out.d[2] = exprBuilder.constant(kNumHeads * kHeadDim);
    return out;
}

bool Ppocrv5RtDetrPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                                    int nbOutputs) noexcept {
    assert(nbInputs == 4);
    assert(nbOutputs == 1);
    assert(pos >= 0 && pos < 5);
    const PluginTensorDesc& desc = inOut[pos];
    return desc.format == TensorFormat::kLINEAR && desc.type == DataType::kFLOAT;
}

void Ppocrv5RtDetrPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                                          const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    assert(nbInputs == 4);
    assert(nbOutputs == 1);
}

size_t Ppocrv5RtDetrPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                                             const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0;
}

int Ppocrv5RtDetrPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                                 const void* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
    int batch = inputDesc[1].dims.d[0];
    int query = inputDesc[1].dims.d[1];
    int totalLength = inputDesc[0].dims.d[1];
    int p3Size = static_cast<int>(std::sqrt(static_cast<float>(totalLength * 16 / 21)) + 0.5f);
    ppocrv5CudaRtDetrDeformableAttention(
            reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<const float*>(inputs[1]),
            reinterpret_cast<const float*>(inputs[2]), reinterpret_cast<const float*>(inputs[3]),
            reinterpret_cast<float*>(outputs[0]), batch, query, totalLength, p3Size, stream);
    CHECK(cudaPeekAtLastError());
    return 0;
}

const char* Ppocrv5RtDetrPlugin::getPluginType() const noexcept {
    return kPluginName;
}

const char* Ppocrv5RtDetrPlugin::getPluginVersion() const noexcept {
    return kPluginVersion;
}

IPluginV2DynamicExt* Ppocrv5RtDetrPlugin::clone() const noexcept {
    Ppocrv5RtDetrPlugin* plugin = new Ppocrv5RtDetrPlugin();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void Ppocrv5RtDetrPlugin::destroy() noexcept {
    delete this;
}

DataType Ppocrv5RtDetrPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
    assert(index == 0);
    assert(nbInputs == 4);
    return inputTypes[0];
}

void Ppocrv5RtDetrPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* Ppocrv5RtDetrPlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace;
}

Ppocrv5RtDetrPluginCreator::Ppocrv5RtDetrPluginCreator() {
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

const char* Ppocrv5RtDetrPluginCreator::getPluginName() const noexcept {
    return kPluginName;
}

const char* Ppocrv5RtDetrPluginCreator::getPluginVersion() const noexcept {
    return kPluginVersion;
}

const PluginFieldCollection* Ppocrv5RtDetrPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* Ppocrv5RtDetrPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    return new Ppocrv5RtDetrPlugin();
}

IPluginV2* Ppocrv5RtDetrPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                         size_t serialLength) noexcept {
    return new Ppocrv5RtDetrPlugin(serialData, serialLength);
}

void Ppocrv5RtDetrPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* Ppocrv5RtDetrPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(Ppocrv5RtDetrPluginCreator);

}  // namespace nvinfer1
