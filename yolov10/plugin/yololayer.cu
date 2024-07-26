#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "types.h"
#include "yololayer.h"

namespace Tn {
template <typename T>
void write(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}
}  // namespace Tn

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

namespace nvinfer1 {
YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const int* strides,
                                 int stridesLength) {

    mClassCount = classCount;
    mYoloV10NetWidth = netWidth;
    mYoloV10netHeight = netHeight;
    mMaxOutObject = maxOut;
    mStridesLength = stridesLength;
    mStrides = new int[stridesLength];
    memcpy(mStrides, strides, stridesLength * sizeof(int));
}

YoloLayerPlugin::~YoloLayerPlugin() {
    if (mStrides != nullptr) {
        delete[] mStrides;
        mStrides = nullptr;
    }
}

YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
    using namespace Tn;
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mClassCount);
    read(d, mThreadCount);
    read(d, mYoloV10NetWidth);
    read(d, mYoloV10netHeight);
    read(d, mMaxOutObject);
    read(d, mStridesLength);
    mStrides = new int[mStridesLength];
    for (int i = 0; i < mStridesLength; ++i) {
        read(d, mStrides[i]);
    }

    assert(d == a + length);
}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {

    using namespace Tn;
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mThreadCount);
    write(d, mYoloV10NetWidth);
    write(d, mYoloV10netHeight);
    write(d, mMaxOutObject);
    write(d, mStridesLength);
    for (int i = 0; i < mStridesLength; ++i) {
        write(d, mStrides[i]);
    }

    assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
    return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mYoloV10netHeight) + sizeof(mYoloV10NetWidth) +
           sizeof(mMaxOutObject) + sizeof(mStridesLength) + sizeof(int) * mStridesLength;
}

int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
    return 0;
}

nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                                    int nbInputDims) TRT_NOEXCEPT {
    int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
    return nvinfer1::Dims3(total_size + 1, 1, 1);
}

void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
    return mPluginNamespace;
}

nvinfer1::DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                      int nbInputs) const TRT_NOEXCEPT {
    return nvinfer1::DataType::kFLOAT;
}

bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
                                                   int nbInputs) const TRT_NOEXCEPT {
    return false;
}

bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {
    return false;
}

void YoloLayerPlugin::configurePlugin(nvinfer1::PluginTensorDesc const* in, int nbInput,
                                      nvinfer1::PluginTensorDesc const* out, int nbOutput) TRT_NOEXCEPT{};

void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                                      IGpuAllocator* gpuAllocator) TRT_NOEXCEPT{};

void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT {

    return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT {
    return "1";
}

void YoloLayerPlugin::destroy() TRT_NOEXCEPT {
    delete this;
}

nvinfer1::IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT {

    YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV10NetWidth, mYoloV10netHeight, mMaxOutObject, mStrides,
                                             mStridesLength);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

int YoloLayerPlugin::enqueue(int batchSize, const void* TRT_CONST_ENQUEUE* inputs, void* const* outputs,
                             void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
    forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, mYoloV10netHeight, mYoloV10NetWidth, batchSize);
    return 0;
}

__device__ float Logist(float data) {
    return 1.0f / (1.0f + expf(-data));
};

__global__ void CalDetection(const float* input, float* output, int numElements, int maxoutobject, const int grid_h,
                             int grid_w, const int stride, int classes, int outputElem) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numElements)
        return;

    int total_grid = grid_h * grid_w;
    int info_len = 4 + classes;
    int batchIdx = idx / total_grid;
    int elemIdx = idx % total_grid;
    const float* curInput = input + batchIdx * total_grid * info_len;
    int outputIdx = batchIdx * outputElem;

    int class_id = 0;
    float max_cls_prob = 0.0;
    for (int i = 4; i < 4 + classes; i++) {
        float p = Logist(curInput[elemIdx + i * total_grid]);
        if (p > max_cls_prob) {
            max_cls_prob = p;
            class_id = i - 4;
        }
    }

    if (max_cls_prob < 0.1)
        return;

    int count = (int)atomicAdd(output + outputIdx, 1);
    if (count >= maxoutobject)
        return;
    char* data = (char*)(output + outputIdx) + sizeof(float) + count * sizeof(Detection);
    Detection* det = (Detection*)(data);

    int row = elemIdx / grid_w;
    int col = elemIdx % grid_w;

    det->conf = max_cls_prob;
    det->class_id = class_id;
    det->bbox[0] = (col + 0.5f - curInput[elemIdx + 0 * total_grid]) * stride;
    det->bbox[1] = (row + 0.5f - curInput[elemIdx + 1 * total_grid]) * stride;
    det->bbox[2] = (col + 0.5f + curInput[elemIdx + 2 * total_grid]) * stride;
    det->bbox[3] = (row + 0.5f + curInput[elemIdx + 3 * total_grid]) * stride;
}

void YoloLayerPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int mYoloV10netHeight,
                                 int mYoloV10NetWidth, int batchSize) {
    int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    for (int idx = 0; idx < batchSize; ++idx) {
        CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
    }
    int numElem = 0;

    //    const int maxGrids = mStridesLength;
    //    int grids[maxGrids][2];
    //    for (int i = 0; i < maxGrids; ++i) {
    //        grids[i][0] = mYoloV10netHeight / mStrides[i];
    //        grids[i][1] = mYoloV10NetWidth / mStrides[i];
    //    }

    int maxGrids = mStridesLength;
    int flatGridsLen = 2 * maxGrids;
    int* flatGrids = new int[flatGridsLen];

    for (int i = 0; i < maxGrids; ++i) {
        flatGrids[2 * i] = mYoloV10netHeight / mStrides[i];
        flatGrids[2 * i + 1] = mYoloV10NetWidth / mStrides[i];
    }

    for (unsigned int i = 0; i < maxGrids; i++) {
        // Access the elements of the original 2D array from the flattened 1D array
        int grid_h = flatGrids[2 * i];      // Corresponds to the access of grids[i][0]
        int grid_w = flatGrids[2 * i + 1];  // Corresponds to the access of grids[i][1]
        int stride = mStrides[i];
        numElem = grid_h * grid_w * batchSize;  // Calculate the total number of elements
        if (numElem < mThreadCount)             // Adjust the thread count if needed
            mThreadCount = numElem;

        // The CUDA kernel call remains unchanged
        CalDetection<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>(
                inputs[i], output, numElem, mMaxOutObject, grid_h, grid_w, stride, mClassCount, outputElem);
    }

    delete[] flatGrids;
}

PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

YoloPluginCreator::YoloPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT {
    return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
    return "1";
}

const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT {
    return &mFC;
}

IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT {
    assert(fc->nbFields == 1);
    assert(strcmp(fc->fields[0].name, "combinedInfo") == 0);
    const int* combinedInfo = static_cast<const int*>(fc->fields[0].data);
    int netinfo_count = 4;
    int class_count = combinedInfo[0];
    int input_w = combinedInfo[1];
    int input_h = combinedInfo[2];
    int max_output_object_count = combinedInfo[3];
    const int* px_arry = combinedInfo + netinfo_count;
    int px_arry_length = fc->fields[0].length - netinfo_count;
    YoloLayerPlugin* obj =
            new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, px_arry, px_arry_length);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                     size_t serialLength) TRT_NOEXCEPT {
    // This object will be deleted when the network is destroyed, which will
    // call YoloLayerPlugin::destroy()
    YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

}  // namespace nvinfer1
