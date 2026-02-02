#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "cuda_utils.h"
#include "types.h"
#include "yololayer.h"

__device__ float d_confThreshold = 0.4f;

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

void setPluginDeviceParams(float confThreshold) {
    cudaMemcpyToSymbol(d_confThreshold, &confThreshold, sizeof(float));
}

YoloLayerPlugin::YoloLayerPlugin(int classCount, int numberOfPoints, int maxDetections, bool isDetection,
                                 bool isSegmentation, bool isPose, bool isObb, int anchor_count) {

    mClassCount = classCount;
    mNumberOfPoints = numberOfPoints;
    mThreadCount = 256;
    mMaxDetections = maxDetections;
    mIsDetection = isDetection;
    mIsSegmentation = isSegmentation;
    mIsPose = isPose;
    mIsObb = isObb;
    mAnchorCount = anchor_count;

    /*
    std::cout << "YoloLayerPlugin created with the following parameters:" << std::endl;
    std::cout << "  Class Count: " << mClassCount << std::endl;
    std::cout << "  Number of Points: " << mNumberOfPoints << std::endl;
    std::cout << "  Confidence Threshold Keypoints: " << mConfThreshold << std::endl;
    std::cout << "  Max Detections: " << mMaxDetections << std::endl;
    std::cout << "  Is Detection: " << mIsDetection << std::endl;
    std::cout << "  Is Segmentation: " << mIsSegmentation << std::endl;
    std::cout << "  Is Pose: " << mIsPose << std::endl;
    std::cout << "  Is OBB: " << mIsObb << std::endl;
    std::cout << "  Anchor Count: " << mAnchorCount << std::endl;
    std::cout << "  Strides: ";
    for (int i = 0; i < mStridesLength; ++i) {
        std::cout << mStrides[i] << " ";
    }
    std::cout << std::endl;
    */
}

YoloLayerPlugin::~YoloLayerPlugin() {}

YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
    using namespace Tn;
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mClassCount);
    read(d, mNumberOfPoints);
    read(d, mThreadCount);
    read(d, mMaxDetections);
    read(d, mIsDetection);
    read(d, mIsSegmentation);
    read(d, mIsPose);
    read(d, mIsObb);
    read(d, mAnchorCount);

    assert(d == a + length);
}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {

    using namespace Tn;
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mNumberOfPoints);
    write(d, mThreadCount);
    write(d, mMaxDetections);
    write(d, mIsDetection);
    write(d, mIsSegmentation);
    write(d, mIsPose);
    write(d, mIsObb);
    write(d, mAnchorCount);

    assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
    return sizeof(mClassCount) + sizeof(mNumberOfPoints) + sizeof(mThreadCount) + sizeof(mMaxDetections) +
           sizeof(mIsDetection) + sizeof(mIsSegmentation) + sizeof(mIsPose) + sizeof(mIsObb) + sizeof(mAnchorCount);
}

int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
    return 0;
}

nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                                    int nbInputDims) TRT_NOEXCEPT {
    int total_size = mMaxDetections * sizeof(Detection) / sizeof(float);
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

void YoloLayerPlugin::configurePlugin(nvinfer1::PluginTensorDesc const* in, int32_t nbInput,
                                      nvinfer1::PluginTensorDesc const* out, int32_t nbOutput) TRT_NOEXCEPT {}

void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                                      IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

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
    YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mNumberOfPoints, mMaxDetections, mIsDetection,
                                             mIsSegmentation, mIsPose, mIsObb, mAnchorCount);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
                             cudaStream_t stream) TRT_NOEXCEPT {
    gatherKernelLauncher(reinterpret_cast<const float* const*>(inputs), reinterpret_cast<float*>(outputs[0]), stream,
                         batchSize);

    return 0;
}

__device__ float Logist(float data) {
    return 1.f / (1.f + expf(-data));
}

__global__ void gatherKernel(const float* input, float* output, int num_elements, int max_out_object, int class_count,
                             int nk, int output_elem, bool is_detection, bool is_segmentation, bool is_pose,
                             bool is_obb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements)
        return;

    int outputIdx = 0 * output_elem;  // TODO: ADD BATCH SUPPORT HERE
    int anchor_size = -1;
    float angle = 0.0f;

    if (is_detection) {
        anchor_size = 4 + class_count;
    } else if (is_obb) {
        anchor_size = 5 + class_count;
        angle = input[idx * (anchor_size) + 4 + class_count];
    }

    float xmin = input[idx * (anchor_size) + 0];
    float ymin = input[idx * (anchor_size) + 1];
    float xmax = input[idx * (anchor_size) + 2];
    float ymax = input[idx * (anchor_size) + 3];

    float score = 0.0f;
    int class_id = -1;
    for (int c = 0; c < class_count; c++) {
        float conf = input[idx * (anchor_size) + 4 + c];
        if (conf > score) {
            score = conf;
            class_id = c;
        }
    }

    if (score < d_confThreshold) {
        return;
    }

    int count = (int)atomicAdd(output + outputIdx, 1);
    if (count >= max_out_object) {
        return;
    }

    int det_size = sizeof(Detection) / sizeof(float);
    Detection* det = (Detection*)(output + outputIdx + 1 + count * det_size);

    /*
    float scale = fminf(640.0f / 1080.0f, 640.0f / 608.0f);    // TODO: GET FROM PARAMETERS WITH SCALE!
    float offset_x = -scale * 1080.0f / 2.0f + 640.0f / 2.0f;  // TODO: GET FROM PARAMETERS WITH OFFSET!
    float offset_y = -scale * 608.0f / 2.0f + 640.0f / 2.0f;   // TODO: GET FROM PARAMETERS WITH OFFSET!
    

    det->conf = score;
    det->class_id = 1;  // TODO: ADD CLASS ID HERE
    det->bbox[0] = (xmin - offset_x) / scale;
    det->bbox[1] = (ymin - offset_y) / scale;
    det->bbox[2] = (xmax - offset_x) / scale;
    det->bbox[3] = (ymax - offset_y) / scale;
    */

    det->conf = score;
    det->class_id = class_id;
    det->bbox[0] = xmin;
    det->bbox[1] = ymin;
    det->bbox[2] = xmax;
    det->bbox[3] = ymax;

    if (is_obb) {
        det->angle = angle;
    }

    // TODO: ADD KEYPOINTS, SEGMENTATION, OBB HERE
}

void YoloLayerPlugin::gatherKernelLauncher(const float* const* inputs, float* outputs, cudaStream_t stream,
                                           int batchSize) {
    // TODO: ADD BATCH SUPPORT, CURRENTLY ONLY BATCH=1 IS SUPPORTED
    // TODO: ADD SEGMENTATION, POSE, OBB SUPPORT
    // TODO: num_elem = batch_size * anchor_num
    const float* input = inputs[0];

    int outputElem = mMaxDetections * sizeof(Detection) / sizeof(float) + 1;
    int num_elem = mAnchorCount;  // Use anchor count from model configuration

    dim3 blockSize(mThreadCount);
    dim3 gridSize((num_elem + mThreadCount - 1) / mThreadCount);

    cudaMemsetAsync(outputs, 0, batchSize * outputElem * sizeof(float), stream);  // TODO: adjust for batch size

    gatherKernel<<<gridSize, blockSize, 0, stream>>>(input, outputs, num_elem, mMaxDetections, mClassCount,
                                                     mNumberOfPoints, outputElem, mIsDetection, mIsSegmentation,
                                                     mIsPose, mIsObb);
}

PluginFieldCollection YoloLayerPluginCreator::mFC{};
std::vector<PluginField> YoloLayerPluginCreator::mPluginAttributes;

YoloLayerPluginCreator::YoloLayerPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloLayerPluginCreator::getPluginName() const TRT_NOEXCEPT {
    return "YoloLayer_TRT";
}

const char* YoloLayerPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
    return "1";
}

const PluginFieldCollection* YoloLayerPluginCreator::getFieldNames() TRT_NOEXCEPT {
    return &mFC;
}

IPluginV2IOExt* YoloLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT {

    assert(fc->nbFields == 1);
    assert(strcmp(fc->fields[0].name, "combinedInfo") == 0);
    const int* combinedInfo = static_cast<const int*>(fc->fields[0].data);
    int net_info_count = fc->fields[0].length;
    int class_count = combinedInfo[0];
    int number_of_points = combinedInfo[1];
    int max_detections = combinedInfo[2];
    bool is_detection = combinedInfo[3];
    bool is_segmentation = combinedInfo[4];
    bool is_pose = combinedInfo[5];
    bool is_obb = combinedInfo[6];
    int anchor_count = combinedInfo[7];

    YoloLayerPlugin* plugin = new YoloLayerPlugin(class_count, number_of_points, max_detections, is_detection,
                                                  is_segmentation, is_pose, is_obb, anchor_count);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2IOExt* YoloLayerPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                          size_t serialLength) TRT_NOEXCEPT {
    YoloLayerPlugin* plugin = new YoloLayerPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

}  // namespace nvinfer1