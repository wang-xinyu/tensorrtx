#include "yololayer.h"
#include "cuda_utils.h"

#include <cassert>
#include <vector>
#include <iostream>

namespace Tn {
template<typename T> 
void write(char*& buffer, const T& val) {
  *reinterpret_cast<T*>(buffer) = val;
  buffer += sizeof(T);
}

template<typename T> 
void read(const char*& buffer, T& val) {
  val = *reinterpret_cast<const T*>(buffer);
  buffer += sizeof(T);
}
}

namespace nvinfer1 {
YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<YoloKernel>& vYoloKernel) {
  mClassCount = classCount;
  mYoloV5NetWidth = netWidth;
  mYoloV5NetHeight = netHeight;
  mMaxOutObject = maxOut;
  is_segmentation_ = is_segmentation;
  mYoloKernel = vYoloKernel;
  mKernelCount = vYoloKernel.size();

  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2;
  for (int ii = 0; ii < mKernelCount; ii++) {
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
  }
}

YoloLayerPlugin::~YoloLayerPlugin() {
  for (int ii = 0; ii < mKernelCount; ii++) {
    CUDA_CHECK(cudaFree(mAnchor[ii]));
  }
  CUDA_CHECK(cudaFreeHost(mAnchor));
}

// create the plugin at runtime from a byte stream
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
  using namespace Tn;
  const char *d = reinterpret_cast<const char *>(data), *a = d;
  read(d, mClassCount);
  read(d, mThreadCount);
  read(d, mKernelCount);
  read(d, mYoloV5NetWidth);
  read(d, mYoloV5NetHeight);
  read(d, mMaxOutObject);
  read(d, is_segmentation_);
  mYoloKernel.resize(mKernelCount);
  auto kernelSize = mKernelCount * sizeof(YoloKernel);
  memcpy(mYoloKernel.data(), d, kernelSize);
  d += kernelSize;
  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2;
  for (int ii = 0; ii < mKernelCount; ii++) {
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
  }
  assert(d == a + length);
}

void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  using namespace Tn;
  char* d = static_cast<char*>(buffer), *a = d;
  write(d, mClassCount);
  write(d, mThreadCount);
  write(d, mKernelCount);
  write(d, mYoloV5NetWidth);
  write(d, mYoloV5NetHeight);
  write(d, mMaxOutObject);
  write(d, is_segmentation_);
  auto kernelSize = mKernelCount * sizeof(YoloKernel);
  memcpy(d, mYoloKernel.data(), kernelSize);
  d += kernelSize;

  assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
  size_t s = sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount);
  s += sizeof(YoloKernel) * mYoloKernel.size();
  s += sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight);
  s += sizeof(mMaxOutObject) + sizeof(is_segmentation_);
  return s;
}

int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
  return 0;
}

Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT {
  //output the result to channel
  int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);
  return Dims3(totalsize + 1, 1, 1);
}

// Set plugin namespace
void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
  mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
  return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {
  return false;
}

void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT {}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

// Detach the plugin object from its execution context.
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

// Clone the plugin
IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT {
  YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, is_segmentation_, mYoloKernel);
  p->setPluginNamespace(mPluginNamespace);
  return p;
}

__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

__global__ void CalDetection(const float *input, float *output, int noElements,
    const int netwidth, const int netheight, int maxoutobject, int yoloWidth,
    int yoloHeight, const float anchors[kNumAnchor * 2], int classes, int outputElem, bool is_segmentation) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= noElements) return;

  int total_grid = yoloWidth * yoloHeight;
  int bnIdx = idx / total_grid;
  idx = idx - total_grid * bnIdx;
  int info_len_i = 5 + classes;
  if (is_segmentation) info_len_i += 32;
  const float* curInput = input + bnIdx * (info_len_i * total_grid * kNumAnchor);

  for (int k = 0; k < kNumAnchor; ++k) {
    float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
    if (box_prob < kIgnoreThresh) continue;
    int class_id = 0;
    float max_cls_prob = 0.0;
    for (int i = 5; i < 5 + classes; ++i) {
      float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
      if (p > max_cls_prob) {
        max_cls_prob = p;
        class_id = i - 5;
      }
    }
    float *res_count = output + bnIdx * outputElem;
    int count = (int)atomicAdd(res_count, 1);
    if (count >= maxoutobject) return;
    char *data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
    Detection *det = (Detection*)(data);

    int row = idx / yoloWidth;
    int col = idx % yoloWidth;

    det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
    det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

    det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
    det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
    det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
    det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
    det->conf = box_prob * max_cls_prob;
    det->class_id = class_id;

    for (int i = 0; is_segmentation && i < 32; i++) {
      det->mask[i] = curInput[idx + k * info_len_i * total_grid + (i + 5 + classes) * total_grid];
    }
  }
}

void YoloLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize) {
  int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
  for (int idx = 0; idx < batchSize; ++idx) {
    CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
  }
  int numElem = 0;
  for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
    const auto& yolo = mYoloKernel[i];
    numElem = yolo.width * yolo.height * batchSize;
    if (numElem < mThreadCount) mThreadCount = numElem;

    CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
      (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem, is_segmentation_);
  }
}


int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
  return 0;
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
  assert(fc->nbFields == 2);
  assert(strcmp(fc->fields[0].name, "netinfo") == 0);
  assert(strcmp(fc->fields[1].name, "kernels") == 0);
  int *p_netinfo = (int*)(fc->fields[0].data);
  int class_count = p_netinfo[0];
  int input_w = p_netinfo[1];
  int input_h = p_netinfo[2];
  int max_output_object_count = p_netinfo[3];
  bool is_segmentation = (bool)p_netinfo[4];
  std::vector<YoloKernel> kernels(fc->fields[1].length);
  memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(YoloKernel));
  YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, is_segmentation, kernels);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call YoloLayerPlugin::destroy()
  YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}
}

