#include "yololayer.h"
#include "types.h"
#include <assert.h>
#include <math.h>

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
}  // namespace Tn


namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut) {
        mClassCount = classCount;
        mYoloV8NetWidth = netWidth;
        mYoloV8netHeight = netHeight;
        mMaxOutObject = maxOut;
    }

    YoloLayerPlugin::~YoloLayerPlugin() {}

    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
        using namespace Tn;
        const char* d = reinterpret_cast<const char*>(data), * a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mYoloV8NetWidth);
        read(d, mYoloV8netHeight);
        read(d, mMaxOutObject);

        assert(d == a + length);
    }


    void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
        
        using namespace Tn;
        char* d = static_cast<char*>(buffer), * a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mYoloV8NetWidth);
        write(d, mYoloV8netHeight);
        write(d, mMaxOutObject);

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mYoloV8netHeight) + sizeof(mYoloV8NetWidth) + sizeof(mMaxOutObject);
    }

    int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
        return 0;
    }

    nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) TRT_NOEXCEPT {
        int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
        return nvinfer1::Dims3(total_size + 1, 1, 1);
    }

    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
        return mPluginNamespace;
    }

    nvinfer1::DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT {
        return nvinfer1::DataType::kFLOAT;
    }


    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT {

        return false;
    }

    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {

        return false;
    }


    void YoloLayerPlugin::configurePlugin(nvinfer1::PluginTensorDesc const* in, int nbInput, nvinfer1::PluginTensorDesc const* out, int nbOutput) TRT_NOEXCEPT {};

    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {};

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

    nvinfer1::IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT

    {
        
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV8NetWidth, mYoloV8netHeight, mMaxOutObject);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }


    int YoloLayerPlugin::enqueue(int batchSize, const void* TRT_CONST_ENQUEUE* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {

        forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, mYoloV8netHeight, mYoloV8NetWidth, batchSize);
        return 0;
    }


    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };


    __global__ void CalDetection(const float* input, float* output, int numElements, int maxoutobject, const int grid_h, int grid_w, const int stride, int classes) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= numElements) return;

        int total_grid = grid_h * grid_w;
        int info_len = 4 + classes;
        const float* curInput = input;

        int class_id = 0;
        float max_cls_prob = 0.0;
        for (int i = 4; i < info_len; i++) {
            float p = Logist(curInput[idx + i * total_grid]);
            if (p > max_cls_prob) {
                max_cls_prob = p;
                class_id = i - 4;
            }
        }

        if (max_cls_prob < 0.1) return;

        int count = (int)atomicAdd(output, 1);
        if (count >= maxoutobject) return;
        char* data = (char*)output + sizeof(float) + count * sizeof(Detection);
        Detection* det = (Detection*)(data);

        int row = idx / grid_w;
        int col = idx % grid_w;

        det->conf = max_cls_prob;
        det->class_id = class_id;
        det->bbox[0] = (col + 0.5f - curInput[idx + 0 * total_grid]) * stride;
        det->bbox[1] = (row + 0.5f - curInput[idx + 1 * total_grid]) * stride;
        det->bbox[2] = (col + 0.5f + curInput[idx + 2 * total_grid]) * stride;
        det->bbox[3] = (row + 0.5f + curInput[idx + 3 * total_grid]) * stride;
    }



    void YoloLayerPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int mYoloV8netHeight,int mYoloV8NetWidth, int batchSize) {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        cudaMemsetAsync(output, 0, sizeof(float), stream);

        int numElem = 0;
        int grids[3][2] = { {mYoloV8netHeight / 8, mYoloV8NetWidth / 8}, {mYoloV8netHeight / 16, mYoloV8NetWidth / 16}, {mYoloV8netHeight / 32, mYoloV8NetWidth / 32} };
        int strides[] = { 8, 16, 32 };
        for (unsigned int i = 0; i < 3; i++) {
            int grid_h = grids[i][0];
            int grid_w = grids[i][1];
            int stride = strides[i];
            numElem = grid_h * grid_w;
            if (numElem < mThreadCount) mThreadCount = numElem;

            CalDetection << <(numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
                (inputs[i], output, numElem, mMaxOutObject, grid_h, grid_w, stride, mClassCount);
        }
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
        assert(strcmp(fc->fields[0].name, "netinfo") == 0);
        int* p_netinfo = (int*)(fc->fields[0].data);
        int class_count = p_netinfo[0];
        int input_w = p_netinfo[1];
        int input_h = p_netinfo[2];
        int max_output_object_count = p_netinfo[3];

        YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count);
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

} // namespace nvinfer1
