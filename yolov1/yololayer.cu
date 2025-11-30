#include <assert.h>
#include "yololayer.h"
#include "utils.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin()
    {
        mS = S;
        mB = B;
        mClasses = CLASSES;

        mOutputSize = OUTPUT_SIZE;

        mThreadCount = 256;
    }

    YoloLayerPlugin::YoloLayerPlugin(const void * data, size_t length)
    {
        using namespace Tn;

        const char * d = reinterpret_cast<const char*>(data);
        const char * start = d;

        read(d, mS);
        read(d, mB);
        read(d, mClasses);
        read(d, mOutputSize);
        read(d, mThreadCount);

        assert(d == start + length);
    }

    YoloLayerPlugin::~YoloLayerPlugin()
    {

    }

    void YoloLayerPlugin::serialize(void * buffer) const TRT_NOEXCEPT
    {
        using namespace Tn;

        char * d = reinterpret_cast<char *>(buffer);
        char * start = d;

        write(d, mS);
        write(d, mB);
        write(d, mClasses);
        write(d, mOutputSize);
        write(d, mThreadCount);

        assert(d == start + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return sizeof(mS) + sizeof(mB) + sizeof(mClasses) + sizeof(mOutputSize) + sizeof(mThreadCount);
    }

    int YoloLayerPlugin::initialize() TRT_NOEXCEPT
    { 
        return 0;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        // YOLOv1 output dims: (C, H, W)
        // C = (5B + C)
        // H = S
        // W = S
        int C = (5 * mB + mClasses);
        int H = mS;
        int W = mS;

        // return Dims3(C, H, W);
        return Dims2(H * H, 5 + mClasses);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

    const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT
    {
        YoloLayerPlugin *p = new YoloLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __global__ void decodeYoloKernel(
        const float* input,  // 1x30xSxS
        float* output,       // 1x(5+C)xSxS
        int S,
        int C
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0) printf("GPU decodeYoloKernel is running!\n");
        int total_grid = S * S;
        if (idx >= total_grid) return;

        int in_stride = 30;
        int out_stride = 5 + C;

        int in_idx = idx * in_stride;
        int out_idx = idx * out_stride;

        // B1
        float b1_x    = input[in_idx + 0];
        float b1_y    = input[in_idx + 1];
        float b1_w    = input[in_idx + 2];
        float b1_h    = input[in_idx + 3];
        float b1_conf = input[in_idx + 4];

        // B2
        float b2_x    = input[in_idx + 5];
        float b2_y    = input[in_idx + 6];
        float b2_w    = input[in_idx + 7];
        float b2_h    = input[in_idx + 8];
        float b2_conf = input[in_idx + 9];

        if (b1_conf > b2_conf)
        {
            output[out_idx + 0] = b1_x;
            output[out_idx + 1] = b1_y;
            output[out_idx + 2] = b1_w;
            output[out_idx + 3] = b1_h;
            output[out_idx + 4] = b1_conf;
        }
        else
        {
            output[out_idx + 0] = b2_x;
            output[out_idx + 1] = b2_y;
            output[out_idx + 2] = b2_w;
            output[out_idx + 3] = b2_h;
            output[out_idx + 4] = b2_conf;
        }

        //  class prob
        for (int c = 0; c < C; c++)
        {
            output[out_idx + 5 + c] = input[in_idx + 10 + c];
        }
    }

    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        const int S = mS;
        const int C = mClasses;
        const int grid_size = S * S;

        const float* input_dev = reinterpret_cast<const float*>(inputs[0]);
        float* output_dev = reinterpret_cast<float*>(outputs[0]);

        for (int b = 0; b < batchSize; ++b)
        {
            const float* input_ptr = input_dev + b * grid_size * 30;
            float* output_ptr      = output_dev + b * grid_size * (5 + C);

            int threads = 128;
            int blocks = (grid_size + threads - 1) / threads;

            decodeYoloKernel<<<blocks, threads, 0, stream>>>(
                input_ptr, output_ptr, S, C
            );
        }

        return 0;
    }













    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char * YoloPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char * YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    const PluginFieldCollection * YoloPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    // call when construct engine 
    IPluginV2IOExt * YoloPluginCreator::createPlugin(const char *name, const PluginFieldCollection * fc) TRT_NOEXCEPT
    {
        YoloLayerPlugin * obj = new YoloLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    // call when load engine. 
    IPluginV2IOExt * YoloPluginCreator::deserializePlugin(const char * name, const void * serialData, size_t serialLength) TRT_NOEXCEPT
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}