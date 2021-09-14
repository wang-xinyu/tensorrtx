#include "decode.h"
#include "stdio.h"

namespace nvinfer1
{
    DecodePlugin::DecodePlugin()
    {
    }

    DecodePlugin::~DecodePlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    DecodePlugin::DecodePlugin(const void* data, size_t length)
    {
    }

    void DecodePlugin::serialize(void* buffer) const TRT_NOEXCEPT
    {
    }

    size_t DecodePlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return 0;
    }

    int DecodePlugin::initialize() TRT_NOEXCEPT
    { 
        return 0;
    }

    Dims DecodePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
    {
        //output the result to channel
        int totalCount = 0;
        totalCount += decodeplugin::INPUT_H / 8 * decodeplugin::INPUT_W / 8 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 16 * decodeplugin::INPUT_W / 16 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 32 * decodeplugin::INPUT_W / 32 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);

        return Dims3(totalCount + 1, 1, 1);
    }

    // Set plugin namespace
    void DecodePlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* DecodePlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType DecodePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool DecodePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool DecodePlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {
        return false;
    }

    void DecodePlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void DecodePlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
    {
    }

    // Detach the plugin object from its execution context.
    void DecodePlugin::detachFromContext() TRT_NOEXCEPT {}

    const char* DecodePlugin::getPluginType() const TRT_NOEXCEPT
    {
        return "Decode_TRT";
    }

    const char* DecodePlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void DecodePlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* DecodePlugin::clone() const TRT_NOEXCEPT
    {
        DecodePlugin *p = new DecodePlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int num_elem, int step, int anchor, int output_elem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        int h = decodeplugin::INPUT_H / step;
        int w = decodeplugin::INPUT_W / step;
        int total_grid = h * w;
        int bn_idx = idx / total_grid;
        idx = idx - bn_idx * total_grid;
        int y = idx / w;
        int x = idx % w;
        const float* cur_input = input + bn_idx * (4 + 2 + 10) * 2 * total_grid;
        const float *bbox_reg = &cur_input[0];
        const float *cls_reg = &cur_input[2 * 4 * total_grid];
        const float *lmk_reg = &cur_input[2 * 4 * total_grid + 2 * 2 * total_grid];

        for (int k = 0; k < 2; ++k) {
            float conf1 = cls_reg[idx + k * total_grid * 2];
            float conf2 = cls_reg[idx + k * total_grid * 2 + total_grid];
            conf2 = expf(conf2) / (expf(conf1) + expf(conf2));
            if (conf2 <= 0.02) continue;

            float *res_count = output + bn_idx * output_elem;
            int count = (int)atomicAdd(res_count, 1);
            char* data = (char *)res_count + sizeof(float) + count * sizeof(decodeplugin::Detection);
            decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

            float prior[4];
            prior[0] = ((float)x + 0.5) / w;
            prior[1] = ((float)y + 0.5) / h;
            prior[2] = (float)anchor * (k + 1) / decodeplugin::INPUT_W;
            prior[3] = (float)anchor * (k + 1) / decodeplugin::INPUT_H;

            //Location
            det->bbox[0] = prior[0] + bbox_reg[idx + k * total_grid * 4] * 0.1 * prior[2];
            det->bbox[1] = prior[1] + bbox_reg[idx + k * total_grid * 4 + total_grid] * 0.1 * prior[3];
            det->bbox[2] = prior[2] * expf(bbox_reg[idx + k * total_grid * 4 + total_grid * 2] * 0.2);
            det->bbox[3] = prior[3] * expf(bbox_reg[idx + k * total_grid * 4 + total_grid * 3] * 0.2);
            det->bbox[0] -= det->bbox[2] / 2;
            det->bbox[1] -= det->bbox[3] / 2;
            det->bbox[2] += det->bbox[0];
            det->bbox[3] += det->bbox[1];
            det->bbox[0] *= decodeplugin::INPUT_W;
            det->bbox[1] *= decodeplugin::INPUT_H;
            det->bbox[2] *= decodeplugin::INPUT_W;
            det->bbox[3] *= decodeplugin::INPUT_H;
            det->class_confidence = conf2;
            for (int i = 0; i < 10; i += 2) {
                det->landmark[i] = prior[0] + lmk_reg[idx + k * total_grid * 10 + total_grid * i] * 0.1 * prior[2];
                det->landmark[i+1] = prior[1] + lmk_reg[idx + k * total_grid * 10 + total_grid * (i + 1)] * 0.1 * prior[3];
                det->landmark[i] *= decodeplugin::INPUT_W;
                det->landmark[i+1] *= decodeplugin::INPUT_H;
            }
        }
    }

    void DecodePlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize)
    {
        int num_elem = 0;
        int base_step = 8;
        int base_anchor = 16;
        int thread_count;

        int totalCount = 1;
        totalCount += decodeplugin::INPUT_H / 8 * decodeplugin::INPUT_W / 8 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 16 * decodeplugin::INPUT_W / 16 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 32 * decodeplugin::INPUT_W / 32 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        for(int idx = 0 ; idx < batchSize; ++idx) {
            cudaMemset(output + idx * totalCount, 0, sizeof(float));
        }

        for (unsigned int i = 0; i < 3; ++i)
        {
            num_elem = batchSize * decodeplugin::INPUT_H / base_step * decodeplugin::INPUT_W / base_step;
            thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
            CalDetection<<< (num_elem + thread_count - 1) / thread_count, thread_count>>>
                (inputs[i], output, num_elem, base_step, base_anchor, totalCount);
            base_step *= 2;
            base_anchor *= 4;
        }
    }

    int DecodePlugin::enqueue(int batchSize, const void*const * inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
        return 0;
    };

    PluginFieldCollection DecodePluginCreator::mFC{};
    std::vector<PluginField> DecodePluginCreator::mPluginAttributes;

    DecodePluginCreator::DecodePluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* DecodePluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "Decode_TRT";
    }

    const char* DecodePluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    const PluginFieldCollection* DecodePluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    IPluginV2IOExt* DecodePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
    {
        DecodePlugin* obj = new DecodePlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* DecodePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
    {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        DecodePlugin* obj = new DecodePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
