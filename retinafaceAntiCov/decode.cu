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

    void DecodePlugin::serialize(void* buffer) const
    {
    }

    size_t DecodePlugin::getSerializationSize() const
    {  
        return 0;
    }

    int DecodePlugin::initialize()
    { 
        return 0;
    }

    Dims DecodePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalCount = 0;
        totalCount += decodeplugin::INPUT_H / 8 * decodeplugin::INPUT_W / 8 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 16 * decodeplugin::INPUT_W / 16 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += decodeplugin::INPUT_H / 32 * decodeplugin::INPUT_W / 32 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);

        return Dims3(totalCount + 1, 1, 1);
    }

    // Set plugin namespace
    void DecodePlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* DecodePlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType DecodePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool DecodePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool DecodePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void DecodePlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void DecodePlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void DecodePlugin::detachFromContext() {}

    const char* DecodePlugin::getPluginType() const
    {
        return "Decode_TRT";
    }

    const char* DecodePlugin::getPluginVersion() const
    {
        return "1";
    }

    void DecodePlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* DecodePlugin::clone() const
    {
        DecodePlugin *p = new DecodePlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int num_elem, int step, int anchor) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        int h = decodeplugin::INPUT_H / step;
        int w = decodeplugin::INPUT_W / step;
        int y = idx / w;
        int x = idx % w;
        const float *cls_reg = &input[2 * num_elem];
        const float *bbox_reg = &input[4 * num_elem];
        const float *lmk_reg = &input[12 * num_elem];
        const float *mask_reg = &input[36 * num_elem];

        for (int k = 0; k < 2; ++k) {
            float conf = cls_reg[idx + k * num_elem];
            if (conf < 0.5) continue;

            float *res_count = output;
            int count = (int)atomicAdd(res_count, 1);
            char* data = (char *)res_count + sizeof(float) + count * sizeof(decodeplugin::Detection);
            decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

            float prior[4];
            prior[0] = 7.5 + (float)(x * step);
            prior[1] = 7.5 + (float)(y * step);
            prior[2] = anchor * 2 / (k + 1);
            prior[3] = prior[2];

            //Location
            det->bbox[0] = prior[0] + bbox_reg[idx + k * num_elem * 4] * prior[2];
            det->bbox[1] = prior[1] + bbox_reg[idx + k * num_elem * 4 + num_elem] * prior[3];
            det->bbox[2] = prior[2] * expf(bbox_reg[idx + k * num_elem * 4 + num_elem * 2]);
            det->bbox[3] = prior[3] * expf(bbox_reg[idx + k * num_elem * 4 + num_elem * 3]);
            det->bbox[0] -= (det->bbox[2] - 1) / 2;
            det->bbox[1] -= (det->bbox[3] - 1) / 2;
            det->bbox[2] += det->bbox[0];
            det->bbox[3] += det->bbox[1];
            det->class_confidence = conf;
            for (int i = 0; i < 10; i += 2) {
                det->landmark[i] = prior[0] + lmk_reg[idx + k * num_elem * 10 + num_elem * i] * 0.2 * prior[2];
                det->landmark[i+1] = prior[1] + lmk_reg[idx + k * num_elem * 10 + num_elem * (i + 1)] * 0.2 * prior[3];
            }
            det->mask_confidence = mask_reg[idx + k * num_elem];;
        }
    }

    void DecodePlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize) 
    {
        int num_elem = 0;
        int base_step = 8;
        int base_anchor = 16;
        int thread_count;
        cudaMemset(output, 0, sizeof(float));
        for (unsigned int i = 0; i < 3; ++i)
        {
            num_elem = decodeplugin::INPUT_H / base_step * decodeplugin::INPUT_W / base_step;
            thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
            CalDetection<<< (num_elem + thread_count - 1) / thread_count, thread_count>>>
                (inputs[i], output, num_elem, base_step, base_anchor);
            base_step *= 2;
            base_anchor *= 4;
        }
    }

    int DecodePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs,(float *)outputs[0],stream,batchSize);

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

    const char* DecodePluginCreator::getPluginName() const
    {
        return "Decode_TRT";
    }

    const char* DecodePluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* DecodePluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* DecodePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        DecodePlugin* obj = new DecodePlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* DecodePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        DecodePlugin* obj = new DecodePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
