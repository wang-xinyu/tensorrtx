#include "decode.h"
#include "stdio.h"

namespace nvinfer1
{
    DecodePlugin::DecodePlugin(const int cudaThread):thread_count_(cudaThread)
    {
    }
    
    DecodePlugin::~DecodePlugin()
    {
    }
    
    // create the plugin at runtime from a byte stream
    DecodePlugin::DecodePlugin(const void* data, size_t length)
    {
    }

    void DecodePlugin::serialize(void* buffer)
    {
    }
    
    size_t DecodePlugin::getSerializationSize()
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

    __device__ float Logist(float data){ return 1./(1. + exp(-data)); };

    __global__ void CalDetection(const float *input, float *output, int num_elem, int step, int anchor) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        int h = decodeplugin::INPUT_H / step;
        int w = decodeplugin::INPUT_W / step;
        int y = idx / w;
        int x = idx % w;
        const float *bbox_reg = &input[0];
        const float *cls_reg = &input[2 * 4 * num_elem];
        const float *lmk_reg = &input[2 * 4 * num_elem + 2 * 2 * num_elem];

        for (int k = 0; k < 2; ++k) {
            float conf1 = cls_reg[idx + k * num_elem * 2];
            float conf2 = cls_reg[idx + k * num_elem * 2 + num_elem];
            conf2 = exp(conf2) / (exp(conf1) + exp(conf2));
            if (conf2 <= 0.02) continue;

            float *res_count = output;
            int count = (int)atomicAdd(res_count, 1);
            char* data = (char *)res_count + sizeof(float) + count * sizeof(decodeplugin::Detection);
            decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

            float prior[4];
            prior[0] = ((float)x + 0.5) / w;
            prior[1] = ((float)y + 0.5) / h;
            prior[2] = (float)anchor * (k + 1) / decodeplugin::INPUT_W;
            prior[3] = (float)anchor * (k + 1) / decodeplugin::INPUT_H;

            //Location
            det->bbox[0] = prior[0] + bbox_reg[idx + k * num_elem * 4] * 0.1 * prior[2];
            det->bbox[1] = prior[1] + bbox_reg[idx + k * num_elem * 4 + num_elem] * 0.1 * prior[3];
            det->bbox[2] = prior[2] * exp(bbox_reg[idx + k * num_elem * 4 + num_elem * 2] * 0.2);
            det->bbox[3] = prior[3] * exp(bbox_reg[idx + k * num_elem * 4 + num_elem * 3] * 0.2);
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
                det->landmark[i] = prior[0] + lmk_reg[idx + k * num_elem * 10 + num_elem * i] * 0.1 * prior[2];
                det->landmark[i+1] = prior[1] + lmk_reg[idx + k * num_elem * 10 + num_elem * (i + 1)] * 0.1 * prior[3];
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

}
