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
        totalCount += input_h_ / 8 * input_w_ / 8 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += input_h_ / 16 * input_w_ / 16 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);
        totalCount += input_h_ / 32 * input_w_ / 32 * 2 * sizeof(decodeplugin::Detection) / sizeof(float);

        return Dims3(totalCount + 1, 1, 1);
    }

    __device__ float Logist(float data){ return 1./(1. + exp(-data)); };

    __global__ void CalDetection(const float *input, float *output, int num_elem, int input_h, int input_w, int step, int anchor) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        int h = input_h / step;
        int w = input_w / step;
        int y = idx / w;
        int x = idx % w;
        const float *bbox_reg = &input[0];
        const float *cls_reg = &input[2 * 4 * num_elem];
        const float *lmk_reg = &input[2 * 4 * num_elem + 2 * 2 * num_elem];

        for (int k = 0; k < 2; ++k) {
            float conf1 = cls_reg[idx + k * num_elem * 2];
            float conf2 = cls_reg[idx + k * num_elem * 2 + num_elem];
            conf2 = exp(conf2) / (exp(conf1) + exp(conf2));
            if (conf2 <= 0.002) continue;

            float *res_count = output;
            int count = (int)atomicAdd(res_count, 1);
            char* data = (char *)res_count + sizeof(float) + count * sizeof(decodeplugin::Detection);
            decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

            float prior[4];
            prior[0] = ((float)x + 0.5) / w;
            prior[1] = ((float)y + 0.5) / h;
            prior[2] = (float)anchor / input_w;
            prior[3] = (float)anchor / input_h;
            printf("prior0, %f\n", prior[0]);
            printf("bbox0, %f\n", bbox_reg[idx + k * num_elem * 4]);

            //Location
            det->bbox[0] = prior[0] + bbox_reg[idx + k * num_elem * 4] * 0.1 * prior[2];
            det->bbox[1] = prior[1] + bbox_reg[idx + k * num_elem * 4 + num_elem] * 0.1 * prior[3];
            det->bbox[2] = prior[2] * exp(bbox_reg[idx + k * num_elem * 4 + num_elem * 2] * 0.2);
            det->bbox[3] = prior[3] * exp(bbox_reg[idx + k * num_elem * 4 + num_elem * 3] * 0.2);
            det->bbox[0] -= det->bbox[2] / 2;
            det->bbox[1] -= det->bbox[3] / 2;
            det->bbox[2] += det->bbox[0];
            det->bbox[3] += det->bbox[1];
            det->bbox[0] *= input_w;
            det->bbox[1] *= input_h;
            det->bbox[2] *= input_w;
            det->bbox[3] *= input_h;
            det->class_confidence = conf2;
            anchor *= 2;
        }
    }
   
    void DecodePlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize) 
    {
        int num_elem = 0;
        int base_step = 8;
        int base_anchor = 16;
        int thread_count;
        for (unsigned int i = 0; i < 3; ++i)
        {
            num_elem = input_h_ / base_step * input_w_ / base_step;
            thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
            CalDetection<<< (num_elem + thread_count - 1) / thread_count, thread_count>>>
                (inputs[i], output, num_elem, input_h_, input_w_, base_step, base_anchor);
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
