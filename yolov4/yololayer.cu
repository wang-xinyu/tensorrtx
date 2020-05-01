#include "yololayer.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(const int cudaThread /*= 512*/):mThreadCount(cudaThread)
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);

        mKernelCount = mYoloKernel.size();
    }
    
    YoloLayerPlugin::~YoloLayerPlugin()
    {
        if(mInputBuffer)
            CUDA_CHECK(cudaFreeHost(mInputBuffer));

        if(mOutputBuffer)
            CUDA_CHECK(cudaFreeHost(mOutputBuffer));
    }
    
    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;

        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer)
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
    
    size_t YoloLayerPlugin::getSerializationSize()
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(Yolo::YoloKernel) * mYoloKernel.size();
    }

    int YoloLayerPlugin::initialize()
    { 
        int totalCount = 0;
        for(const auto& yolo : mYoloKernel)
            totalCount += (LOCATIONS + 1) * yolo.width*yolo.height * CHECK_COUNT;
        CUDA_CHECK(cudaHostAlloc(&mInputBuffer, totalCount * sizeof(float), cudaHostAllocDefault));

        totalCount = 0;//detection count
        for(const auto& yolo : mYoloKernel)
            totalCount += yolo.width*yolo.height * CHECK_COUNT;
        CUDA_CHECK(cudaHostAlloc(&mOutputBuffer, sizeof(float) + totalCount * sizeof(Detection), cudaHostAllocDefault));
        return 0;
    }
    
    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalCount = 0;
        for(const auto& yolo : mYoloKernel)
            totalCount += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);

        return Dims3(totalCount + 1, 1, 1);
    }

    /*void YoloLayerPlugin::forwardCpu(const float*const * inputs, float* outputs, cudaStream_t stream,int batchSize)
    {
        auto Logist = [=](float data){
            return 1./(1. + exp(-data));
        };

        int totalOutputCount = 0;
            int i = 0;
        int totalCount = 0;
            for(const auto& yolo : mYoloKernel)
            {
            totalOutputCount += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);
            totalCount += (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
            ++ i;
        }

        for (int idx = 0; idx < batchSize;idx++)
        {
            i = 0;
            float* inputData = (float *)mInputBuffer;// + idx *totalCount; //if create more batch size
            for(const auto& yolo : mYoloKernel)
            {
                int size = (LOCATIONS + 1 + mClassCount) * yolo.width*yolo.height * CHECK_COUNT;
                CUDA_CHECK(cudaMemcpyAsync(inputData, (float *)inputs[i] + idx * size, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
                inputData += size;
                ++ i;
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));

            inputData = (float *)mInputBuffer ;//+ idx *totalCount; //if create more batch size
            std::vector <Detection> result;
            for (const auto& yolo : mYoloKernel)
            {
                int stride = yolo.width*yolo.height;
                for (int j = 0;j < stride ;++j)
                {
                    for (int k = 0;k < CHECK_COUNT; ++k )
                    {
                        int beginIdx = (LOCATIONS + 1 + mClassCount)* stride *k + j;
                        int objIndex = beginIdx + LOCATIONS*stride;
                        
                        //check obj
                        float objProb = Logist(inputData[objIndex]);   
                        if(objProb <= IGNORE_THRESH)
                            continue;

                        //classes
                        int classId = -1;
                        float maxProb = IGNORE_THRESH;
                        for (int c = 0;c< mClassCount;++c){
                            float cProb =  Logist(inputData[beginIdx + (5 + c) * stride]) * objProb;
                            if(cProb > maxProb){
                                maxProb = cProb;
                                classId = c;
                            }
                        }
            
                        if(classId >= 0) {
                            Detection det;
                            int row = j / yolo.width;
                            int cols = j % yolo.width;
    
                            //Location
                            det.bbox[0] = (cols + Logist(inputData[beginIdx]))/ yolo.width;
                            det.bbox[1] = (row + Logist(inputData[beginIdx+stride]))/ yolo.height;
                            det.bbox[2] = exp(inputData[beginIdx+2*stride]) * yolo.anchors[2*k];
                            det.bbox[3] = exp(inputData[beginIdx+3*stride]) * yolo.anchors[2*k + 1];
                            //det.classId = classId;
                            det.prob = maxProb;

                            result.emplace_back(det);
                        }
                    }
                }

                inputData += (LOCATIONS + 1 + mClassCount) * stride * CHECK_COUNT;
            }

            
            int detCount =result.size();
            auto data = (float *)mOutputBuffer;// + idx*(totalOutputCount + 1); //if create more batch size
            float * begin = data;
            //copy count;
            data[0] = (float)detCount;
            data++;
            //copy result
            memcpy(data,result.data(),result.size()*sizeof(Detection));

            //(count + det result)
            CUDA_CHECK(cudaMemcpyAsync(outputs, begin,sizeof(float) + result.size()*sizeof(Detection), cudaMemcpyHostToDevice, stream));

            outputs += totalOutputCount + 1;
        }
    };*/

    __device__ float Logist(float data){ return 1./(1. + exp(-data)); };

    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int info_len_i = 5 + classes;
        //int info_len_o = 7;
        int input_col = idx;
        //int out_row = input_col;

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(input[input_col + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(input[input_col + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output;
            int count = (int)atomicAdd(res_count, 1);
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(input[input_col + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row + Logist(input[input_col + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = exp(input[input_col + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = exp(input[input_col + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }
   
    void YoloLayerPlugin::forwardGpu(const float *const * inputs,float * output,cudaStream_t stream,int batchSize) {
        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));

        int outputElem = 1;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            outputElem += yolo.width*yolo.height * CHECK_COUNT * sizeof(Detection) / sizeof(float);
        }

        for(int idx = 0 ;idx < batchSize;++idx)
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));

        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < 256)
                mThreadCount = numElem;
            CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount ,outputElem);
        }

        CUDA_CHECK(cudaFree(devAnchor));
    }


    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs,(float *)outputs[0],stream,batchSize);

        //CPU
        //forwardCpu((const float *const *)inputs,(float *)outputs[0],stream,batchSize);
        return 0;
    };

}
