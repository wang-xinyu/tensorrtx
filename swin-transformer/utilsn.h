#ifndef UTILSN_H
#define UTILSN_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cudnn.h>
#include <NvInfer.h>
#include "myhpp.h"
using namespace std;

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

namespace Tn
{
    class Profiler : public nvinfer1::IProfiler
    {
    public:
        void printLayerTimes(int itrationsTimes)
        {
            float totalTime = 0;
            for (size_t i = 0; i < mProfile.size(); i++)
            {
                printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / itrationsTimes);
                totalTime += mProfile[i].second;
            }
            printf("Time over all layers: %4.3f\n", totalTime / itrationsTimes);
        }
    private:
        typedef std::pair<std::string, float> Record;
        std::vector<Record> mProfile;

        virtual void reportLayerTime(const char* layerName, float ms)
        {
            auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
            if (record == mProfile.end())
               { mProfile.push_back(std::make_pair(layerName, ms));}
            else
                record->second += ms;
        }
    };

    template<typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

//    void* copyToDevice(const void* data, size_t count)
//    {
//        void* deviceData;
//        cudaMalloc(&deviceData, count);
//        cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice);
//        return deviceData;
//    }
//    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
//    {
//        deviceWeights = copyToDevice(hostBuffer, size);
//        hostBuffer += size;
//    }
//    size_t type2size(nvinfer1::DataType type) { return sizeof(float); }
//    void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights& weights)
//    {
//        memcpy(buffer, weights.values, weights.count * type2size(weights.type));
//        buffer += weights.count * type2size(weights.type);
//    }


}

#endif // UTILSN_H
