#ifndef ENTROPY_CALIBRATOR_H
#define ENTROPY_CALIBRATOR_H

#include <string>
#include <vector>
#include "NvInfer.h"
#include "macros.h"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
   public:
    Int8EntropyCalibrator2(int batchSize, int inputW, int inputH, const char* imgDir, const char* calibTableName,
                           const char* inputBlobName, bool readCache = true);

    virtual ~Int8EntropyCalibrator2();

    int getBatchSize() const TRT_NOEXCEPT override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) TRT_NOEXCEPT override;

    const void* readCalibrationCache(size_t& length) TRT_NOEXCEPT override;

    void writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT override;

   private:
    int mBatchSize;
    int mInputW;
    int mInputH;
    int mImgIdx;
    std::string mImgDir;
    std::vector<std::string> mImgFiles;
    size_t mInputCount;
    std::string mCalibTableName;
    const char* mInputBlobName;
    bool mReadCache;
    void* mDeviceInput;
    std::vector<char> mCalibCache;
};

#endif  // ENTROPY_CALIBRATOR_H