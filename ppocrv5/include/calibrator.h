#pragma once

#include "NvInfer.h"

#include <string>
#include <vector>

#ifndef TRT_NOEXCEPT
#define TRT_NOEXCEPT noexcept
#endif

enum class CalibratorPreprocessType {
    kOCRDet,
    kOCRRec,
    kImageNet,
};

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
   public:
    Int8EntropyCalibrator2(int batchsize, int inputW, int inputH, const char* imgDir, const char* calibTableName,
                           const char* inputBlobName, bool readCache = true);
    Int8EntropyCalibrator2(int batchsize, int inputW, int inputH, const char* imgDir, const char* calibTableName,
                           const char* inputBlobName, CalibratorPreprocessType preprocessType, bool readCache = true);
    ~Int8EntropyCalibrator2() override;

    int getBatchSize() const TRT_NOEXCEPT override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) TRT_NOEXCEPT override;
    const void* readCalibrationCache(size_t& length) TRT_NOEXCEPT override;
    void writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT override;

   private:
    void preprocessImage(const std::string& imagePath, float* dst) const;

    int batchsize_;
    int input_w_;
    int input_h_;
    int img_idx_;
    std::string img_dir_;
    std::vector<std::string> img_files_;
    size_t input_count_;
    std::string calib_table_name_;
    std::string input_blob_name_;
    CalibratorPreprocessType preprocess_type_;
    bool read_cache_;
    void* device_input_;
    std::vector<char> calib_cache_;
    std::vector<float> input_host_;
};
