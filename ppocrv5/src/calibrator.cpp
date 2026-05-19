#include "calibrator.h"

#include "cuda_utils.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace {

void resizeToFixed(const cv::Mat& image, cv::Mat& resized, int inputW, int inputH) {
    cv::resize(image, resized, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);
}

void resizeRec(const cv::Mat& image, cv::Mat& resized, int inputW, int inputH, int& validW) {
    float ratio = static_cast<float>(image.cols) / static_cast<float>(image.rows);
    validW = std::max(1, static_cast<int>(std::ceil(inputH * ratio)));
    validW = std::min(validW, inputW);
    cv::resize(image, resized, cv::Size(validW, inputH), 0, 0, cv::INTER_LINEAR);
}

void fillCHW(const cv::Mat& image, float* dst, int inputW, int inputH, CalibratorPreprocessType preprocessType) {
    int area = inputW * inputH;
    for (int y = 0; y < inputH; ++y) {
        const cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputW; ++x) {
            int index = y * inputW + x;
            float b = static_cast<float>(row[x][0]);
            float g = static_cast<float>(row[x][1]);
            float r = static_cast<float>(row[x][2]);

            if (preprocessType == CalibratorPreprocessType::kOCRRec) {
                dst[index] = b / 127.5f - 1.0f;
                dst[index + area] = g / 127.5f - 1.0f;
                dst[index + area * 2] = r / 127.5f - 1.0f;
            } else if (preprocessType == CalibratorPreprocessType::kImageNet) {
                dst[index] = (b / 255.0f - 0.485f) / 0.229f;
                dst[index + area] = (g / 255.0f - 0.456f) / 0.224f;
                dst[index + area * 2] = (r / 255.0f - 0.406f) / 0.225f;
            } else {
                dst[index] = (b / 255.0f - 0.485f) / 0.229f;
                dst[index + area] = (g / 255.0f - 0.456f) / 0.224f;
                dst[index + area * 2] = (r / 255.0f - 0.406f) / 0.225f;
            }
        }
    }
}

}  // namespace

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int inputW, int inputH, const char* imgDir,
                                               const char* calibTableName, const char* inputBlobName, bool readCache)
    : Int8EntropyCalibrator2(batchsize, inputW, inputH, imgDir, calibTableName, inputBlobName,
                             CalibratorPreprocessType::kOCRDet, readCache) {}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int inputW, int inputH, const char* imgDir,
                                               const char* calibTableName, const char* inputBlobName,
                                               CalibratorPreprocessType preprocessType, bool readCache)
    : batchsize_(batchsize),
      input_w_(inputW),
      input_h_(inputH),
      img_idx_(0),
      img_dir_(imgDir),
      input_count_(static_cast<size_t>(batchsize) * 3 * inputW * inputH),
      calib_table_name_(calibTableName),
      input_blob_name_(inputBlobName),
      preprocess_type_(preprocessType),
      read_cache_(readCache),
      device_input_(nullptr),
      input_host_(input_count_) {
    if (batchsize_ <= 0 || input_w_ <= 0 || input_h_ <= 0) {
        throw std::runtime_error("invalid calibrator input shape");
    }
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    img_files_ = listImages(img_dir_);
    if (img_files_.empty()) {
        throw std::runtime_error("no calibration images found: " + img_dir_);
    }
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    if (device_input_) {
        CUDA_CHECK(cudaFree(device_input_));
        device_input_ = nullptr;
    }
}

int Int8EntropyCalibrator2::getBatchSize() const TRT_NOEXCEPT {
    return batchsize_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) TRT_NOEXCEPT {
    if (img_idx_ + batchsize_ > static_cast<int>(img_files_.size())) {
        return false;
    }

    size_t singleInputCount = static_cast<size_t>(3 * input_w_ * input_h_);
    try {
        for (int i = 0; i < batchsize_; ++i) {
            const std::string& imagePath = img_files_[img_idx_ + i];
            std::cout << imagePath << "  " << img_idx_ + i << std::endl;
            preprocessImage(imagePath, input_host_.data() + i * singleInputCount);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
    img_idx_ += batchsize_;

    CUDA_CHECK(cudaMemcpy(device_input_, input_host_.data(), input_count_ * sizeof(float), cudaMemcpyHostToDevice));

    int inputIndex = -1;
    for (int i = 0; i < nbBindings; ++i) {
        if (std::strcmp(names[i], input_blob_name_.c_str()) == 0) {
            inputIndex = i;
            break;
        }
    }
    if (inputIndex < 0) {
        return false;
    }
    bindings[inputIndex] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) TRT_NOEXCEPT {
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    if (read_cache_ && input.good()) {
        std::copy(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>(),
                  std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT {
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

void Int8EntropyCalibrator2::preprocessImage(const std::string& imagePath, float* dst) const {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("calibration image cannot open: " + imagePath);
    }

    cv::Mat resized;
    if (preprocess_type_ == CalibratorPreprocessType::kOCRRec) {
        std::fill(dst, dst + 3 * input_w_ * input_h_, 0.0f);
        int validW = 0;
        resizeRec(image, resized, input_w_, input_h_, validW);
        int area = input_w_ * input_h_;
        for (int y = 0; y < input_h_; ++y) {
            const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < validW; ++x) {
                int index = y * input_w_ + x;
                dst[index] = static_cast<float>(row[x][0]) / 127.5f - 1.0f;
                dst[index + area] = static_cast<float>(row[x][1]) / 127.5f - 1.0f;
                dst[index + area * 2] = static_cast<float>(row[x][2]) / 127.5f - 1.0f;
            }
        }
        return;
    }

    resizeToFixed(image, resized, input_w_, input_h_);
    fillCHW(resized, dst, input_w_, input_h_, preprocess_type_);
}
