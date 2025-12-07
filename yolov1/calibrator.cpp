#include "calibrator.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/dnn/dnn.hpp>
#include "cuda_runtime_api.h"
#include "utils.h"

/**
 * @brief Construct a new Int8EntropyCalibrator2 object.
 *
 * This calibrator implements TensorRT's IInt8EntropyCalibrator2 interface.
 * It loads calibration images from a directory, preprocesses them, and provides
 * batches of input data to TensorRT during INT8 calibration.
 *
 * @param batchSize Number of images per calibration batch.
 * @param inputW Input image width expected by the network.
 * @param inputH Input image height expected by the network.
 * @param imgDir Directory containing calibration images.
 * @param calibTableName File name for saving or loading the calibration table.
 * @param inputBlobName Name of the network input tensor.
 * @param readCache If true, read existing calibration cache instead of recalibrating.
 */
Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchSize, int inputW, int inputH, const char* imgDir,
                                               const char* calibTableName, const char* inputBlobName, bool readCache)
    : mBatchSize(batchSize),
      mInputW(inputW),
      mInputH(inputH),
      mImgIdx(0),
      mImgDir(imgDir),
      mCalibTableName(calibTableName),
      mInputBlobName(inputBlobName),
      mReadCache(readCache) {
    mInputCount = 3 * inputW * inputH * batchSize;
    CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    readFilesInDir(imgDir, mImgFiles);
}

/**
 * @brief Destroy the Int8EntropyCalibrator2 object.
 *
 * Frees allocated GPU memory for calibration batches.
 */
Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    CUDA_CHECK(cudaFree(mDeviceInput));
}

/**
 * @brief Get the batch size used for INT8 calibration.
 *
 * @return The batch size.
 */
int Int8EntropyCalibrator2::getBatchSize() const TRT_NOEXCEPT {
    return mBatchSize;
}

/**
 * @brief Provide a batch of preprocessed input images to TensorRT.
 *
 * This method loads images, preprocesses them, converts them to CHW format,
 * uploads them to GPU memory, and binds the device buffer to TensorRT.
 *
 * TensorRT repeatedly calls this until the method returns false.
 *
 * @param bindings TensorRT binding array to fill with device pointers.
 * @param names Names of the network bindings.
 * @param nbBindings Number of bindings.
 * @return true if a batch is successfully prepared, false when no more data is available.
 */
bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) TRT_NOEXCEPT {
    if (mImgIdx + mBatchSize > (int)mImgFiles.size()) {
        return false;
    }

    std::vector<cv::Mat> inputImgs;
    for (int i = mImgIdx; i < mImgIdx + mBatchSize; i++) {
        std::cout << mImgFiles[i] << " " << i << std::endl;
        cv::Mat temp = cv::imread(mImgDir + mImgFiles[i]);
        if (temp.empty()) {
            std::cerr << "Fatal error : image cannot open." << std::endl;
            return false;
        }

        cv::Mat preImg = preprocessImg(temp, mInputW, mInputH);

        inputImgs.push_back(preImg);
    }
    mImgIdx += mBatchSize;

    // HWC → CHW
    cv::Mat blob =
            cv::dnn::blobFromImages(inputImgs, 1.0, cv::Size(mInputW, mInputH), cv::Scalar(0, 0, 0), false, false);

    // 5. Host → Device
    CUDA_CHECK(cudaMemcpy(mDeviceInput, blob.ptr<float>(0), mInputCount * sizeof(float), cudaMemcpyHostToDevice));

    // 6. TensorRT binding
    assert(!strcmp(names[0], mInputBlobName));
    bindings[0] = mDeviceInput;

    return true;
}

/**
 * @brief Read an existing INT8 calibration cache file.
 *
 * If readCache is true and the file exists, TensorRT will reuse the
 * calibration table, skipping the calibration phase.
 *
 * @param length Output parameter containing size of the calibration buffer.
 * @return Pointer to the calibration cache data, or nullptr if none exists.
 */
const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) TRT_NOEXCEPT {
    std::cout << "reading calib cache" << mCalibTableName << std::endl;

    mCalibCache.clear();
    std::ifstream input(mCalibTableName, std::ios::binary);

    input >> std::noskipws;
    if (mReadCache && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibCache));
    }
    length = mCalibCache.size();
    return length ? mCalibCache.data() : nullptr;
}

/**
 * @brief Write the generated INT8 calibration cache to a file.
 *
 * TensorRT calls this method after calibration to store the calibration table.
 *
 * @param cache Pointer to calibration data provided by TensorRT.
 * @param length Length of the calibration data in bytes.
 */
void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT {
    std::cout << "writing calib cache:" << mCalibTableName << "size:" << length << std::endl;

    std::ofstream output(mCalibTableName, std::ios::binary);

    output.write(reinterpret_cast<const char*>(cache), length);
}
