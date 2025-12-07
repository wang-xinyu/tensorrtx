#ifndef __TRT_UTILS_H_
#define __TRT_UTILS_H_

#include <cudnn.h>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "macros.h"

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

namespace Tn {
template <typename T>
void write(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}
}  // namespace Tn

/**
 * @brief Preprocess an image: BGR → RGB, resize, normalize to float32[0,1].
 *
 * This function converts the input image from BGR to RGB, resizes it to the
 * specified dimensions, normalizes pixel values to the range [0,1], and
 * returns the result as a float32 HWC format cv::Mat.
 *
 * @param img Input image in BGR format.
 * @param inputW Target width.
 * @param inputH Target height.
 * @return cv::Mat Preprocessed image in HWC format, type CV_32FC3.
 */
static inline cv::Mat preprocessImg(cv::Mat& img, int inputW, int inputH) {
    // 1. BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // 2. Resize
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(inputW, inputH));

    // 3. float32 + normalize(0~1)
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    return resized;  // HWC, CV_32FC3
}

/**
 * @brief Convert an image from HWC layout to CHW layout.
 *
 * This function splits a float32 HWC cv::Mat into 3 separate channels (R, G, B)
 * and copies them sequentially into the output buffer in CHW format.
 *
 * @param img Input image in HWC format (CV_32FC3).
 * @param data Output pointer to store CHW formatted float data.
 */
inline void hwcToChw(const cv::Mat& img, float* data) {
    int channels = 3;
    int imgH = img.rows;
    int imgW = img.cols;
    int imgSize = imgH * imgW;

    std::vector<cv::Mat> splitChannels;
    cv::split(img, splitChannels);

    for (int c = 0; c < channels; c++) {
        memcpy(data + c * imgSize, splitChannels[c].data, imgSize * sizeof(float));
    }
}

/**
 * @brief Complete preprocessing pipeline for TRT inference.
 *
 * Includes BGR→RGB conversion, resizing, normalization to [0,1],
 * and conversion from HWC to CHW layout directly into the provided buffer.
 *
 * @param img Input BGR image.
 * @param data Pointer to output float buffer of size 3 * inputH * inputW.
 * @param inputH Target height.
 * @param inputW Target width.
 */
static inline void preprocess(const cv::Mat& img, float* data, int inputH, int inputW) {
    // 1. BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // 2. Resize to input size
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(inputW, inputH));

    // 3. Convert to float32
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 4. HWC -> CHW
    int channels = 3;
    int imgSize = inputH * inputW;
    std::vector<cv::Mat> splitChannels;
    cv::split(resized, splitChannels);  // R, G, B channels

    for (int c = 0; c < channels; ++c) {
        memcpy(data + c * imgSize, splitChannels[c].data, imgSize * sizeof(float));
    }
}

/**
 * @brief Read all file names inside a directory (non-recursive).
 *
 * This function scans the given directory and collects all file names
 * except "." and "..". Returned names are **not prefixed** with the directory path.
 *
 * @param pDirName Path of directory to scan.
 * @param fileNames Output vector to store file names.
 * @return int 0 on success, -1 if directory cannot be opened.
 */
static inline int readFilesInDir(const char* pDirName, std::vector<std::string>& fileNames) {
    DIR* pDir = opendir(pDirName);
    if (pDir == nullptr) {
        return -1;
    }

    struct dirent* pFile = nullptr;
    while ((pFile = readdir(pDir)) != nullptr) {
        if (strcmp(pFile->d_name, ".") != 0 && strcmp(pFile->d_name, "..") != 0) {
            //std::string curFileName(pDirName);
            //curFileName += "/";
            //curFileName += pFile->d_name;
            std::string curFileName(pFile->d_name);
            fileNames.push_back(curFileName);
        }
    }

    closedir(pDir);
    return 0;
}

#endif
