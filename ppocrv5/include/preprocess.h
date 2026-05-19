#pragma once

#include <opencv2/opencv.hpp>
#include "types.h"

DetPreprocessResult preprocessDet(const cv::Mat& image);
RecPreprocessResult preprocessRec(const cv::Mat& crop, int maxWidth);
void cuda_preprocess_init(int maxImageSize);
void cuda_preprocess_destroy();
