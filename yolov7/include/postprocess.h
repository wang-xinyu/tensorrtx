#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);

