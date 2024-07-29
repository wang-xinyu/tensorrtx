#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

void batch_topk(std::vector<std::vector<Detection>>& res_batch, float* output, int batch_size, int output_size,
                float conf_thresh, int topk = 300);
