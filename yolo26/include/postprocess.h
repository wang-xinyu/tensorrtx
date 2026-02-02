#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"

// Preprocessing functions
cv::Rect get_rect(cv::Mat& img, float bbox[4]);

// NMS functions
void decode(std::vector<Detection>& res, float* output);

void batch_decode(std::vector<std::vector<Detection>>& res_batch, float* output, int batch_size, int output_size);

void decode_obb(std::vector<Detection>& res, float* output);

void batch_decode_obb(std::vector<std::vector<Detection>>& batch_res, float* output, int batch_size, int output_size);

// Drawing functions
void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

void draw_bbox_obb(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

void draw_bbox_keypoints_line(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks,
                    std::unordered_map<int, std::string>& labels_map);