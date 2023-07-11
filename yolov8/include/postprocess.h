#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh = 0.5);

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

cv::Rect processDetection_cuda_rect(const float* decode_ptr_host, int i, int bbox_element, cv::Mat& img, Detection& det, int& boxes_count);

void draw_bbox_cuda_process_single(const float* decode_ptr_host, int bbox_element, cv::Mat& img);

void draw_bbox_cuda_process_batch(float *decode_ptr_host_batch,int bbox_element,const std::vector<cv::Mat>& img_batch);