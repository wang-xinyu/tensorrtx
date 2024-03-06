#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
cv::Rect get_rect(cv::Mat& img, float bbox[4]);

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float *output, int batch_size, int output_size, float conf_thresh, float nms_thresh = 0.5);

void draw_bbox(std::vector<cv::Mat>& img_batch, std::vector<std::vector<Detection>>& res_batch);

std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);

void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks, std::unordered_map<int, std::string>& labels_map);
// cuda NMS
void cuda_decode(float* predict, int num_bboxes, float confidence_threshold,float* parray,int max_objects, cudaStream_t stream);
void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);
void batch_process(std::vector<std::vector<Detection>> &res_batch, const float* decode_ptr_host, int batch_size, int bbox_element, const std::vector<cv::Mat>& img_batch);