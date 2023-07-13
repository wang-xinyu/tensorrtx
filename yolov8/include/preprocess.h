#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"
#include <map>


struct AffineMatrix {
    float value[6];
};

const int bbox_element = sizeof(AffineMatrix) / sizeof(float)+1;      // left, top, right, bottom, confidence, class, keepflag

void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height, cudaStream_t stream);

void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch, float *dst, int dst_width, int dst_height, cudaStream_t stream);

void cuda_decode(float* predict, int num_bboxes, float confidence_threshold,float* parray,int max_objects, cudaStream_t stream);

void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);