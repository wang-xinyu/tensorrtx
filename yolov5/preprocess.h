#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"


using namespace cv;

struct AffineMatrix{
	float value[6];
};


void preprocess_kernel_img(uint8_t* src, int src_width, int src_height, 
        float* dst, int dst_width, int dst_height,
	cudaStream_t stream);
#endif  // PREPROCESS_H
