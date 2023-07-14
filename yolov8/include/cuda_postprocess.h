//
// Created by lindsay on 23-7-17.
//

#ifndef YOLOV8_CUDA_POSTPROCESS_H
#define YOLOV8_CUDA_POSTPROCESS_H

#include "types.h"

void cuda_decode(float* predict, int num_bboxes, float confidence_threshold,float* parray,int max_objects, cudaStream_t stream);

void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);


#endif //YOLOV8_CUDA_POSTPROCESS_H
