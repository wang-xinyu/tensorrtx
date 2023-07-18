//
// Created by lindsay on 23-7-17.
//
#include "types.h"
#include "postprocess.h"

static __global__ void
decode_kernel(float *predict, int num_bboxes, float confidence_threshold, float *parray, int max_objects) {

    float count = predict[0];
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= count)
        return;
    float *pitem = predict + 1 + position * 6;
    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;
    float confidence = pitem[4];
    if (confidence < confidence_threshold)
        return;
    float left = pitem[0];
    float top = pitem[1];
    float right = pitem[2];
    float bottom = pitem[3];
    float label = pitem[5];
    float *pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float
box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom) {

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold) {

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = bboxes[0];

    // float count = 0.0f;
    if (position >= count)
        return;

    float *pcurrent = bboxes + 1 + position * bbox_element;
    for (int i = 1; i < count; ++i) {
        float *pitem = bboxes + 1 + i * bbox_element;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0], pitem[1], pitem[2], pitem[3]
            );

            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void cuda_decode(float *predict, int num_bboxes, float confidence_threshold, float *parray, int max_objects,
                 cudaStream_t stream) {
    int block = 256;
    int grid = ceil(num_bboxes / (float) block);
    decode_kernel << <
    grid, block, 0, stream >> > ((float *) predict, num_bboxes, confidence_threshold, parray, max_objects);

}

void cuda_nms(float *parray, float nms_threshold, int max_objects, cudaStream_t stream) {
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float) block);
    nms_kernel << < grid, block, 0, stream >> > (parray, max_objects, nms_threshold);

}
