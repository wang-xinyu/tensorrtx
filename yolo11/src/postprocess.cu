//
// Created by lindsay on 23-7-17.
//
#include "postprocess.h"
#include "types.h"

static __global__ void decode_kernel_obb(float* predict, int num_bboxes, float confidence_threshold, float* parray,
                                         int max_objects) {
    float count = predict[0];
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= count)
        return;

    float* pitem = predict + 1 + position * (sizeof(Detection) / sizeof(float));
    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float confidence = pitem[4];

    if (confidence < confidence_threshold)
        return;
    //[center_x center_y w h conf class_id  mask[32] keypoints[51] angle]
    float cx = pitem[0];
    float cy = pitem[1];
    float width = pitem[2];
    float height = pitem[3];
    float label = pitem[5];
    float angle = pitem[89];

    float* pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = cx;
    *pout_item++ = cy;
    *pout_item++ = width;
    *pout_item++ = height;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = angle;
}

static __global__ void decode_kernel(float* predict, int num_bboxes, float confidence_threshold, float* parray,
                                     int max_objects) {
    float count = predict[0];
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= count)
        return;

    float* pitem = predict + 1 + position * (sizeof(Detection) / sizeof(float));
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

    float* pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom) {
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

static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = bboxes[0];
    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * bbox_element;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * bbox_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;
            float iou =
                    box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);
            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

static __device__ void convariance_matrix(float w, float h, float r, float& a, float& b, float& c) {
    float a_val = w * w / 12.0f;
    float b_val = h * h / 12.0f;
    float cos_r = cosf(r);
    float sin_r = sinf(r);

    a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
    b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
    c = (a_val - b_val) * sin_r * cos_r;
}

static __device__ float box_probiou(float cx1, float cy1, float w1, float h1, float r1, float cx2, float cy2, float w2,
                                    float h2, float r2, float eps = 1e-7) {

    // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
    float a1, b1, c1, a2, b2, c2;
    convariance_matrix(w1, h1, r1, a1, b1, c1);
    convariance_matrix(w2, h2, r2, a2, b2, c2);

    float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) /
               ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
    float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) /
                            (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) +
                    eps);
    float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
    bd = fmaxf(fminf(bd, 100.0f), eps);
    float hd = sqrtf(1.0f - expf(-bd) + eps);
    return 1 - hd;
}

static __global__ void nms_kernel_obb(float* bboxes, int max_objects, float threshold) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = bboxes[0];
    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * bbox_element;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * bbox_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;
            float iou = box_probiou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pcurrent[7], pitem[0], pitem[1],
                                    pitem[2], pitem[3], pitem[7]);
            if (iou > threshold) {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void cuda_decode(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                 cudaStream_t stream) {
    int block = 256;
    int grid = ceil(num_bboxes / (float)block);
    decode_kernel<<<grid, block, 0, stream>>>((float*)predict, num_bboxes, confidence_threshold, parray, max_objects);
}

void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream) {
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}

void cuda_decode_obb(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                     cudaStream_t stream) {
    int block = 256;
    int grid = ceil(num_bboxes / (float)block);
    decode_kernel_obb<<<grid, block, 0, stream>>>((float*)predict, num_bboxes, confidence_threshold, parray,
                                                  max_objects);
}

void cuda_nms_obb(float* parray, float nms_threshold, int max_objects, cudaStream_t stream) {
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel_obb<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}
