//
// Created by lindsay on 23-7-17.
//
#include "cuda_utils.h"
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
    int count = min(static_cast<int>(bboxes[0]), max_objects);
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
    int count = min(static_cast<int>(bboxes[0]), max_objects);
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

// ======================================================================================
// GPU Segmentation Kernels (Ported from StiQy)
// ======================================================================================

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __global__ void compact_and_gather_masks_kernel(float* nms_output, int* final_count, float* compacted_masks_out,
                                                       int* mapping_out, int max_objects) {
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= max_objects)
        return;

    // The number of items to check is stored at nms_output[0]
    int count = (int)nms_output[0];
    if (position >= count)
        return;

    float* pcurrent = nms_output + 1 + position * bbox_element;
    int keep_flag = (int)pcurrent[4 + 1 + 1];  // index 6: 0=x,1=y,2=w,3=h,4=conf,5=cls,6=keep

    if (keep_flag == 1) {
        // This detection was kept by NMS. Get its new, compacted index.
        int final_index = atomicAdd(final_count, 1);

        // The mask coefficients start at index 7.
        float* mask_src = pcurrent + 7;
        float* mask_dst = compacted_masks_out + final_index * 32;

        for (int i = 0; i < 32; ++i) {
            mask_dst[i] = mask_src[i];
        }
        // record mapping from nms slot -> final compacted index
        mapping_out[position] = final_index;
    }
}

void cuda_compact_and_gather_masks(const float* decode_ptr_device, int* final_count_device,
                                   float* compacted_masks_device, int* mask_mapping_device, int max_objects,
                                   cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(final_count_device, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(mask_mapping_device, -1, sizeof(int) * max_objects, stream));
    int threads = 256;
    int blocks = (max_objects + threads - 1) / threads;
    // Note: cast const away for decode_ptr_device as kernel assumes non-const (though it treats it as input)
    compact_and_gather_masks_kernel<<<blocks, threads, 0, stream>>>(
            (float*)decode_ptr_device, final_count_device, compacted_masks_device, mask_mapping_device, max_objects);
}

// Integrated Kernel with Strict Clipping and Bilinear Interpolation
__global__ void process_mask_kernel(const float* proto, const float* masks_in, const float* bboxes, float* masks_out,
                                    int num_dets, int proto_h, int proto_w, int out_h, int out_w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= out_w || y >= out_h || det_idx >= num_dets)
        return;

    // Strict Clipping Logic
    float x1 = bboxes[det_idx * 4 + 0];
    float y1 = bboxes[det_idx * 4 + 1];
    float x2 = bboxes[det_idx * 4 + 2];
    float y2 = bboxes[det_idx * 4 + 3];

    // Check if pixel is outside the bounding box
    if (x < x1 || x > x2 || y < y1 || y > y2) {
        masks_out[det_idx * (out_h * out_w) + y * out_w + x] = 0.0f;
        return;
    }

    // Bilinear Interpolation
    float proto_x_float = ((float)x + 0.5f) / 4.0f - 0.5f;
    float proto_y_float = ((float)y + 0.5f) / 4.0f - 0.5f;

    int proto_x1 = (int)floorf(proto_x_float);
    int proto_y1 = (int)floorf(proto_y_float);
    int proto_x2 = proto_x1 + 1;
    int proto_y2 = proto_y1 + 1;

    float w_x = proto_x_float - proto_x1;
    float w_y = proto_y_float - proto_y1;

    const float* mask_weights = masks_in + det_idx * 32;
    float mask_val = 0.0f;

    for (int j = 0; j < 32; ++j) {
        const float* proto_channel = proto + j * (proto_h * proto_w);
        float p1 = (proto_x1 >= 0 && proto_y1 >= 0) ? proto_channel[proto_y1 * proto_w + proto_x1] : 0.0f;
        float p2 = (proto_x2 < proto_w && proto_y1 >= 0) ? proto_channel[proto_y1 * proto_w + proto_x2] : 0.0f;
        float p3 = (proto_x1 >= 0 && proto_y2 < proto_h) ? proto_channel[proto_y2 * proto_w + proto_x1] : 0.0f;
        float p4 = (proto_x2 < proto_w && proto_y2 < proto_h) ? proto_channel[proto_y2 * proto_w + proto_x2] : 0.0f;

        float interpolated_p =
                p1 * (1 - w_x) * (1 - w_y) + p2 * w_x * (1 - w_y) + p3 * (1 - w_x) * w_y + p4 * w_x * w_y;
        mask_val += mask_weights[j] * interpolated_p;
    }

    masks_out[det_idx * (out_h * out_w) + y * out_w + x] = sigmoid(mask_val);
}

void cuda_process_mask(const float* proto, const float* masks_in, const float* bboxes_in, float* masks_out,
                       int num_dets, int proto_h, int proto_w, int out_h, int out_w, cudaStream_t stream) {
    if (num_dets == 0)
        return;
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x, (out_h + block_dim.y - 1) / block_dim.y,
                  (num_dets + block_dim.z - 1) / block_dim.z);
    process_mask_kernel<<<grid_dim, block_dim, 0, stream>>>(proto, masks_in, bboxes_in, masks_out, num_dets, proto_h,
                                                            proto_w, out_h, out_w);
}

// Box Blur Kernels
__global__ void box_blur_horizontal(const float* src, float* dst, int w, int h, int num_masks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || m >= num_masks)
        return;
    int r = 1;
    float sum = 0.0f;
    int count = 0;
    int base = m * w * h + y * w;
    for (int dx = -r; dx <= r; ++dx) {
        int nx = x + dx;
        if (nx >= 0 && nx < w) {
            sum += src[base + nx];
            count++;
        }
    }
    dst[base + x] = sum / count;
}
__global__ void box_blur_vertical(const float* src, float* dst, int w, int h, int num_masks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || m >= num_masks)
        return;
    int r = 1;
    float sum = 0.0f;
    int count = 0;
    int base = m * w * h;
    for (int dy = -r; dy <= r; ++dy) {
        int ny = y + dy;
        if (ny >= 0 && ny < h) {
            sum += src[base + ny * w + x];
            count++;
        }
    }
    dst[base + y * w + x] = sum / count;
}
void cuda_blur_masks(float* masks_device, int num_dets, int mask_h, int mask_w, cudaStream_t stream) {
    if (num_dets <= 0)
        return;
    float* tmp = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&tmp, sizeof(float) * num_dets * mask_h * mask_w));
    dim3 block(16, 16, 1);
    dim3 grid((mask_w + block.x - 1) / block.x, (mask_h + block.y - 1) / block.y, (num_dets + block.z - 1) / block.z);
    box_blur_horizontal<<<grid, block, 0, stream>>>(masks_device, tmp, mask_w, mask_h, num_dets);
    box_blur_vertical<<<grid, block, 0, stream>>>(tmp, masks_device, mask_w, mask_h, num_dets);
    CUDA_CHECK(cudaFree(tmp));
}

// Optimized Drawing Kernels
__global__ void gather_kept_bboxes_kernel(const float* nms_output, const int* mask_mapping, float* dense_bboxes,
                                          int max_bboxes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_bboxes)
        return;
    int compacted_idx = mask_mapping[i];
    if (compacted_idx != -1) {
        const float* pcurrent = nms_output + 1 + i * bbox_element;
        dense_bboxes[compacted_idx * 4 + 0] = pcurrent[0];
        dense_bboxes[compacted_idx * 4 + 1] = pcurrent[1];
        dense_bboxes[compacted_idx * 4 + 2] = pcurrent[2];
        dense_bboxes[compacted_idx * 4 + 3] = pcurrent[3];
    }
}

void cuda_gather_kept_bboxes(const float* nms_output, const int* mask_mapping, float* dense_bboxes, int max_bboxes,
                             cudaStream_t stream) {
    int threads = 256;
    int blocks = (max_bboxes + threads - 1) / threads;
    gather_kept_bboxes_kernel<<<blocks, threads, 0, stream>>>(nms_output, mask_mapping, dense_bboxes, max_bboxes);
}

__global__ void draw_results_on_image_kernel(float* image_buffer, const float* final_masks, const float* dense_bboxes,
                                             int num_dets, int mask_mode, float mask_thresh) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= kInputW || y >= kInputH)
        return;

    bool pixel_is_covered = false;
    for (int i = 0; i < num_dets; ++i) {
        float left = dense_bboxes[i * 4 + 0];
        float top = dense_bboxes[i * 4 + 1];
        float right = dense_bboxes[i * 4 + 2];
        float bottom = dense_bboxes[i * 4 + 3];

        float box_w = right - left;
        float box_h = bottom - top;
        float padding = 0.03f * fmaxf(box_w, box_h);
        float padded_left = left - padding;
        float padded_top = top - padding;
        float padded_right = right + padding;
        float padded_bottom = bottom + padding;

        if (x >= padded_left && x < padded_right && y >= padded_top && y < padded_bottom) {
            float mask_val = final_masks[i * (kInputW * kInputH) + y * kInputW + x];
            if (mask_val > mask_thresh) {
                pixel_is_covered = true;
                break;
            }
        }
    }

    if (mask_mode == 1) {  // "Mask-Out" mode
        if (!pixel_is_covered) {
            int area = kInputW * kInputH;
            image_buffer[y * kInputW + x] = 0.0f;             // R
            image_buffer[area + y * kInputW + x] = 0.0f;      // G
            image_buffer[2 * area + y * kInputW + x] = 0.0f;  // B
        }
    } else {  // "White Mask" mode
        if (pixel_is_covered) {
            int area = kInputW * kInputH;
            float* r_ptr = image_buffer + y * kInputW + x;
            float* g_ptr = r_ptr + area;
            float* b_ptr = g_ptr + area;
            *r_ptr = *r_ptr * 0.5f + 1.0f * 0.5f;
            *g_ptr = *g_ptr * 0.5f + 1.0f * 0.5f;
            *b_ptr = *b_ptr * 0.5f + 1.0f * 0.5f;
        }
    }
}

void cuda_draw_results(float* image_buffer, const float* final_masks, const float* nms_output, const int* mask_mapping,
                       int num_dets, int mask_mode, float mask_thresh, cudaStream_t stream) {
    if (num_dets == 0)
        return;

    float* dense_bboxes = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dense_bboxes, num_dets * 4 * sizeof(float), stream));

    int threads = 256;
    int blocks = (kMaxNumOutputBbox + threads - 1) / threads;
    gather_kept_bboxes_kernel<<<blocks, threads, 0, stream>>>(nms_output, mask_mapping, dense_bboxes,
                                                              kMaxNumOutputBbox);

    dim3 block_dim(16, 16);
    dim3 grid_dim((kInputW + block_dim.x - 1) / block_dim.x, (kInputH + block_dim.y - 1) / block_dim.y);

    draw_results_on_image_kernel<<<grid_dim, block_dim, 0, stream>>>(image_buffer, final_masks, dense_bboxes, num_dets,
                                                                     mask_mode, mask_thresh);

    CUDA_CHECK(cudaFreeAsync(dense_bboxes, stream));
}
