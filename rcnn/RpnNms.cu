#include "RpnNmsPlugin.h"
#include "cuda_utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <cmath>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>

namespace nvinfer1 {

    __global__ void rpn_nms_kernel(
        const int num_per_thread, const float threshold, const int num_detections,
        const int *indices, float *scores, const float4 *boxes) {
        // Go through detections by descending score
        for (int m = 0; m < num_detections; m++) {
            for (int n = 0; n < num_per_thread; n++) {
                int i = threadIdx.x * num_per_thread + n;
                if (i < num_detections && m < i && scores[m] > -FLT_MAX) {
                    int idx = indices[i];
                    int max_idx = indices[m];

                    float4 ibox = boxes[idx];
                    float4 mbox = boxes[max_idx];
                    float x1 = max(ibox.x, mbox.x);
                    float y1 = max(ibox.y, mbox.y);
                    float x2 = min(ibox.z, mbox.z);
                    float y2 = min(ibox.w, mbox.w);
                    float w = max(0.0f, x2 - x1);
                    float h = max(0.0f, y2 - y1);
                    float iarea = (ibox.z - ibox.x) * (ibox.w - ibox.y);
                    float marea = (mbox.z - mbox.x) * (mbox.w - mbox.y);
                    float inter = w * h;
                    float overlap = inter / (iarea + marea - inter);
                    if (overlap > threshold) {
                        scores[i] = -FLT_MAX;
                    }
                }
            }

            // Sync discarded detections
            __syncthreads();
        }
    }

    int rpnNms(int batch_size,
        const void *const *inputs, void **outputs,
        size_t pre_nms_topk, int post_nms_topk, float nms_thresh,
        void *workspace, size_t workspace_size, cudaStream_t stream) {
        if (!workspace || !workspace_size) {
            // Return required scratch space size cub style
            workspace_size += get_size_aligned<int>(pre_nms_topk);   // indices
            workspace_size += get_size_aligned<int>(pre_nms_topk);   // indices_sorted
            workspace_size += get_size_aligned<float>(pre_nms_topk);  // scores
            workspace_size += get_size_aligned<float>(pre_nms_topk);  // scores_sorted

            size_t temp_size_sort = 0;
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
                (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, pre_nms_topk);
            workspace_size += temp_size_sort;

            return workspace_size;
        }

        auto on_stream = thrust::cuda::par.on(stream);

        auto indices = get_next_ptr<int>(pre_nms_topk, workspace, workspace_size);
        std::vector<int> indices_h(pre_nms_topk);
        for (int i = 0; i < pre_nms_topk; i++)
            indices_h[i] = i;
        cudaMemcpyAsync(indices, indices_h.data(), pre_nms_topk * sizeof * indices, cudaMemcpyHostToDevice, stream);
        auto indices_sorted = get_next_ptr<int>(pre_nms_topk, workspace, workspace_size);
        auto scores = get_next_ptr<float>(pre_nms_topk, workspace, workspace_size);
        auto scores_sorted = get_next_ptr<float>(pre_nms_topk, workspace, workspace_size);

        for (int batch = 0; batch < batch_size; batch++) {
            auto in_scores = static_cast<const float *>(inputs[0]) + batch * pre_nms_topk;
            auto in_boxes = static_cast<const float4 *>(inputs[1]) + batch * pre_nms_topk;

            auto out_boxes = static_cast<float4 *>(outputs[0]) + batch * post_nms_topk;

            int num_detections = pre_nms_topk;
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                in_scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores_sorted) * 8, stream);

            // Launch actual NMS kernel - 1 block with each thread handling n detections
            // TODO: different device has differnet max threads
            const int max_threads = 1024;
            int num_per_thread = ceil((float)num_detections / max_threads);
            rpn_nms_kernel << <1, max_threads, 0, stream >> > (num_per_thread, nms_thresh, num_detections,
                indices_sorted, scores_sorted, in_boxes);

            // Re-sort with updated scores
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                scores_sorted, scores, indices_sorted, indices, num_detections, 0, sizeof(*scores_sorted) * 8, stream);

            // Gather filtered scores, boxes, classes
            num_detections = min(post_nms_topk, num_detections);
            thrust::gather(on_stream, indices, indices + num_detections, in_boxes, out_boxes);
        }

        return 0;
    }
}  // namespace nvinfer1
