#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include "BatchedNmsPlugin.h"
#include "./cuda_utils.h"

namespace nvinfer1 {

__global__ void batched_nms_kernel(
    const float threshold, const int num_detections,
    const int *indices, float *scores, const float *classes, const float4 *boxes) {

    // Go through detections by descending score
    for (int m = 0; m < num_detections; m++) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_detections && m < i && scores[m] > 0.0f) {
            int idx = indices[i];
            int max_idx = indices[m];
            int icls = classes[idx];
            int mcls = classes[max_idx];
            if (mcls == icls) {
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
                    scores[i] = 0.0f;
                }
            }
        }

        // Sync discarded detections
        __syncthreads();
    }
}

int batchedNms(int batch_size,
    const void *const *inputs, void **outputs,
    size_t count, int detections_per_im, float nms_thresh,
    void *workspace, size_t workspace_size, cudaStream_t stream) {

    if (!workspace || !workspace_size) {
        // Return required scratch space size cub style
        workspace_size += get_size_aligned<int>(count);   // indices
        workspace_size += get_size_aligned<int>(count);   // indices_sorted
        workspace_size += get_size_aligned<float>(count);  // scores_sorted

        size_t temp_size_sort = 0;
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
            static_cast<void*>(nullptr), temp_size_sort,
            static_cast<float*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<int*>(nullptr),
            static_cast<int*>(nullptr), count);
        workspace_size += temp_size_sort;

        return workspace_size;
    }

    auto on_stream = thrust::cuda::par.on(stream);

    auto indices = get_next_ptr<int>(count, workspace, workspace_size);
    std::vector<int> indices_h(count);
    for (int i = 0; i < count; i++)
        indices_h[i] = i;
    cudaMemcpyAsync(indices, indices_h.data(), count * sizeof * indices, cudaMemcpyHostToDevice, stream);
    auto indices_sorted = get_next_ptr<int>(count, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(count, workspace, workspace_size);

    for (int batch = 0; batch < batch_size; batch++) {
        auto in_scores = static_cast<const float *>(inputs[0]) + batch * count;
        auto in_boxes = static_cast<const float4 *>(inputs[1]) + batch * count;
        auto in_classes = static_cast<const float *>(inputs[2]) + batch * count;

        auto out_scores = static_cast<float *>(outputs[0]) + batch * detections_per_im;
        auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * detections_per_im;
        auto out_classes = static_cast<float *>(outputs[2]) + batch * detections_per_im;

        // Sort scores and corresponding indices
        int num_detections = count;
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            in_scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores_sorted) * 8, stream);

        // Launch actual NMS kernel - 1 block with each thread handling n detections
        // TODO: different device has differnet max threads
        const int max_threads = 1024;
        int num_per_thread = ceil(static_cast<float>(num_detections) / max_threads);
        batched_nms_kernel << <num_per_thread, max_threads, 0, stream >> > (nms_thresh, num_detections,
            indices_sorted, scores_sorted, in_classes, in_boxes);

        // Re-sort with updated scores
        thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            scores_sorted, scores_sorted, indices_sorted, indices,
            num_detections, 0, sizeof(*scores_sorted) * 8, stream);

        // Gather filtered scores, boxes, classes
        num_detections = min(detections_per_im, num_detections);
        cudaMemcpyAsync(out_scores, scores_sorted, num_detections * sizeof *scores_sorted,
        cudaMemcpyDeviceToDevice, stream);
        if (num_detections < detections_per_im) {
            thrust::fill_n(on_stream, out_scores + num_detections, detections_per_im - num_detections, 0);
        }
        thrust::gather(on_stream, indices, indices + num_detections, in_boxes, out_boxes);
        thrust::gather(on_stream, indices, indices + num_detections, in_classes, out_classes);
    }

    return 0;
}
}  // namespace nvinfer1
