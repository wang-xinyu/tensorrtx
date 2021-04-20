#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>

#include "RpnDecodePlugin.h"
#include "./cuda_utils.h"

namespace nvinfer1 {

int rpnDecode(int batch_size,
    const void *const *inputs, void **outputs,
    size_t height, size_t width, size_t image_height, size_t image_width, float stride,
    const std::vector<float> &anchors, int top_n,
    void *workspace, size_t workspace_size, cudaStream_t stream) {

    size_t num_anchors = anchors.size() / 4;
    int scores_size = num_anchors * height * width;

    if (!workspace || !workspace_size) {
        // Return required scratch space size cub style
        workspace_size = get_size_aligned<float>(anchors.size());  // anchors
        workspace_size += get_size_aligned<int>(scores_size);      // indices
        workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
        workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

        size_t temp_size_sort = 0;
        if (scores_size > top_n) {
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
                static_cast<void*>(nullptr), temp_size_sort,
                static_cast<float*>(nullptr),
                static_cast<float*>(nullptr),
                static_cast<int*>(nullptr),
                static_cast<int*>(nullptr), scores_size);
            workspace_size += temp_size_sort;
        }

        return workspace_size;
    }

    auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
    cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);

    auto on_stream = thrust::cuda::par.on(stream);

    auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
    // TODO: how to generate sequence on gpu directly?
    std::vector<int> indices_h(scores_size);
    for (int i = 0; i < scores_size; i++)
        indices_h[i] = i;
    cudaMemcpyAsync(indices, indices_h.data(), scores_size * sizeof * indices, cudaMemcpyHostToDevice, stream);
    auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

    for (int batch = 0; batch < batch_size; batch++) {
        auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
        auto in_boxes = static_cast<const float *>(inputs[1]) + batch * scores_size * 4;

        auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
        auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * top_n;

        // Only keep top n scores
        int num_detections = scores_size;
        auto indices_filtered = indices;
        if (num_detections > top_n) {
            thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                in_scores, scores_sorted, indices, indices_sorted, scores_size, 0, sizeof(*scores_sorted) * 8, stream);
            indices_filtered = indices_sorted;
            num_detections = top_n;
        }

        // Gather boxes
        bool has_anchors = !anchors.empty();
        thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
            thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes)),
            [=] __device__(int i) {
            int x = i % width;
            int y = (i / width) % height;
            int a = (i / height / width) % num_anchors;
            float4 box = float4{
              in_boxes[((a * 4 + 0) * height + y) * width + x],
              in_boxes[((a * 4 + 1) * height + y) * width + x],
              in_boxes[((a * 4 + 2) * height + y) * width + x],
              in_boxes[((a * 4 + 3) * height + y) * width + x]
            };

            if (has_anchors) {
                // Add anchors offsets to deltas
                float x = (i % width) * stride;
                float y = ((i / width) % height) * stride;
                float *d = anchors_d + 4 * a;

                float x1 = x + d[0];
                float y1 = y + d[1];
                float x2 = x + d[2];
                float y2 = y + d[3];
                float w = x2 - x1;
                float h = y2 - y1;
                float pred_ctr_x = box.x * w + x1 + 0.5f * w;
                float pred_ctr_y = box.y * h + y1 + 0.5f * h;
                float pred_w = exp(box.z) * w;
                float pred_h = exp(box.w) * h;

                // TODO: set image size as parameter
                box = float4{
                  max(0.0f, pred_ctr_x - 0.5f * pred_w),
                  max(0.0f, pred_ctr_y - 0.5f * pred_h),
                  min(pred_ctr_x + 0.5f * pred_w, static_cast<float>(image_width)),
                  min(pred_ctr_y + 0.5f * pred_h, static_cast<float>(image_height))
                };
            }
            // filter empty boxes
            if (box.z - box.x <= 0.0f || box.w - box.y <= 0.0f)
                return thrust::make_tuple(-FLT_MAX, box);
            else
                return thrust::make_tuple(in_scores[i], box);
        });

        // Zero-out unused scores
        if (num_detections < top_n) {
            thrust::fill(on_stream, out_scores + num_detections,
                out_scores + top_n, -FLT_MAX);
        }
    }

    return 0;
}
}  // namespace nvinfer1
