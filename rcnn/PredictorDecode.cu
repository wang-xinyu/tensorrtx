#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>

#include <algorithm>
#include <cstdint>

#include "PredictorDecodePlugin.h"
#include "./cuda_utils.h"
#include "macros.h"

#ifdef CUDA_11
#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#else
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>
namespace cub = thrust::cuda_cub::cub;
#endif

namespace nvinfer1 {

int predictorDecode(int batchSize, const void *const *inputs,
void *TRT_CONST_ENQUEUE*outputs, unsigned int num_boxes, unsigned int num_classes,
unsigned int image_height, unsigned int image_width,
const std::vector<float>& bbox_reg_weights, void *workspace,
size_t workspace_size, cudaStream_t stream) {
    int scores_size = num_boxes * num_classes;

    if (!workspace || !workspace_size) {
        // Return required scratch space size cub style
        workspace_size = get_size_aligned<float>(bbox_reg_weights.size());  // anchors
        workspace_size += get_size_aligned<int>(scores_size);      // indices
        workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
        workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

        size_t temp_size_sort = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            static_cast<void*>(nullptr), temp_size_sort,
            static_cast<float*>(nullptr),
            static_cast<float*>(nullptr),
            static_cast<int*>(nullptr),
            static_cast<int*>(nullptr),
            scores_size);
        workspace_size += temp_size_sort;

        return workspace_size;
    }

    auto bbox_reg_weights_d = get_next_ptr<float>(bbox_reg_weights.size(), workspace, workspace_size);
    cudaMemcpyAsync(bbox_reg_weights_d, bbox_reg_weights.data(),
    bbox_reg_weights.size() * sizeof *bbox_reg_weights_d,
    cudaMemcpyHostToDevice, stream);

    auto on_stream = thrust::cuda::par.on(stream);

    auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
    std::vector<int> indices_h(scores_size, 0);
    for (int i = 0; i < scores_size; i++) indices_h[i] = i;
    cudaMemcpyAsync(indices, indices_h.data(), scores_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
    auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

    for (int batch = 0; batch < batchSize; batch++) {
        auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
        auto in_boxes = static_cast<const float4 *>(inputs[1]) + batch * scores_size;
        auto in_proposals = static_cast<const float4 *>(inputs[2]) + batch * num_boxes;

        auto out_scores = static_cast<float *>(outputs[0]) + batch * num_boxes;
        auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * num_boxes;
        auto out_classes = static_cast<float *>(outputs[2]) + batch * num_boxes;

        // Only keep top n scores
        cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            in_scores, scores_sorted, indices, indices_sorted, scores_size, 0, sizeof(*scores_sorted) * 8, stream);

        // Gather boxes
        thrust::transform(on_stream, indices_sorted, indices_sorted + num_boxes,
            thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes, out_classes)),
            [=] __device__(int i) {
            int cls = i % num_classes;
            int n = i / num_classes;
            float4 deltas = in_boxes[i];

            float4 boxes = in_proposals[n];

            float w = boxes.z - boxes.x;
            float h = boxes.w - boxes.y;
            float pred_ctr_x = (deltas.x / bbox_reg_weights_d[0]) * w + boxes.x + 0.5f * w;
            float pred_ctr_y = (deltas.y / bbox_reg_weights_d[1]) * h + boxes.y + 0.5f * h;
            float pred_w = exp(deltas.z / bbox_reg_weights_d[2]) * w;
            float pred_h = exp(deltas.w / bbox_reg_weights_d[3]) * h;

            boxes = float4{
              max(0.0f, pred_ctr_x - 0.5f * pred_w),
              max(0.0f, pred_ctr_y - 0.5f * pred_h),
              min(pred_ctr_x + 0.5f * pred_w, static_cast<float>(image_width)),
              min(pred_ctr_y + 0.5f * pred_h, static_cast<float>(image_width))
            };

            // filter empty boxes
            if (boxes.z - boxes.x <= 0.0f || boxes.w - boxes.y <= 0.0f) return thrust::make_tuple(0.0f, boxes, cls);
            else
                return thrust::make_tuple(in_scores[i], boxes, cls);
        });
    }

    return 0;
}

}  // namespace nvinfer1
