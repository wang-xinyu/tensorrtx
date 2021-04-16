#include "RoiAlignPlugin.h"
#include "cuda_utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <cmath>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>

namespace nvinfer1 {
template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0;
    }

    if (y <= 0) {
        y = 0;
    }
    if (x <= 0) {
        x = 0;
    }

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = bottom_data[y_low * width + x_low];
    T v2 = bottom_data[y_low * width + x_high];
    T v3 = bottom_data[y_high * width + x_low];
    T v4 = bottom_data[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;  // mode Avg

    return val;
}

__global__ void RoIAlignForward(
    const int nthreads,
    const float* bottom_data,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float4* bottom_rois,
    float* top_data) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        const float4* offset_bottom_rois = bottom_rois + n;

        // Do not using rounding; this implementation detail is critical
        float roi_offset = 0.5f;
        float roi_start_w = offset_bottom_rois->x * spatial_scale - roi_offset;
        float roi_start_h = offset_bottom_rois->y * spatial_scale - roi_offset;
        float roi_end_w = offset_bottom_rois->z * spatial_scale - roi_offset;
        float roi_end_h = offset_bottom_rois->w * spatial_scale - roi_offset;

        float roi_width = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;

        float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        const float* offset_bottom_data =
            bottom_data + static_cast<int>(c * height * width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / pooled_height);  // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const float count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

        float output_val = 0.f;
        bool max_flag = false;
        // e.g., iy = 0, 1
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const float y = roi_start_h + ph * bin_size_h +
                static_cast<float>(iy + .5f) * bin_size_h /
                static_cast<float>(roi_bin_grid_h);  // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const float x = roi_start_w + pw * bin_size_w +
                    static_cast<float>(ix + .5f) * bin_size_w /
                    static_cast<float>(roi_bin_grid_w);

                float val = bilinear_interpolate(
                    offset_bottom_data, height, width, y, x);

                output_val += val;
            }
        }

        output_val /= count;

        top_data[index] = output_val;
    }
}

int roiAlign(int batchSize, const void *const *inputs, void **outputs, int pooler_resolution, float spatial_scale,
    int sampling_ratio, int num_proposals, int out_channels, int feature_h, int feature_w, cudaStream_t stream) {
    for (int batch = 0; batch < batchSize; batch++) {
        auto in_boxes = static_cast<const float4 *>(inputs[0]) + batch * num_proposals;
        auto in_features = static_cast<const float *>(inputs[1]) + batch * out_channels * feature_h * feature_w;

        int nthreads = num_proposals * out_channels * pooler_resolution * pooler_resolution;
        auto out_features = static_cast<float *>(outputs[0]) + batch * nthreads;
        const int max_threads = 1024;

        int blocksPerGrid = ceil((float)nthreads / max_threads);
        RoIAlignForward<< <blocksPerGrid, max_threads, 0 >> > (
            nthreads,
            in_features,
            spatial_scale,
            out_channels,
            feature_h,
            feature_w,
            pooler_resolution,
            pooler_resolution,
            sampling_ratio,
            in_boxes,
            out_features);
        cudaDeviceSynchronize();
    }

    return 0;
}
}  // namespace nvinfer1
