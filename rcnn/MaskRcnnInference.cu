#include "MaskRcnnInferencePlugin.h"

namespace nvinfer1 {

__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); }

__global__ void MaskRcnnInferenceKernel(
    const int nthreads,
    const int detections_per_im,
    const int output_size,
    const int num_classes,
    const float* indices,
    const float* masks,
    float* out_masks) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nthreads) {
        int ind = index / output_size / output_size / num_classes;
        int ind_class = indices[ind];
        int cur_class = index / output_size / output_size % num_classes;
        if (ind_class == cur_class) {
            int w = index % output_size;
            int h = index / output_size % output_size;
            int tmp = ind * num_classes * output_size * output_size +
              cur_class * output_size*output_size + h * output_size + w;
            float maskVal = masks[ind * num_classes * output_size *
              output_size + cur_class * output_size * output_size +
              h * output_size + w];
            out_masks[ind * output_size * output_size + h * output_size + w] = Logist(maskVal);
        }
    }
}

int maskRcnnInference(int batchSize,
    const void *const *inputs, void **outputs,
    int detections_per_im, int output_size, int num_classes, cudaStream_t stream) {

    for (int batch = 0; batch < batchSize; batch++) {
        auto in_indices = static_cast<const float *>(inputs[0]) + batch * detections_per_im;
        auto in_masks = static_cast<const float *>(inputs[1]) + batch * detections_per_im *
          num_classes * output_size * output_size;

        auto out_masks = static_cast<float *>(outputs[0]) + batch * detections_per_im * output_size * output_size;

        int nthreads = detections_per_im * num_classes * output_size * output_size;
        const int max_threads = 1024;
        int blocksPerGrid = ceil(static_cast<float>(nthreads) / max_threads);
        // TODO: can implement this function with thrust?
        MaskRcnnInferenceKernel << <blocksPerGrid, max_threads, 0, stream >> > (
            nthreads,
            detections_per_im,
            output_size,
            num_classes,
            in_indices,
            in_masks,
            out_masks);
        cudaDeviceSynchronize();
    }

    return 0;
}

}  // namespace nvinfer1
