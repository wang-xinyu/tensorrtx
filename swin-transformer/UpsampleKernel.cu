#include "UpsmapleKernel.h"


/**
 * @brief caculate the number of cuda kernel for upsample. (Cite from: 《GPU高性能编程CUDA实战》P46,P47)
 * 
 * @param total_thread_num: the number of cuda thread of you want to used for upsample
 * @param max_thread_num: the gpu device property
 * @return int  the number of cuda kernel for upsample
 */
int get_kernel_num(int total_thread_num, int max_thread_num)
{
    return (total_thread_num + max_thread_num - 1)/max_thread_num;
}

int get_max_thread_num()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}

__host__ __forceinline__ float linear_upsampling_compute_scale(int input_size, int output_size)
{
    return float(input_size)/float(output_size) ;
}

__device__ __forceinline__ float linear_upsampling_compute_source_index(float scale, int dst_index, int intput_size)
{
    float src_idx = scale * (dst_index + 0.5)-0.5;
    return (src_idx>=0) ? src_idx : 0;
}


__device__ __forceinline__ int get_index(const int batch_idx, const int channel_idx, const int height_idx, const int width_idx, 
                const int batch_total, const int channel_total, const int width)
{
    int ret_idx = batch_idx * batch_total
                    + channel_idx * channel_total
                    + height_idx * width
                    + width_idx;
    return ret_idx;
}

/**
 * @brief 
 * 
 * @tparam T 
 * @param n 
 * @param input_shape: input data shape. such as [batch, channel, height, width] 
 * @param rate_h 
 * @param rate_w 
 * @param inputs 
 * @param outputs 
 * @return __global__ BilinearKernel 
 * @TODO: 
 *  
 */


template <typename T>
__global__ void BilinearKernel(
        const int n,
        int input_b,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        const float rate_h,
        const float rate_w,
        const T* inputs,
        T* outputs)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n)
    {
        const int w2 = index % output_w;
        const int h2 = index / output_w;


        const float h1r = linear_upsampling_compute_source_index(rate_h, h2, input_h);
        const int h1 = int(h1r);
        const int h1p = (h1 < input_h - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1 - h1lambda;

        const float w1r = linear_upsampling_compute_source_index(rate_w, w2, input_w);
        const int w1 = int(w1r);
        const int w1p = (w1 < input_w - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1 - w1lambda;

        int s_batch_total_1 = input_c * input_h * input_w;
        int s_channel_total_1 = input_h * input_w;

        int s_batch_total_2 = input_c * output_h * output_w;
        int s_channel_total_2 = output_h * output_w;


        const int batch_size = input_b;
        const int channel_size = input_c;

        for(int b_idx=0; b_idx<batch_size; b_idx++)
        {
            for(int c=0; c<channel_size; c++)
            {
                const T val = h0lambda * (w0lambda * inputs[get_index(b_idx, c, h1, w1, s_batch_total_1, s_channel_total_1, input_w)]
                                    + w1lambda * inputs[get_index(b_idx, c, h1, w1+w1p, s_batch_total_1, s_channel_total_1, input_w)])
                                    + h1lambda * (w0lambda * inputs[get_index(b_idx, c, h1+h1p, w1, s_batch_total_1, s_channel_total_1, input_w)]
                                    + w1lambda * inputs[get_index(b_idx, c, h1+h1p, w1+w1p, s_batch_total_1, s_channel_total_1, input_w)]);
                outputs[get_index(b_idx, c, h2, w2, s_batch_total_2, s_channel_total_2, output_w)] = val;
                
            }
        }
    }
}


int UpsampleInference(
    cudaStream_t stream,
    int n,
    int input_b,
    int input_c,
    int input_h,
    int input_w,
    float scale_h,
    float scale_w,
    const void* inputs,
    void* outputs)
{
    int output_h = int(input_h * scale_h);
    int output_w = int(input_w * scale_w);
    int max_threads = get_max_thread_num();
    int kernel_num = get_kernel_num(n, max_threads);
    float rate_h = linear_upsampling_compute_scale(input_h, output_h);
    float rate_w = linear_upsampling_compute_scale(input_w, output_w);

    BilinearKernel<float><<< kernel_num, max_threads, 0, stream>>>(n,input_b,input_c,input_h,input_w,
                                                                                    output_h, output_w, 
                                                                                    rate_h, rate_w,
                                                                                    static_cast<const float*>(inputs),
                                                                                    static_cast<float*>(outputs));
    return 0;
}
