#include "preprocess.h"

#include "config.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;
static int img_buffer_size = 0;

static float* chw_buffer_device = nullptr;
static int chw_buffer_size = 0;

namespace {

__device__ void bilinearSample(uint8_t* src, int src_line_size, int src_width, int src_height, float src_x, float src_y,
                               float* c0, float* c1, float* c2) {
    int x_low = floorf(src_x);
    int y_low = floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    float lx = src_x - x_low;
    float ly = src_y - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;
    float w1 = hy * hx;
    float w2 = hy * lx;
    float w3 = ly * hx;
    float w4 = ly * lx;

    uint8_t const_value[] = {0, 0, 0};
    uint8_t* v1 = const_value;
    uint8_t* v2 = const_value;
    uint8_t* v3 = const_value;
    uint8_t* v4 = const_value;

    if (y_low >= 0) {
        if (x_low >= 0) {
            v1 = src + y_low * src_line_size + x_low * 3;
        }
        if (x_high < src_width) {
            v2 = src + y_low * src_line_size + x_high * 3;
        }
    }

    if (y_high < src_height) {
        if (x_low >= 0) {
            v3 = src + y_high * src_line_size + x_low * 3;
        }
        if (x_high < src_width) {
            v4 = src + y_high * src_line_size + x_high * 3;
        }
    }

    *c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    *c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    *c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
}

__global__ void det_preprocess_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst,
                                      int dst_width, int dst_height, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) {
        return;
    }

    int dx = position % dst_width;
    int dy = position / dst_width;
    float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
    float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
    float src_x = (dx + 0.5f) * scale_x - 0.5f;
    float src_y = (dy + 0.5f) * scale_y - 0.5f;
    src_x = fminf(fmaxf(src_x, 0.0f), static_cast<float>(src_width - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), static_cast<float>(src_height - 1));

    float c0, c1, c2;
    bilinearSample(src, src_line_size, src_width, src_height, src_x, src_y, &c0, &c1, &c2);

    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = (c0 - 0.485f) / 0.229f;
    *pdst_c1 = (c1 - 0.456f) / 0.224f;
    *pdst_c2 = (c2 - 0.406f) / 0.225f;
}

__global__ void rec_preprocess_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst,
                                      int resize_width, int dst_width, int dst_height, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) {
        return;
    }

    int dx = position % resize_width;
    int dy = position / resize_width;
    float scale_x = static_cast<float>(src_width) / static_cast<float>(resize_width);
    float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
    float src_x = (dx + 0.5f) * scale_x - 0.5f;
    float src_y = (dy + 0.5f) * scale_y - 0.5f;
    src_x = fminf(fmaxf(src_x, 0.0f), static_cast<float>(src_width - 1));
    src_y = fminf(fmaxf(src_y, 0.0f), static_cast<float>(src_height - 1));

    float c0, c1, c2;
    bilinearSample(src, src_line_size, src_width, src_height, src_x, src_y, &c0, &c1, &c2);

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0 / 127.5f - 1.0f;
    *pdst_c1 = c1 / 127.5f - 1.0f;
    *pdst_c2 = c2 / 127.5f - 1.0f;
}

int roundToMultiple32(int value) {
    return std::max(32, static_cast<int>(std::round(value / 32.0f)) * 32);
}

void ensureImageBuffer(int imageSize) {
    if (imageSize <= img_buffer_size) {
        return;
    }
    if (img_buffer_host) {
        CUDA_CHECK(cudaFreeHost(img_buffer_host));
        img_buffer_host = nullptr;
    }
    if (img_buffer_device) {
        CUDA_CHECK(cudaFree(img_buffer_device));
        img_buffer_device = nullptr;
    }
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&img_buffer_host), imageSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&img_buffer_device), imageSize));
    img_buffer_size = imageSize;
}

void ensureChwBuffer(int count) {
    if (count <= chw_buffer_size) {
        return;
    }
    if (chw_buffer_device) {
        CUDA_CHECK(cudaFree(chw_buffer_device));
        chw_buffer_device = nullptr;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&chw_buffer_device), count * sizeof(float)));
    chw_buffer_size = count;
}

void cudaPreprocessUpload(const cv::Mat& image, cudaStream_t stream) {
    int imageSize = image.cols * image.rows * 3;
    ensureImageBuffer(imageSize);
    std::memcpy(img_buffer_host, image.ptr(), imageSize);
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, imageSize, cudaMemcpyHostToDevice, stream));
}

void cudaPreprocessDet(const cv::Mat& image, float* dst, int dstWidth, int dstHeight, cudaStream_t stream) {
    cudaPreprocessUpload(image, stream);
    int jobs = dstWidth * dstHeight;
    int threads = 256;
    int blocks = static_cast<int>(std::ceil(jobs / static_cast<float>(threads)));
    det_preprocess_kernel<<<blocks, threads, 0, stream>>>(img_buffer_device, image.cols * 3, image.cols, image.rows,
                                                          dst, dstWidth, dstHeight, jobs);
}

void cudaPreprocessRec(const cv::Mat& image, float* dst, int resizeWidth, int dstWidth, int dstHeight,
                       cudaStream_t stream) {
    cudaPreprocessUpload(image, stream);
    CUDA_CHECK(cudaMemsetAsync(dst, 0, dstWidth * dstHeight * 3 * sizeof(float), stream));
    int jobs = resizeWidth * dstHeight;
    int threads = 256;
    int blocks = static_cast<int>(std::ceil(jobs / static_cast<float>(threads)));
    rec_preprocess_kernel<<<blocks, threads, 0, stream>>>(img_buffer_device, image.cols * 3, image.cols, image.rows,
                                                          dst, resizeWidth, dstWidth, dstHeight, jobs);
}

void downloadChw(std::vector<float>& chw, int count, cudaStream_t stream) {
    chw.resize(count);
    CUDA_CHECK(cudaMemcpyAsync(chw.data(), chw_buffer_device, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace

DetPreprocessResult preprocessDet(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("empty detection image");
    }

    int srcH = image.rows;
    int srcW = image.cols;
    float ratio = 1.0f;
    if (std::max(srcH, srcW) > kDetResizeLong) {
        ratio = static_cast<float>(kDetResizeLong) / static_cast<float>(std::max(srcH, srcW));
    }
    int resizeH = roundToMultiple32(static_cast<int>(srcH * ratio));
    int resizeW = roundToMultiple32(static_cast<int>(srcW * ratio));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    ensureChwBuffer(3 * resizeH * resizeW);
    cudaPreprocessDet(image, chw_buffer_device, resizeW, resizeH, stream);

    DetPreprocessResult result;
    result.input_h = resizeH;
    result.input_w = resizeW;
    result.src_h = srcH;
    result.src_w = srcW;
    result.ratio_h = static_cast<float>(resizeH) / static_cast<float>(srcH);
    result.ratio_w = static_cast<float>(resizeW) / static_cast<float>(srcW);
    downloadChw(result.chw, 3 * resizeH * resizeW, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

RecPreprocessResult preprocessRec(const cv::Mat& crop, int maxWidth) {
    if (crop.empty()) {
        throw std::runtime_error("empty recognition crop");
    }

    int targetH = kRecInputH;
    float ratio = static_cast<float>(crop.cols) / static_cast<float>(crop.rows);
    int resizeW = std::max(1, static_cast<int>(std::ceil(targetH * ratio)));
    resizeW = std::min(resizeW, maxWidth);
    int paddedW = std::max(kRecOptW, resizeW);
    paddedW = std::min(paddedW, maxWidth);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    ensureChwBuffer(3 * targetH * paddedW);
    cudaPreprocessRec(crop, chw_buffer_device, resizeW, paddedW, targetH, stream);

    RecPreprocessResult result;
    result.input_h = targetH;
    result.input_w = paddedW;
    result.valid_w = resizeW;
    downloadChw(result.chw, 3 * targetH * paddedW, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

void cuda_preprocess_init(int maxImageSize) {
    ensureImageBuffer(maxImageSize * 3);
}

void cuda_preprocess_destroy() {
    if (img_buffer_device) {
        CUDA_CHECK(cudaFree(img_buffer_device));
        img_buffer_device = nullptr;
    }
    if (img_buffer_host) {
        CUDA_CHECK(cudaFreeHost(img_buffer_host));
        img_buffer_host = nullptr;
    }
    if (chw_buffer_device) {
        CUDA_CHECK(cudaFree(chw_buffer_device));
        chw_buffer_device = nullptr;
    }
    img_buffer_size = 0;
    chw_buffer_size = 0;
}
