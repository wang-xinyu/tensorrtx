/*
* Copyright 2021 Zuru Tech HK Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* istributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <utility>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <PillowResize.hpp>

double PillowResize::BoxFilter::filter(double x) const
{
    constexpr double half_pixel = 0.5;
    if (x > -half_pixel && x <= half_pixel) {
        return 1.0;
    }
    return 0.0;
}

double PillowResize::BilinearFilter::filter(double x) const
{
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

double PillowResize::HammingFilter::filter(double x) const
{
    if (x < 0.0) {
        x = -x;
    }
    if (x == 0.0) {
        return 1.0;
    }
    if (x >= 1.0) {
        return 0.0;
    }
    x = x * M_PI;
    return sin(x) / x * (0.54 + 0.46 * cos(x));    // NOLINT
}

double PillowResize::BicubicFilter::filter(double x) const
{
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    constexpr double a = -0.5;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {                                         // NOLINT
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;    // NOLINT
    }
    if (x < 2.0) {                                 // NOLINT
        return (((x - 5) * x + 8) * x - 4) * a;    // NOLINT
    }
    return 0.0;
}

double PillowResize::LanczosFilter::_sincFilter(double x)
{
    if (x == 0.0) {
        return 1.0;
    }
    x = x * M_PI;
    return sin(x) / x;
}

double PillowResize::LanczosFilter::filter(double x) const
{
    // Truncated sinc.
    // According to Jim Blinn, the Lanczos kernel (with a = 3)
    // "keeps low frequencies and rejects high frequencies better
    // than any (achievable) filter we've seen so far."[3]
    // (https://en.wikipedia.org/wiki/Lanczos_resampling#Advantages)
    constexpr double lanczos_a_param = 3.0;
    if (-lanczos_a_param <= x && x < lanczos_a_param) {
        return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
    }
    return 0.0;
}

int32_t PillowResize::_precomputeCoeffs(int32_t in_size,
                                        double in0,
                                        double in1,
                                        int32_t out_size,
                                        const std::shared_ptr<Filter>& filterp,
                                        std::vector<int32_t>& bounds,
                                        std::vector<double>& kk)
{
    // Prepare for horizontal stretch.
    const double scale = (in1 - in0) / static_cast<double>(out_size);
    double filterscale = scale;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    // Determine support size (length of resampling filter).
    const double support = filterp->support() * filterscale;

    // Maximum number of coeffs.
    const auto k_size = static_cast<int32_t>(ceil(support)) * 2 + 1;

    // Check for overflow
    if (out_size >
        INT32_MAX / (k_size * static_cast<int32_t>(sizeof(double)))) {
        throw std::runtime_error("Memory error");
    }

    // Coefficient buffer.
    kk.resize(out_size * k_size);

    // Bounds vector.
    bounds.resize(out_size * 2);

    int32_t x = 0;
    constexpr double half_pixel = 0.5;
    for (int32_t xx = 0; xx < out_size; ++xx) {
        double center = in0 + (xx + half_pixel) * scale;
        double ww = 0.0;
        double ss = 1.0 / filterscale;
        // Round the value.
        auto xmin = static_cast<int32_t>(center - support + half_pixel);
        if (xmin < 0) {
            xmin = 0;
        }
        // Round the value.
        auto xmax = static_cast<int32_t>(center + support + half_pixel);
        if (xmax > in_size) {
            xmax = in_size;
        }
        xmax -= xmin;
        double* k = &kk[xx * k_size];
        for (x = 0; x < xmax; ++x) {
            double w = filterp->filter((x + xmin - center + half_pixel) * ss);
            k[x] = w;    // NOLINT
            ww += w;
        }
        for (x = 0; x < xmax; ++x) {
            if (ww != 0.0) {
                k[x] /= ww;    // NOLINT
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < k_size; ++x) {
            k[x] = 0;    // NOLINT
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    return k_size;
}

std::vector<double> PillowResize::_normalizeCoeffs8bpc(
    const std::vector<double>& prekk)
{
    std::vector<double> kk;
    kk.reserve(prekk.size());

    constexpr auto shifted_coeff = static_cast<double>(1U << precision_bits);

    constexpr double half_pixel = 0.5;
    for (const auto& k : prekk) {
        if (k < 0) {
            kk.emplace_back(trunc(-half_pixel + k * shifted_coeff));
        }
        else {
            kk.emplace_back(trunc(half_pixel + k * shifted_coeff));
        }
    }
    return kk;
}

cv::Mat PillowResize::resize(const cv::Mat& src,
                             const cv::Size& out_size,
                             int32_t filter)
{
    cv::Rect2f box(0.F, 0.F, static_cast<float>(src.size().width),
                   static_cast<float>(src.size().height));
    return resize(src, out_size, filter, box);
}

cv::Mat PillowResize::resize(const cv::Mat& src,
                             const cv::Size& out_size,
                             int32_t filter,
                             const cv::Rect2f& box)
{
    // Box = x0,y0,w,h
    // Rect = x0,y0,x1,y1
    const cv::Vec4f rect(box.x, box.y, box.x + box.width, box.y + box.height);

    const int32_t x_size = out_size.width;
    const int32_t y_size = out_size.height;
    if (x_size < 1 || y_size < 1) {
        throw std::runtime_error("Height and width must be > 0");
    }

    if (rect[0] < 0.F || rect[1] < 0.F) {
        throw std::runtime_error("Box offset can't be negative");
    }

    if (static_cast<int32_t>(rect[2]) > src.size().width ||
        static_cast<int32_t>(rect[3]) > src.size().height) {
        throw std::runtime_error("Box can't exceed original image size");
    }

    if (box.width < 0 || box.height < 0) {
        throw std::runtime_error("Box can't be empty");
    }

    // If box's coordinates are int and box size matches requested size
    if (static_cast<int32_t>(box.width) == x_size &&
        static_cast<int32_t>(box.height) == y_size) {
        cv::Rect roi = box;
        return cv::Mat(src, roi);
    }
    if (filter == INTERPOLATION_NEAREST) {
        return _nearestResample(src, x_size, y_size, rect);
    }
    std::shared_ptr<Filter> filter_p;

    // Check filter.
    switch (filter) {
        case INTERPOLATION_BOX:
            filter_p = std::make_shared<BoxFilter>(BoxFilter());
            break;
        case INTERPOLATION_BILINEAR:
            filter_p = std::make_shared<BilinearFilter>(BilinearFilter());
            break;
        case INTERPOLATION_HAMMING:
            filter_p = std::make_shared<HammingFilter>(HammingFilter());
            break;
        case INTERPOLATION_BICUBIC:
            filter_p = std::make_shared<BicubicFilter>(BicubicFilter());
            break;
        case INTERPOLATION_LANCZOS:
            filter_p = std::make_shared<LanczosFilter>(LanczosFilter());
            break;
        default:
            throw std::runtime_error("unsupported resampling filter");
    }

    return PillowResize::_resample(src, x_size, y_size, filter_p, rect);
}

cv::Mat PillowResize::_nearestResample(const cv::Mat& im_in,
                                       int32_t x_size,
                                       int32_t y_size,
                                       const cv::Vec4f& rect)
{
    auto rx0 = static_cast<int32_t>(rect[0]);
    auto ry0 = static_cast<int32_t>(rect[1]);
    auto rx1 = static_cast<int32_t>(rect[2]);
    auto ry1 = static_cast<int32_t>(rect[3]);
    rx0 = std::max(rx0, 0);
    ry0 = std::max(ry0, 0);
    rx1 = std::min(rx1, im_in.size().width);
    ry1 = std::min(ry1, im_in.size().height);

    // Affine tranform matrix.
    cv::Mat m = cv::Mat::zeros(2, 3, CV_64F);
    m.at<double>(0, 0) =
        static_cast<double>(rx1 - rx0) / static_cast<double>(x_size);
    m.at<double>(0, 2) = static_cast<double>(rx0);
    m.at<double>(1, 1) =
        static_cast<double>(ry1 - ry0) / static_cast<double>(y_size);
    m.at<double>(1, 2) = static_cast<double>(ry0);

    cv::Mat im_out = cv::Mat::zeros(y_size, x_size, im_in.type());

    // Check pixel type and determine the pixel size
    // (element size * number of channels).
    size_t pixel_size = 0;
    switch (_getPixelType(im_in)) {
        case CV_8U:
            pixel_size = sizeof(uint8_t);
            break;
        case CV_8S:
            pixel_size = sizeof(int8_t);
            break;
        case CV_16U:
            pixel_size = sizeof(uint16_t);
            break;
        case CV_16S:
            pixel_size = sizeof(int16_t);
            break;
        case CV_32S:
            pixel_size = sizeof(int32_t);
            break;
        case CV_32F:
            pixel_size = sizeof(float);
            break;
        default:
            throw std::runtime_error("Pixel type not supported");
    }
    pixel_size *= im_in.channels();

    const int32_t x0 = 0;
    const int32_t y0 = 0;
    const int32_t x1 = x_size;
    const int32_t y1 = y_size;

    double xo = m.at<double>(0, 2) + m.at<double>(0, 0) * 0.5;
    double yo = m.at<double>(1, 2) + m.at<double>(1, 1) * 0.5;

    auto coord = [](double x) -> int32_t {
        return x < 0. ? -1 : static_cast<int32_t>(x);
    };

    std::vector<int> xintab;
    xintab.resize(im_out.size().width);

    /* Pretabulate horizontal pixel positions */
    int32_t xmin = x1;
    int32_t xmax = x0;
    for (int32_t x = x0; x < x1; ++x) {
        int32_t xin = coord(xo);
        if (xin >= 0 && xin < im_in.size().width) {
            xmax = x + 1;
            if (x < xmin) {
                xmin = x;
            }
            xintab[x] = xin;
        }
        xo += m.at<double>(0, 0);
    }

    for (int32_t y = y0; y < y1; ++y) {
        int32_t yi = coord(yo);
        if (yi >= 0 && yi < im_in.size().height) {
            for (int32_t x = xmin; x < xmax; ++x) {
                memcpy(im_out.ptr(y, x), im_in.ptr(yi, xintab[x]), pixel_size);
            }
        }
        yo += m.at<double>(1, 1);
    }

    return im_out;
}

cv::Mat PillowResize::_resample(const cv::Mat& im_in,
                                int32_t x_size,
                                int32_t y_size,
                                const std::shared_ptr<Filter>& filter_p,
                                const cv::Vec4f& rect)
{
    cv::Mat im_out;
    cv::Mat im_temp;

    std::vector<int32_t> bounds_horiz;
    std::vector<int32_t> bounds_vert;
    std::vector<double> kk_horiz;
    std::vector<double> kk_vert;

    const bool need_horizontal = x_size != im_in.size().width ||
                                 (rect[0] != 0.0F) ||
                                 static_cast<int32_t>(rect[2]) != x_size;
    const bool need_vertical = y_size != im_in.size().height ||
                               (rect[1] != 0.0F) ||
                               static_cast<int32_t>(rect[3]) != y_size;

    // Compute horizontal filter coefficients.
    const int32_t ksize_horiz =
        _precomputeCoeffs(im_in.size().width, rect[0], rect[2], x_size,
                          filter_p, bounds_horiz, kk_horiz);

    // Compute vertical filter coefficients.
    const int32_t ksize_vert =
        _precomputeCoeffs(im_in.size().height, rect[1], rect[3], y_size,
                          filter_p, bounds_vert, kk_vert);

    // First used row in the source image.
    const int32_t ybox_first = bounds_vert[0];
    // Last used row in the source image.
    const int32_t ybox_last =
        bounds_vert[y_size * 2 - 2] + bounds_vert[y_size * 2 - 1];

    // Two-pass resize, horizontal pass.
    if (need_horizontal) {
        // Shift bounds for vertical pass.
        for (int32_t i = 0; i < y_size; ++i) {
            bounds_vert[i * 2] -= ybox_first;
        }

        // Create destination image with desired ouput width and same input pixel type.
        im_temp.create(ybox_last - ybox_first, x_size, im_in.type());
        if (!im_temp.empty()) {
            _resampleHorizontal(im_temp, im_in, ybox_first, ksize_horiz,
                                bounds_horiz, kk_horiz);
        }
        else {
            return cv::Mat();
        }
        im_out = im_temp;
    }

    // Vertical pass.
    if (need_vertical) {
        // Create destination image with desired ouput size and same input pixel type.

        const auto new_w =
            (im_temp.size().width != 0) ? im_temp.size().width : x_size;
        im_out.create(y_size, new_w, im_in.type());
        if (!im_out.empty()) {
            if (im_temp.empty()) {
                im_temp = im_in;
            }
            // Input can be the original image or horizontally resampled one.
            _resampleVertical(im_out, im_temp, ksize_vert, bounds_vert,
                              kk_vert);
        }
        else {
            return cv::Mat();
        }
    }

    // None of the previous steps are performed, copying.
    if (im_out.empty()) {
        im_out = im_in;
    }

    return im_out;
}

void PillowResize::_resampleHorizontal(cv::Mat& im_out,
                                       const cv::Mat& im_in,
                                       int32_t offset,
                                       int32_t ksize,
                                       const std::vector<int32_t>& bounds,
                                       const std::vector<double>& prekk)
{
    // Check pixel type.
    switch (_getPixelType(im_in)) {
        case CV_8U:
            return _resampleHorizontal<uint8_t>(
                im_out, im_in, offset, ksize, bounds, prekk,
                _normalizeCoeffs8bpc,
                static_cast<double>(1U << (precision_bits - 1U)), _clip8);
        case CV_8S:
            return _resampleHorizontal<int8_t>(im_out, im_in, offset, ksize,
                                               bounds, prekk, nullptr, 0.,
                                               _roundUp<int8_t>);
        case CV_16U:
            return _resampleHorizontal<uint16_t>(im_out, im_in, offset, ksize,
                                                 bounds, prekk);
        case CV_16S:
            return _resampleHorizontal<int16_t>(im_out, im_in, offset, ksize,
                                                bounds, prekk, nullptr, 0.,
                                                _roundUp<int16_t>);
        case CV_32S:
            return _resampleHorizontal<int32_t>(im_out, im_in, offset, ksize,
                                                bounds, prekk, nullptr, 0.,
                                                _roundUp<int32_t>);
        case CV_32F:
            return _resampleHorizontal<float>(im_out, im_in, offset, ksize,
                                              bounds, prekk);
        default:
            throw std::runtime_error("Pixel type not supported");
    }
}

void PillowResize::_resampleVertical(cv::Mat& im_out,
                                     const cv::Mat& im_in,
                                     int32_t ksize,
                                     const std::vector<int32_t>& bounds,
                                     const std::vector<double>& prekk)
{
    im_out = im_out.t();
    _resampleHorizontal(im_out, im_in.t(), 0, ksize, bounds, prekk);
    im_out = im_out.t();
}
