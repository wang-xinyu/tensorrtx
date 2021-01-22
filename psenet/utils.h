#ifndef TENSORRTX_UTILS_H
#define TENSORRTX_UTILS_H

#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "assert.h"
#include <fstream>

using namespace nvinfer1;

std::map<std::string, Weights> loadWeights(const std::string file);

cv::RotatedRect expandBox(const cv::RotatedRect& inBox, float ratio = 1.0);

void drawRects(cv::Mat& image, std::vector<cv::RotatedRect> boxes, float stride, float ratio_h, float ratio_w, float expand_ratio);

cv::Mat renderSegment(cv::Mat image, const cv::Mat& mask);

// <============== Operator =============>
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    Logger() : Logger(Severity::kWARNING) {}

    Logger(Severity severity) : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{ Severity::kWARNING };
};

#endif
