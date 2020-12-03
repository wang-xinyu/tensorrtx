#ifndef TENSORRTX_UTILS_H
#define TENSORRTX_UTILS_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <map>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

std::map<std::string, Weights> loadWeights(const std::string file);

cv::RotatedRect expandBox(const cv::RotatedRect &inBox, float ratio = 1.0);

void drawRects(cv::Mat &image, cv::Mat mask, float ratio_h, float ratio_w, int stride, float expand_ratio = 1.4);

cv::Mat renderSegment(cv::Mat image, const cv::Mat &mask);

// <============== Operator =============>
struct InferDeleter
{
	template <typename T>
	void operator()(T *obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

#endif
