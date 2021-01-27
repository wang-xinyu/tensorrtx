#ifndef TENSORRTX_PSENET_H
#define TENSORRTX_PSENET_H
#include <memory>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "layers.h"
class PSENet
{
public:
	PSENet(int max_side_len, int min_side_len, float threshold, int num_kernel, int stride);
	~PSENet();

	ICudaEngine* createEngine(IBuilder* builder, IBuilderConfig* config);
	void serializeEngine();
	void deserializeEngine();
	void init();
	void inferenceOnce(IExecutionContext& context, float* input, float* output, int input_h, int input_w);
	void detect(std::string image_path);
	float* preProcess(cv::Mat image, int& resize_h, int& resize_w, float& ratio_h, float& ratio_w);
	std::vector<cv::RotatedRect> postProcess(float* origin_output, int resize_h, int resize_w);

private:
	Logger gLogger;
	std::shared_ptr<nvinfer1::IRuntime> mRuntime;
	std::shared_ptr<nvinfer1::ICudaEngine> mCudaEngine;
	std::shared_ptr<nvinfer1::IExecutionContext> mContext;
	DataType dt = DataType::kFLOAT;
	const char* input_name_ = "input";
	const char* output_name_ = "output";
	int max_side_len_ = 1024;
	int min_side_len_ = 640;
	float post_threshold_ = 0.9;
	int num_kernels_ = 6;
	int stride_ = 4;
};

#endif // TENSORRTX_PSENET_H
