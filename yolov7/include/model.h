#pragma once

#include "NvInfer.h"
#include <string>

nvinfer1::ICudaEngine* build_engine_yolov7e6e(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
nvinfer1::ICudaEngine* build_engine_yolov7d6(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
nvinfer1::ICudaEngine* build_engine_yolov7e6(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
nvinfer1::ICudaEngine* build_engine_yolov7w6(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
nvinfer1::ICudaEngine* build_engine_yolov7x(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
nvinfer1::ICudaEngine* build_engine_yolov7(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
nvinfer1::ICudaEngine* build_engine_yolov7_tiny(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string& wts_name);
