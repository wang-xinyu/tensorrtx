#pragma once
#include "NvInfer.h"
#include <string>
#include <assert.h>

nvinfer1::IHostMemory* buildEngineYolov8n(const int& batchsize, nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8s(const int& batchsize, nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8m(const int& batchsize, nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8l(const int& batchsize, nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8x(const int& batchsize, nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
