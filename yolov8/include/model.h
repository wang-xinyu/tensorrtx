#pragma once
#include "NvInfer.h"
#include <string>
#include <assert.h>

nvinfer1::IHostMemory* buildEngineYolov8n(nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8s(nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8m(nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8l(nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);

nvinfer1::IHostMemory* buildEngineYolov8x(nvinfer1::IBuilder* builder,
nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, const std::string& wts_path);
