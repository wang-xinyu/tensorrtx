#pragma once

#include <NvInfer.h>
#include <string>
nvinfer1::IHostMemory* build_engine_yolov9_e(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string& wts_name);
nvinfer1::IHostMemory* build_engine_yolov9_c(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string& wts_name);