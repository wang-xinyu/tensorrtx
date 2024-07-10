#pragma once

#include <NvInfer.h>
#include <string>

nvinfer1::IHostMemory* build_cls_engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, float& gd, float& gw, std::string& wts_name);
