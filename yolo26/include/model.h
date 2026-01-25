#pragma once

#include <assert.h>
#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildEngineYolo26Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);