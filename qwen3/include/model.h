#pragma once

#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildEngineQwen3_0_6B(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_dir);
