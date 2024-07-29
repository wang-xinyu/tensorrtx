#pragma once

#include <assert.h>
#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildEngineYolov10DetN(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels);

nvinfer1::IHostMemory* buildEngineYolov10DetS(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels);

nvinfer1::IHostMemory* buildEngineYolov10DetM(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels);

nvinfer1::IHostMemory* buildEngineYolov10DetBL(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                               nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                               int& max_channels);

nvinfer1::IHostMemory* buildEngineYolov10DetX(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                              int& max_channels);
