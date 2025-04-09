#pragma once

#include <assert.h>
#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildEngineYolo11Cls(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            std::string& type, int max_channels);

nvinfer1::IHostMemory* buildEngineYolo11Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);

nvinfer1::IHostMemory* buildEngineYolo11Seg(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);

nvinfer1::IHostMemory* buildEngineYolo11Pose(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                             int& max_channels, std::string& type);

nvinfer1::IHostMemory* buildEngineYolo11Obb(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                            nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                            int& max_channels, std::string& type);
