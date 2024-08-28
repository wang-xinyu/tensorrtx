#pragma once

#include <NvInfer.h>
#include <string>
// yolov9
nvinfer1::IHostMemory* build_engine_yolov9_t(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                             std::string& wts_name, bool isConvert = false);
nvinfer1::IHostMemory* build_engine_yolov9_s(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                             std::string& wts_name, bool isConvert = false);
nvinfer1::IHostMemory* build_engine_yolov9_m(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                             std::string& wts_name, bool isConvert = false);
nvinfer1::IHostMemory* build_engine_yolov9_c(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                             std::string& wts_name);
nvinfer1::IHostMemory* build_engine_yolov9_e(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                             nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                             std::string& wts_name);
// gelan
nvinfer1::IHostMemory* build_engine_gelan_t(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                            nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                            std::string& wts_name);
nvinfer1::IHostMemory* build_engine_gelan_m(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                            nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                            std::string& wts_name);
nvinfer1::IHostMemory* build_engine_gelan_c(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                            nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                            std::string& wts_name);
nvinfer1::IHostMemory* build_engine_gelan_e(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                            nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                            std::string& wts_name);
