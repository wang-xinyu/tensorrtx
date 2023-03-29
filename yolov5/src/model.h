#pragma once

#include <NvInfer.h>
#include <string>

nvinfer1::ICudaEngine* build_det_engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                        nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                        float& gd, float& gw, std::string& wts_name);

nvinfer1::ICudaEngine* build_det_p6_engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                           nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                           float& gd, float& gw, std::string& wts_name);

nvinfer1::ICudaEngine* build_cls_engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, float& gd, float& gw, std::string& wts_name);

nvinfer1::ICudaEngine* build_seg_engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, float& gd, float& gw, std::string& wts_name);
