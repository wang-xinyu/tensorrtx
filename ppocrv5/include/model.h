#pragma once

#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildPPOCRv5MobileDet(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildPPOCRv5ServerDet(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildPPOCRv5MobileRec(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildPPOCRv5ServerRec(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildPPOCRv5Model(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                         nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildSLANeXtWiredModel(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                              nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildPPFormulaNetEncoderDirect(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                                      nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildPPFormulaNetDecoderDirect(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                                      nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildEnginePPOCRv5Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildEnginePPOCRv5Det(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath,
                                             const std::string& variant);

nvinfer1::IHostMemory* buildEnginePPOCRv5Rec(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildEnginePPOCRv5Rec(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wtsPath,
                                             const std::string& variant);

nvinfer1::IHostMemory* buildEnginePPOCRv5Model(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                               nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildEnginePPFormulaNetEncoder(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                                      nvinfer1::DataType dt, const std::string& wtsPath);

nvinfer1::IHostMemory* buildEnginePPFormulaNetDecoder(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                                      nvinfer1::DataType dt, const std::string& wtsPath);
