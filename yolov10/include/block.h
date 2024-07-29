#pragma once

#include <map>
#include <string>
#include <vector>
#include "NvInfer.h"

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                      std::string lname, float eps);

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                        int ch, int k, int s, std::string lname, int g = 1);

nvinfer1::IElementWiseLayer* C2F(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                 int c2, int n, bool shortcut, float e, std::string lname);

nvinfer1::IElementWiseLayer* C2(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c1,
                                int c2, int n, bool shortcut, float e, std::string lname);

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int k, std::string lname);

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname);

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network, std::vector<nvinfer1::ILayer*> dets,
                                       const int* px_arry, int px_arry_num);

nvinfer1::ILayer* SCDown(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, int k, int s, std::string lname);

nvinfer1::ILayer* PSA(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                      nvinfer1::ITensor& input, int ch, std::string lname);

nvinfer1::ILayer* C2fCIB(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int c1, int c2, int n, bool shortcut, bool lk, float e,
                         std::string lname);
