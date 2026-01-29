#pragma once

#include <map>
#include <string>
#include <vector>
#include "NvInfer.h"

using namespace std;
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                      std::string lname, float eps);

nvinfer1::IElementWiseLayer* convBnSiLU(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                        int ch, std::vector<int> k, int s, std::string lname, int g = 1);

nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int n, bool c3k, bool shortcut, bool atnn, float e, std::string lname);

nvinfer1::IElementWiseLayer* SPPF(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                                  int c2, int k, bool shortcut, std::string lname);

nvinfer1::IElementWiseLayer* C2PSA(nvinfer1::INetworkDefinition* network,
                                   std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                   int c1, int c2, int n, float e, std::string lname);

nvinfer1::ILayer* DWConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, std::vector<int> k, int s, std::string lname);

nvinfer1::ILayer* conv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int ch, std::vector<int> k, int s, std::string lname, int g = 1,
                       bool act = true);

nvinfer1::IPluginV2Layer* addYoloLayer(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       const std::vector<int>& strides, const std::vector<int>& fm_sizes,
                                       int stridesLength, bool is_segmentation, bool is_pose, bool is_obb,
                                       int anchorCount);