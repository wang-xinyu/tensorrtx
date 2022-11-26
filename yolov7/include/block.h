#pragma once

#include "NvInfer.h"
#include <string>
#include <vector>
#include <map>

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

nvinfer1::IElementWiseLayer* convBnSilu(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c2, int k, int s, int p, std::string lname);

nvinfer1::ILayer* ReOrg(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int inch);

nvinfer1::ILayer* DownC(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c1, int c2, const std::string& lname);

nvinfer1::IElementWiseLayer* SPPCSPC(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c2, const std::string& lname);

nvinfer1::IElementWiseLayer* RepConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c2, int k, int s, const std::string& lname);

nvinfer1::IActivationLayer* convBlockLeakRelu(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int outch, int ksize, int s, int p, std::string lname);

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, std::string lname, std::vector<nvinfer1::IConvolutionLayer*> dets);

