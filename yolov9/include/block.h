#include "config.h"
#include "yololayer.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
void PrintDim(const ILayer* layer, std::string log = "");
std::map<std::string, Weights> loadWeights(const std::string file);
int get_width(int x, float gw, int divisor = 8);
int get_depth(int x, float gd);
ILayer* Proto(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c_, int c2,
              std::string lname);
std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname);
// ----------------------------------------------------------------
nvinfer1::ILayer* convBnSiLU(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                             nvinfer1::ITensor& input, int ch, int k, int s, int p, std::string lname, int g = 1);
ILayer* ELAN1(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2,
              int c3, int c4, std::string lname);
ILayer* RepNCSPELAN4(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1,
                     int c2, int c3, int c4, int c5, std::string lname);
ILayer* ADown(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2,
              std::string lname);
ILayer* AConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c2,
              std::string lname);
std::vector<ILayer*> CBLinear(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                              std::vector<int> c2s, int k, int s, int p, int g, std::string lname);
ILayer* CBFuse(INetworkDefinition* network, std::vector<std::vector<ILayer*>> input, std::vector<int> idx,
               std::vector<int> strides);
ILayer* SPPELAN(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2,
                int c3, std::string lname);
std::vector<IConcatenationLayer*> DualDDetect(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                              std::vector<ILayer*> dets, int cls, std::vector<int> ch,
                                              std::string lname);
nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                       std::vector<nvinfer1::IConcatenationLayer*> dets, bool is_segmentation);
nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int k, int s, int p, std::string lname);
nvinfer1::ILayer* convBnNoAct(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int ch,
                              int k, int s, int p, std::string lname, int g);
std::vector<IConcatenationLayer*> DDetect(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                          std::vector<ILayer*> dets, int cls, std::vector<int> ch, std::string lname);
