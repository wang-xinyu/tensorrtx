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
                                        int ch, std::vector<int> k, int s, std::string lname,int p = 0, int g = 1,
                                        int d = 1);


nvinfer1::ILayer* Conv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int c_out, std::string lname, int k = 1, int s = 1, int padding = 0,
                       int g = 1, bool act = true);

nvinfer1::IShuffleLayer* DFL(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             nvinfer1::ITensor& input, int ch, int grid, int k, int s, int p, std::string lname);

nvinfer1::IPluginV2Layer* addYoLoLayer(nvinfer1::INetworkDefinition* network,
                                       std::vector<nvinfer1::IConcatenationLayer*> dets, const int* px_arry,
                                       int px_arry_num, bool is_segmentation, bool is_pose, bool is_obb);

nvinfer1::IElementWiseLayer* C3k(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c2,
                                 std::string lname, int n = 1, bool shortcut = true, int g = 1, float e = 0.5,
                                 int k = 3);

nvinfer1::IElementWiseLayer* C3K2(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input, int c2,
                                  int n, std::string lname, bool c3k = false, float e = 0.5, int g = 1,
                                  bool shortcut = true);

nvinfer1::ILayer* AAttn(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int dim, int num_heads, std::string lname, int area = 1);


nvinfer1::ILayer* DWConv(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int ch, std::vector<int> k, int s, std::string lname);


nvinfer1::IElementWiseLayer* ABlock(nvinfer1::INetworkDefinition* network,
                                    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                    int dim, int num_heads, std::string lname, float mlp_ratio=1.2, int area=1);

nvinfer1::ILayer* A2C2f(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>,
                        nvinfer1::ITensor& input, int c2, int n, std::string lname, bool a2 = true, int area = 1,
                        bool residual = false, float mlp_ratio = 2.0, float e = 0.5, int g = 1, bool shortcut = true);




nvinfer1::IElementWiseLayer* DSConv(nvinfer1::INetworkDefinition* network,
                                    std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                    int c_in, int c_out, std::string lname, int k = 3, int s = 1, int p = 0, int d = 1,
                                    bool bias = false);

nvinfer1::ILayer* DSBottleneck(nvinfer1::INetworkDefinition* network,
                               std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input, int c1,
                               int c2, std::string lname, bool shortcut = true, float e = 0.5, int k1 = 3, int k2 = 5,
                               int d2 = 1);

nvinfer1::ILayer* DSC3k(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                        nvinfer1::ITensor& input, int c2, int n, std::string lname, bool shortcut = true, int g = 1,
                        float e = 0.5, int k1 = 3, int k2 = 5, int d2 = 1);


nvinfer1::ILayer* DSC3K2(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                         nvinfer1::ITensor& input, int c2, std::string lname, int n = 1, bool dsc3k = false,
                         float e = 0.5, int g = 1, bool shortcut = true, int k1 = 3, int k2 = 7, int d2 = 1);

nvinfer1::ILayer* FuseModule(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                             std::vector<nvinfer1::ITensor*>& input, int c_in, bool channel_adjust, std::string lname);


nvinfer1::ISoftMaxLayer* AdaHyperedgeGen(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                         int node_dim, int num_hyperedges, std::string lname, int num_heads = 4,
                                         std::string context = "both");

nvinfer1::IElementWiseLayer* GELU(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input);

nvinfer1::IElementWiseLayer* AdaHGConv(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                       int embed_dim, std::string lname, int num_hyperedges = 16, int num_heads = 4,
                                       std::string context = "both");

nvinfer1::IShuffleLayer* AdaHGComputation(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                          int embed_dim, std::string lname, int num_hyperedges = 16, int num_heads = 8,
                                          std::string context = "both");

nvinfer1::ILayer* C3AH(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                       nvinfer1::ITensor& input, int c2, std::string lname, float e = 1.0, int num_hyperedges = 8,
                       std::string context = "both");

nvinfer1::ILayer* HyperACE(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
                           std::vector<nvinfer1::ITensor*> input, int c1, int c2, std::string lname, int n = 1,
                           int num_hyperedges = 8, bool dsc3k = false, bool shortcut = false, float e1 = 0.5,
                           float e2 = 1, std::string context = "both", bool channel_adjust = true);

nvinfer1::ILayer* DownsampleConv(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                 int in_channels, std::string lname, bool channel_adjust);


nvinfer1::IElementWiseLayer* FullPad_Tunnel(nvinfer1::INetworkDefinition* network,
                                            std::map<std::string, nvinfer1::Weights> weightMap,
                                            std::vector<nvinfer1::ITensor*> input, std::string lname);

nvinfer1::ILayer* DownsampleConv(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                 int in_channels, std::string lname, bool channel_adjust = true);


void cout_dim(nvinfer1::ITensor& input);