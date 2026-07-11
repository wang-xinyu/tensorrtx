#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "NvInfer.h"

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);

void freeWeights(std::map<std::string, nvinfer1::Weights>& weightMap);

nvinfer1::IElementWiseLayer* Qwen3RMSNorm(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights> weightMap, nvinfer1::ITensor& input,
                                          int ndDims, const std::string& lname);

nvinfer1::ILayer* addLinear(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                            nvinfer1::ITensor& input, int out_features, const std::string& lname);

nvinfer1::ILayer* Qwen3Embedding(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input_ids);

nvinfer1::ILayer* GatherLastToken(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& hidden_states);

nvinfer1::ILayer* Qwen3MLP(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
                           nvinfer1::ITensor& input, const std::string& lname);

std::tuple<nvinfer1::ILayer*, nvinfer1::ILayer*, nvinfer1::ILayer*> Qwen3DecoderLayer(
        nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap,
        nvinfer1::ITensor& hidden_states, nvinfer1::ITensor& cos, nvinfer1::ITensor& sin,
        nvinfer1::ITensor& k_cache_input, nvinfer1::ITensor& v_cache_input, nvinfer1::ITensor& mask_input,
        int layer_idx);

std::tuple<nvinfer1::ILayer*, nvinfer1::ILayer*, nvinfer1::ILayer*> Qwen3Attention(
        nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights> weightMap,
        nvinfer1::ITensor& hidden_states, nvinfer1::ITensor& cos, nvinfer1::ITensor& sin,
        nvinfer1::ITensor& k_cache_input, nvinfer1::ITensor& v_cache_input, nvinfer1::ITensor& mask_input,
        int layer_idx, const std::string& lname);
