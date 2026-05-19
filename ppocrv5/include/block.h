#pragma once

#include "NvInfer.h"

#include <map>
#include <string>
#include <vector>

int calculateP(int ksize);

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);
void freeWeights(std::map<std::string, nvinfer1::Weights>& weightMap);
nvinfer1::Weights getWeights(std::map<std::string, nvinfer1::Weights>& weightMap, const std::string& name);
nvinfer1::Weights getWeightsByPrefix(std::map<std::string, nvinfer1::Weights>& weightMap, const std::string& name);

nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      const std::string& lname, float eps);
nvinfer1::IScaleLayer* addBatchNorm2dByPrefix(nvinfer1::INetworkDefinition* network,
                                              std::map<std::string, nvinfer1::Weights>& weightMap,
                                              nvinfer1::ITensor& input, const std::string& lname, float eps);

nvinfer1::IConvolutionLayer* addConv2d(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       const std::string& wname, int outChannels, nvinfer1::DimsHW ksize,
                                       nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups);

nvinfer1::IConvolutionLayer* addConv2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       nvinfer1::Weights kernel, nvinfer1::Weights bias, int outChannels,
                                       nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
                                       nvinfer1::DimsHW dilation, int groups, const std::string& lname);

nvinfer1::IConvolutionLayer* convBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, int k, int s, int p, const std::string& convName);

nvinfer1::IConvolutionLayer* convBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, int k, int s, int p, int groups, const std::string& convName);

nvinfer1::IConvolutionLayer* convBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                      nvinfer1::DimsHW padding, int groups, const std::string& convName);

nvinfer1::IScaleLayer* convBn(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                              int outChannels, int k, int s, int p, const std::string& convName,
                              const std::string& bnName);

nvinfer1::IScaleLayer* convBn(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                              int outChannels, int k, int s, int p, int groups, const std::string& convName,
                              const std::string& bnName);

nvinfer1::IScaleLayer* convBn(nvinfer1::INetworkDefinition* network,
                              std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                              int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                              nvinfer1::DimsHW padding, int groups, const std::string& convName,
                              const std::string& bnName);

nvinfer1::IElementWiseLayer* convBnHSwish(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int outChannels, int k, int s, int p, const std::string& convName,
                                          const std::string& bnName);

nvinfer1::IElementWiseLayer* convBnHSwish(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int outChannels, int k, int s, int p, int groups, const std::string& convName,
                                          const std::string& bnName);
nvinfer1::IElementWiseLayer* convBnHSwishByPrefix(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, int outChannels, int k, int s, int p,
                                                  int groups, const std::string& convName, const std::string& bnName);
nvinfer1::IElementWiseLayer* convBnHSwishByPrefix(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, int outChannels, nvinfer1::DimsHW ksize,
                                                  nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups,
                                                  const std::string& convName, const std::string& bnName);

nvinfer1::IElementWiseLayer* convBnSwish(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                         nvinfer1::DimsHW padding, int groups, const std::string& convName,
                                         const std::string& bnName);

nvinfer1::IDeconvolutionLayer* addDeconv2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                           nvinfer1::Weights kernel, nvinfer1::Weights bias, int outChannels,
                                           nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride, nvinfer1::DimsHW padding,
                                           int groups, const std::string& lname);

nvinfer1::IDeconvolutionLayer* deconvBias(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int outChannels, int k, int s, int p, const std::string& weightName);

nvinfer1::IScaleLayer* addChannelScale(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       nvinfer1::Weights shift, nvinfer1::Weights scale, nvinfer1::Weights power,
                                       const std::string& lname);

nvinfer1::IScaleLayer* addChannelBias(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      const std::string& biasName, const std::string& lname);

nvinfer1::IScaleLayer* addUniformAffine(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                        const std::string& scaleName, const std::string& biasName,
                                        const std::string& lname);

nvinfer1::IScaleLayer* learnableRepLayer(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int outChannels, int k, int s, int p, int groups, const std::string& convName,
                                         int affineIndex, bool withAct);

nvinfer1::IScaleLayer* learnableRepLayer(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                         nvinfer1::DimsHW padding, int groups, const std::string& convName,
                                         int affineIndex, bool withAct);

nvinfer1::IReduceLayer* addGlobalAvgPool2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                           const std::string& lname);

nvinfer1::IElementWiseLayer* seLayer(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                     int squeezeChannels, int outChannels, const std::string& conv0Name,
                                     const std::string& conv1Name);
nvinfer1::IElementWiseLayer* seLayerByPrefix(nvinfer1::INetworkDefinition* network,
                                             std::map<std::string, nvinfer1::Weights>& weightMap,
                                             nvinfer1::ITensor& input, int squeezeChannels, int outChannels,
                                             const std::string& conv0Name, const std::string& conv1Name);

nvinfer1::IElementWiseLayer* rseLayer(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int outChannels, int squeezeChannels, int k, int p, const std::string& convName,
                                      const std::string& conv0Name, const std::string& conv1Name);

nvinfer1::IElementWiseLayer* ppLcNetBlock(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int inChannels, int outChannels, int dwName, int dwBnName, int pwName,
                                          int pwBnName, int kernel, nvinfer1::DimsHW stride, bool useSe);
nvinfer1::IElementWiseLayer* slanetLcNetBlock(nvinfer1::INetworkDefinition* network,
                                              std::map<std::string, nvinfer1::Weights>& weightMap,
                                              nvinfer1::ITensor& input, int inChannels, int outChannels, int dwName,
                                              int dwBnName, int pwName, int pwBnName, int kernel, int stride,
                                              bool useSe);
nvinfer1::ITensor* addSvtrAttention(nvinfer1::INetworkDefinition* network,
                                    std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                    const std::string& qkvName, const std::string& projName, const std::string& lname);
nvinfer1::ITensor* addSvtrBlock(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                const std::string& ln0Name, const std::string& qkvName, const std::string& projName,
                                const std::string& ln1Name, const std::string& mlp0Name, const std::string& mlp1Name,
                                const std::string& lname);
nvinfer1::ITensor* addHgConvBlock(nvinfer1::INetworkDefinition* network,
                                  std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                  int bodyChannels, int bodyCount, int bodyStartName, int bodyStartBnName,
                                  int squeezeName, int squeezeBnName, int squeezeChannels, int exciteName,
                                  int exciteBnName, int exciteChannels);
nvinfer1::ITensor* addHgStandardBlock(nvinfer1::INetworkDefinition* network,
                                      std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                      int bodyChannels, int bodyCount, int firstDwName, int firstDwBnName,
                                      int firstDwChannels, nvinfer1::DimsHW firstStride, int bodyStartName,
                                      int bodyStartBnName, int squeezeName, int squeezeBnName, int squeezeChannels,
                                      int exciteName, int exciteBnName, int exciteChannels);
nvinfer1::ITensor* addHgLightBlock(nvinfer1::INetworkDefinition* network,
                                   std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& identity,
                                   int bodyChannels, int bodyCount, int bodyStartName, int bodyStartBnName,
                                   int squeezeName, int squeezeBnName, int squeezeChannels, int exciteName,
                                   int exciteBnName, int exciteChannels, bool residual);
nvinfer1::ITensor* addHgConvBlockByPrefix(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int bodyChannels, int bodyCount, int bodyStartName, int bodyStartBnName,
                                          int squeezeName, int squeezeBnName, int squeezeChannels, int exciteName,
                                          int exciteBnName, int exciteChannels);
nvinfer1::ITensor* addHgStandardBlockByPrefix(nvinfer1::INetworkDefinition* network,
                                              std::map<std::string, nvinfer1::Weights>& weightMap,
                                              nvinfer1::ITensor& input, int bodyChannels, int bodyCount,
                                              int firstDwName, int firstDwBnName, int firstDwChannels,
                                              nvinfer1::DimsHW firstStride, int bodyStartName, int bodyStartBnName,
                                              int squeezeName, int squeezeBnName, int squeezeChannels, int exciteName,
                                              int exciteBnName, int exciteChannels);
nvinfer1::ITensor* addHgLightBlockByPrefix(nvinfer1::INetworkDefinition* network,
                                           std::map<std::string, nvinfer1::Weights>& weightMap,
                                           nvinfer1::ITensor& identity, int bodyChannels, int bodyCount,
                                           int bodyStartName, int bodyStartBnName, int squeezeName, int squeezeBnName,
                                           int squeezeChannels, int exciteName, int exciteBnName, int exciteChannels,
                                           bool residual);
nvinfer1::ITensor* addLargeKernelBranch(nvinfer1::INetworkDefinition* network,
                                        std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                        int conv0Name, nvinfer1::DimsHW conv0Kernel, nvinfer1::DimsHW conv0Padding,
                                        int conv1Name, nvinfer1::DimsHW conv1Kernel, nvinfer1::DimsHW conv1Padding,
                                        int conv2Name, nvinfer1::DimsHW conv2Kernel, nvinfer1::DimsHW conv2Padding);
nvinfer1::ITensor* addLargeKernelBlock(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int reduceName, int branch7Name0, int branch7Name1, int branch7Name2,
                                       int branch5Name0, int branch5Name1, int branch5Name2, int branch3Name0,
                                       int branch3Name1, int branch3Name2, int expandName, const std::string& bnName);
nvinfer1::ITensor* addUvdocResidualBlock(nvinfer1::INetworkDefinition* network,
                                         std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                         int channels, int conv0Name, int bn0Name, int conv1Name, int bn1Name,
                                         int dilation);
nvinfer1::ITensor* addUvdocDownBlock(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                     int channels, int skipConvName, int skipBnName, int conv0Name, int bn0Name,
                                     int conv1Name, int bn1Name);
nvinfer1::ITensor* addUvdocReflectPad2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, int batch,
                                        int channels, int height, int width, int pad, const std::string& lname);
nvinfer1::ITensor* addUvdocPRelu(nvinfer1::INetworkDefinition* network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                 const std::string& preluName, const std::string& lname);

nvinfer1::IConcatenationLayer* addConcat(nvinfer1::INetworkDefinition* network,
                                         const std::vector<nvinfer1::ITensor*>& inputs, int axis,
                                         const std::string& lname);

nvinfer1::IResizeLayer* addNearestResize(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                         const std::vector<float>& scales, const std::string& lname);

nvinfer1::IElementWiseLayer* addSum(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input0,
                                    nvinfer1::ITensor& input1, const std::string& lname);

nvinfer1::IPoolingLayer* addPool2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   nvinfer1::PoolingType type, int k, int s, int p, const std::string& lname);
nvinfer1::IPoolingLayer* addPool2d(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                   nvinfer1::PoolingType type, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                   nvinfer1::DimsHW padding, const std::string& lname);

nvinfer1::IActivationLayer* addRelu(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                    const std::string& lname);
nvinfer1::IActivationLayer* addSigmoid(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       const std::string& lname);
nvinfer1::IElementWiseLayer* addSwish(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                      const std::string& lname);
nvinfer1::IElementWiseLayer* addHardSwish(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input);
nvinfer1::IActivationLayer* addHardSigmoid(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input);

nvinfer1::IElementWiseLayer* addLinear(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int inChannels, int outChannels, const std::string& linearName);
nvinfer1::IElementWiseLayer* addLinear(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                       nvinfer1::Weights kernel, nvinfer1::Weights bias, int inChannels,
                                       int outChannels, const std::string& lname);
nvinfer1::IElementWiseLayer* addLinear2dByPrefix(nvinfer1::INetworkDefinition* network,
                                                 std::map<std::string, nvinfer1::Weights>& weightMap,
                                                 nvinfer1::ITensor& input, int inChannels, int outChannels,
                                                 const std::string& linearName);
nvinfer1::Weights sliceLinearKernel(std::map<std::string, nvinfer1::Weights>& weightMap, const std::string& linearName,
                                    int inChannels, int outChannels, int part);
nvinfer1::Weights sliceLinearBias(std::map<std::string, nvinfer1::Weights>& weightMap, const std::string& linearName,
                                  int outChannels, int part);
nvinfer1::Weights sliceLinearKernelByPrefix(std::map<std::string, nvinfer1::Weights>& weightMap,
                                            const std::string& linearName, int inChannels, int outChannels, int part);
nvinfer1::Weights sliceLinearBiasByPrefix(std::map<std::string, nvinfer1::Weights>& weightMap,
                                          const std::string& linearName, int outChannels, int part);
nvinfer1::IElementWiseLayer* addLinearPart(nvinfer1::INetworkDefinition* network,
                                           std::map<std::string, nvinfer1::Weights>& weightMap,
                                           nvinfer1::ITensor& input, int inChannels, int outChannels,
                                           const std::string& linearName, int part, const std::string& lname);
nvinfer1::IElementWiseLayer* addLinearPartByPrefix(nvinfer1::INetworkDefinition* network,
                                                   std::map<std::string, nvinfer1::Weights>& weightMap,
                                                   nvinfer1::ITensor& input, int inChannels, int outChannels,
                                                   const std::string& linearName, int part, const std::string& lname);
nvinfer1::IElementWiseLayer* addLinearByPrefix(nvinfer1::INetworkDefinition* network,
                                               std::map<std::string, nvinfer1::Weights>& weightMap,
                                               nvinfer1::ITensor& input, int inChannels, int outChannels,
                                               const std::string& linearName, const std::string& lname);
int deepcopyOrder(const std::string& key, const std::string& prefix);
nvinfer1::Weights getWeightsByPrefixOrder(std::map<std::string, nvinfer1::Weights>& weightMap,
                                          const std::string& prefix, int order);
nvinfer1::Weights sliceLinearKernelByPrefixOrder(std::map<std::string, nvinfer1::Weights>& weightMap,
                                                 const std::string& linearName, int order, int inChannels,
                                                 int outChannels, int part);
nvinfer1::Weights sliceLinearBiasByPrefixOrder(std::map<std::string, nvinfer1::Weights>& weightMap,
                                               const std::string& linearName, int order, int outChannels, int part);
nvinfer1::IElementWiseLayer* addLinearPartByPrefixOrder(nvinfer1::INetworkDefinition* network,
                                                        std::map<std::string, nvinfer1::Weights>& weightMap,
                                                        nvinfer1::ITensor& input, int inChannels, int outChannels,
                                                        const std::string& linearName, int order, int part,
                                                        const std::string& lname);
nvinfer1::IElementWiseLayer* addLinearByPrefixOrder(nvinfer1::INetworkDefinition* network,
                                                    std::map<std::string, nvinfer1::Weights>& weightMap,
                                                    nvinfer1::ITensor& input, int inChannels, int outChannels,
                                                    const std::string& linearName, int order, const std::string& lname);
nvinfer1::IElementWiseLayer* addLayerNormByPrefixOrder(nvinfer1::INetworkDefinition* network,
                                                       std::map<std::string, nvinfer1::Weights>& weightMap,
                                                       nvinfer1::ITensor& input, int channels,
                                                       const std::string& layerNormName, int order,
                                                       const std::string& lname);
nvinfer1::IElementWiseLayer* addLayerNormByPrefix(nvinfer1::INetworkDefinition* network,
                                                  std::map<std::string, nvinfer1::Weights>& weightMap,
                                                  nvinfer1::ITensor& input, int channels,
                                                  const std::string& layerNormName, const std::string& lname);
nvinfer1::IElementWiseLayer* addScalarMul(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          float value, const std::string& lname);
nvinfer1::ITensor* addSiluTensor(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                 const std::string& lname);
nvinfer1::ITensor* addGeluTensor(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                 const std::string& lname);
nvinfer1::ITensor* addConvBnTensor(nvinfer1::INetworkDefinition* network,
                                   std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                   int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                   nvinfer1::DimsHW padding, int groups, const std::string& convName,
                                   const std::string& bnName);
nvinfer1::ITensor* addConvBnReluTensor(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                       nvinfer1::DimsHW padding, int groups, const std::string& convName,
                                       const std::string& bnName);
nvinfer1::ITensor* addConvNoBiasTensor(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                       nvinfer1::DimsHW padding, int groups, const std::string& convName);
nvinfer1::ITensor* addConvNoBiasTensorByPrefix(nvinfer1::INetworkDefinition* network,
                                               std::map<std::string, nvinfer1::Weights>& weightMap,
                                               nvinfer1::ITensor& input, int outChannels, nvinfer1::DimsHW ksize,
                                               nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups,
                                               const std::string& convName);
nvinfer1::ITensor* addConvBiasTensor(nvinfer1::INetworkDefinition* network,
                                     std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                     int outChannels, nvinfer1::DimsHW ksize, nvinfer1::DimsHW stride,
                                     nvinfer1::DimsHW padding, const std::string& convName);
nvinfer1::ITensor* addConvBiasTensorByPrefix(nvinfer1::INetworkDefinition* network,
                                             std::map<std::string, nvinfer1::Weights>& weightMap,
                                             nvinfer1::ITensor& input, int outChannels, nvinfer1::DimsHW ksize,
                                             nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups,
                                             const std::string& convName);
nvinfer1::ITensor* addSameConvBnReluTensor(nvinfer1::INetworkDefinition* network,
                                           std::map<std::string, nvinfer1::Weights>& weightMap,
                                           nvinfer1::ITensor& input, int outChannels, int kernel,
                                           const std::string& convName, const std::string& bnName);
nvinfer1::ITensor* addConvBnTensorByPrefix(nvinfer1::INetworkDefinition* network,
                                           std::map<std::string, nvinfer1::Weights>& weightMap,
                                           nvinfer1::ITensor& input, int outChannels, nvinfer1::DimsHW ksize,
                                           nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups,
                                           const std::string& convName, const std::string& bnName);
nvinfer1::ITensor* addConvBnReluTensorByPrefix(nvinfer1::INetworkDefinition* network,
                                               std::map<std::string, nvinfer1::Weights>& weightMap,
                                               nvinfer1::ITensor& input, int outChannels, nvinfer1::DimsHW ksize,
                                               nvinfer1::DimsHW stride, nvinfer1::DimsHW padding, int groups,
                                               const std::string& convName, const std::string& bnName);
nvinfer1::ITensor* addSameConvBnReluTensorByPrefix(nvinfer1::INetworkDefinition* network,
                                                   std::map<std::string, nvinfer1::Weights>& weightMap,
                                                   nvinfer1::ITensor& input, int outChannels, int kernel,
                                                   const std::string& convName, const std::string& bnName);
nvinfer1::ITensor* addBilinearResizeTensor(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                           int channels, int height, int width, const std::string& lname);
nvinfer1::ITensor* addConvBiasBnTensor(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int outChannels, int kernel, int stride, int padding, int dilation,
                                       const std::string& convName, const std::string& bnName);
nvinfer1::ITensor* addConvBiasBnReluTensor(nvinfer1::INetworkDefinition* network,
                                           std::map<std::string, nvinfer1::Weights>& weightMap,
                                           nvinfer1::ITensor& input, int outChannels, int kernel, int stride,
                                           int padding, int dilation, const std::string& convName,
                                           const std::string& bnName);
nvinfer1::ITensor* addConvBnReluTensor(nvinfer1::INetworkDefinition* network,
                                       std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                       int outChannels, int kernel, int stride, int padding, int dilation,
                                       const std::string& convName, const std::string& bnName);
nvinfer1::IElementWiseLayer* addLayerNorm(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int channels, const std::string& layerNormName);
nvinfer1::IElementWiseLayer* addLayerNorm(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap, nvinfer1::ITensor& input,
                                          int channels, const std::string& layerNormName, float eps);
nvinfer1::ISoftMaxLayer* addSoftmax(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input, int axes,
                                    const std::string& lname);
nvinfer1::IShuffleLayer* addShuffle(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                    nvinfer1::Dims reshapeDims, const std::string& lname);
nvinfer1::IShuffleLayer* addPermute(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor& input,
                                    nvinfer1::Permutation permutation, const std::string& lname);
