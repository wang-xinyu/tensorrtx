// PixelShufflePlugin.cpp
//
// #include "pixel_shuffle/pixel_shuffle.hpp"
// #include <cstring>
// #include <cassert>
//
// PixelShufflePlugin::PixelShufflePlugin(int upscaleFactor)
//         : mUpscaleFactor(upscaleFactor) {
//     // Initialize other members
// }
//
// PixelShufflePlugin::PixelShufflePlugin(const void* data, size_t length) {
//     // Deserialize data to initialize members
//     const char* d = static_cast<const char*>(data);
//     mUpscaleFactor = *reinterpret_cast<const int*>(d);
//     d += sizeof(int);
//     mInputVolume = *reinterpret_cast<const size_t*>(d);
//     d += sizeof(size_t);
//     mOutputVolume = *reinterpret_cast<const size_t*>(d);
// }
//
// int PixelShufflePlugin::getNbOutputs() const {
//     return 1;
// }
//
// nvinfer1::Dims PixelShufflePlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) {
//     assert(index == 0);
//     assert(nbInputDims == 1);
//     int c = inputs[0].d[0];
//     int h = inputs[0].d[1];
//     int w = inputs[0].d[2];
//     int upscaleFactor = mUpscaleFactor;
//
//     assert(c % (upscaleFactor * upscaleFactor) == 0);
//     int newC = c / (upscaleFactor * upscaleFactor);
//     int newH = h * upscaleFactor;
//     int newW = w * upscaleFactor;
//
//     return nvinfer1::Dims3(newC, newH, newW);
// }
//
// int PixelShufflePlugin::initialize() {
//     return 0;
// }
//
// void PixelShufflePlugin::terminate() {
//     // Clean up
// }
//
// size_t PixelShufflePlugin::getWorkspaceSize(int maxBatchSize) const {
//     return 0;
// }
//
// int PixelShufflePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
//     // Launch CUDA kernel for PixelShuffle
//     // Assume inputs[0] and outputs[0] are pointers to device memory
//     const float* input = static_cast<const float*>(inputs[0]);
//     float* output = static_cast<float*>(outputs[0]);
//
//     int c = mInputVolume / (mUpscaleFactor * mUpscaleFactor);
//     int h = mOutputVolume / (c * mUpscaleFactor);
//     int w = h; // Assuming square input for simplicity
//     int upscaleFactor = mUpscaleFactor;
//
//     // Launch CUDA kernel (to be implemented)
//     // pixelShuffleKernel(input, output, c, h, w, upscaleFactor, stream);
//
//     return 0;
// }
//
// size_t PixelShufflePlugin::getSerializationSize() const {
//     return sizeof(int) + sizeof(size_t) * 2;
// }
//
// void PixelShufflePlugin::serialize(void* buffer) const {
//     char* d = static_cast<char*>(buffer);
//     *reinterpret_cast<int*>(d) = mUpscaleFactor;
//     d += sizeof(int);
//     *reinterpret_cast<size_t*>(d) = mInputVolume;
//     d += sizeof(size_t);
//     *reinterpret_cast<size_t*>(d) = mOutputVolume;
// }
//
// void PixelShufflePlugin::destroy() {
//     delete this;
// }
//
// const char* PixelShufflePlugin::getPluginType() const {
//     return "PixelShufflePlugin";
// }
//
// const char* PixelShufflePlugin::getPluginVersion() const {
//     return "1";
// }
//
// void PixelShufflePlugin::setPluginNamespace(const char* pluginNamespace) {
//     mPluginNamespace = pluginNamespace;
// }
//
// const char* PixelShufflePlugin::getPluginNamespace() const {
//     return mPluginNamespace;
// }
//
// nvinfer1::IPluginV2IOExt* PixelShufflePlugin::clone() const {
//     return new PixelShufflePlugin(mUpscaleFactor);
// }
//
// bool PixelShufflePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const {
//     return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
// }
//
// void PixelShufflePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
//     // Configure the plugin based on the input and output descriptions
//     mInputVolume = in[0].desc.volume();
//     mOutputVolume = out[0].desc.volume();
// }
//
// nvinfer1::DataType PixelShufflePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
//     return inputTypes[0];
// }
//
// bool PixelShufflePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {
//     return false;
// }
//
// bool PixelShufflePlugin::canBroadcastInputAcrossBatch(int inputIndex) const {
//     return false;
// }
