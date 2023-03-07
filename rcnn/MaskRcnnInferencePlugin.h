#pragma once

#include <NvInfer.h>

#include <vector>
#include <cassert>
#include "macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "MaskRcnnInference"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {
int maskRcnnInference(int batchSize,
    const void *const *inputs, void *TRT_CONST_ENQUEUE*outputs,
    int detections_per_im, int output_size, int num_classes, cudaStream_t stream);
/*
    input1: indices{C, 1} C->topk
    input2: masks{C, NUM_CLASS, size, size} C->topk format:XYXY
    output1: masks{C, 1, size, size} C->detections_per_img
    Description: implement index select
*/

class MaskRcnnInferencePlugin : public IPluginV2Ext {
    int _detections_per_im;
    int _output_size;
    int _num_classes = 1;

 protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _detections_per_im);
        read(d, _output_size);
        read(d, _num_classes);
    }
    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_detections_per_im) + sizeof(_output_size) + sizeof(_num_classes);
    }
    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char* d = static_cast<char*>(buffer);
        write(d, _detections_per_im);
        write(d, _output_size);
        write(d, _num_classes);
    }

 public:
    MaskRcnnInferencePlugin(int detections_per_im, int output_size)
        : _detections_per_im(detections_per_im), _output_size(output_size) {
        assert(detections_per_im > 0);
        assert(output_size > 0);
    }
    MaskRcnnInferencePlugin(int detections_per_im, int output_size, int num_classes)
        : _detections_per_im(detections_per_im), _output_size(output_size), _num_classes(num_classes) {
        assert(detections_per_im > 0);
        assert(output_size > 0);
        assert(num_classes > 0);
    }
    MaskRcnnInferencePlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }
    const char *getPluginType() const TRT_NOEXCEPT override {
        return PLUGIN_NAME;
    }
    const char *getPluginVersion() const TRT_NOEXCEPT override {
        return PLUGIN_VERSION;
    }
    int getNbOutputs() const TRT_NOEXCEPT override {
        return 1;
    }
    Dims getOutputDimensions(int index,
        const Dims *inputs, int nbInputDims) TRT_NOEXCEPT override {
        assert(index < this->getNbOutputs());
        return Dims4(_detections_per_im, 1, _output_size, _output_size);
    }
    bool supportsFormat(DataType type, PluginFormat format) const TRT_NOEXCEPT override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }
    int initialize() TRT_NOEXCEPT override { return 0; }
    void terminate() TRT_NOEXCEPT override {}
    size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override {
        return 0;
    }
    int enqueue(int batchSize,
        const void *const *inputs, void *TRT_CONST_ENQUEUE*outputs,
        void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        return maskRcnnInference(batchSize, inputs, outputs,
            _detections_per_im, _output_size, _num_classes, stream);
    }
    void destroy() TRT_NOEXCEPT override {
        delete this;
    }
    const char *getPluginNamespace() const TRT_NOEXCEPT override {
        return PLUGIN_NAMESPACE;
    }
    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {
    }
    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < 1);
        return DataType::kFLOAT;
    }
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
        int nbInputs) const TRT_NOEXCEPT override {
        return false;
    }
    bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override { return false; }
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) TRT_NOEXCEPT override {
        assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
            floatFormat == nvinfer1::PluginFormat::kLINEAR);
        assert(nbInputs == 2);
        assert(inputDims[0].d[0] == _detections_per_im);
        assert(inputDims[1].d[0] == _detections_per_im);
        assert(inputDims[1].d[2] == _output_size);
        assert(inputDims[1].d[3] == _output_size);
        _num_classes = inputDims[1].d[1];
    }
    IPluginV2Ext *clone() const TRT_NOEXCEPT override {
        return new MaskRcnnInferencePlugin(_detections_per_im, _output_size, _num_classes);
    }

 private:
    template<typename T> void write(char*& buffer, const T& val) const {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }
    template<typename T> void read(const char*& buffer, T& val) {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
};

class MaskRcnnInferencePluginCreator : public IPluginCreator {
 public:
    MaskRcnnInferencePluginCreator() {}
    const char *getPluginNamespace() const TRT_NOEXCEPT override {
        return PLUGIN_NAMESPACE;
    }
    const char *getPluginName() const TRT_NOEXCEPT override {
        return PLUGIN_NAME;
    }
    const char *getPluginVersion() const TRT_NOEXCEPT override {
        return PLUGIN_VERSION;
    }
    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new MaskRcnnInferencePlugin(serialData, serialLength);
    }
    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(MaskRcnnInferencePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
