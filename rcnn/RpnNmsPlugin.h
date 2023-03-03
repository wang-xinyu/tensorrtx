#pragma once

#include <NvInfer.h>

#include <vector>
#include <cassert>
#include "macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "RpnNms"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

int rpnNms(int batchSize,
    const void *const *inputs, void *TRT_CONST_ENQUEUE*outputs,
    size_t pre_nms_topk, int post_nms_topk, float nms_thresh,
    void *workspace, size_t workspace_size, cudaStream_t stream);

/*
    input1: scores{C, 1} C->pre_nms_topk
    input2: boxes{C, 4} C->pre_nms_topk format:XYXY
    output1: boxes{C, 4} C->post_nms_topk format:XYXY
    Description: implement rpn nms
*/
class RpnNmsPlugin : public IPluginV2Ext {
    float _nms_thresh;
    int _post_nms_topk;

    size_t _pre_nms_topk = 1;
    mutable int size = -1;

 protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _nms_thresh);
        read(d, _post_nms_topk);
        read(d, _pre_nms_topk);
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_nms_thresh) + sizeof(_post_nms_topk)
            + sizeof(_pre_nms_topk);
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char* d = static_cast<char*>(buffer);
        write(d, _nms_thresh);
        write(d, _post_nms_topk);
        write(d, _pre_nms_topk);
    }

 public:
    RpnNmsPlugin(float nms_thresh, int post_nms_topk)
        : _nms_thresh(nms_thresh), _post_nms_topk(post_nms_topk) {
        assert(nms_thresh > 0);
        assert(post_nms_topk > 0);
    }

    RpnNmsPlugin(float nms_thresh, int post_nms_topk, size_t pre_nms_topk)
        : _nms_thresh(nms_thresh), _post_nms_topk(post_nms_topk), _pre_nms_topk(pre_nms_topk) {
        assert(nms_thresh > 0);
        assert(post_nms_topk > 0);
        assert(pre_nms_topk > 0);
    }

    RpnNmsPlugin(void const* data, size_t length) {
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
        assert(nbInputDims == 2);
        assert(index < this->getNbOutputs());
        return Dims2(_post_nms_topk, 4);
    }

    bool supportsFormat(DataType type, PluginFormat format) const TRT_NOEXCEPT override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }

    int initialize() TRT_NOEXCEPT override { return 0; }

    void terminate() TRT_NOEXCEPT override {}

    size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override {
        if (size < 0) {
            size = rpnNms(maxBatchSize, nullptr, nullptr, _pre_nms_topk,
                _post_nms_topk, _nms_thresh,
                nullptr, 0, nullptr);
        }
        return size;
    }

    int enqueue(int batchSize,
        const void *const *inputs, void *TRT_CONST_ENQUEUE*outputs,
        void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        return rpnNms(batchSize, inputs, outputs, _pre_nms_topk,
            _post_nms_topk, _nms_thresh,
            workspace, getWorkspaceSize(batchSize), stream);
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
        assert(inputDims[0].d[0] == inputDims[1].d[0]);
        _pre_nms_topk = inputDims[0].d[0];
    }

    IPluginV2Ext *clone() const TRT_NOEXCEPT override {
        return new RpnNmsPlugin(_nms_thresh, _post_nms_topk, _pre_nms_topk);
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

class RpnNmsPluginCreator : public IPluginCreator {
 public:
    RpnNmsPluginCreator() {}

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
        return new RpnNmsPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(RpnNmsPluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
