#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

using namespace nvinfer1;

#define PLUGIN_NAME "RpnDecode"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

int rpnDecode(int batchSize, const void *const *inputs,
void **outputs, size_t height, size_t width, size_t image_height,
size_t image_width, float stride, const std::vector<float> &anchors,
int top_n, void *workspace, size_t workspace_size, cudaStream_t stream);

/*
    input1: scores{C,H,W} C->anchors
    input2: boxes{C,H,W} C->4*anchors
    output1: scores{C, 1} C->topk
    output2: boxes{C, 4} C->topk format:XYXY
    Description: implement anchor decode
*/
class RpnDecodePlugin : public IPluginV2Ext {
    int _top_n;
    std::vector<float> _anchors;
    float _stride;

    size_t _height;
    size_t _width;
    size_t _image_height;  // for cliping the boxes by limiting y coordinates to the range [0, height]
    size_t _image_width;  // for cliping the boxes by limiting x coordinates to the range [0, width]
    mutable int size = -1;

 protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _top_n);
        size_t anchors_size;
        read(d, anchors_size);
        while (anchors_size--) {
            float val;
            read(d, val);
            _anchors.push_back(val);
        }
        read(d, _stride);
        read(d, _height);
        read(d, _width);
        read(d, _image_height);
        read(d, _image_width);
    }

    size_t getSerializationSize() const override {
        return sizeof(_top_n)
            + sizeof(size_t) + sizeof(float) * _anchors.size() + sizeof(_stride)
            + sizeof(_height) + sizeof(_width) + sizeof(_image_height) + sizeof(_image_width);
    }

    void serialize(void *buffer) const override {
        char* d = static_cast<char*>(buffer);
        write(d, _top_n);
        write(d, _anchors.size());
        for (auto &val : _anchors) {
            write(d, val);
        }
        write(d, _stride);
        write(d, _height);
        write(d, _width);
        write(d, _image_height);
        write(d, _image_width);
    }

 public:
    RpnDecodePlugin(int top_n, std::vector<float> const& anchors, float stride, size_t image_height, size_t image_width)
        :  _top_n(top_n), _anchors(anchors), _stride(stride), _image_height(image_height), _image_width(image_width) {}

    RpnDecodePlugin(int top_n, std::vector<float> const& anchors, float stride,
        size_t height, size_t width, size_t image_height, size_t image_width)
        : _top_n(top_n), _anchors(anchors), _stride(stride),
        _height(height), _width(width), _image_height(image_height), _image_width(image_width) {}

    RpnDecodePlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }

    const char *getPluginType() const override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override {
        return PLUGIN_VERSION;
    }

    int getNbOutputs() const override {
        return 2;
    }

    Dims getOutputDimensions(int index,
        const Dims *inputs, int nbInputDims) override {
        assert(nbInputDims == 2);
        assert(index < this->getNbOutputs());
        return Dims2(_top_n, (index == 1 ? 4 : 1));
    }

    bool supportsFormat(DataType type, PluginFormat format) const override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override {
        if (size < 0) {
            size = rpnDecode(maxBatchSize, nullptr, nullptr, _height, _width, _image_height, _image_width, _stride,
                _anchors, _top_n,
                nullptr, 0, nullptr);
        }
        return size;
    }

    int enqueue(int batchSize,
        const void *const *inputs, void **outputs,
        void *workspace, cudaStream_t stream) override {
        return rpnDecode(batchSize, inputs, outputs, _height, _width, _image_height, _image_width, _stride,
            _anchors, _top_n, workspace, getWorkspaceSize(batchSize), stream);
    }

    void destroy() override {
        delete this;
    };

    const char *getPluginNamespace() const override {
        return PLUGIN_NAMESPACE;
    }

    void setPluginNamespace(const char *N) override {
    }

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const {
        assert(index < 3);
        return DataType::kFLOAT;
    }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
        int nbInputs) const {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const { return false; }

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
        assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
            floatFormat == nvinfer1::PluginFormat::kLINEAR);
        assert(nbInputs == 2);
        assert(nbOutputs == 2);
        auto const& scores_dims = inputDims[0];
        auto const& boxes_dims = inputDims[1];
        assert(scores_dims.d[1] == boxes_dims.d[1]);
        assert(scores_dims.d[2] == boxes_dims.d[2]);
        _height = scores_dims.d[1];
        _width = scores_dims.d[2];
    }

    IPluginV2Ext *clone() const override {
        return new RpnDecodePlugin(_top_n, _anchors, _stride, _height, _width, _image_height, _image_width);
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

class RpnDecodePluginCreator : public IPluginCreator {
 public:
    RpnDecodePluginCreator() {}

    const char *getPluginName() const override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override {
        return PLUGIN_VERSION;
    }

    const char *getPluginNamespace() const override {
        return PLUGIN_NAMESPACE;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override {
        return new RpnDecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) override {}
    const PluginFieldCollection *getFieldNames() override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(RpnDecodePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
