#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>
#include "macros.h"

using namespace nvinfer1;

#define PLUGIN_NAME "PredictorDecode"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {

int predictorDecode(int batchSize,
const void *const *inputs, void *TRT_CONST_ENQUEUE*outputs, unsigned int num_boxes,
unsigned int num_classes, unsigned int image_height,
unsigned int image_width, const std::vector<float>& bbox_reg_weights,
void *workspace, size_t workspace_size, cudaStream_t stream);

/*
    input1: scores{N,C,1,1} N->nums C->num of classes
    input2: boxes{N,C*4,1,1} N->nums C->num of classes
    input3: proposals{N,4} N->nums
    output1: scores{N, 1} N->nums
    output2: boxes{N, 4} N->nums format:XYXY
    output3: classes{N, 1} N->nums
    Description: implement fast rcnn decode
*/
class PredictorDecodePlugin : public IPluginV2Ext {
    unsigned int _num_boxes;
    unsigned int _num_classes;
    unsigned int _image_height;
    unsigned int _image_width;
    std::vector<float> _bbox_reg_weights;
    mutable int size = -1;

 protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _num_boxes);
        read(d, _num_classes);
        read(d, _image_height);
        read(d, _image_width);
        size_t bbox_reg_weights_size;
        read(d, bbox_reg_weights_size);
        while (bbox_reg_weights_size--) {
            float val;
            read(d, val);
            _bbox_reg_weights.push_back(val);
        }
    }

    size_t getSerializationSize() const TRT_NOEXCEPT override {
        return sizeof(_num_boxes) + sizeof(_num_classes) +
        sizeof(_image_height) + sizeof(_image_width) + sizeof(size_t) +
        sizeof(float)*_bbox_reg_weights.size();
    }

    void serialize(void *buffer) const TRT_NOEXCEPT override {
        char* d = static_cast<char*>(buffer);
        write(d, _num_boxes);
        write(d, _num_classes);
        write(d, _image_height);
        write(d, _image_width);
        write(d, _bbox_reg_weights.size());
        for (auto &val : _bbox_reg_weights) {
            write(d, val);
        }
    }

 public:
    PredictorDecodePlugin(unsigned int num_boxes, unsigned int image_height,
    unsigned int image_width, std::vector<float> const& bbox_reg_weights)
        : _num_boxes(num_boxes), _image_height(image_height),
        _image_width(image_width), _bbox_reg_weights(bbox_reg_weights) {}

    PredictorDecodePlugin(unsigned int num_boxes, unsigned int num_classes,
    unsigned int image_height, unsigned int image_width,
    std::vector<float> const& bbox_reg_weights)
        : _num_boxes(num_boxes), _num_classes(num_classes),
        _image_height(image_height), _image_width(image_width),
        _bbox_reg_weights(bbox_reg_weights) {}

    PredictorDecodePlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }

    const char *getPluginType() const TRT_NOEXCEPT override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const TRT_NOEXCEPT override {
        return PLUGIN_VERSION;
    }

    int getNbOutputs() const TRT_NOEXCEPT override {
        return 3;
    }

    Dims getOutputDimensions(int index,
        const Dims *inputs, int nbInputDims) TRT_NOEXCEPT override {
        assert(nbInputDims == 3);
        assert(index < this->getNbOutputs());
        return Dims2(_num_boxes, (index == 1 ? 4 : 1));
    }

    bool supportsFormat(DataType type, PluginFormat format) const TRT_NOEXCEPT override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }

    int initialize() TRT_NOEXCEPT override { return 0; }

    void terminate() TRT_NOEXCEPT override {}

    size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override {
        if (size < 0) {
            size = predictorDecode(maxBatchSize, nullptr, nullptr,
            _num_boxes, _num_classes, _image_height, _image_width,
            _bbox_reg_weights, nullptr, 0, nullptr);
        }
        return size;
    }

    int enqueue(int batchSize,
        const void *const *inputs, void *TRT_CONST_ENQUEUE*outputs,
        void *workspace, cudaStream_t stream) TRT_NOEXCEPT override {
        return predictorDecode(batchSize, inputs, outputs, _num_boxes,
        _num_classes, _image_height, _image_width, _bbox_reg_weights,
        workspace, getWorkspaceSize(batchSize), stream);
    }

    void destroy() TRT_NOEXCEPT override {
        delete this;
    };

    const char *getPluginNamespace() const TRT_NOEXCEPT override {
        return PLUGIN_NAMESPACE;
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override {
        assert(index < this->getNbOutputs());
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
        assert(nbInputs == 3);
        assert(nbOutputs == 3);
        auto const& scores_dims = inputDims[0];
        auto const& boxes_dims = inputDims[1];
        auto const& proposals_dims = inputDims[2];
        assert(scores_dims.d[0] == _num_boxes);
        assert(scores_dims.d[0] == boxes_dims.d[0]);
        assert(scores_dims.d[0] == proposals_dims.d[0]);
        assert(scores_dims.d[1] * 4 == boxes_dims.d[1]);
        assert(proposals_dims.d[1] == 4);
        _num_classes = scores_dims.d[1];
    }

    IPluginV2Ext *clone() const TRT_NOEXCEPT override {
        return new PredictorDecodePlugin(_num_boxes, _num_classes, _image_height, _image_width, _bbox_reg_weights);
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

class PredictorDecodePluginCreator : public IPluginCreator {
 public:
    PredictorDecodePluginCreator() {}

    const char *getPluginName() const TRT_NOEXCEPT override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const TRT_NOEXCEPT override {
        return PLUGIN_VERSION;
    }

    const char *getPluginNamespace() const TRT_NOEXCEPT override {
        return PLUGIN_NAMESPACE;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT override {
        return new PredictorDecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) TRT_NOEXCEPT override {}
    const PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(PredictorDecodePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
