#pragma once

#include <NvInfer.h>

#include <cassert>
#include <vector>

using namespace nvinfer1;

#define PLUGIN_NAME "RoiAlign"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {
int roiAlign(int batchSize, const void *const *inputs, void **outputs, int pooler_resolution, float spatial_scale,
    int sampling_ratio, int num_proposals, int out_channels, int feature_h, int feature_w, cudaStream_t stream);

    /*
        input1: boxes{N,4} N->post_nms_topk
        input2: features{C,H,W} C->num of feature map channels
        output1: features{N, C, H, W} N:nums of proposals C:output out_channels H,W:roialign size
        Description: roialign
    */
class RoiAlignPlugin : public IPluginV2Ext {
    int _pooler_resolution;
    float _spatial_scale;
    int _sampling_ratio;
    int _num_proposals;
    int _out_channels;
    int _feature_h;
    int _feature_w;

protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _pooler_resolution);
        read(d, _spatial_scale);
        read(d, _sampling_ratio);
        read(d, _num_proposals);
        read(d, _out_channels);
        read(d, _feature_h);
        read(d, _feature_w);
    }

    size_t getSerializationSize() const override {
        return sizeof(_pooler_resolution) + sizeof(_spatial_scale) + sizeof(_sampling_ratio) +
            sizeof(_num_proposals) + sizeof(_out_channels) + sizeof(_feature_h) + sizeof(_feature_w);
    }

    void serialize(void *buffer) const override {
        char* d = static_cast<char*>(buffer);
        write(d, _pooler_resolution);
        write(d, _spatial_scale);
        write(d, _sampling_ratio);
        write(d, _num_proposals);
        write(d, _out_channels);
        write(d, _feature_h);
        write(d, _feature_w);
    }

public:
    RoiAlignPlugin(int pooler_resolution, float spatial_scale, int sampling_ratio, int num_proposals,
        int out_channels)
        : _pooler_resolution(pooler_resolution), _spatial_scale(spatial_scale), _sampling_ratio(sampling_ratio),
        _num_proposals(num_proposals), _out_channels(out_channels) {}

    RoiAlignPlugin(int pooler_resolution, float spatial_scale, int sampling_ratio, int num_proposals,
        int out_channels, int feature_h, int feature_w)
        : _pooler_resolution(pooler_resolution), _spatial_scale(spatial_scale), _sampling_ratio(sampling_ratio),
        _num_proposals(num_proposals), _out_channels(out_channels), _feature_h(feature_h), _feature_w(feature_w) {}

    RoiAlignPlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }

    const char *getPluginType() const override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override {
        return PLUGIN_VERSION;
    }

    int getNbOutputs() const override {
        return 1;
    }

    Dims getOutputDimensions(int index,
        const Dims *inputs, int nbInputDims) override {
        assert(index < this->getNbOutputs());
        return Dims4(_num_proposals, _out_channels, _pooler_resolution, _pooler_resolution);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override {
        return 0;
    }

    int enqueue(int batchSize,
        const void *const *inputs, void **outputs,
        void *workspace, cudaStream_t stream) override {
        return roiAlign(batchSize, inputs, outputs, _pooler_resolution, _spatial_scale, _sampling_ratio,
            _num_proposals, _out_channels, _feature_h, _feature_w, stream);
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
        assert(index < this->getNbOutputs());
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
        assert(nbOutputs == 1);
        auto const& boxes_dims = inputDims[0];
        auto const& feature_dims = inputDims[1];
        assert(_num_proposals == boxes_dims.d[0]);
        assert(_out_channels == feature_dims.d[0]);
        _feature_h = feature_dims.d[1];
        _feature_w = feature_dims.d[2];
    }

    IPluginV2Ext *clone() const override {
        return new RoiAlignPlugin(_pooler_resolution, _spatial_scale, _sampling_ratio, _num_proposals,
            _out_channels, _feature_h, _feature_w);
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

class RoiAlignPluginCreator : public IPluginCreator {
public:
    RoiAlignPluginCreator() {}

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
        return new RoiAlignPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) override {}
    const PluginFieldCollection *getFieldNames() override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(RoiAlignPluginCreator);
}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
