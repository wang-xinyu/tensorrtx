#pragma once

#include <NvInfer.h>

#include <vector>
#include <cassert>

using namespace nvinfer1;

#define PLUGIN_NAME "BatchedNms"
#define PLUGIN_VERSION "1"
#define PLUGIN_NAMESPACE ""

namespace nvinfer1 {
int batchedNms(int batchSize,
    const void *const *inputs, void **outputs,
    size_t count, int detections_per_im, float nms_thresh,
    void *workspace, size_t workspace_size, cudaStream_t stream);

/*
    input1: scores{C, 1} C->topk
    input2: boxes{C, 4} C->topk format:XYXY
    input3: classes{C, 1} C->topk
    output1: scores{C, 1} C->detections_per_img
    output2: boxes{C, 4} C->detections_per_img format:XYXY
    output3: classes{C, 1} C->detections_per_img
    Description: implement batched nms
*/
class BatchedNmsPlugin : public IPluginV2Ext {
    float _nms_thresh;
    int _detections_per_im;

    size_t _count;

 protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _nms_thresh);
        read(d, _detections_per_im);
        read(d, _count);
    }

    size_t getSerializationSize() const override {
        return sizeof(_nms_thresh) + sizeof(_detections_per_im)
            + sizeof(_count);
    }

    void serialize(void *buffer) const override {
        char* d = static_cast<char*>(buffer);
        write(d, _nms_thresh);
        write(d, _detections_per_im);
        write(d, _count);
    }

 public:
    BatchedNmsPlugin(float nms_thresh, int detections_per_im)
        : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im) {
        assert(nms_thresh > 0);
        assert(detections_per_im > 0);
    }

    BatchedNmsPlugin(float nms_thresh, int detections_per_im, size_t count)
        : _nms_thresh(nms_thresh), _detections_per_im(detections_per_im), _count(count) {
        assert(nms_thresh > 0);
        assert(detections_per_im > 0);
        assert(count > 0);
    }

    BatchedNmsPlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }

    const char *getPluginType() const override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override {
        return PLUGIN_VERSION;
    }

    int getNbOutputs() const override {
        return 3;
    }

    Dims getOutputDimensions(int index,
        const Dims *inputs, int nbInputDims) override {
        assert(nbInputDims == 3);
        assert(index < this->getNbOutputs());
        return Dims2(_detections_per_im, index == 1 ? 4 : 1);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override {
        static int size = -1;
        if (size < 0) {
            size = batchedNms(maxBatchSize, nullptr, nullptr, _count,
                _detections_per_im, _nms_thresh,
                nullptr, 0, nullptr);
        }
        return size;
    }

    int enqueue(int batchSize,
        const void *const *inputs, void **outputs,
        void *workspace, cudaStream_t stream) override {
        return batchedNms(batchSize, inputs, outputs, _count,
            _detections_per_im, _nms_thresh,
            workspace, getWorkspaceSize(batchSize), stream);
    }

    void destroy() override {
        delete this;
    }

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
        assert(nbInputs == 3);
        assert(inputDims[0].d[0] == inputDims[2].d[0]);
        assert(inputDims[1].d[0] == inputDims[2].d[0]);
        _count = inputDims[0].d[0];
    }

    IPluginV2Ext *clone() const override {
        return new BatchedNmsPlugin(_nms_thresh, _detections_per_im, _count);
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

class BatchedNmsPluginCreator : public IPluginCreator {
 public:
    BatchedNmsPluginCreator() {}

    const char *getPluginNamespace() const override {
        return PLUGIN_NAMESPACE;
    }
    const char *getPluginName() const override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const override {
        return PLUGIN_VERSION;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override {
        return new BatchedNmsPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) override {}
    const PluginFieldCollection *getFieldNames() override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(BatchedNmsPluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
