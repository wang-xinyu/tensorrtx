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
void *const *outputs, size_t height, size_t width, size_t image_height,
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
    // 反序列化
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
    // 得到序列化大小
    size_t getSerializationSize() const noexcept override {
        return sizeof(_top_n)
            + sizeof(size_t) + sizeof(float) * _anchors.size() + sizeof(_stride)
            + sizeof(_height) + sizeof(_width) + sizeof(_image_height) + sizeof(_image_width);
    }
    // 序列化
    void serialize(void *buffer) const noexcept override {
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
    // 构造函数 1 用于parse阶段
    RpnDecodePlugin(int top_n, std::vector<float> const& anchors, float stride, size_t image_height, size_t image_width)
        :  _top_n(top_n), _anchors(anchors), _stride(stride), _image_height(image_height), _image_width(image_width) {}

    // 构造函数 2 用于clone阶段
    RpnDecodePlugin(int top_n, std::vector<float> const& anchors, float stride,
        size_t height, size_t width, size_t image_height, size_t image_width)
        : _top_n(top_n), _anchors(anchors), _stride(stride),
        _height(height), _width(width), _image_height(image_height), _image_width(image_width) {}

    // 构造函数 1 用于deserialize阶段
    RpnDecodePlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }
    // 返回plugin类型
    const char *getPluginType() const noexcept override {
        return PLUGIN_NAME;
    }
    // 返回plugin版本
    const char *getPluginVersion() const noexcept override {
        return PLUGIN_VERSION;
    }
    // 返回输出tensor数量
    int getNbOutputs() const noexcept override {
        return 2;
    }
    // 得到输出维度
    Dims getOutputDimensions(int index,
        const Dims *inputs, int nbInputDims) noexcept override {
        assert(nbInputDims == 2);
        assert(index < this->getNbOutputs());
        return Dims2(_top_n, (index == 1 ? 4 : 1));
    }

    // 判断支持的数据格式。
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override {
        return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
    }

    // 初始化 返回0为成功，否则失败
    int initialize() noexcept override { return 0; }

    // 销毁操作  这个方法通常用于在程序出现异常或崩溃时，强制终止 CUDA 操作，以避免资源泄漏或数据损坏，正常不要使用 而是用delete or destroy()
    void terminate() noexcept override {}

    // 申请最大显存空间
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        if (size < 0) {
            size = rpnDecode(maxBatchSize, nullptr, nullptr, _height, _width, _image_height, _image_width, _stride,
                _anchors, _top_n,
                nullptr, 0, nullptr);
        }
        return size;
    }

    // 算法的具体实现
    int enqueue(int batchSize,
        const void *const *inputs, void *const *outputs,
        void *workspace, cudaStream_t stream) noexcept override {
        return rpnDecode(batchSize, inputs, outputs, _height, _width, _image_height, _image_width, _stride,
            _anchors, _top_n, workspace, getWorkspaceSize(batchSize), stream);
    }

    // 注销
    void destroy() noexcept override {
        delete this;
    };

    // 得到plugin的命名空间
    const char *getPluginNamespace() const noexcept override {
        return PLUGIN_NAMESPACE;
    }

    // 设置plugin的命名空间
    void setPluginNamespace(const char *N) noexcept override {
    }

    // 得到输出数据类型
    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        assert(index < 3);
        return DataType::kFLOAT;
    }

    // 如果输出张量在批处理中被广播，则返回true。
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
        int nbInputs) const noexcept override {
        return false;
    }

    // 如果插件可以使用跨批广播而不复制的输入，则返回true。
    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }

    // 配置这个插件op，判断输入和输出类型数量是否正确。
    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override {
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

    // 拷贝构造
    IPluginV2Ext *clone() const noexcept override {
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

    const char *getPluginName() const noexcept override {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override {
        return PLUGIN_VERSION;
    }

    const char *getPluginNamespace() const noexcept override {
        return PLUGIN_NAMESPACE;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
        return new RpnDecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) noexcept override {}
    const PluginFieldCollection *getFieldNames() noexcept override { return nullptr; }
    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(RpnDecodePluginCreator);

}  // namespace nvinfer1

#undef PLUGIN_NAME
#undef PLUGIN_VERSION
#undef PLUGIN_NAMESPACE
