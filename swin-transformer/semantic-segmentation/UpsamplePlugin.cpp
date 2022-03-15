#include <iostream>
#include "UpsmapleKernel.h"
#include "UpsamplePlugin.h"

#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Upsample plugin specific constants
namespace {
    static const char* UPSAMPLE_PLUGIN_VERSION{"1"};
    static const char* UPSAMPLE_PLUGIN_NAME{"UpsamplePlugin"};
}

// Static class fields initialization
PluginFieldCollection UpsamplePluginCreator::mFC{};
std::vector<PluginField> UpsamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(UpsamplePluginCreator);

template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

UpsamplePlugin::UpsamplePlugin(const std::string name, float scale_h, float scale_w)
    : mLayerName(name)
    , mScaleFactor_h(scale_h)
    , mScaleFactor_w(scale_w)
{
    mInputShape.c() = -1;
    mInputShape.h() = -1;
    mInputShape.w() = -1;
    mInputVolume = 0;
}

UpsamplePlugin::UpsamplePlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    mScaleFactor_h = readFromBuffer<float>(d);
    mScaleFactor_w = readFromBuffer<float>(d);
    mInputVolume = readFromBuffer<size_t>(d);
    mInputShape.c() = readFromBuffer<int>(d);
    mInputShape.h() = readFromBuffer<int>(d);
    mInputShape.w() = readFromBuffer<int>(d);

    assert(d == (a + length));

}

const char* UpsamplePlugin::getPluginType() const
{
    return UPSAMPLE_PLUGIN_NAME;
}

const char* UpsamplePlugin::getPluginVersion() const
{
    return UPSAMPLE_PLUGIN_VERSION;
}

int UpsamplePlugin::getNbOutputs() const
{
    return 1;
}

Dims UpsamplePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0);
    assert(nbInputDims == 1);
    assert(inputs[0].nbDims == 3);
    return nvinfer1::DimsCHW{inputs[0].d[0],int(inputs[0].d[1]*mScaleFactor_h), int(inputs[0].d[2]*mScaleFactor_w)};
}

int UpsamplePlugin::initialize()
{
    //printf("UpsamplePlugin::initialize\n");
    return 0;
}


int UpsamplePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    //printf("UpsamplePlugin::enqueue\n");
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = UpsampleInference(stream, mInputVolume, 
                                batchSize, mInputShape.c(), mInputShape.h(), mInputShape.w(),
                                mScaleFactor_h,mScaleFactor_w,
                                inputs[0], output);
    return status;
}

size_t UpsamplePlugin::getSerializationSize() const
{
    //printf("UpsamplePlugin::getSerializationSize\n");
    return sizeof(mScaleFactor_h)  + sizeof(mScaleFactor_w) +
            sizeof(mInputVolume) + sizeof(mInputShape.c()) + 
            sizeof(mInputShape.h()) + sizeof(mInputShape.w());
}


void UpsamplePlugin::serialize(void* buffer) const 
{
    //printf("UpsamplePlugin::serialize\n");
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, mScaleFactor_h);
    writeToBuffer(d, mScaleFactor_w);
    writeToBuffer(d, mInputVolume);
    writeToBuffer(d, mInputShape.c());
    writeToBuffer(d, mInputShape.h());
    writeToBuffer(d, mInputShape.w());

    assert(d == a + getSerializationSize());
}

void UpsamplePlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kNCHW);
    assert(inputs[0].nbDims == 3);

    size_t volume = int(inputs[0].d[1]*mScaleFactor_h) * int(inputs[0].d[2]*mScaleFactor_w);
    mInputVolume = volume;
    mInputShape.c() = inputs[0].d[0];
    mInputShape.h() = inputs[0].d[1];
    mInputShape.w() = inputs[0].d[2];
}

bool UpsamplePlugin::supportsFormat(DataType type, PluginFormat format) const
{
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
        return false;
}

void UpsamplePlugin::terminate() {}

void UpsamplePlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* UpsamplePlugin::clone() const
{
    return new UpsamplePlugin(mLayerName, mScaleFactor_h, mScaleFactor_w);
}

void UpsamplePlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* UpsamplePlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

UpsamplePluginCreator::UpsamplePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("scaleFactor", nullptr, PluginFieldType::kFLOAT32, 2));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
const char* UpsamplePluginCreator::getPluginName() const
{
    return UPSAMPLE_PLUGIN_NAME;
}

const char* UpsamplePluginCreator::getPluginVersion() const
{
    return UPSAMPLE_PLUGIN_VERSION;
}

const PluginFieldCollection* UpsamplePluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* UpsamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float scaleFactor_h = 0.f;
    float scaleFactor_w = 0.f;
    const PluginField* fields = fc->fields;

    assert(fc->nbFields == 1);
    for (int i = 0; i < fc->nbFields; i++){
    
        if (strcmp(fields[i].name, "scaleFactor") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            scaleFactor_h = *(static_cast<const float*>(fields[i].data));
            scaleFactor_w = *(static_cast<const float*>(fields[i].data)+1);
            //std::cout<<scaleFactor_h<< " , "<<scaleFactor_w<<std::endl;
        } 
    }
    return new UpsamplePlugin(name, scaleFactor_h, scaleFactor_w);
}

IPluginV2* UpsamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new UpsamplePlugin(name, serialData, serialLength);
}

void UpsamplePluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* UpsamplePluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}
