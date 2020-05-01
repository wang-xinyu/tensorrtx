#ifndef MY_PLUGIN_FACTORY_H
#define MY_PLUGIN_FACTORY_H
#include <NvInfer.h>

namespace nvinfer1 {
class PluginFactory : public IPluginFactory {
    public:
        IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
};

}
#endif
