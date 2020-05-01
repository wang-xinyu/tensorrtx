#include "common.h"
#include "plugin_factory.h"
#include "NvInferPlugin.h"
#include "yololayer.h"
#include "mish.h"

using namespace nvinfer1;
using nvinfer1::PluginFactory;

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    IPlugin *plugin = nullptr;
    if (strstr(layerName, "leaky") != NULL) {
        plugin = plugin::createPReLUPlugin(serialData, serialLength);
    } else if (strstr(layerName, "yolo") != NULL) {
        plugin = new YoloLayerPlugin(serialData, serialLength);
    } else if (strstr(layerName, "mish") != NULL) {
        plugin = new MishPlugin(serialData, serialLength);
    }
    return plugin;
}
