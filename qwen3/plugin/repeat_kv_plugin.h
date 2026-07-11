#pragma once

#include "NvInfer.h"

nvinfer1::IPluginV2* createRepeatKVPlugin(int groups);
