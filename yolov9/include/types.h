#pragma once

#include "config.h"

struct YoloKernel {
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
    float bbox[4];  // center_x center_y w h
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float mask[32];
};
const int bbox_element = 7; // center_x, center_y, w, h, conf, cls, obj
