#pragma once
#include "config.h"

struct alignas(float) Detection {
  //center_x center_y w h
  float bbox[4];
  float conf;  // bbox_conf * cls_conf
  float class_id;
};

struct AffineMatrix {
    float value[6];
};

const int bbox_element = sizeof(AffineMatrix) / sizeof(float)+1;      // left, top, right, bottom, confidence, class, keepflag
