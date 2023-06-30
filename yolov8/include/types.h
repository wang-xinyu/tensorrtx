#pragma once
#include "config.h"

struct alignas(float) Detection {
  //center_x center_y w h
  float bbox[4];
  float conf;  // bbox_conf * cls_conf
  float class_id;
};

