#pragma once
#include "config.h"

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float mask[32];
    float keypoints[kNumberOfPoints * 3];  // 17*3 keypoints
    float angle;                           // obb angle
};

struct AffineMatrix {
    float value[6];
};

// For pose detection: bbox[4] + conf + class_id + keepflag + keypoints[17*3]
const int bbox_element = 4 + 1 + 1 + 1 + (kNumberOfPoints * 3);  // Total elements per detection
