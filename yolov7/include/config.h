#pragma once

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

const static int kNumClass = 80;

// Yolo's input width and height must by divisible by 32
const static int kInputH = 640;
const static int kInputW = 640;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
const static int kMaxNumOutputBbox = 1000;

const static int kNumAnchor = 3;

// The bboxes whose conf lower kIgnoreThresh will be ignored in yololayer plugin.
const static float kIgnoreThresh = 0.1f;

