#pragma once

/* --------------------------------------------------------
 * These configs are related to tensorrt model, if these are changed,
 * please re-compile and re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// For INT8, you need prepare the calibration dataset, please refer to
// https://github.com/wang-xinyu/tensorrtx/tree/master/yolov7#int8-quantization
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

const static int kNumClass = 80;
const static int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
const static int kInputH = 640;
const static int kInputW = 640;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
const static int kMaxNumOutputBbox = 1000;

const static int kNumAnchor = 3;

// The bboxes whose confidence is lower than kIgnoreThresh will be ignored in yololayer plugin.
const static float kIgnoreThresh = 0.1f;

/* --------------------------------------------------------
 * These configs are not related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// NMS overlapping thresh and final detection confidence thresh
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

const static int kGpuId = 0;

// If your image size is larger than 4096 * 3112, please increase this value
const static int kMaxInputImageSize = 4096 * 3112;

