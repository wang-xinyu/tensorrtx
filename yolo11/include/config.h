//#define USE_FP16
// #define USE_FP32
#define USE_INT8

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output";
const static char* kProtoTensorName = "proto";
const static int kNumClass = 80;
const static int kPoseNumClass = 1;
const static int kNumberOfPoints = 17;  // number of keypoints total
const static int kBatchSize = 1;
const static int kGpuId = 0;
const static int kInputH = 640;
const static int kInputW = 640;
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;
const static float kConfThreshKeypoints = 0.5f;  // keypoints confidence
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;
//Quantization input image folder path
const static char* kInputQuantizationFolder = "./coco_calib";

// Classfication model's number of classes
constexpr static int kClsNumClass = 1000;
// Classfication model's input shape
constexpr static int kClsInputH = 224;
constexpr static int kClsInputW = 224;
