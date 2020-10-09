#ifndef BOXUTILS_H
#define BOXUTILS_H
#include<vector>

namespace ssd{
  struct alignas(float) Detection{
      std::vector<float> bbox;  // x1 y1 x2 y2
      float class_id; // 0 background
      float conf;     // classification confidence
  };
  static const int INPUT_H = 300;
  static const int INPUT_W = 300;
  static const int INPUT_C = 3;

  static const int LOCATIONS = 4;
  static const int NUM_DETECTIONS = 3000;
  static const int NUM_CLASSES = 21;
};

std::vector<ssd::Detection> post_process_output(float* prob, float* locations, float conf_thresh);
void prepare_input(std::string filename, float* data);
void save_inference(std::string filename, std::vector<ssd::Detection> res);

#endif
