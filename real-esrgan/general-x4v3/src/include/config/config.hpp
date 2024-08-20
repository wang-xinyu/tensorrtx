#ifndef REAL_ESRGAN_TRT_CONFIG_HPP
#define REAL_ESRGAN_TRT_CONFIG_HPP

#include <string>

//std::string INPUT_BLOB_NAME = "input";
//std::string OUTPUT_BLOB_NAME = "output";

const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";

const bool USE_FP16 = false;

static const int BATCH_SIZE = 1;
static const int INPUT_C = 3;
static const int INPUT_H = 450;
static const int INPUT_W = 300;
static const int OUT_SCALE = 4;
//static const int OUTPUT_SIZE = INPUT_C * INPUT_H * OUT_SCALE * INPUT_W * OUT_SCALE;
static const int OUTPUT_SIZE = BATCH_SIZE * 48 * 450 * 300;
//INPUT_C * INPUT_H * OUT_SCALE * INPUT_W * OUT_SCALE;
#endif  //REAL_ESRGAN_TRT_CONFIG_HPP
