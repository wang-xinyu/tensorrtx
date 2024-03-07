#pragma once

const static char *kInputTensorName = "data";
const static char *kOutputTensorName = "prob";
const static char *kEngineFile = "./csrnet.engine";

const static int kBatchSize = 1;

const static int MAX_INPUT_SIZE = 1440; // 32x
const static int MIN_INPUT_SIZE = 608;
const static int OPT_INPUT_W = 1152;
const static int OPT_INPUT_H = 640;

constexpr static int kMaxInputImageSize = MAX_INPUT_SIZE * MAX_INPUT_SIZE * 3;
constexpr static int kMaxOutputProbSize =
    (MAX_INPUT_SIZE * MAX_INPUT_SIZE) >> 6;