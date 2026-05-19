#pragma once

#include <string>
#include <vector>
#include "types.h"

std::vector<TextBox> dbPostprocess(const float* prob, int outH, int outW, const DetPreprocessResult& meta);
RecResult ctcDecode(const float* prob, int timeSteps, int classCount, const std::vector<std::string>& dict);
