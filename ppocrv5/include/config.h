#pragma once

#include <cstddef>
#include <string>

static constexpr const char* kDetInputTensorName = "x";
static constexpr const char* kDetOutputTensorName = "det_out";
static constexpr const char* kRecInputTensorName = "x";
static constexpr const char* kRecOutputTensorName = "rec_out";

static constexpr int kDefaultBatchSize = 1;

static constexpr std::size_t kMaxBuilderWorkspaceMiB = 2048;
static constexpr int kDefaultBuilderOptLevel = 0;
static constexpr int kMaxBuilderOptLevel = 5;

static constexpr int kDetResizeLong = 960;
static constexpr float kDetThresh = 0.3f;
static constexpr float kDetBoxThresh = 0.6f;
static constexpr float kDetUnclipRatio = 1.5f;
static constexpr int kDetMaxCandidates = 1000;

static constexpr int kRecInputH = 48;
static constexpr int kRecMinW = 160;
static constexpr int kRecOptW = 320;
static constexpr int kRecMaxW = 3200;
static constexpr int kRecClassCount = 18385;

static constexpr int kFormulaInputH = 768;
static constexpr int kFormulaInputW = 768;
static constexpr int kFormulaMaxLength = 2560;
static constexpr int kFormulaStateCount = 38;
static constexpr int kFormulaBosId = 0;
static constexpr int kFormulaEosId = 2;

static const std::string kDefaultDetModelDir = "official_models/PP-OCRv5_mobile_det";
static const std::string kDefaultRecModelDir = "official_models/PP-OCRv5_mobile_rec";
