#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "types.h"

std::vector<std::string> listImages(const std::string& path);
std::string makeOutputPath(const std::string& imagePath, const std::string& suffix);
bool fileExists(const std::string& path);
std::string siblingPath(const std::string& anchorPath, const std::string& fileName);
std::vector<std::string> loadDictionary(const std::string& path);
std::string basenameNoExt(const std::string& path);

void saveEngine(const std::string& engineName, const nvinfer1::IHostMemory* serializedEngine);
std::vector<char> readBinaryFile(const std::string& fileName);
std::string findIOTensorName(nvinfer1::ICudaEngine* engine, nvinfer1::TensorIOMode mode);
cv::Mat cropTextBox(const cv::Mat& image, const TextBox& box);
void drawOcrResult(cv::Mat& image, const std::vector<TextBox>& boxes, const std::vector<RecResult>& recResults);
