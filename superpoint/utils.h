#pragma once

#include <map>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "assert.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace nvinfer1;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)


int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
std::map<std::string, Weights> loadWeights(const std::string file);
void tokenize(const std::string &str, std::vector<std::string> &tokens, const std::string &delimiters = ",");