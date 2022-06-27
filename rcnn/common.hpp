#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <dirent.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "./logging.h"
#include "./cuda_utils.h"

static Logger gLogger;

using namespace nvinfer1;

void loadWeights(const std::string file, std::map<std::string, Weights>& weightMap) {
    std::cout << "Loading weights: " << file << std::endl;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            // std::string cur_file_name(p_dir_name);
            // cur_file_name += "/";
            // cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

static inline cv::Mat preprocessImg(cv::Mat& img, int input_w, int input_h, int& X_LEFT_PAD, int& X_RIGHT_PAD, int& Y_TOP_PAD, int& Y_BOTTOM_PAD) {
    int w, h;
    float x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);

    // this code can also support left-right and top-bottom padding if you need
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0.0;
        y = (input_h - h) / 2.f;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2.f;
        y = 0.0;
    }

    // support both odd and even cases
    X_LEFT_PAD = (int)(round(x - 0.1));
    X_RIGHT_PAD = (int)(round(x + 0.1));
    Y_TOP_PAD = (int)(round(y - 0.1));
    Y_BOTTOM_PAD = (int)(round(y + 0.1));

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(X_LEFT_PAD, Y_TOP_PAD, re.cols, re.rows)));

    return out;
}