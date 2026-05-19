#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct TextBox {
    cv::Point2f points[4];
    float score;
};

struct DetPreprocessResult {
    std::vector<float> chw;
    int input_h;
    int input_w;
    int src_h;
    int src_w;
    float ratio_h;
    float ratio_w;
};

struct RecPreprocessResult {
    std::vector<float> chw;
    int input_h;
    int input_w;
    int valid_w;
};

struct RecResult {
    std::string text;
    float score;
};
