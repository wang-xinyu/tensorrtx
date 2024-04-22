#pragma once

#include <opencv2/opencv.hpp>

void genHeatMap(cv::Mat originImg, cv::Mat& anomalyGrayMap, cv::Mat& HeatMap) {
    cv::Mat colorMap;
    cv::applyColorMap(colorMap, anomalyGrayMap, cv::COLORMAP_JET);
    cv::addWeighted(originImg, 0.5, colorMap, 0.5, 0, HeatMap);
}
