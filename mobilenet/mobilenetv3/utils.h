#ifndef MOBILENET_UTILS_H
#define MOBILENET_UTILS_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define INPUT_H 224
#define INPUT_W 224

/**
 * Preprocesses an image using the standard PyTorch MobileNet preprocessing pipeline:
 * 1. Load image and convert BGR to RGB
 * 2. Resize to 256x256
 * 3. Center crop to 224x224  
 * 4. Convert to float [0,1] (ToTensor equivalent)
 * 5. Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
 * 
 * @param imagePath Path to input image file
 * @return Preprocessed image data as CHW float vector
 */
std::vector<float> preprocessImage(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << std::endl;
        exit(1);
    }

    std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;

    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // PyTorch preprocessing: Resize to 256, then center crop to 224
    int resize_size = 256;
    cv::resize(image, image, cv::Size(resize_size, resize_size));

    // Center crop to 224x224
    int crop_size = 224;
    int start_x = (resize_size - crop_size) / 2;
    int start_y = (resize_size - crop_size) / 2;
    cv::Rect crop_rect(start_x, start_y, crop_size, crop_size);
    image = image(crop_rect);

    // Convert to float and normalize to [0, 1] (ToTensor equivalent)
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    // ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    std::vector<float> input_data(3 * INPUT_H * INPUT_W);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < INPUT_H; h++) {
            for (int w = 0; w < INPUT_W; w++) {
                int dst_idx = c * INPUT_H * INPUT_W + h * INPUT_W + w;
                float pixel_value = image.at<cv::Vec3f>(h, w)[c];
                input_data[dst_idx] = (pixel_value - mean[c]) / std[c];
            }
        }
    }

    return input_data;
}

/**
 * Applies softmax to output logits and prints top-k predictions
 * 
 * @param output Raw model output logits
 * @param output_size Size of output array
 * @param k Number of top predictions to print
 */
void printTopPredictions(float* output, int output_size, int k = 5) {
    std::vector<float> softmax_output(output_size);
    float max_val = *std::max_element(output, output + output_size);

    float sum_exp = 0.0f;
    for (int i = 0; i < output_size; i++) {
        softmax_output[i] = std::exp(output[i] - max_val);
        sum_exp += softmax_output[i];
    }

    for (int i = 0; i < output_size; i++) {
        softmax_output[i] /= sum_exp;
    }

    std::vector<std::pair<float, int>> prob_index_pairs;
    for (int i = 0; i < output_size; i++) {
        prob_index_pairs.push_back({softmax_output[i], i});
    }

    std::sort(prob_index_pairs.begin(), prob_index_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

    std::cout << "\nTop " << k << " predictions:" << std::endl;
    for (int i = 0; i < k && i < output_size; i++) {
        std::cout << "  " << i + 1 << ". Class " << prob_index_pairs[i].second << ": " << prob_index_pairs[i].first
                  << std::endl;
    }

    std::cout << std::endl;
}

#endif  // MOBILENET_UTILS_H