#pragma once
#include <dirent.h>
#include <fstream>
#include <opencv2/opencv.hpp>

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
            // std::string cur_file_name(p_dir_name);
            // cur_file_name += "/";
            // cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            //            std::cout << "Found file: " << cur_file_name << std::endl;
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// Function to trim leading and trailing whitespace from a string
static inline std::string trim_leading_whitespace(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// Src: https://stackoverflow.com/questions/16605967
static inline std::string to_string_with_precision(const float a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

static inline int read_labels(const std::string labels_filename, std::unordered_map<int, std::string>& labels_map) {
    std::ifstream file(labels_filename);
    // Read each line of the file
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        // Strip the line of any leading or trailing whitespace
        line = trim_leading_whitespace(line);

        // Add the stripped line to the labels_map, using the loop index as the key
        labels_map[index] = line;
        index++;
    }
    // Close the file
    file.close();

    return 0;
}

static inline bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir,
                              std::string& type, float& gd, float& gw, int& max_channels) {
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto sub_type = std::string(argv[4]);

        if (sub_type[0] == 'n') {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        } else if (sub_type[0] == 's') {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        } else if (sub_type[0] == 'm') {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        } else if (sub_type[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        } else if (sub_type[0] == 'x') {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        } else {
            return false;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}