#include "utils.h"

#include <sys/stat.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

bool isDirectory(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool hasImageExt(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) {
        return false;
    }
    std::string ext = path.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp";
}

bool isGeneratedOutputImage(const std::string& path) {
    auto slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    return name.find("_ppocrv5_") != std::string::npos || name.find("_ppocr_system_") != std::string::npos;
}

std::string trimAscii(const std::string& text) {
    size_t first = 0;
    while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) {
        ++first;
    }
    size_t last = text.size();
    while (last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) {
        --last;
    }
    return text.substr(first, last - first);
}

bool startsWith(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

std::string unquoteYamlScalar(const std::string& raw) {
    std::string text = trimAscii(raw);
    if (text.size() >= 2 && text.front() == '\'' && text.back() == '\'') {
        std::string out;
        for (size_t i = 1; i + 1 < text.size(); ++i) {
            if (text[i] == '\'' && i + 1 < text.size() - 1 && text[i + 1] == '\'') {
                out.push_back('\'');
                ++i;
            } else {
                out.push_back(text[i]);
            }
        }
        return out;
    }
    if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
        std::string out;
        for (size_t i = 1; i + 1 < text.size(); ++i) {
            if (text[i] == '\\' && i + 1 < text.size() - 1) {
                char e = text[++i];
                if (e == 'n') {
                    out.push_back('\n');
                } else if (e == 't') {
                    out.push_back('\t');
                } else {
                    out.push_back(e);
                }
            } else {
                out.push_back(text[i]);
            }
        }
        return out;
    }
    return text;
}

std::vector<std::string> parseCharacterDictYaml(const std::vector<std::string>& lines) {
    std::vector<std::string> dict;
    bool inCharacterDict = false;
    for (const auto& rawLine : lines) {
        std::string line = rawLine;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::string trimmed = trimAscii(line);
        if (!inCharacterDict) {
            if (trimmed == "character_dict:") {
                inCharacterDict = true;
            }
            continue;
        }
        if (trimmed.empty()) {
            continue;
        }
        if (startsWith(trimmed, "-")) {
            dict.push_back(unquoteYamlScalar(trimmed.substr(1)));
            continue;
        }
        break;
    }
    return dict;
}

}  // namespace

std::vector<std::string> listImages(const std::string& path) {
    std::vector<std::string> files;
    if (!isDirectory(path)) {
        if (hasImageExt(path)) {
            files.push_back(path);
        }
        return files;
    }

    std::vector<cv::String> cvFiles;
    cv::glob(path + "/*", cvFiles, false);
    for (const auto& file : cvFiles) {
        std::string item = file;
        if (hasImageExt(item) && !isGeneratedOutputImage(item)) {
            files.push_back(item);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string basenameNoExt(const std::string& path) {
    auto slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    auto dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

std::string makeOutputPath(const std::string& imagePath, const std::string& suffix) {
    auto slash = imagePath.find_last_of("/\\");
    std::string dir = slash == std::string::npos ? "." : imagePath.substr(0, slash);
    return dir + "/" + basenameNoExt(imagePath) + suffix;
}

bool fileExists(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && !S_ISDIR(st.st_mode);
}

std::string siblingPath(const std::string& anchorPath, const std::string& fileName) {
    auto slash = anchorPath.find_last_of("/\\");
    std::string candidate = slash == std::string::npos ? fileName : anchorPath.substr(0, slash + 1) + fileName;
    if (fileExists(candidate) || !fileExists(fileName)) {
        return candidate;
    }
    return fileName;
}

std::vector<std::string> loadDictionary(const std::string& path) {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open dictionary: " + path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }

    std::vector<std::string> dict = parseCharacterDictYaml(lines);
    if (dict.empty()) {
        dict = lines;
    }
    if (dict.empty() || dict.back() != " ") {
        dict.push_back(" ");
    }
    return dict;
}

void saveEngine(const std::string& engineName, const nvinfer1::IHostMemory* serializedEngine) {
    std::ofstream file(engineName, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("could not open engine output file: " + engineName);
    }
    file.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
}

std::vector<char> readBinaryFile(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("failed to open binary file: " + fileName);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    return data;
}

std::string findIOTensorName(nvinfer1::ICudaEngine* engine, nvinfer1::TensorIOMode mode) {
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == mode) {
            return name;
        }
    }
    throw std::runtime_error("failed to find requested TensorRT IO tensor");
}

cv::Mat cropTextBox(const cv::Mat& image, const TextBox& box) {
    float widthA = cv::norm(box.points[0] - box.points[1]);
    float widthB = cv::norm(box.points[2] - box.points[3]);
    float heightA = cv::norm(box.points[0] - box.points[3]);
    float heightB = cv::norm(box.points[1] - box.points[2]);
    int width = std::max(1, static_cast<int>(std::max(widthA, widthB)));
    int height = std::max(1, static_cast<int>(std::max(heightA, heightB)));

    std::vector<cv::Point2f> src = {box.points[0], box.points[1], box.points[2], box.points[3]};
    std::vector<cv::Point2f> dst = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(static_cast<float>(width), 0.0f),
            cv::Point2f(static_cast<float>(width), static_cast<float>(height)),
            cv::Point2f(0.0f, static_cast<float>(height)),
    };

    cv::Mat transform = cv::getPerspectiveTransform(src, dst);
    cv::Mat crop;
    cv::warpPerspective(image, crop, transform, cv::Size(width, height), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    if (crop.rows >= crop.cols * 1.5) {
        cv::rotate(crop, crop, cv::ROTATE_90_CLOCKWISE);
    }
    return crop;
}

void drawOcrResult(cv::Mat& image, const std::vector<TextBox>& boxes, const std::vector<RecResult>& recResults) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        for (int j = 0; j < 4; ++j) {
            cv::line(image, box.points[j], box.points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        std::string label = i < recResults.size() ? recResults[i].text : "";
        if (!label.empty()) {
            cv::putText(image, label, box.points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }
}
