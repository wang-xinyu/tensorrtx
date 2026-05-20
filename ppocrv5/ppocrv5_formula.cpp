#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;

namespace {

struct FormulaPreprocessResult {
    std::vector<float> chw;
    int input_h;
    int input_w;
};

struct DeviceBuffer {
    void* ptr{nullptr};
    size_t bytes{0};

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
        other.ptr = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    void allocate(size_t requestedBytes) {
        release();
        bytes = std::max<size_t>(1, requestedBytes);
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }

    void release() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
};

struct TensorState {
    DataType dtype{DataType::kFLOAT};
    Dims dims{};
    DeviceBuffer buffer;
};

int64_t volume(const Dims& dims) {
    int64_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        count *= dims.d[i];
    }
    return count;
}

size_t dataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT8:
            return 1;
        case DataType::kINT32:
            return 4;
        case DataType::kBOOL:
            return 1;
        case DataType::kUINT8:
            return 1;
#if NV_TENSORRT_MAJOR >= 10
        case DataType::kINT64:
            return 8;
#endif
        default:
            throw std::runtime_error("unsupported TensorRT data type");
    }
}

size_t tensorBytes(DataType dtype, const Dims& dims) {
    int64_t count = volume(dims);
    if (count < 0) {
        throw std::runtime_error("tensor has unresolved dynamic dimension");
    }
    return static_cast<size_t>(count) * dataTypeSize(dtype);
}

Dims makeDims(const std::vector<int>& shape) {
    Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

std::vector<int> dimsToVector(const Dims& dims) {
    std::vector<int> shape;
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

std::string dimsToString(const Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) {
            oss << ",";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

bool isInput(ICudaEngine* engine, const std::string& name) {
    return engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
}

std::vector<std::string> tensorNames(ICudaEngine* engine, TensorIOMode mode) {
    std::vector<std::string> names;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == mode) {
            names.push_back(name);
        }
    }
    return names;
}

bool startsWith(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

int stateIndexFromName(const std::string& name, const std::string& prefix) {
    if (!startsWith(name, prefix)) {
        return -1;
    }
    return std::atoi(name.substr(prefix.size()).c_str());
}

std::string appendBeforeExtension(const std::string& path, const std::string& suffix) {
    auto slash = path.find_last_of("/\\");
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path + suffix;
    }
    return path.substr(0, dot) + suffix + path.substr(dot);
}

std::string trim(const std::string& text) {
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

std::string unquoteYamlScalar(const std::string& raw) {
    std::string text = trim(raw);
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

int indentation(const std::string& line) {
    int count = 0;
    while (count < static_cast<int>(line.size()) && line[count] == ' ') {
        ++count;
    }
    return count;
}

std::vector<std::string> readLines(const std::string& path) {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open tokenizer yaml: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

std::map<uint32_t, unsigned char> buildByteDecoder() {
    std::vector<int> bytes;
    for (int c = '!'; c <= '~'; ++c) {
        bytes.push_back(c);
    }
    for (int c = 0xA1; c <= 0xAC; ++c) {
        bytes.push_back(c);
    }
    for (int c = 0xAE; c <= 0xFF; ++c) {
        bytes.push_back(c);
    }

    std::set<int> used(bytes.begin(), bytes.end());
    std::vector<int> unicode = bytes;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (used.find(b) == used.end()) {
            bytes.push_back(b);
            unicode.push_back(256 + extra++);
        }
    }

    std::map<uint32_t, unsigned char> decoder;
    for (size_t i = 0; i < bytes.size(); ++i) {
        decoder[static_cast<uint32_t>(unicode[i])] = static_cast<unsigned char>(bytes[i]);
    }
    return decoder;
}

bool nextUtf8Codepoint(const std::string& text, size_t& pos, uint32_t& cp) {
    if (pos >= text.size()) {
        return false;
    }
    unsigned char c = static_cast<unsigned char>(text[pos++]);
    if ((c & 0x80) == 0) {
        cp = c;
        return true;
    }
    if ((c & 0xE0) == 0xC0 && pos < text.size()) {
        cp = (c & 0x1F) << 6;
        cp |= static_cast<unsigned char>(text[pos++]) & 0x3F;
        return true;
    }
    if ((c & 0xF0) == 0xE0 && pos + 1 < text.size()) {
        cp = (c & 0x0F) << 12;
        cp |= (static_cast<unsigned char>(text[pos++]) & 0x3F) << 6;
        cp |= static_cast<unsigned char>(text[pos++]) & 0x3F;
        return true;
    }
    if ((c & 0xF8) == 0xF0 && pos + 2 < text.size()) {
        cp = (c & 0x07) << 18;
        cp |= (static_cast<unsigned char>(text[pos++]) & 0x3F) << 12;
        cp |= (static_cast<unsigned char>(text[pos++]) & 0x3F) << 6;
        cp |= static_cast<unsigned char>(text[pos++]) & 0x3F;
        return true;
    }
    cp = c;
    return true;
}

class FormulaTokenizer {
   public:
    explicit FormulaTokenizer(const std::string& yamlPath) { load(yamlPath); }

    std::string decode(const std::vector<int64_t>& ids) const {
        std::string text;
        for (int64_t rawId : ids) {
            int id = static_cast<int>(rawId);
            if (id == kFormulaEosId) {
                break;
            }
            if (specialIds_.find(id) != specialIds_.end()) {
                continue;
            }
            if (id < 0 || id >= static_cast<int>(idToToken_.size())) {
                continue;
            }
            text += byteLevelDecode(idToToken_[id]);
        }
        return postProcess(text);
    }

   private:
    void load(const std::string& yamlPath) {
        auto lines = readLines(yamlPath);
        idToToken_.assign(50000, "");
        parseAddedTokens(lines);
        parseVocab(lines);
        if (idToToken_[33].empty() || idToToken_[kFormulaBosId].empty()) {
            throw std::runtime_error("failed to parse FormulaNet tokenizer from: " + yamlPath);
        }
        byteDecoder_ = buildByteDecoder();
    }

    void parseAddedTokens(const std::vector<std::string>& lines) {
        std::string content;
        bool inAddedToken = false;
        for (const auto& line : lines) {
            std::string t = trim(line);
            if (startsWith(t, "- content:")) {
                content = unquoteYamlScalar(t.substr(std::string("- content:").size()));
                inAddedToken = true;
                continue;
            }
            if (inAddedToken && startsWith(t, "id:")) {
                int id = std::atoi(trim(t.substr(3)).c_str());
                ensureTokenSize(id);
                idToToken_[id] = content;
                specialIds_.insert(id);
                inAddedToken = false;
            }
        }
    }

    void parseVocab(const std::vector<std::string>& lines) {
        bool inVocab = false;
        int vocabIndent = -1;
        for (const auto& line : lines) {
            std::string t = trim(line);
            if (!inVocab) {
                if (t == "vocab:") {
                    inVocab = true;
                    vocabIndent = indentation(line);
                }
                continue;
            }
            if (t.empty()) {
                continue;
            }
            int indent = indentation(line);
            if (indent <= vocabIndent) {
                break;
            }
            size_t sep = t.rfind(": ");
            if (sep == std::string::npos) {
                continue;
            }
            std::string token = unquoteYamlScalar(t.substr(0, sep));
            int id = std::atoi(trim(t.substr(sep + 2)).c_str());
            ensureTokenSize(id);
            idToToken_[id] = token;
        }
    }

    void ensureTokenSize(int id) {
        if (id >= static_cast<int>(idToToken_.size())) {
            idToToken_.resize(id + 1);
        }
    }

    std::string byteLevelDecode(const std::string& token) const {
        std::string out;
        size_t pos = 0;
        uint32_t cp = 0;
        while (nextUtf8Codepoint(token, pos, cp)) {
            auto it = byteDecoder_.find(cp);
            if (it != byteDecoder_.end()) {
                out.push_back(static_cast<char>(it->second));
            }
        }
        return out;
    }

    std::string postProcess(const std::string& text) const {
        std::string out = text;
        out = std::regex_replace(out, std::regex("\\\\text\\s*\\{([^{}]*)\\}"), "\\\\text{$1}");
        bool changed = true;
        while (changed) {
            std::string old = out;
            out = std::regex_replace(out, std::regex("([^A-Za-z])\\s+([^A-Za-z])"), "$1$2");
            out = std::regex_replace(out, std::regex("([^A-Za-z])\\s+([A-Za-z])"), "$1$2");
            out = std::regex_replace(out, std::regex("([A-Za-z])\\s+([^A-Za-z])"), "$1$2");
            changed = out != old;
        }
        return trim(out);
    }

    std::vector<std::string> idToToken_;
    std::set<int> specialIds_;
    std::map<uint32_t, unsigned char> byteDecoder_;
};

FormulaPreprocessResult preprocessFormula(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("empty formula image");
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(gray, &minValue, &maxValue);

    cv::Rect cropRect(0, 0, image.cols, image.rows);
    if (maxValue > minValue) {
        cv::Mat normalized;
        gray.convertTo(normalized, CV_32F);
        normalized = (normalized - minValue) * (255.0 / (maxValue - minValue));
        cv::Mat mask = normalized < 200.0;
        std::vector<cv::Point> coords;
        cv::findNonZero(mask, coords);
        if (!coords.empty()) {
            cropRect = cv::boundingRect(coords) & cv::Rect(0, 0, image.cols, image.rows);
        }
    }

    cv::Mat cropped = image(cropRect).clone();
    float scale = std::min(static_cast<float>(kFormulaInputW) / static_cast<float>(cropped.cols),
                           static_cast<float>(kFormulaInputH) / static_cast<float>(cropped.rows));
    int resizedW = std::max(1, static_cast<int>(std::round(cropped.cols * scale)));
    int resizedH = std::max(1, static_cast<int>(std::round(cropped.rows * scale)));
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(resizedW, resizedH), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat padded(kFormulaInputH, kFormulaInputW, CV_8UC3, cv::Scalar(0, 0, 0));
    int padX = (kFormulaInputW - resizedW) / 2;
    int padY = (kFormulaInputH - resizedH) / 2;
    resized.copyTo(padded(cv::Rect(padX, padY, resizedW, resizedH)));

    FormulaPreprocessResult result;
    result.input_h = kFormulaInputH;
    result.input_w = kFormulaInputW;
    result.chw.resize(kFormulaInputH * kFormulaInputW);
    const float mean = 0.7931f;
    const float stdv = 0.1738f;
    for (int y = 0; y < kFormulaInputH; ++y) {
        const cv::Vec3b* row = padded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kFormulaInputW; ++x) {
            float b = static_cast<float>(row[x][0]);
            float g = static_cast<float>(row[x][1]);
            float r = static_cast<float>(row[x][2]);
            float grayValue = 0.114f * r + 0.587f * g + 0.299f * b;
            result.chw[y * kFormulaInputW + x] = (grayValue / 255.0f - mean) / stdv;
        }
    }
    return result;
}

ICudaEngine* loadEngine(IRuntime* runtime, const std::string& path) {
    auto data = readBinaryFile(path);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!engine) {
        throw std::runtime_error("failed to deserialize engine: " + path);
    }
    return engine;
}

void uploadZeros(TensorState& state) {
    state.buffer.allocate(tensorBytes(state.dtype, state.dims));
    CUDA_CHECK(cudaMemset(state.buffer.ptr, 0, state.buffer.bytes));
}

void uploadBool(TensorState& state, bool value) {
    state.buffer.allocate(sizeof(bool));
    CUDA_CHECK(cudaMemcpy(state.buffer.ptr, &value, sizeof(bool), cudaMemcpyHostToDevice));
}

template <typename T>
void uploadValues(TensorState& state, const std::vector<T>& values) {
    size_t bytes = values.size() * sizeof(T);
    state.buffer.allocate(bytes);
    CUDA_CHECK(cudaMemcpy(state.buffer.ptr, values.data(), bytes, cudaMemcpyHostToDevice));
}

void uploadIndexValues(TensorState& state, const std::vector<int64_t>& values) {
#if NV_TENSORRT_MAJOR >= 10
    if (state.dtype == DataType::kINT64) {
        uploadValues(state, values);
        return;
    }
#endif
    std::vector<int32_t> values32(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        values32[i] = static_cast<int32_t>(values[i]);
    }
    uploadValues(state, values32);
}

TensorState makeInitialState(ICudaEngine* decoder, int index) {
    std::string name = "state_" + std::to_string(index);
    TensorState state;
    state.dtype = decoder->getTensorDataType(name.c_str());

    if (index == 1) {
        state.dims = makeDims({});
        uploadBool(state, false);
    } else if (index == 2) {
        state.dims = makeDims({1});
        uploadIndexValues(state, std::vector<int64_t>{0});
    } else if (index == 3 || index == 5) {
        state.dims = makeDims({1, 1});
        uploadIndexValues(state, std::vector<int64_t>{kFormulaBosId});
    } else if (index == 4) {
        state.dims = makeDims({});
        uploadValues(state, std::vector<float>{0.0f});
    } else if (index >= 6 && index <= 37) {
        int slot = (index - 6) % 4;
        int seqLen = slot < 2 ? 0 : 144;
        state.dims = makeDims({1, 16, seqLen, 32});
        uploadZeros(state);
    } else if (index == 38) {
        state.dims = makeDims({1});
        uploadIndexValues(state, std::vector<int64_t>{1});
    } else {
        throw std::runtime_error("unexpected FormulaNet decoder state index");
    }
    return state;
}

TensorState makeOutputTensor(ICudaEngine* engine, IExecutionContext* context, const std::string& name) {
    TensorState state;
    state.dtype = engine->getTensorDataType(name.c_str());
    state.dims = context->getTensorShape(name.c_str());
    for (int i = 0; i < state.dims.nbDims; ++i) {
        if (state.dims.d[i] < 0) {
            throw std::runtime_error("unresolved output shape for " + name + ": " + dimsToString(state.dims));
        }
    }
    state.buffer.allocate(tensorBytes(state.dtype, state.dims));
    return state;
}

TensorState runEncoder(IExecutionContext* context, ICudaEngine* engine, const cv::Mat& image, cudaStream_t stream) {
    FormulaPreprocessResult meta = preprocessFormula(image);
    std::string inputName = findIOTensorName(engine, TensorIOMode::kINPUT);
    std::string outputName = findIOTensorName(engine, TensorIOMode::kOUTPUT);
    Dims inputDims = makeDims({1, 1, meta.input_h, meta.input_w});
    if (!context->setInputShape(inputName.c_str(), inputDims)) {
        throw std::runtime_error("failed to set FormulaNet encoder input shape");
    }

    TensorState input;
    input.dtype = engine->getTensorDataType(inputName.c_str());
    input.dims = inputDims;
    uploadValues(input, meta.chw);

    TensorState output;
    output.dtype = engine->getTensorDataType(outputName.c_str());
    output.dims = context->getTensorShape(outputName.c_str());
    output.buffer.allocate(tensorBytes(output.dtype, output.dims));

    context->setTensorAddress(inputName.c_str(), input.buffer.ptr);
    context->setTensorAddress(outputName.c_str(), output.buffer.ptr);
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("FormulaNet encoder enqueue failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return output;
}

std::vector<int64_t> downloadInt64Tensor(const TensorState& state) {
    int64_t count = volume(state.dims);
    std::vector<int64_t> values(static_cast<size_t>(std::max<int64_t>(0, count)));
    if (!values.empty()) {
#if NV_TENSORRT_MAJOR >= 10
        if (state.dtype == DataType::kINT64) {
            CUDA_CHECK(cudaMemcpy(values.data(), state.buffer.ptr, values.size() * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost));
            return values;
        }
#endif
        if (state.dtype == DataType::kINT32) {
            std::vector<int32_t> values32(values.size());
            CUDA_CHECK(cudaMemcpy(values32.data(), state.buffer.ptr, values32.size() * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = values32[i];
            }
            return values;
        }
        throw std::runtime_error("unexpected FormulaNet index tensor data type");
    }
    return values;
}

bool downloadCondition(const TensorState& state) {
    if (state.dtype == DataType::kBOOL) {
        bool value = false;
        CUDA_CHECK(cudaMemcpy(&value, state.buffer.ptr, sizeof(bool), cudaMemcpyDeviceToHost));
        return value;
    }
    if (state.dtype == DataType::kINT32) {
        int32_t value = 0;
        CUDA_CHECK(cudaMemcpy(&value, state.buffer.ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
        return value != 0;
    }
    throw std::runtime_error("FormulaNet condition output has unsupported type");
}

std::vector<int64_t> runDecoder(IExecutionContext* context, ICudaEngine* engine, const TensorState& formulaMemory,
                                cudaStream_t stream) {
    std::map<int, TensorState> states;
    for (int i = 1; i <= kFormulaStateCount; ++i) {
        std::string name = "state_" + std::to_string(i);
        if (isInput(engine, name)) {
            states[i] = makeInitialState(engine, i);
        }
    }

    std::vector<std::string> outputNames = tensorNames(engine, TensorIOMode::kOUTPUT);
    int steps = 0;
    bool keepGoing = true;
    while (keepGoing && steps < kFormulaMaxLength) {
        if (!context->setInputShape("formula_memory", formulaMemory.dims)) {
            throw std::runtime_error("failed to set formula_memory shape");
        }
        context->setTensorAddress("formula_memory", formulaMemory.buffer.ptr);

        for (auto& item : states) {
            std::string name = "state_" + std::to_string(item.first);
            if (!context->setInputShape(name.c_str(), item.second.dims)) {
                throw std::runtime_error("failed to set " + name + " shape " + dimsToString(item.second.dims));
            }
            context->setTensorAddress(name.c_str(), item.second.buffer.ptr);
        }

        std::map<int, TensorState> nextStates;
        std::unique_ptr<TensorState> condition(new TensorState());
        for (const auto& outputName : outputNames) {
            if (outputName == "condition") {
                *condition = makeOutputTensor(engine, context, outputName);
                context->setTensorAddress(outputName.c_str(), condition->buffer.ptr);
                continue;
            }
            int stateIndex = stateIndexFromName(outputName, "next_state_");
            if (stateIndex > 0) {
                TensorState out = makeOutputTensor(engine, context, outputName);
                context->setTensorAddress(outputName.c_str(), out.buffer.ptr);
                nextStates[stateIndex] = std::move(out);
            }
        }
        if (!condition->buffer.ptr) {
            throw std::runtime_error("FormulaNet decoder condition output not found");
        }

        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("FormulaNet decoder enqueue failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        keepGoing = downloadCondition(*condition);
        states.swap(nextStates);
        ++steps;

        auto generated = states.find(5);
        if (generated != states.end()) {
            auto ids = downloadInt64Tensor(generated->second);
            if (!ids.empty() && ids.back() == kFormulaEosId) {
                keepGoing = false;
            }
            if (std::getenv("PPOCRV5_FORMULA_DEBUG")) {
                std::cout << "step=" << steps << " tokens=" << ids.size();
                if (!ids.empty()) {
                    std::cout << " last=" << ids.back();
                }
                std::cout << " keep_going=" << keepGoing << std::endl;
            }
        }
    }

    auto it = states.find(5);
    if (it == states.end()) {
        throw std::runtime_error("FormulaNet decoder did not produce state_5");
    }
    return downloadInt64Tensor(it->second);
}

void serializeEngine(const std::string& wtsName, const std::string& engineName) {
    IBuilder* encoderBuilder = createInferBuilder(gLogger);
    IBuilderConfig* encoderConfig = encoderBuilder->createBuilderConfig();
    IHostMemory* encoder = buildEnginePPFormulaNetEncoder(encoderBuilder, encoderConfig, DataType::kFLOAT, wtsName);
    saveEngine(engineName, encoder);
    delete encoder;
    delete encoderConfig;
    delete encoderBuilder;

    std::string decoderName = appendBeforeExtension(engineName, ".decoder");
    IBuilder* decoderBuilder = createInferBuilder(gLogger);
    IBuilderConfig* decoderConfig = decoderBuilder->createBuilderConfig();
    IHostMemory* decoder = buildEnginePPFormulaNetDecoder(decoderBuilder, decoderConfig, DataType::kFLOAT, wtsName);
    saveEngine(decoderName, decoder);
    delete decoder;
    delete decoderConfig;
    delete decoderBuilder;
    std::cout << "FormulaNet decoder engine: " << decoderName << std::endl;
}

std::string defaultTokenizerPath(const std::string& encoderEngine) {
    std::string sibling = siblingPath(encoderEngine, "pp_formulanet_plus_l_inference.yml");
    if (fileExists(sibling)) {
        return sibling;
    }
    std::vector<std::string> candidates = {
            "../official_models/PP-FormulaNet_plus-L/inference.yml",
            "official_models/PP-FormulaNet_plus-L/inference.yml",
    };
    for (const auto& path : candidates) {
        if (fileExists(path)) {
            return path;
        }
    }
    return sibling;
}

void inferImages(const std::string& encoderEngine, const std::string& decoderEngine, const std::string& imagePath,
                 const std::string& tokenizerPath) {
    FormulaTokenizer tokenizer(tokenizerPath);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* encoder = loadEngine(runtime, encoderEngine);
    ICudaEngine* decoder = loadEngine(runtime, decoderEngine);
    IExecutionContext* encoderContext = encoder->createExecutionContext();
    IExecutionContext* decoderContext = decoder->createExecutionContext();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        TensorState formulaMemory = runEncoder(encoderContext, encoder, image, stream);
        std::vector<int64_t> tokenIds = runDecoder(decoderContext, decoder, formulaMemory, stream);
        std::string latex = tokenizer.decode(tokenIds);
        std::cout << file << " tokens=" << tokenIds.size() << " latex=" << latex << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete decoderContext;
    delete encoderContext;
    delete decoder;
    delete encoder;
    delete runtime;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0] << " -s formula.wts formula.engine\n  " << argv[0]
                  << " -d formula.engine formula.decoder.engine image_or_dir [inference.yml]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "-s" && argc == 4) {
        serializeEngine(argv[2], argv[3]);
        return 0;
    }
    if (mode == "-d" && (argc == 5 || argc == 6)) {
        std::string tokenizerPath = argc == 6 ? argv[5] : defaultTokenizerPath(argv[2]);
        inferImages(argv[2], argv[3], argv[4], tokenizerPath);
        return 0;
    }

    std::cerr << "Invalid arguments" << std::endl;
    return 1;
}
