#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "ppocrv5_db_layer.h"
#include "ppocrv5_rtdetr_layer.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;

namespace {

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

struct HostTensor {
    std::string name;
    DataType dtype{DataType::kFLOAT};
    Dims dims{};
    std::vector<char> data;
};

struct InputTensor {
    std::string name;
    Dims dims{};
    std::vector<float> data;
};

struct FormulaPreprocessResult {
    std::vector<float> chw;
    int input_h;
    int input_w;
};

int64_t volume(const Dims& dims) {
    int64_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("dynamic dimension was not resolved");
        }
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
    return static_cast<size_t>(volume(dims)) * dataTypeSize(dtype);
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

std::string dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT:
            return "float32";
        case DataType::kHALF:
            return "float16";
        case DataType::kINT8:
            return "int8";
        case DataType::kINT32:
            return "int32";
        case DataType::kBOOL:
            return "bool";
        case DataType::kUINT8:
            return "uint8";
#if NV_TENSORRT_MAJOR >= 10
        case DataType::kINT64:
            return "int64";
#endif
        default:
            return "unknown";
    }
}

bool hasDynamicDim(const Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string parentPath(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return ".";
    }
    return path.substr(0, slash);
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

bool startsWith(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

std::string safeFileTag(const std::string& tag) {
    std::string safe = tag;
    for (char& c : safe) {
        unsigned char ch = static_cast<unsigned char>(c);
        if (!std::isalnum(ch) && c != '_' && c != '-' && c != '.') {
            c = '_';
        }
    }
    return safe;
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

ICudaEngine* deserializeEngine(IRuntime* runtime, const std::string& path) {
    ppocrv5EnsureDbPlugin();
    ppocrv5EnsureRtDetrPlugin();

    auto data = readBinaryFile(path);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!engine) {
        throw std::runtime_error("failed to deserialize engine: " + path);
    }
    return engine;
}

class OutputAllocator : public IOutputAllocator {
   public:
    ~OutputAllocator() override {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    void* reallocateOutput(char const*, void*, uint64_t size, uint64_t) noexcept override {
        if (size > bytes_) {
            if (ptr_) {
                cudaFree(ptr_);
                ptr_ = nullptr;
            }
            if (cudaMalloc(&ptr_, size) != cudaSuccess) {
                ptr_ = nullptr;
                bytes_ = 0;
                return nullptr;
            }
            bytes_ = size;
        }
        return ptr_;
    }

#if NV_TENSORRT_MAJOR >= 10
    void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
                                cudaStream_t) noexcept override {
        return reallocateOutput(tensorName, currentMemory, size, alignment);
    }
#endif

    void notifyShape(char const*, Dims const& dims) noexcept override { dims_ = dims; }

    void* ptr() const { return ptr_; }

    const Dims& dims() const { return dims_; }

   private:
    void* ptr_{nullptr};
    uint64_t bytes_{0};
    Dims dims_{};
};

class EngineSession {
   public:
    EngineSession(IRuntime* runtime, const std::string& enginePath) : path_(enginePath) {
        engine_ = deserializeEngine(runtime, enginePath);
        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("failed to create execution context: " + enginePath);
        }
        inputs_ = tensorNames(engine_, TensorIOMode::kINPUT);
        outputs_ = tensorNames(engine_, TensorIOMode::kOUTPUT);
    }

    ~EngineSession() {
        delete context_;
        delete engine_;
    }

    const std::vector<std::string>& inputs() const { return inputs_; }

    Dims tensorShape(const std::string& name) const { return engine_->getTensorShape(name.c_str()); }

    std::vector<HostTensor> infer(const std::vector<InputTensor>& inputs, cudaStream_t stream) {
        std::map<std::string, const InputTensor*> byName;
        for (const auto& input : inputs) {
            byName[input.name] = &input;
        }

        std::vector<DeviceBuffer> inputBuffers(inputs_.size());
        for (size_t i = 0; i < inputs_.size(); ++i) {
            auto it = byName.find(inputs_[i]);
            if (it == byName.end()) {
                throw std::runtime_error("missing input tensor: " + inputs_[i]);
            }
            const InputTensor& input = *it->second;
            if (!context_->setInputShape(input.name.c_str(), input.dims)) {
                throw std::runtime_error("failed to set input shape for " + input.name + ": " +
                                         dimsToString(input.dims));
            }
            inputBuffers[i].allocate(input.data.size() * sizeof(float));
            CUDA_CHECK(cudaMemcpyAsync(inputBuffers[i].ptr, input.data.data(), inputBuffers[i].bytes,
                                       cudaMemcpyHostToDevice, stream));
            context_->setTensorAddress(input.name.c_str(), inputBuffers[i].ptr);
        }

        std::vector<DeviceBuffer> outputBuffers(outputs_.size());
        std::vector<std::unique_ptr<OutputAllocator>> allocators(outputs_.size());
        std::vector<HostTensor> results(outputs_.size());
        std::vector<bool> dynamic(outputs_.size(), false);
        for (size_t i = 0; i < outputs_.size(); ++i) {
            results[i].name = outputs_[i];
            results[i].dtype = engine_->getTensorDataType(outputs_[i].c_str());
            results[i].dims = context_->getTensorShape(outputs_[i].c_str());
            dynamic[i] = hasDynamicDim(results[i].dims);
            if (dynamic[i]) {
                allocators[i].reset(new OutputAllocator());
                context_->setOutputAllocator(outputs_[i].c_str(), allocators[i].get());
            } else {
                outputBuffers[i].allocate(tensorBytes(results[i].dtype, results[i].dims));
                context_->setTensorAddress(outputs_[i].c_str(), outputBuffers[i].ptr);
            }
        }

        if (!context_->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueue failed: " + path_);
        }

        for (size_t i = 0; i < outputs_.size(); ++i) {
            if (dynamic[i]) {
                results[i].dims = allocators[i]->dims();
                results[i].data.resize(tensorBytes(results[i].dtype, results[i].dims));
                CUDA_CHECK(cudaMemcpyAsync(results[i].data.data(), allocators[i]->ptr(), results[i].data.size(),
                                           cudaMemcpyDeviceToHost, stream));
            } else {
                results[i].data.resize(outputBuffers[i].bytes);
                CUDA_CHECK(cudaMemcpyAsync(results[i].data.data(), outputBuffers[i].ptr, results[i].data.size(),
                                           cudaMemcpyDeviceToHost, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return results;
    }

   private:
    std::string path_;
    ICudaEngine* engine_{nullptr};
    IExecutionContext* context_{nullptr};
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

std::vector<float> hostTensorAsFloat(const HostTensor& tensor) {
    if (tensor.dtype != DataType::kFLOAT) {
        return std::vector<float>();
    }
    std::vector<float> values(tensor.data.size() / sizeof(float));
    if (!values.empty()) {
        std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
    }
    return values;
}

std::vector<float> normalizeImageToChw(const cv::Mat& image, int height, int width) {
    if (image.empty()) {
        throw std::runtime_error("empty image");
    }
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> chw(3 * height * width);
    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            float b = row[x][0] / 255.0f;
            float g = row[x][1] / 255.0f;
            float r = row[x][2] / 255.0f;
            chw[0 * height * width + y * width + x] = (b - mean[0]) / stdv[0];
            chw[1 * height * width + y * width + x] = (g - mean[1]) / stdv[1];
            chw[2 * height * width + y * width + x] = (r - mean[2]) / stdv[2];
        }
    }
    return chw;
}

Dims defaultInputShape(const std::string& modelName) {
    if (modelName == "pp_lcnet_x1_0_doc_ori" || modelName == "pp_lcnet_x1_0_table_cls") {
        return makeDims({1, 3, 224, 224});
    }
    if (modelName == "pp_lcnet_x1_0_textline_ori") {
        return makeDims({1, 3, 80, 160});
    }
    if (modelName == "pp_doclayout_plus_l" || modelName == "slanet_plus" || modelName == "uvdoc") {
        return makeDims({1, 3, 800, 800});
    }
    if (modelName == "slanext_wired") {
        return makeDims({1, 3, 512, 512});
    }
    return makeDims({1, 3, 640, 640});
}

InputTensor makeImageTensor(const std::string& name, const cv::Mat& image, const Dims& wantedShape) {
    if (wantedShape.nbDims != 4 || wantedShape.d[1] != 3) {
        throw std::runtime_error("expected NCHW image shape for " + name + ": " + dimsToString(wantedShape));
    }
    InputTensor input;
    input.name = name;
    input.dims = wantedShape;
    input.data = normalizeImageToChw(image, wantedShape.d[2], wantedShape.d[3]);
    return input;
}

std::vector<InputTensor> makeGenericInputs(EngineSession& session, const std::string& modelName, const cv::Mat& image) {
    Dims imageShape = defaultInputShape(modelName);
    std::vector<InputTensor> inputs;
    for (const auto& name : session.inputs()) {
        Dims shape = session.tensorShape(name);
        if (hasDynamicDim(shape)) {
            shape = name == "im_shape" || name == "scale_factor" ? makeDims({1, 2}) : imageShape;
        }
        if (name == "im_shape") {
            InputTensor input;
            input.name = name;
            input.dims = shape;
            input.data = {static_cast<float>(imageShape.d[2]), static_cast<float>(imageShape.d[3])};
            inputs.push_back(input);
        } else if (name == "scale_factor") {
            InputTensor input;
            input.name = name;
            input.dims = shape;
            input.data = {static_cast<float>(imageShape.d[2]) / static_cast<float>(image.rows),
                          static_cast<float>(imageShape.d[3]) / static_cast<float>(image.cols)};
            inputs.push_back(input);
        } else {
            inputs.push_back(makeImageTensor(name, image, shape));
        }
    }
    return inputs;
}

void printTopClass(const std::vector<float>& values) {
    if (values.empty()) {
        return;
    }
    int best = static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
    std::cout << " top=" << best << " score=" << values[best];
}

void printFloatStats(const std::vector<float>& values) {
    if (values.empty()) {
        return;
    }
    auto mm = std::minmax_element(values.begin(), values.end());
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::cout << " min=" << *mm.first << " max=" << *mm.second << " mean=" << sum / values.size();
}

void printDetectionPreview(const HostTensor& tensor, const std::vector<float>& values) {
    if (values.empty() || tensor.dims.nbDims < 2) {
        return;
    }
    int last = tensor.dims.d[tensor.dims.nbDims - 1];
    if (last < 6 || last > 16 || values.size() < static_cast<size_t>(last)) {
        return;
    }
    int rows = static_cast<int>(values.size()) / last;
    std::cout << " preview_rows=" << std::min(rows, 3);
    for (int r = 0; r < std::min(rows, 3); ++r) {
        std::cout << " [";
        for (int c = 0; c < std::min(last, 8); ++c) {
            if (c) {
                std::cout << ",";
            }
            std::cout << values[r * last + c];
        }
        std::cout << "]";
    }
}

void printTensorSummary(const std::string& file, const std::string& modelName, const std::vector<HostTensor>& outputs) {
    for (const auto& out : outputs) {
        std::cout << file << " model=" << modelName << " output=" << out.name << " dtype=" << dtypeName(out.dtype)
                  << " shape=" << dimsToString(out.dims);
        std::vector<float> values = hostTensorAsFloat(out);
        if (!values.empty()) {
            printFloatStats(values);
            if (values.size() <= 1024) {
                printTopClass(values);
            }
            printDetectionPreview(out, values);
        }
        std::cout << std::endl;
    }
}

std::vector<TextBox> runDet(EngineSession& det, const cv::Mat& image, cudaStream_t stream) {
    auto meta = preprocessDet(image);
    InputTensor input;
    input.name = det.inputs().front();
    input.dims = makeDims({1, 3, meta.input_h, meta.input_w});
    input.data = meta.chw;
    auto outputs = det.infer(std::vector<InputTensor>{input}, stream);
    if (outputs.empty()) {
        throw std::runtime_error("OCR detection engine has no outputs");
    }
    auto values = hostTensorAsFloat(outputs[0]);
    return dbPostprocess(values.data(), outputs[0].dims.d[2], outputs[0].dims.d[3], meta);
}

RecResult runRec(EngineSession& rec, const cv::Mat& crop, const std::vector<std::string>& dict, cudaStream_t stream) {
    auto meta = preprocessRec(crop, kRecMaxW);
    InputTensor input;
    input.name = rec.inputs().front();
    input.dims = makeDims({1, 3, meta.input_h, meta.input_w});
    input.data = meta.chw;
    auto outputs = rec.infer(std::vector<InputTensor>{input}, stream);
    if (outputs.empty()) {
        throw std::runtime_error("OCR recognition engine has no outputs");
    }
    auto values = hostTensorAsFloat(outputs[0]);
    int timeSteps = outputs[0].dims.nbDims == 3 ? outputs[0].dims.d[1] : outputs[0].dims.d[0];
    int classCount = outputs[0].dims.nbDims == 3 ? outputs[0].dims.d[2] : outputs[0].dims.d[1];
    return ctcDecode(values.data(), timeSteps, classCount, dict);
}

std::string defaultRecDict(const std::string& recEnginePath) {
    std::string engineDir = parentPath(recEnginePath);
    std::string recStem = basenameNoExt(recEnginePath);
    std::vector<std::string> candidates = {
            joinPath(engineDir, recStem + "_dict.txt"),
            joinPath(parentPath(engineDir), "all_models/" + recStem + "_dict.txt"),
            "all_models/ppocrv5_mobile_rec_dict.txt",
            "all_models/ppocrv5_server_rec_dict.txt",
            joinPath(engineDir, "ppocrv5_mobile_rec_dict.txt"),
            joinPath(engineDir, "ppocrv5_server_rec_dict.txt"),
            "../official_models/PP-OCRv5_mobile_rec/inference.yml",
            "../official_models/PP-OCRv5_server_rec/inference.yml",
            "official_models/PP-OCRv5_mobile_rec/inference.yml",
            "official_models/PP-OCRv5_server_rec/inference.yml",
            "ppocrv5_rec_dict.txt",
    };
    for (const auto& path : candidates) {
        if (fileExists(path)) {
            return path;
        }
    }
    return candidates.front();
}

std::string defaultFormulaYaml(const std::string& engineDir) {
    std::vector<std::string> candidates = {
            joinPath(engineDir, "pp_formulanet_plus_l_inference.yml"),
            joinPath(parentPath(engineDir), "all_models/pp_formulanet_plus_l_inference.yml"),
            "../official_models/PP-FormulaNet_plus-L/inference.yml",
            "official_models/PP-FormulaNet_plus-L/inference.yml",
    };
    for (const auto& path : candidates) {
        if (fileExists(path)) {
            return path;
        }
    }
    return candidates.front();
}

void runOcrPair(IRuntime* runtime, const std::string& detPath, const std::string& recPath, const std::string& imagePath,
                const std::string& dictPath, const std::string& tag, cudaStream_t stream) {
    if (!fileExists(detPath) || !fileExists(recPath)) {
        std::cout << "ocr skipped missing det/rec engine det=" << detPath << " rec=" << recPath << std::endl;
        return;
    }

    std::vector<std::string> dict = loadDictionary(dictPath.empty() ? defaultRecDict(recPath) : dictPath);
    EngineSession det(runtime, detPath);
    EngineSession rec(runtime, recPath);
    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        std::vector<TextBox> boxes = runDet(det, image, stream);
        std::vector<RecResult> recResults;
        for (const auto& box : boxes) {
            recResults.push_back(runRec(rec, cropTextBox(image, box), dict, stream));
        }
        for (size_t i = 0; i < boxes.size(); ++i) {
            std::cout << file << " function=" << tag << " box=" << i << " det_conf=" << boxes[i].score
                      << " rec_conf=" << recResults[i].score << " text=" << recResults[i].text << std::endl;
        }
        cv::Mat vis = image.clone();
        drawOcrResult(vis, boxes, recResults);
        std::string outPath = makeOutputPath(file, "_ppocr_system_" + safeFileTag(tag) + ".jpg");
        cv::imwrite(outPath, vis);
        std::cout << file << " function=" << tag << " boxes=" << boxes.size() << " output=" << outPath << std::endl;
    }
}

void runOcrGroup(IRuntime* runtime, const std::string& engineDir, const std::string& imagePath,
                 const std::string& dictPath, const std::string& prefix, cudaStream_t stream) {
    runOcrPair(runtime, joinPath(engineDir, prefix + "_det.engine"), joinPath(engineDir, prefix + "_rec.engine"),
               imagePath, dictPath, prefix, stream);
}

void runOcrModels(IRuntime* runtime, const std::string& detModel, const std::string& detPath,
                  const std::string& recModel, const std::string& recPath, const std::string& imagePath,
                  const std::string& dictPath, const std::string& prefix, cudaStream_t stream) {
    std::string tag = prefix + ":" + detModel + "+" + recModel;
    runOcrPair(runtime, detPath, recPath, imagePath, dictPath, tag, stream);
}

void runGenericModel(IRuntime* runtime, const std::string& engineDir, const std::string& modelName,
                     const std::string& imagePath, cudaStream_t stream) {
    std::string enginePath = joinPath(engineDir, modelName + ".engine");
    if (!fileExists(enginePath)) {
        std::cout << "model=" << modelName << " skipped missing engine" << std::endl;
        return;
    }
    EngineSession session(runtime, enginePath);
    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        auto inputs = makeGenericInputs(session, modelName, image);
        auto outputs = session.infer(inputs, stream);
        printTensorSummary(file, modelName, outputs);
    }
}

void runGenericModelPath(IRuntime* runtime, const std::string& modelName, const std::string& enginePath,
                         const std::string& imagePath, cudaStream_t stream) {
    if (!fileExists(enginePath)) {
        std::cout << "model=" << modelName << " skipped missing engine path=" << enginePath << std::endl;
        return;
    }
    EngineSession session(runtime, enginePath);
    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        auto inputs = makeGenericInputs(session, modelName, image);
        auto outputs = session.infer(inputs, stream);
        printTensorSummary(file, modelName, outputs);
    }
}

void runModelGroup(IRuntime* runtime, const std::string& engineDir, const std::string& imagePath,
                   const std::vector<std::string>& models, cudaStream_t stream) {
    for (const auto& model : models) {
        runGenericModel(runtime, engineDir, model, imagePath, stream);
    }
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
                out.push_back(e == 'n' ? '\n' : e == 't' ? '\t' : e);
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

bool isInput(ICudaEngine* engine, const std::string& name) {
    return engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
}

int stateIndexFromName(const std::string& name, const std::string& prefix) {
    if (!startsWith(name, prefix)) {
        return -1;
    }
    return std::atoi(name.substr(prefix.size()).c_str());
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

TensorState runFormulaEncoder(IExecutionContext* context, ICudaEngine* engine, const cv::Mat& image,
                              cudaStream_t stream) {
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

std::vector<int64_t> runFormulaDecoder(IExecutionContext* context, ICudaEngine* engine,
                                       const TensorState& formulaMemory, cudaStream_t stream) {
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
        }
    }

    auto it = states.find(5);
    if (it == states.end()) {
        throw std::runtime_error("FormulaNet decoder did not produce state_5");
    }
    return downloadInt64Tensor(it->second);
}

void runFormulaPair(IRuntime* runtime, const std::string& encoderPath, const std::string& decoderPath,
                    const std::string& imagePath, const std::string& yamlPath, cudaStream_t stream) {
    if (!fileExists(encoderPath) || !fileExists(decoderPath)) {
        std::cout << "formula skipped missing encoder/decoder engine" << std::endl;
        return;
    }

    FormulaTokenizer tokenizer(yamlPath.empty() ? defaultFormulaYaml(parentPath(encoderPath)) : yamlPath);
    ICudaEngine* encoder = deserializeEngine(runtime, encoderPath);
    ICudaEngine* decoder = deserializeEngine(runtime, decoderPath);
    IExecutionContext* encoderContext = encoder->createExecutionContext();
    IExecutionContext* decoderContext = decoder->createExecutionContext();

    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        TensorState formulaMemory = runFormulaEncoder(encoderContext, encoder, image, stream);
        std::vector<int64_t> tokenIds = runFormulaDecoder(decoderContext, decoder, formulaMemory, stream);
        std::string latex = tokenizer.decode(tokenIds);
        std::cout << file << " function=formula tokens=" << tokenIds.size() << " latex=" << latex << std::endl;
    }

    delete decoderContext;
    delete encoderContext;
    delete decoder;
    delete encoder;
}

void runFormulaGroup(IRuntime* runtime, const std::string& engineDir, const std::string& imagePath,
                     const std::string& yamlPath, cudaStream_t stream) {
    runFormulaPair(runtime, joinPath(engineDir, "pp_formulanet_plus_l.engine"),
                   joinPath(engineDir, "pp_formulanet_plus_l.decoder.engine"), imagePath, yamlPath, stream);
}

bool isOcrVariantToken(const std::string& text) {
    return text == "m" || text == "s";
}

std::string ocrPrefixFromVariant(const std::string& variant) {
    if (variant == "m") {
        return "ppocrv5_mobile";
    }
    if (variant == "s") {
        return "ppocrv5_server";
    }
    throw std::runtime_error("unknown OCR variant, use m or s: " + variant);
}

void runSystem(const std::string& mode, const std::string& engineDir, const std::string& imagePath,
               const std::string& dictPath, const std::string& formulaYaml, const std::string& ocrPrefix) {
    IRuntime* runtime = createInferRuntime(gLogger);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (mode == "all" || mode == "ocr") {
        runOcrGroup(runtime, engineDir, imagePath, dictPath, ocrPrefix, stream);
    }
    if (mode == "all" || mode == "classify") {
        runModelGroup(runtime, engineDir, imagePath,
                      {"pp_lcnet_x1_0_doc_ori", "pp_lcnet_x1_0_table_cls", "pp_lcnet_x1_0_textline_ori"}, stream);
    }
    if (mode == "all" || mode == "layout") {
        runModelGroup(runtime, engineDir, imagePath, {"pp_docblocklayout", "pp_doclayout_plus_l"}, stream);
    }
    if (mode == "all" || mode == "table") {
        runModelGroup(
                runtime, engineDir, imagePath,
                {"rt_detr_l_wired_table_cell_det", "rt_detr_l_wireless_table_cell_det", "slanet_plus", "slanext_wired"},
                stream);
    }
    if (mode == "all" || mode == "uvdoc") {
        runModelGroup(runtime, engineDir, imagePath, {"uvdoc"}, stream);
    }
    if (mode == "all" || mode == "formula") {
        runFormulaGroup(runtime, engineDir, imagePath, formulaYaml, stream);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete runtime;
}

void runExplicitOcrSystem(const std::string& variant, const std::string& detModel, const std::string& detPath,
                          const std::string& recModel, const std::string& recPath, const std::string& imagePath,
                          const std::string& dictPath) {
    IRuntime* runtime = createInferRuntime(gLogger);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    runOcrModels(runtime, detModel, detPath, recModel, recPath, imagePath, dictPath, ocrPrefixFromVariant(variant),
                 stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete runtime;
}

void runExplicitModelSystem(const std::string& modelName, const std::string& enginePath, const std::string& imagePath) {
    IRuntime* runtime = createInferRuntime(gLogger);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    runGenericModelPath(runtime, modelName, enginePath, imagePath, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete runtime;
}

void runExplicitFormulaSystem(const std::string&, const std::string& encoderPath, const std::string& decoderPath,
                              const std::string& imagePath, const std::string& yamlPath) {
    IRuntime* runtime = createInferRuntime(gLogger);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    runFormulaPair(runtime, encoderPath, decoderPath, imagePath, yamlPath, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete runtime;
}

void printUsage(const char* app) {
    std::cerr << "Usage:\n"
              << "  " << app
              << " -all m|s engine_dir image_or_dir [rec_dict.txt|rec_inference.yml] [formula_inference.yml]\n"
              << "  " << app << " -ocr m|s engine_dir image_or_dir [rec_dict.txt|rec_inference.yml]\n"
              << "  " << app
              << " -ocr m|s det_model det.engine rec_model rec.engine image_or_dir [rec_dict.txt|rec_inference.yml]\n"
              << "  " << app << " -model model_name model.engine image_or_dir\n"
              << "  " << app << " -classify engine_dir image_or_dir\n"
              << "  " << app << " -layout engine_dir image_or_dir\n"
              << "  " << app << " -table engine_dir image_or_dir\n"
              << "  " << app << " -uvdoc engine_dir image_or_dir\n"
              << "  " << app << " -formula engine_dir image_or_dir [formula_inference.yml]\n"
              << "  " << app
              << " -formula model_name encoder.engine decoder.engine image_or_dir [formula_inference.yml]\n"
              << "m=mobile, s=server" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modeArg = argv[1];
    if (!modeArg.empty() && modeArg[0] == '-') {
        modeArg = modeArg.substr(1);
    }

    static const std::set<std::string> kModes = {"all",    "ocr",   "model", "classify",
                                                 "layout", "table", "uvdoc", "formula"};
    if (kModes.find(modeArg) == kModes.end()) {
        printUsage(argv[0]);
        return 1;
    }

    std::string engineDir;
    std::string imagePath;
    std::string dictPath;
    std::string formulaYaml;
    std::string ocrPrefix;

    if (modeArg == "ocr" && (argc == 8 || argc == 9)) {
        if (!isOcrVariantToken(argv[2])) {
            printUsage(argv[0]);
            return 1;
        }
        dictPath = argc == 9 ? argv[8] : "";
        runExplicitOcrSystem(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], dictPath);
        return 0;
    }

    if (modeArg == "model") {
        if (argc != 5) {
            printUsage(argv[0]);
            return 1;
        }
        runExplicitModelSystem(argv[2], argv[3], argv[4]);
        return 0;
    }

    if (modeArg == "formula" && (argc == 6 || argc == 7)) {
        formulaYaml = argc == 7 ? argv[6] : "";
        runExplicitFormulaSystem(argv[2], argv[3], argv[4], argv[5], formulaYaml);
        return 0;
    }

    if (modeArg == "all" || modeArg == "ocr") {
        if (argc < 5 || !isOcrVariantToken(argv[2])) {
            printUsage(argv[0]);
            return 1;
        }
        ocrPrefix = ocrPrefixFromVariant(argv[2]);
        engineDir = argv[3];
        imagePath = argv[4];
        int nextArg = 5;
        if (argc > nextArg) {
            dictPath = argv[nextArg++];
        }
        if (modeArg == "all" && argc > nextArg) {
            formulaYaml = argv[nextArg++];
        }
        if (argc > nextArg) {
            printUsage(argv[0]);
            return 1;
        }
    } else {
        if (argc < 4) {
            printUsage(argv[0]);
            return 1;
        }
        engineDir = argv[2];
        imagePath = argv[3];
        if (modeArg == "formula" && argc >= 5) {
            formulaYaml = argv[4];
        }
        if ((modeArg == "formula" && argc > 5) || (modeArg != "formula" && argc > 4)) {
            printUsage(argv[0]);
            return 1;
        }
    }

    runSystem(modeArg, engineDir, imagePath, dictPath, formulaYaml, ocrPrefix);
    return 0;
}
