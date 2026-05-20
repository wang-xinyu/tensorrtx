#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "ppocrv5_db_layer.h"
#include "ppocrv5_rtdetr_layer.h"
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

bool isFormulaNet(const std::string& path) {
    return path.find("formulanet") != std::string::npos || path.find("FormulaNet") != std::string::npos;
}

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

std::string dimsToString(const Dims& dims) {
    std::string text = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) {
            text += ",";
        }
        text += std::to_string(dims.d[i]);
    }
    text += "]";
    return text;
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

std::string appendBeforeExtension(const std::string& path, const std::string& suffix) {
    auto slash = path.find_last_of("/\\");
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path + suffix;
    }
    return path.substr(0, dot) + suffix + path.substr(dot);
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

void serializeEngine(const std::string& wtsName, const std::string& engineName) {
    if (isFormulaNet(wtsName)) {
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
        return;
    }

    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* engine = buildEnginePPOCRv5Model(builder, config, DataType::kFLOAT, wtsName);
    saveEngine(engineName, engine);
    delete engine;
    delete config;
    delete builder;
}

void inferModel(const std::string& modelName, const std::string& engineName, const std::string& imagePath) {
    if (isFormulaNet(modelName) || isFormulaNet(engineName)) {
        throw std::runtime_error("use ppocrv5_formula -d for FormulaNet encoder/decoder validation");
    }
    IRuntime* runtime = createInferRuntime(gLogger);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    EngineSession session(runtime, engineName);
    std::vector<std::string> images = listImages(imagePath);
    if (images.empty()) {
        throw std::runtime_error("no validation images found: " + imagePath);
    }
    for (const auto& file : images) {
        cv::Mat image = cv::imread(file);
        if (image.empty()) {
            throw std::runtime_error("failed to read validation image: " + file);
        }
        auto inputs = makeGenericInputs(session, modelName, image);
        auto outputs = session.infer(inputs, stream);
        printTensorSummary(file, modelName, outputs);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete runtime;
}

void printUsage(const char* app) {
    std::cerr << "Usage:\n"
              << "  " << app << " -s model.wts model.engine\n"
              << "  " << app << " -d model.engine image_or_dir\n"
              << "  " << app << " -d model_name model.engine image_or_dir" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    try {
        if (mode == "-s" && argc == 4) {
            serializeEngine(argv[2], argv[3]);
            return 0;
        }
        if (mode == "-d" && argc == 4) {
            inferModel(basenameNoExt(argv[2]), argv[2], argv[3]);
            return 0;
        }
        if (mode == "-d" && argc == 5) {
            inferModel(argv[2], argv[3], argv[4]);
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    printUsage(argv[0]);
    return 1;
}
