#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "cuda_utils.h"
#include "logging.h"
#include "ppocrv5_db_layer.h"
#include "ppocrv5_rtdetr_layer.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;

namespace {

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

int elementSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT8:
        case DataType::kBOOL:
            return 1;
        case DataType::kINT32:
            return 4;
        default:
            throw std::runtime_error("unsupported TensorRT data type for dump");
    }
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
        default:
            return "unknown";
    }
}

std::vector<int> parseShape(const std::string& text) {
    std::vector<int> dims;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find_first_of("xX,", start);
        std::string token = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty()) {
            dims.push_back(std::stoi(token));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    if (dims.empty()) {
        throw std::runtime_error("input shape is empty");
    }
    return dims;
}

std::vector<std::string> split(const std::string& text, char delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find(delim, start);
        parts.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

Dims makeDims(const std::vector<int>& shape) {
    Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = shape[i];
    }
    return dims;
}

std::string shapeText(const Dims& dims) {
    std::string text;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i != 0) {
            text += "x";
        }
        text += std::to_string(dims.d[i]);
    }
    return text;
}

std::vector<float> makeDeterministicInput(int64_t count) {
    std::vector<float> data(count);
    for (int64_t i = 0; i < count; ++i) {
        data[i] = std::sin(static_cast<float>(i % 251) * 0.013f) * 0.5f + 0.25f;
    }
    return data;
}

std::vector<float> makeInputData(const std::string& name, const Dims& dims, const std::vector<std::string>& inputNames,
                                 const std::vector<Dims>& inputDims) {
    int64_t count = volume(dims);
    if (name == "scale_factor" && count == 2) {
        return std::vector<float>{1.0f, 1.0f};
    }
    if (name == "im_shape" && count == 2) {
        for (size_t i = 0; i < inputNames.size(); ++i) {
            if (inputNames[i] == "image" && inputDims[i].nbDims == 4) {
                return std::vector<float>{static_cast<float>(inputDims[i].d[2]), static_cast<float>(inputDims[i].d[3])};
            }
        }
    }
    return makeDeterministicInput(count);
}

bool hasDynamicDim(const Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

class DumpOutputAllocator : public IOutputAllocator {
   public:
    ~DumpOutputAllocator() override {
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

std::vector<float> readFloatInput(const std::string& path, int64_t count) {
    auto bytes = readBinaryFile(path);
    if (bytes.size() != static_cast<size_t>(count * sizeof(float))) {
        throw std::runtime_error("input binary size does not match requested shape: " + path);
    }
    std::vector<float> data(count);
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return data;
}

void writeBinary(const std::string& path, const void* data, size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    file.write(reinterpret_cast<const char*>(data), bytes);
}

void writeShapeFile(const std::string& path, const std::string& name, DataType dtype, const Dims& dims) {
    std::ofstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open shape file: " + path);
    }
    file << "name " << name << "\n";
    file << "dtype " << dtypeName(dtype) << "\n";
    file << "shape " << shapeText(dims) << "\n";
    file << "count " << volume(dims) << "\n";
}

std::vector<std::string> ioTensorNames(ICudaEngine* engine, TensorIOMode mode) {
    std::vector<std::string> names;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == mode) {
            names.push_back(name);
        }
    }
    return names;
}

std::vector<Dims> parseInputDims(const std::vector<std::string>& inputNames, const std::string& spec) {
    std::vector<Dims> inputDims(inputNames.size());
    if (spec.find('=') == std::string::npos) {
        if (inputNames.size() != 1) {
            throw std::runtime_error("multi-input engines need name=shape;name=shape specs");
        }
        inputDims[0] = makeDims(parseShape(spec));
        return inputDims;
    }

    std::map<std::string, Dims> dimsByName;
    for (const auto& item : split(spec, ';')) {
        if (item.empty()) {
            continue;
        }
        size_t eq = item.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("invalid input spec item: " + item);
        }
        dimsByName[item.substr(0, eq)] = makeDims(parseShape(item.substr(eq + 1)));
    }

    for (size_t i = 0; i < inputNames.size(); ++i) {
        auto it = dimsByName.find(inputNames[i]);
        if (it == dimsByName.end()) {
            throw std::runtime_error("missing input spec for tensor: " + inputNames[i]);
        }
        inputDims[i] = it->second;
    }
    return inputDims;
}

void dumpEngine(const std::string& enginePath, const std::string& shapeArg, const std::string& outPrefix,
                const std::string& inputPath) {
    ppocrv5EnsureDbPlugin();
    ppocrv5EnsureRtDetrPlugin();

    auto engineData = readBinaryFile(enginePath);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        throw std::runtime_error("failed to deserialize engine: " + enginePath);
    }
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        delete engine;
        delete runtime;
        throw std::runtime_error("failed to create execution context: " + enginePath);
    }

    auto inputs = ioTensorNames(engine, TensorIOMode::kINPUT);
    auto outputs = ioTensorNames(engine, TensorIOMode::kOUTPUT);
    if (outputs.empty()) {
        throw std::runtime_error("engine has no output tensors");
    }

    std::vector<Dims> inputDims = parseInputDims(inputs, shapeArg);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<std::vector<float>> inputHosts(inputs.size());
    std::vector<void*> inputDevices(inputs.size(), nullptr);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!context->setInputShape(inputs[i].c_str(), inputDims[i])) {
            throw std::runtime_error("failed to set input shape for tensor: " + inputs[i]);
        }
        int64_t inputCount = volume(inputDims[i]);
        std::string itemInputPath;
        if (!inputPath.empty()) {
            itemInputPath = inputs.size() == 1 ? inputPath : inputPath + "." + std::to_string(i) + ".bin";
        }
        inputHosts[i] = itemInputPath.empty() ? makeInputData(inputs[i], inputDims[i], inputs, inputDims)
                                              : readFloatInput(itemInputPath, inputCount);
        if (inputPath.empty()) {
            std::string prefix = inputs.size() == 1 ? outPrefix + ".input" : outPrefix + ".input." + std::to_string(i);
            writeBinary(prefix + ".bin", inputHosts[i].data(), inputHosts[i].size() * sizeof(float));
            writeShapeFile(prefix + ".shape", inputs[i], DataType::kFLOAT, inputDims[i]);
        }
        CUDA_CHECK(cudaMalloc(&inputDevices[i], inputHosts[i].size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inputDevices[i], inputHosts[i].data(), inputHosts[i].size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        context->setTensorAddress(inputs[i].c_str(), inputDevices[i]);
    }

    std::vector<void*> outputDevices(outputs.size(), nullptr);
    std::vector<std::unique_ptr<DumpOutputAllocator>> outputAllocators(outputs.size());
    std::vector<std::vector<char>> outputHosts(outputs.size());
    std::vector<Dims> outputDims(outputs.size());
    std::vector<bool> dynamicOutputs(outputs.size(), false);
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputDims[i] = context->getTensorShape(outputs[i].c_str());
        DataType dtype = engine->getTensorDataType(outputs[i].c_str());
        dynamicOutputs[i] = hasDynamicDim(outputDims[i]);
        if (dynamicOutputs[i]) {
            outputAllocators[i].reset(new DumpOutputAllocator());
            context->setOutputAllocator(outputs[i].c_str(), outputAllocators[i].get());
        } else {
            size_t bytes = static_cast<size_t>(volume(outputDims[i]) * elementSize(dtype));
            outputHosts[i].resize(bytes);
            CUDA_CHECK(cudaMalloc(&outputDevices[i], bytes));
            context->setTensorAddress(outputs[i].c_str(), outputDevices[i]);
        }
    }

    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("TensorRT enqueue failed");
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        DataType dtype = engine->getTensorDataType(outputs[i].c_str());
        if (dynamicOutputs[i]) {
            outputDims[i] = outputAllocators[i]->dims();
            size_t bytes = static_cast<size_t>(volume(outputDims[i]) * elementSize(dtype));
            outputHosts[i].resize(bytes);
            CUDA_CHECK(cudaMemcpyAsync(outputHosts[i].data(), outputAllocators[i]->ptr(), bytes, cudaMemcpyDeviceToHost,
                                       stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(outputHosts[i].data(), outputDevices[i], outputHosts[i].size(),
                                       cudaMemcpyDeviceToHost, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string prefix = outputs.size() == 1 ? outPrefix : outPrefix + "." + std::to_string(i);
        DataType dtype = engine->getTensorDataType(outputs[i].c_str());
        writeBinary(prefix + ".bin", outputHosts[i].data(), outputHosts[i].size());
        writeShapeFile(prefix + ".shape", outputs[i], dtype, outputDims[i]);
        std::cout << outputs[i] << " dtype=" << dtypeName(dtype) << " shape=" << shapeText(outputDims[i])
                  << " file=" << prefix << ".bin" << std::endl;
    }

    for (void* ptr : outputDevices) {
        CUDA_CHECK(cudaFree(ptr));
    }
    for (void* ptr : inputDevices) {
        CUDA_CHECK(cudaFree(ptr));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "Usage:\n  " << argv[0] << " -d model.engine 1x3x224x224 out_prefix [input.bin]\n"
                  << "  " << argv[0] << " -d model.engine 'image=1x3x800x800;im_shape=1x2;scale_factor=1x2' out_prefix"
                  << std::endl;
        return 1;
    }
    if (std::string(argv[1]) != "-d") {
        std::cerr << "Invalid arguments" << std::endl;
        return 1;
    }
    try {
        dumpEngine(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : "");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
