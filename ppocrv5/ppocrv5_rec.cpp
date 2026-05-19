#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

Logger gLogger;
using namespace nvinfer1;

namespace {

int64_t volume(const Dims& dims) {
    int64_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        count *= dims.d[i];
    }
    return count;
}

void printTopIds(const float* output, int timeSteps, int classCount) {
    if (!std::getenv("PPOCRV5_DEBUG_CTC")) {
        return;
    }
    std::cout << "top_ids=";
    int last = -1;
    for (int t = 0; t < timeSteps; ++t) {
        const float* row = output + t * classCount;
        int index = static_cast<int>(std::max_element(row, row + classCount) - row);
        if (index != last) {
            std::cout << index << ",";
        }
        last = index;
    }
    std::cout << std::endl;
}

void serializeEngine(const std::string& wtsName, const std::string& engineName) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* engine = buildEnginePPOCRv5Rec(builder, config, DataType::kFLOAT, wtsName);
    saveEngine(engineName, engine);
    delete engine;
    delete config;
    delete builder;
}

void inferImages(const std::string& engineName, const std::string& imagePath, const std::string& dictPath) {
    auto dict = loadDictionary(dictPath);
    auto engineData = readBinaryFile(engineName);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();
    std::string inputName = findIOTensorName(engine, TensorIOMode::kINPUT);
    std::string outputName = findIOTensorName(engine, TensorIOMode::kOUTPUT);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (const auto& file : listImages(imagePath)) {
        cv::Mat crop = cv::imread(file);
        auto meta = preprocessRec(crop, kRecMaxW);
        Dims4 inputDims{1, 3, meta.input_h, meta.input_w};
        context->setInputShape(inputName.c_str(), inputDims);
        Dims outputDims = context->getTensorShape(outputName.c_str());

        float* inputDevice = nullptr;
        float* outputDevice = nullptr;
        std::vector<float> outputHost(volume(outputDims));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputDevice), meta.chw.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputDevice), outputHost.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inputDevice, meta.chw.data(), meta.chw.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        context->setTensorAddress(inputName.c_str(), inputDevice);
        context->setTensorAddress(outputName.c_str(), outputDevice);
        context->enqueueV3(stream);
        CUDA_CHECK(cudaMemcpyAsync(outputHost.data(), outputDevice, outputHost.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int timeSteps = outputDims.nbDims == 3 ? outputDims.d[1] : outputDims.d[0];
        int classCount = outputDims.nbDims == 3 ? outputDims.d[2] : outputDims.d[1];
        printTopIds(outputHost.data(), timeSteps, classCount);
        auto result = ctcDecode(outputHost.data(), timeSteps, classCount, dict);
        std::cout << file << " conf=" << result.score << " text=" << result.text << std::endl;

        CUDA_CHECK(cudaFree(inputDevice));
        CUDA_CHECK(cudaFree(outputDevice));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0] << " -s rec.wts rec.engine\n  " << argv[0]
                  << " -d rec.engine image_or_dir rec_dict.txt|rec_inference.yml" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "-s" && argc == 4) {
        serializeEngine(argv[2], argv[3]);
        return 0;
    }
    if (mode == "-d" && argc == 5) {
        inferImages(argv[2], argv[3], argv[4]);
        return 0;
    }

    std::cerr << "Invalid arguments" << std::endl;
    return 1;
}
