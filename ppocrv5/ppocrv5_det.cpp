#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "ppocrv5_db_layer.h"
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

void serializeEngine(const std::string& wtsName, const std::string& engineName) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* engine = buildEnginePPOCRv5Det(builder, config, DataType::kFLOAT, wtsName);
    saveEngine(engineName, engine);
    delete engine;
    delete config;
    delete builder;
}

void inferImages(const std::string& engineName, const std::string& imagePath) {
    ppocrv5EnsureDbPlugin();

    auto engineData = readBinaryFile(engineName);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        throw std::runtime_error("failed to deserialize engine: " + engineName);
    }
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        delete engine;
        delete runtime;
        throw std::runtime_error("failed to create execution context: " + engineName);
    }
    std::string inputName = findIOTensorName(engine, TensorIOMode::kINPUT);
    std::string outputName = findIOTensorName(engine, TensorIOMode::kOUTPUT);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        auto meta = preprocessDet(image);
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

        int outH = outputDims.d[2];
        int outW = outputDims.d[3];
        auto boxes = dbPostprocess(outputHost.data(), outH, outW, meta);
        std::vector<RecResult> empty(boxes.size());
        drawOcrResult(image, boxes, empty);
        std::string outPath = makeOutputPath(file, "_ppocrv5_det.jpg");
        cv::imwrite(outPath, image);
        std::cout << file << " boxes=" << boxes.size() << " output=" << outPath << std::endl;

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
        std::cerr << "Usage:\n  " << argv[0] << " -s det.wts det.engine\n  " << argv[0] << " -d det.engine image_or_dir"
                  << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "-s" && argc == 4) {
        serializeEngine(argv[2], argv[3]);
        return 0;
    }
    if (mode == "-d" && argc == 4) {
        inferImages(argv[2], argv[3]);
        return 0;
    }

    std::cerr << "Invalid arguments" << std::endl;
    return 1;
}
