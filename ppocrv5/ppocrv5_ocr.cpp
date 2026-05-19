#include <algorithm>
#include <cctype>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

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

std::string ocrPrefixFromVariant(const std::string& variant) {
    if (variant == "m") {
        return "ppocrv5_mobile";
    }
    if (variant == "s") {
        return "ppocrv5_server";
    }
    throw std::runtime_error("unknown OCR variant, use m or s: " + variant);
}

std::string lowerText(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

std::string inferOcrRole(const std::string& wtsPath, const std::string& enginePath) {
    std::string tag = lowerText(basenameNoExt(wtsPath) + " " + basenameNoExt(enginePath));
    bool isDet = tag.find("det") != std::string::npos;
    bool isRec = tag.find("rec") != std::string::npos;
    if (isDet == isRec) {
        throw std::runtime_error("cannot infer OCR role, use det or rec in wts/engine name: " + wtsPath + " -> " +
                                 enginePath);
    }
    return isDet ? "det" : "rec";
}

void warnIfSystemNameMismatch(const std::string& enginePath, const std::string& variant, const std::string& role) {
    if (variant.empty()) {
        return;
    }

    std::string prefix = ocrPrefixFromVariant(variant);
    std::string expected = prefix + "_" + role;
    if (basenameNoExt(enginePath) != expected) {
        std::cerr << "Warning: ppocr_system loads OCR engines by name. For variant " << variant << ", use " << expected
                  << ".engine in the engine_dir." << std::endl;
    }
}

void serializeOne(const std::string& wtsPath, const std::string& enginePath, const std::string& variant) {
    std::string role = inferOcrRole(wtsPath, enginePath);
    warnIfSystemNameMismatch(enginePath, variant, role);

    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* plan = role == "det" ? buildEnginePPOCRv5Det(builder, config, DataType::kFLOAT, wtsPath, variant)
                                      : buildEnginePPOCRv5Rec(builder, config, DataType::kFLOAT, wtsPath, variant);
    saveEngine(enginePath, plan);
    delete plan;
    delete config;
    delete builder;
}

void serializeBoth(const std::string& detWts, const std::string& recWts, const std::string& detEngine,
                   const std::string& recEngine, const std::string& variant = std::string()) {
    IBuilder* detBuilder = createInferBuilder(gLogger);
    IBuilderConfig* detConfig = detBuilder->createBuilderConfig();
    IHostMemory* detPlan = variant.empty()
                                   ? buildEnginePPOCRv5Det(detBuilder, detConfig, DataType::kFLOAT, detWts)
                                   : buildEnginePPOCRv5Det(detBuilder, detConfig, DataType::kFLOAT, detWts, variant);
    saveEngine(detEngine, detPlan);
    delete detPlan;
    delete detConfig;
    delete detBuilder;

    IBuilder* recBuilder = createInferBuilder(gLogger);
    IBuilderConfig* recConfig = recBuilder->createBuilderConfig();
    IHostMemory* recPlan = variant.empty()
                                   ? buildEnginePPOCRv5Rec(recBuilder, recConfig, DataType::kFLOAT, recWts)
                                   : buildEnginePPOCRv5Rec(recBuilder, recConfig, DataType::kFLOAT, recWts, variant);
    saveEngine(recEngine, recPlan);
    delete recPlan;
    delete recConfig;
    delete recBuilder;
}

ICudaEngine* loadEngine(IRuntime* runtime, const std::string& path) {
    ppocrv5EnsureDbPlugin();

    auto data = readBinaryFile(path);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), data.size());
    if (!engine) {
        throw std::runtime_error("failed to deserialize engine: " + path);
    }
    return engine;
}

std::vector<TextBox> runDet(IExecutionContext* context, const std::string& inputName, const std::string& outputName,
                            const cv::Mat& image, cudaStream_t stream) {
    auto meta = preprocessDet(image);
    Dims4 inputDims{1, 3, meta.input_h, meta.input_w};
    context->setInputShape(inputName.c_str(), inputDims);
    Dims outputDims = context->getTensorShape(outputName.c_str());

    float* inputDevice = nullptr;
    float* outputDevice = nullptr;
    std::vector<float> outputHost(volume(outputDims));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputDevice), meta.chw.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputDevice), outputHost.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(inputDevice, meta.chw.data(), meta.chw.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
    context->setTensorAddress(inputName.c_str(), inputDevice);
    context->setTensorAddress(outputName.c_str(), outputDevice);
    context->enqueueV3(stream);
    CUDA_CHECK(cudaMemcpyAsync(outputHost.data(), outputDevice, outputHost.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(inputDevice));
    CUDA_CHECK(cudaFree(outputDevice));

    return dbPostprocess(outputHost.data(), outputDims.d[2], outputDims.d[3], meta);
}

RecResult runRec(IExecutionContext* context, const std::string& inputName, const std::string& outputName,
                 const cv::Mat& crop, const std::vector<std::string>& dict, cudaStream_t stream) {
    auto meta = preprocessRec(crop, kRecMaxW);
    Dims4 inputDims{1, 3, meta.input_h, meta.input_w};
    context->setInputShape(inputName.c_str(), inputDims);
    Dims outputDims = context->getTensorShape(outputName.c_str());

    float* inputDevice = nullptr;
    float* outputDevice = nullptr;
    std::vector<float> outputHost(volume(outputDims));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&inputDevice), meta.chw.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputDevice), outputHost.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(inputDevice, meta.chw.data(), meta.chw.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
    context->setTensorAddress(inputName.c_str(), inputDevice);
    context->setTensorAddress(outputName.c_str(), outputDevice);
    context->enqueueV3(stream);
    CUDA_CHECK(cudaMemcpyAsync(outputHost.data(), outputDevice, outputHost.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(inputDevice));
    CUDA_CHECK(cudaFree(outputDevice));

    int timeSteps = outputDims.nbDims == 3 ? outputDims.d[1] : outputDims.d[0];
    int classCount = outputDims.nbDims == 3 ? outputDims.d[2] : outputDims.d[1];
    return ctcDecode(outputHost.data(), timeSteps, classCount, dict);
}

void runOcr(const std::string& detEngine, const std::string& recEngine, const std::string& imagePath,
            const std::string& userDictPath = std::string()) {
    std::string dictPath =
            userDictPath.empty() ? siblingPath(recEngine, basenameNoExt(recEngine) + "_dict.txt") : userDictPath;
    if (!fileExists(dictPath)) {
        std::string recStem = basenameNoExt(recEngine);
        size_t recPos = recStem.find("_rec");
        if (recPos != std::string::npos) {
            dictPath = siblingPath(recEngine, recStem.substr(0, recPos + 4) + "_dict.txt");
        }
    }
    if (!fileExists(dictPath)) {
        dictPath = siblingPath(recEngine, "ppocrv5_rec_dict.txt");
    }
    if (!fileExists(dictPath)) {
        std::string recStem = basenameNoExt(recEngine);
        dictPath = recStem.find("server") != std::string::npos ? "../official_models/PP-OCRv5_server_rec/inference.yml"
                                                               : "../official_models/PP-OCRv5_mobile_rec/inference.yml";
    }
    std::vector<std::string> dict = loadDictionary(dictPath);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* det = loadEngine(runtime, detEngine);
    ICudaEngine* rec = loadEngine(runtime, recEngine);
    IExecutionContext* detContext = det->createExecutionContext();
    IExecutionContext* recContext = rec->createExecutionContext();
    std::string detInputName = findIOTensorName(det, TensorIOMode::kINPUT);
    std::string detOutputName = findIOTensorName(det, TensorIOMode::kOUTPUT);
    std::string recInputName = findIOTensorName(rec, TensorIOMode::kINPUT);
    std::string recOutputName = findIOTensorName(rec, TensorIOMode::kOUTPUT);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (const auto& file : listImages(imagePath)) {
        cv::Mat image = cv::imread(file);
        std::vector<TextBox> boxes = runDet(detContext, detInputName, detOutputName, image, stream);
        std::vector<RecResult> results;
        for (const auto& box : boxes) {
            cv::Mat crop = cropTextBox(image, box);
            results.push_back(runRec(recContext, recInputName, recOutputName, crop, dict, stream));
        }

        for (size_t i = 0; i < boxes.size(); ++i) {
            std::cout << file << " box=";
            for (int j = 0; j < 4; ++j) {
                std::cout << boxes[i].points[j].x << "," << boxes[i].points[j].y;
                if (j != 3) {
                    std::cout << " ";
                }
            }
            std::cout << " conf=" << boxes[i].score << " text=" << results[i].text << std::endl;
        }

        cv::Mat vis = image.clone();
        drawOcrResult(vis, boxes, results);
        std::string outPath = makeOutputPath(file, "_ppocrv5_ocr.jpg");
        cv::imwrite(outPath, vis);
        std::cout << file << " boxes=" << boxes.size() << " output=" << outPath << std::endl;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    delete recContext;
    delete detContext;
    delete rec;
    delete det;
    delete runtime;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0] << " -s det_or_rec.wts det_or_rec.engine m|s\n  " << argv[0]
                  << " -d det.engine rec.engine image_or_dir [rec_dict.txt|rec_inference.yml]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    try {
        if (mode == "-s" && argc == 5) {
            serializeOne(argv[2], argv[3], argv[4]);
            return 0;
        }
        if (mode == "-d" && (argc == 5 || argc == 6)) {
            runOcr(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : std::string());
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cerr << "Invalid arguments" << std::endl;
    return 1;
}
