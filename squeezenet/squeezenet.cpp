#include <NvInfer.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>
#include "logging.h"
#include "utils.h"

// stuff we know about squeezenet
static constexpr const int N = 1;
static constexpr const int INPUT_H = 224;
static constexpr const int INPUT_W = 224;
static constexpr const int SIZES[] = {3 * INPUT_H * INPUT_W, N * 1000};
static constexpr const char* NAMES[] = {"data", "prob"};
static constexpr const bool TRT_PREPROCESS = TRT_VERSION >= 8510 ? true : false;
static constexpr const float mean[3] = {0.485f, 0.456f, 0.406f};
static constexpr const float stdv[3] = {0.229f, 0.224f, 0.225f};

static constexpr const char* WTS_PATH = "../models/squeezenet.wts";
static constexpr const char* ENGINE_PATH = "../models/squeezenet.engine";
static constexpr const char* LABELS_PATH = "../assets/imagenet1000_clsidx_to_labels.txt";

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;

static Logger gLogger;

ILayer* fire(INetworkDefinition* network, WeightMap& m, ITensor& input, const std::string& lname,
             int32_t squeeze_planes, int32_t e1x1_planes, int32_t e3x3_planes) {
    auto* conv1 = network->addConvolutionNd(input, squeeze_planes, DimsHW{1, 1}, m[lname + "squeeze.weight"],
                                            m[lname + "squeeze.bias"]);
    assert(conv1);
    auto* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU)->getOutput(0);

    std::string _c = lname + "expand1x1";
    auto* conv2 = network->addConvolutionNd(*relu1, e1x1_planes, DimsHW{1, 1}, m[_c + ".weight"], m[_c + ".bias"]);
    assert(conv2);
    auto* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    _c = lname + "expand3x3";
    auto* conv3 = network->addConvolutionNd(*relu1, e3x3_planes, DimsHW{3, 3}, m[_c + ".weight"], m[_c + ".bias"]);
    assert(conv3);
    conv3->setPaddingNd(DimsHW{1, 1});
    auto* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    ITensor* inputTensors[] = {relu2->getOutput(0), relu3->getOutput(0)};
    auto* concat = network->addConcatenation(inputTensors, 2);
    assert(concat);
    return concat;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    auto weightMap = loadWeights(WTS_PATH);
#if TRT_VERSION >= 10000
    auto* network = builder->createNetworkV2(0);
#else
    auto* network = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

    ITensor* data{nullptr};
    if constexpr (TRT_PREPROCESS) {
#if TRT_VERSION > 8510
        dt = DataType::kUINT8;
#else
        dt = DataType::kINT8;
#endif
        data = network->addInput(NAMES[0], dt, Dims4{N, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(network, *data, true, mean, stdv);
        data = trans->getOutput(0);
    } else {
        data = network->addInput(NAMES[0], dt, Dims4{N, 3, INPUT_H, INPUT_W});
    }
    assert(data);

    auto* conv1 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["features.0.weight"],
                                            weightMap["features.0.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    auto* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    auto* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    auto* cat1 = fire(network, weightMap, *pool1->getOutput(0), "features.3.", 16, 64, 64);
    cat1 = fire(network, weightMap, *cat1->getOutput(0), "features.4.", 16, 64, 64);

    auto* pool2 = network->addPoolingNd(*cat1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});
    pool2->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    // pool2->setPostPadding(DimsHW{1, 1});

    cat1 = fire(network, weightMap, *pool2->getOutput(0), "features.6.", 32, 128, 128);
    cat1 = fire(network, weightMap, *cat1->getOutput(0), "features.7.", 32, 128, 128);

    auto* pool3 = network->addPoolingNd(*cat1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool3);
    pool3->setStrideNd(DimsHW{2, 2});
    pool3->setPostPadding(DimsHW{1, 1});
    pool3->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    cat1 = fire(network, weightMap, *pool3->getOutput(0), "features.9.", 48, 192, 192);
    cat1 = fire(network, weightMap, *cat1->getOutput(0), "features.10.", 48, 192, 192);
    cat1 = fire(network, weightMap, *cat1->getOutput(0), "features.11.", 64, 256, 256);
    cat1 = fire(network, weightMap, *cat1->getOutput(0), "features.12.", 64, 256, 256);

    // classifier
    auto* conv2 = network->addConvolutionNd(*cat1->getOutput(0), 1000, DimsHW{1, 1}, weightMap["classifier.1.weight"],
                                            weightMap["classifier.1.bias"]);
    assert(conv2);
    auto* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    auto* pool4 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{14, 14});
    assert(pool4);

    pool4->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*pool4->getOutput(0));

    // Build engine
#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    auto* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete network;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    auto* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif
    std::cout << "build out" << std::endl;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(int32_t N, IRuntime* runtime, IHostMemory** modelStream) {
    // Create builder
    auto* builder = createInferBuilder(gLogger);
    auto* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    auto* engine = createEngine(N, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

#if TRT_VERSION >= 8000
    delete engine;
    delete config;
    delete builder;
#else
    engine->destroy();
    config->destroy();
    builder->destroy();
#endif
}

std::vector<std::vector<float>> doInference(IExecutionContext& context, void* input, int32_t batch_size) {
    const auto& engine = context.getEngine();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<void*> buffers;

#if TRT_VERSION >= 8000
    const int32_t nIO = engine.getNbIOTensors();
#else
    const int32_t nIO = engine.getNbBindings();
#endif

    buffers.resize(nIO);
    for (auto i = 0; i < nIO; ++i) {
        std::size_t size = 0;
#if TRT_VERSION >= 8000
        const auto* tensor_name = engine.getIOTensorName(i);
        auto s = getSize(engine.getTensorDataType(tensor_name));
        size = s * batch_size * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
        context.setTensorAddress(tensor_name, buffers[i]);
#else
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        auto s = getSize(engine.getBindingDataType(idx));
        assert(idx == i);
        size = s * batch_size * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
#endif
    }

#if TRT_VERSION >= 8000
    assert(context.enqueueV3(stream));
#else
    assert(context.enqueueV2(buffers.data(), stream, nullptr));
#endif

    std::vector<std::vector<float>> prob;
    for (int i = 1; i < nIO; ++i) {
        std::vector<float> tmp(batch_size * SIZES[i], std::nan(""));
        std::size_t size = batch_size * SIZES[i] * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(tmp);
    }
    CHECK(cudaStreamSynchronize(stream));

    cudaStreamDestroy(stream);
    for (auto i = 0; i < nIO; ++i) {
        CHECK(cudaFree(buffers[i]));
    }
    return prob;
}

int main(int argc, char** argv) {
    checkTrtEnv();
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./squeezenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./squeezenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    auto* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, runtime, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(ENGINE_PATH, std::ios::binary | std::ios::trunc);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
#if TRT_VERSION >= 8000
        delete modelStream;
#else
        modelStream->destroy();
#endif
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file(ENGINE_PATH, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

#if TRT_VERSION >= 8000
    auto* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    auto* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    auto* context = engine->createExecutionContext();
    assert(context != nullptr);

    void* input = nullptr;
    std::vector<float> flat_img;
    cv::Mat img;
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        img = cv::imread("../assets/cats.jpg", cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        img = cv::imread("../assets/cats.jpg", cv::IMREAD_COLOR);
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }

    for (int32_t i = 0; i < 100; ++i) {
        auto _start = std::chrono::system_clock::now();
        auto prob = doInference(*context, input, N);
        auto _end = std::chrono::system_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
        std::cout << "Execution time: " << _time << "us" << std::endl;

        for (auto vector : prob) {
            int idx = 0;
            for (auto v : vector) {
                std::cout << std::setprecision(4) << v << ", " << std::flush;
                if (++idx > 20) {
                    std::cout << "\n====" << std::endl;
                    break;
                }
            }
        }

        if (i == 99) {
            std::cout << "prediction result: " << std::endl;
            auto labels = loadImagenetLabelMap(LABELS_PATH);
            int _top = 0;
            for (auto& [idx, logits] : topk(prob[0], 3)) {
                std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits
                          << ", label: " << labels[idx] << std::endl;
            }
        }
    }

    delete[] trtModelStream;
    // Destroy the engine
#if TRT_VERSION >= 8000
    delete context;
    delete engine;
    delete runtime;
#else
    context->destroy();
    engine->destroy();
    runtime->destroy();
#endif
    return 0;
}
