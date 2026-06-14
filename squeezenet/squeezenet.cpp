#include <NvInfer.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "logging.h"
#include "utils.h"

static constexpr const std::size_t WORKSPACE_SIZE = 16 << 20;

static constexpr const int64_t N = 1;
static constexpr const int32_t INPUT_H = 224;
static constexpr const int32_t INPUT_W = 224;
static constexpr const std::array<int32_t, 2> SIZES = {3 * INPUT_H * INPUT_W, N * 1000};
static constexpr const std::array<const char*, 2> NAMES = {"data", "prob"};
static constexpr const bool TRT_PREPROCESS = TRT_VERSION_GE(8, 5, 1) ? true : false;
static constexpr const std::array<const float, 3> mean = {0.485f, 0.456f, 0.406f};
static constexpr const std::array<const float, 3> stdv = {0.229f, 0.224f, 0.225f};

static constexpr const char* WTS_PATH = "models/squeezenet.wts";
static constexpr const char* ENGINE_PATH = "models/squeezenet.engine";
static constexpr const char* LABELS_PATH = "assets/imagenet1000_clsidx_to_labels.txt";

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;
using NDCF = nvinfer1::NetworkDefinitionCreationFlag;

static Logger gLogger;

static auto fire(INetworkDefinition* network, WeightMap& weights, ITensor& input, const std::string& lname,
                 int32_t squeeze_planes, int32_t e1x1_planes, int32_t e3x3_planes) -> ILayer* {
    auto* conv1 = network->addConvolutionNd(input, squeeze_planes, DimsHW{1, 1}, weights.at(lname + "squeeze.weight"),
                                            weights.at(lname + "squeeze.bias"));
    assert(conv1);
    auto* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    std::string _c = lname + "expand1x1";
    auto* conv2 = network->addConvolutionNd(*relu1->getOutput(0), e1x1_planes, DimsHW{1, 1}, weights.at(_c + ".weight"),
                                            weights.at(_c + ".bias"));
    assert(conv2);
    auto* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    _c = lname + "expand3x3";
    auto* conv3 = network->addConvolutionNd(*relu1->getOutput(0), e3x3_planes, DimsHW{3, 3}, weights.at(_c + ".weight"),
                                            weights.at(_c + ".bias"));
    assert(conv3);
    conv3->setPaddingNd(DimsHW{1, 1});
    auto* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    std::array<ITensor*, 2> inputTensors = {relu2->getOutput(0), relu3->getOutput(0)};
    auto* concat = network->addConcatenation(inputTensors.data(), 2);
    assert(concat);
    return concat;
}

// Create the engine using only the API and not any parser.
static auto createEngine(int32_t batch_size, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config,
                         DataType dt) -> ICudaEngine* {
    auto weightMap = loadWeights(WTS_PATH);

#if TRT_VERSION_GE(10, 12, 0)
    auto flag = 1U << static_cast<int>(NDCF::kSTRONGLY_TYPED);
#elif TRT_VERSION_GE(10, 0, 0)
    auto flag = 0U;
#else
    auto flag = 1U << static_cast<int>(NDCF::kEXPLICIT_BATCH);
#endif
    auto* network = builder->createNetworkV2(flag);

    ITensor* data{nullptr};
    if constexpr (TRT_PREPROCESS) {
        dt = DataType::kUINT8;
        data = network->addInput(NAMES[0], dt, Dims4{batch_size, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(network, *data, true, mean, stdv);
        data = trans->getOutput(0);
    } else {
        data = network->addInput(NAMES[0], dt, Dims4{batch_size, 3, INPUT_H, INPUT_W});
    }
    assert(data);

    auto* conv1 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap.at("features.0.weight"),
                                            weightMap.at("features.0.bias"));
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
    auto* conv2 = network->addConvolutionNd(*cat1->getOutput(0), 1000, DimsHW{1, 1},
                                            weightMap.at("classifier.1.weight"), weightMap.at("classifier.1.bias"));
    assert(conv2);
    auto* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    auto* pool4 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{14, 14});
    assert(pool4);

    pool4->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*pool4->getOutput(0));

    // Build engine
#if TRT_VERSION_GE(8, 0, 0)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    auto* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete mem;
    delete network;
#else
    builder->setMaxBatchSize(batch_size);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    auto* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif
    std::cout << "build finished\n";

    // Release host memory
    for (auto& mem : weightMap) {
        delete[] static_cast<const uint32_t*>(mem.second.values);
    }

    return engine;
}

static void destroyRuntime(IRuntime* runtime) {
#if TRT_VERSION_GE(8, 0, 0)
    delete runtime;
#else
    runtime->destroy();
#endif
}

static void destroyHostMemory(IHostMemory* memory) {
#if TRT_VERSION_GE(8, 0, 0)
    delete memory;
#else
    memory->destroy();
#endif
}

static void APIToModel(int32_t batch_size, IRuntime* runtime, IHostMemory** modelStream) {
    auto* builder = createInferBuilder(gLogger);
    auto* config = builder->createBuilderConfig();

    auto* engine = createEngine(batch_size, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    (*modelStream) = engine->serialize();

#if TRT_VERSION_GE(8, 0, 0)
    delete engine;
    delete config;
    delete builder;
#else
    engine->destroy();
    config->destroy();
    builder->destroy();
#endif
}

static auto doInference(IExecutionContext& context, void* input,
                        int64_t batch_size) -> std::vector<std::vector<float>> {
    const auto& engine = context.getEngine();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<void*> buffers;

#if TRT_VERSION_GE(8, 0, 0)
    const int32_t nIO = engine.getNbIOTensors();
#else
    const int32_t nIO = engine.getNbBindings();
#endif

    buffers.resize(nIO);
    for (auto i = 0; i < nIO; ++i) {
        std::size_t size = 0;
#if TRT_VERSION_GE(8, 0, 0)
        const auto* tensor_name = engine.getIOTensorName(i);
        auto s = getSize(engine.getTensorDataType(tensor_name));
        size = s * batch_size * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
        if (!context.setTensorAddress(tensor_name, buffers[i])) {
            std::cerr << "setTensorAddress failed\n";
            std::abort();
        }
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

#if TRT_VERSION_GE(8, 0, 0)
    if (!context.enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed\n";
        std::abort();
    }
#else
    if (!context.enqueueV2(buffers.data(), stream, nullptr)) {
        std::cerr << "enqueueV2 failed\n";
        std::abort();
    }
#endif

    std::vector<std::vector<float>> prob;
    for (int i = 1; i < nIO; ++i) {
        std::vector<float> tmp(batch_size * SIZES[i], std::nanf(""));
        std::size_t size = batch_size * SIZES[i] * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(tmp);
    }
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaStreamDestroy(stream));
    for (auto i = 0; i < nIO; ++i) {
        CHECK(cudaFree(buffers[i]));
    }
    return prob;
}

int main(int argc, char** argv) {
    checkTrtEnv();
    if (argc != 2) {
        std::cerr << "arguments not right!\n";
        std::cerr << "./squeezenet -s   // serialize model to plan file\n";
        std::cerr << "./squeezenet -d   // deserialize plan file and run inference\n";
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    auto* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trtModelStream{nullptr};
    std::streamsize size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, runtime, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(ENGINE_PATH, std::ios::binary | std::ios::trunc);
        if (!p) {
            std::cerr << "could not open plan output file\n";
            destroyHostMemory(modelStream);
            destroyRuntime(runtime);
            return -1;
        }
        if (modelStream->size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::cerr << "this model is too large to serialize\n";
            destroyHostMemory(modelStream);
            destroyRuntime(runtime);
            return -1;
        }
        const auto* data_ptr = reinterpret_cast<const char*>(modelStream->data());
        auto data_size = static_cast<std::streamsize>(modelStream->size());
        p.write(data_ptr, data_size);
        destroyHostMemory(modelStream);
        destroyRuntime(runtime);
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
        } else {
            std::cerr << "could not open engine file\n";
            destroyRuntime(runtime);
            return -1;
        }
    } else {
        destroyRuntime(runtime);
        return -1;
    }

#if TRT_VERSION_GE(8, 0, 0)
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
        img = cv::imread("assets/cats.jpg", cv::IMREAD_COLOR);
        assert(!img.empty());
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        img = cv::imread("assets/cats.jpg", cv::IMREAD_COLOR);
        assert(!img.empty());
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }
    assert(input);

    auto firstProb = doInference(*context, input, N);
    printFirstOutputs("squeezenet", firstProb[0].data(), firstProb[0].size());
    std::cout << "prediction result:\n";
    auto labels = loadImagenetLabelMap(LABELS_PATH);
    int _top = 0;
    for (const auto& [idx, logits] : topk(firstProb[0], 3)) {
        std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits << ", label: " << labels[idx]
                  << "\n";
    }

    std::vector<double> latencies;
    latencies.reserve(kBenchmarkRuns);
    for (int32_t i = 0; i < kBenchmarkRuns; ++i) {
        auto _start = std::chrono::steady_clock::now();
        (void)doInference(*context, input, N);
        auto _end = std::chrono::steady_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
        latencies.push_back(static_cast<double>(_time) / 1000.0);
    }
    printBenchmark("squeezenet", latencies, N);

    delete[] trtModelStream;
#if TRT_VERSION_GE(8, 0, 0)
    delete context;
    delete engine;
#else
    context->destroy();
    engine->destroy();
#endif
    destroyRuntime(runtime);
    return 0;
}
