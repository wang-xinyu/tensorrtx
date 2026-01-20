#include <NvInfer.h>
#include <math.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include "logging.h"
#include "utils.h"

// stuff we know about alexnet
constexpr const int32_t N = 1;
constexpr const int32_t INPUT_H = 224;
constexpr const int32_t INPUT_W = 224;
constexpr const int32_t SIZES[] = {3 * INPUT_H * INPUT_W, 1000};

constexpr const char* NAMES[] = {"data", "prob"};
constexpr const char ENGINE_PATH[] = "../models/alexnet.engine";
constexpr const char WTS_PATH[] = "../models/alexnet.wts";
constexpr const char LABELS_PATH[] = "../assets/imagenet1000_clsidx_to_labels.txt";
static constexpr const bool TRT_PREPROCESS = TRT_VERSION >= 8510 ? true : false;
static constexpr const float mean[3] = {0.485f, 0.456f, 0.406f};
static constexpr const float stdv[3] = {0.229f, 0.224f, 0.225f};

using WeightMap = std::map<std::string, Weights>;
using M = nvinfer1::MatrixOperation;
using E = nvinfer1::ElementWiseOperation;

static Logger gLogger;

/**
 * @brief Create the engine using TensorRT API and without any parser.
 *
 * @param N max batch size
 * @param builder
 * @param config
 * @param dt
 * @return ICudaEngine*
 */
ICudaEngine* createEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    WeightMap weightMap = loadWeights(WTS_PATH);
#if TRT_VERSION >= 10000
    auto* network = builder->createNetworkV2(0);
#else
    auto* network = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

    // Create input tensor
    ITensor* input{nullptr};
    if constexpr (TRT_PREPROCESS) {
#if TRT_VERSION > 8510
        dt = DataType::kUINT8;
#else
        dt = DataType::kINT8;
#endif
        input = network->addInput(NAMES[0], dt, Dims4{N, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(network, *input, true, mean, stdv);
        input = trans->getOutput(0);
    } else {
        input = network->addInput(NAMES[0], dt, Dims4{N, 3, INPUT_H, INPUT_W});
    }
    assert(input);

    // CRP (Conv-Relu-Pool)
    auto* conv1 = network->addConvolutionNd(*input, 64, DimsHW{11, 11}, weightMap["features.0.weight"],
                                            weightMap["features.0.bias"]);
    auto* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    auto* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv1 && relu1 && pool1);
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    // CRP
    auto* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5, 5}, weightMap["features.3.weight"],
                                            weightMap["features.3.bias"]);
    auto* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    auto* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv2 && pool2 && relu2);
    conv2->setPaddingNd(DimsHW{2, 2});
    pool2->setStrideNd(DimsHW{2, 2});

    // CR
    auto* conv3 = network->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3}, weightMap["features.6.weight"],
                                            weightMap["features.6.bias"]);
    auto* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(conv3 && relu3);
    conv3->setPaddingNd(DimsHW{1, 1});

    // CR
    auto* conv4 = network->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.8.weight"],
                                            weightMap["features.8.bias"]);
    auto* relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    assert(conv4 && relu4);
    conv4->setPaddingNd(DimsHW{1, 1});

    // CRP
    auto* conv5 = network->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.10.weight"],
                                            weightMap["features.10.bias"]);
    auto* relu5 = network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
    assert(conv5);
    auto* pool3 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv5 && relu5 && pool3);
    conv5->setPaddingNd(DimsHW{1, 1});
    pool3->setStrideNd(DimsHW{2, 2});

    // adaptive avgerage pooling
    auto* adaptive_pool = network->addPoolingNd(*pool3->getOutput(0), PoolingType::kAVERAGE, DimsHW{1, 1});
    assert(adaptive_pool);
    IShuffleLayer* shuffle = network->addShuffle(*adaptive_pool->getOutput(0));
    assert(shuffle);
    shuffle->setReshapeDimensions(Dims2{N, -1});  // "-1" means "256 * 6 * 6"

    // all classifier tensors
    auto* fc1w = network->addConstant(DimsHW{4096, 256 * 6 * 6}, weightMap["classifier.1.weight"])->getOutput(0);
    auto* fc1b = network->addConstant(DimsHW{1, 4096}, weightMap["classifier.1.bias"])->getOutput(0);
    auto* fc2w = network->addConstant(DimsHW{4096, 4096}, weightMap["classifier.4.weight"])->getOutput(0);
    auto* fc2b = network->addConstant(DimsHW{1, 4096}, weightMap["classifier.4.bias"])->getOutput(0);
    auto* fc3w = network->addConstant(DimsHW{1000, 4096}, weightMap["classifier.6.weight"])->getOutput(0);
    auto* fc3b = network->addConstant(DimsHW{1, 1000}, weightMap["classifier.6.bias"])->getOutput(0);
    assert(fc1w && fc1b && fc2w && fc2b && fc3w && fc3b);

    // all layers in classifier
    auto* fc1_0 = network->addMatrixMultiply(*shuffle->getOutput(0), M::kNONE, *fc1w, M::kTRANSPOSE);
    auto* fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, E::kSUM);
    auto* relu6 = network->addActivation(*fc1_1->getOutput(0), ActivationType::kRELU);
    assert(fc1_0 && fc1_1 && relu6);
    fc1_0->setName("fc1_0");  // set name here, only for debug purpose
    auto* fc2_0 = network->addMatrixMultiply(*relu6->getOutput(0), M::kNONE, *fc2w, M::kTRANSPOSE);
    auto* fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, E::kSUM);
    auto* relu7 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    assert(fc2_0 && fc2_1 && relu7);
    fc2_0->setName("fc2_0");
    auto* fc3_0 = network->addMatrixMultiply(*relu7->getOutput(0), M::kNONE, *fc3w, M::kTRANSPOSE);
    auto* fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, E::kSUM);
    assert(fc3_0 && fc3_1);
    fc3_0->setName("fc3_0");

    fc3_1->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*fc3_1->getOutput(0));

    // Build engine
#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    auto* host_mem = builder->buildSerializedNetwork(*network, *config);
    auto* engine = runtime->deserializeCudaEngine(host_mem->data(), host_mem->size());
    delete network;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    auto* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif

    std::cout << "build finished" << std::endl;
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(int32_t N, IRuntime* runtime, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(N, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
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

std::vector<std::vector<float>> doInference(IExecutionContext& context, const std::string& img_path,
                                            int32_t batchSize) {
    static std::vector<float> flat_img;
    auto img = cv::imread(img_path, cv::IMREAD_COLOR);
    void* input = nullptr;

    // use preprocess from gpu(TensorRT) or cpu(OpenCV)
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }
    assert(input);

    const ICudaEngine& engine = context.getEngine();
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
#if TRT_VERSION >= 8000
        auto* tensor_name = engine.getIOTensorName(i);
        auto s = getSize(engine.getTensorDataType(tensor_name));
        std::size_t size = s * batchSize * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
        context.setTensorAddress(tensor_name, buffers[i]);
#else
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        auto s = getSize(engine.getBindingDataType(idx));
        assert(idx == i);
        std::size_t size = s * batchSize * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
#endif
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
    }

#if TRT_VERSION >= 8000
    assert(context.enqueueV3(stream));
#else
    assert(context.enqueueV2(buffers.data(), stream, nullptr));
#endif

    std::vector<std::vector<float>> prob;
    for (int i = 1; i < nIO; ++i) {
        std::vector<float> tmp(batchSize * SIZES[i], std::nan(""));
        std::size_t size = batchSize * SIZES[i] * sizeof(float);
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
        std::cerr << "./alexnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./alexnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(N, runtime, &modelStream);
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
        return 1;
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
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    const std::string img_path = "../assets/cats.jpg";
    for (int32_t i = 0; i < 100; ++i) {
        auto _start = std::chrono::system_clock::now();
        auto prob = doInference(*context, img_path, N);
        auto _end = std::chrono::system_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
        std::cout << "Execution time: " << _time << "ms" << std::endl;

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
