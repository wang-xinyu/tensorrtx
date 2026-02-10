#include <NvInfer.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <map>
#include <opencv2/opencv.hpp>
#include <vector>
#include "logging.h"
#include "utils.h"

using M = nvinfer1::MatrixOperation;
using E = nvinfer1::ElementWiseOperation;

// parameters we know about the lenet-5
constexpr static const int64_t INPUT_H = 32;
constexpr static const int64_t INPUT_W = 32;
constexpr static const std::array<const char*, 2> NAMES = {"data", "prob"};
constexpr static const std::array<const int64_t, 2> SIZES = {1ll * INPUT_H * INPUT_W, 10};
constexpr static const char* WTS_PATH = "../models/lenet.wts";
constexpr static const char* ENGINE_PATH = "../models/lenet.engine";

static Logger gLogger;

/**
 * @brief Creat the engine using only the API and not any parser.
 *
 * @param N max batch size
 * @param runtime runtime
 * @param builder builder
 * @param config config
 * @param dt data type
 * @return ICudaEngine*
 */
ICudaEngine* createLenetEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
#if TRT_VERSION >= 11200
    auto flag = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#elif TRT_VERSION >= 10000
    auto flag = 0U;
#else
    auto flag = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto* network = builder->createNetworkV2(flag);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_NAME
    ITensor* data = network->addInput(NAMES[0], dt, Dims4{N, 1, INPUT_H, INPUT_W});
    assert(data);

    // Add convolution layer with 6 outputs and a 5x5 filter.
    std::filesystem::path wts_path{WTS_PATH};
    wts_path = std::filesystem::absolute(wts_path);
    std::map<std::string, Weights> weightMap = loadWeights(wts_path.string());
    auto* conv1 = network->addConvolutionNd(*data, 6, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setName("conv1");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    relu1->setName("relu1");

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setName("pool1");

    // Add second convolution layer with 16 outputs and a 5x5 filter.
    auto* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 16, DimsHW{5, 5}, weightMap["conv2.weight"],
                                            weightMap["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setName("conv2");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x2>
    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});
    pool2->setName("pool2");

    // Add fully connected layer
    auto* flatten = network->addShuffle(*pool2->getOutput(0));
    flatten->setReshapeDimensions(Dims2{-1, 400});
    auto* tensor_fc1w = network->addConstant(Dims2{120, 400}, weightMap["fc1.weight"])->getOutput(0);
    auto* fc1w = network->addMatrixMultiply(*tensor_fc1w, M::kNONE, *flatten->getOutput(0), M::kTRANSPOSE);
    assert(tensor_fc1w && fc1w);
    auto tensor_fc1b = network->addConstant(Dims2{120, 1}, weightMap["fc1.bias"])->getOutput(0);
    auto* fc1b = network->addElementWise(*fc1w->getOutput(0), *tensor_fc1b, E::kSUM);
    fc1b->setName("fc1b");
    assert(tensor_fc1b && fc1b);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu3 = network->addActivation(*fc1b->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    auto* flatten_relu3 = network->addShuffle(*relu3->getOutput(0));
    flatten_relu3->setReshapeDimensions(Dims2{-1, 120});

    auto* fc2w = network->addConstant(Dims2{84, 120}, weightMap["fc2.weight"])->getOutput(0);
    auto* fc2b = network->addConstant(Dims2{84, 1}, weightMap["fc2.bias"])->getOutput(0);
    auto* fc3w = network->addConstant(Dims2{10, 84}, weightMap["fc3.weight"])->getOutput(0);
    auto* fc3b = network->addConstant(Dims2{10, 1}, weightMap["fc3.bias"])->getOutput(0);
    assert(fc2w && fc2b && fc3w && fc3b);

    // fully connected layer with relu
    auto* fc2_0 = network->addMatrixMultiply(*fc2w, M::kNONE, *flatten_relu3->getOutput(0), M::kTRANSPOSE);
    assert(fc2_0);
    fc2_0->setName("fc2");
    auto* fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, E::kSUM);
    assert(fc2_1);
    IActivationLayer* relu4 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    assert(relu4);
    auto* shuffle = network->addShuffle(*relu4->getOutput(0));
    shuffle->setReshapeDimensions(Dims2{-1, 84});
    auto* fc3_0 = network->addMatrixMultiply(*fc3w, M::kNONE, *shuffle->getOutput(0), M::kTRANSPOSE);
    assert(fc3_0);
    auto* fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, E::kSUM);
    assert(fc3_1);
    // clang-format on

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*fc3_1->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*prob->getOutput(0));

#if TRT_VERSION >= 8400
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
#else
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    builder->setMaxBatchSize(N);
#endif

    // Build engine
#if TRT_VERSION >= 8000
    IHostMemory* serialized_mem = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serialized_mem->data(), serialized_mem->size());
    delete network;
#else
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

/**
 * @brief create a model using the API directly and serialize it to a stream
 *
 * @param N max batch size
 * @param runtime runtime
 * @param modelStream
 */
void APIToModel(int32_t N, IRuntime* runtime, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createLenetEngine(N, runtime, builder, config, DataType::kFLOAT);
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

std::vector<std::vector<float>> doInference(IExecutionContext& context, void* input, int64_t batchSize) {
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
        auto* tensor_name = engine.getIOTensorName(i);
        auto s = getSize(engine.getTensorDataType(tensor_name));
        size = s * batchSize * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
        context.setTensorAddress(tensor_name, buffers[i]);
#else
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        auto s = getSize(engine.getBindingDataType(idx));
        assert(idx == i);
        size = s * batchSize * SIZES[i];
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
        std::vector<float> tmp(batchSize * SIZES[i], std::nanf(""));
        std::size_t size = batchSize * SIZES[i] * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(tmp);
    }
    CHECK(cudaStreamSynchronize(stream));

    for (auto& buffer : buffers) {
        CHECK(cudaFree(buffer));
    }
    CHECK(cudaStreamDestroy(stream));
    return prob;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "arguments not right!\n";
            std::cerr << "./lenet -s   // serialize model to plan file\n";
            std::cerr << "./lenet -d   // deserialize plan file and run inference\n";
            return -1;
        }

        IRuntime* runtime = createInferRuntime(gLogger);
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
                return -1;
            }
            if (modelStream->size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                std::cerr << "this model is too large to serialize\n";
                return -1;
            }
            const auto* data_ptr = reinterpret_cast<const char*>(modelStream->data());
            auto data_size = static_cast<std::streamsize>(modelStream->size());
            p.write(data_ptr, data_size);

#if TRT_VERSION >= 8000
            delete modelStream;
#else
            modelStream->destroy();
#endif
            std::cout << "serialized weights to lenet5.engine\n";
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

        // prepare input/output data
        auto img = cv::imread("../assets/6.pgm", cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);
        assert(img.channels() == 1);
        img.convertTo(img, CV_32FC1, 0.00392156f, -0.1307f);
        img = img / cv::Scalar(0.3081);
        assert(img.total() * img.elemSize() == SIZES[0] * sizeof(float));

#if TRT_VERSION >= 8000
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);

        // Run inference
        for (int32_t i = 0; i < 100; ++i) {
            auto _start = std::chrono::system_clock::now();
            auto prob = doInference(*context, img.data, 1);
            auto _end = std::chrono::system_clock::now();
            auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
            std::cout << "Execution time: " << _time << "us\n";

            for (const auto& vector : prob) {
                int idx = 0;
                for (auto v : vector) {
                    std::cout << std::setprecision(4) << v << ", " << std::flush;
                    if (++idx > 9) {
                        std::cout << "\n====\n";
                        break;
                    }
                }
            }

            if (i == 99) {
                std::cout << "prediction result:\n";
                int _top = 0;
                for (auto& [idx, logits] : topk(prob[0], 3)) {
                    std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits << ", label: " << idx
                              << "\n";
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
    } catch (const std::exception& err) {
        std::cerr << "fatal error: " << err.what() << '\n';
        return -1;
    } catch (...) {
        std::cerr << "fatal error: unknown exception\n";
        return -1;
    }
}
