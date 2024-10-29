#include <math.h>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>
#include "logging.h"
#include "utils.h"

// stuff we know about alexnet
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_SIZE (INPUT_H * INPUT_W)
#define OUTPUT_SIZE 1000
#define INPUT_NAME "data"
#define OUTPUT_NAME "prob"

#define ENGINE_PATH "../models/alexnet.engine"
#define WTS_PATH "../models/alexnet.wts"

using namespace nvinfer1;

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
    INetworkDefinition* network = builder->createNetworkV2(1u);
    // clang-format off

    // Create input tensor of shape { N, 1, INPUT_H, INPUT_W }
    ITensor* data = network->addInput(INPUT_NAME, dt, Dims4{N, 3, INPUT_H, INPUT_W});
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(WTS_PATH);

    // CRP (Conv-Relu-Pool)
    auto* conv1 = network->addConvolutionNd(*data, 64, DimsHW{11, 11}, weightMap["features.0.weight"], weightMap["features.0.bias"]);
    auto* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    auto* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv1 && relu1 && pool1);
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    // CRP
    auto* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5, 5}, weightMap["features.3.weight"], weightMap["features.3.bias"]);
    auto* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    auto* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv2 && pool2 && relu2);
    conv2->setPaddingNd(DimsHW{2, 2});
    pool2->setStrideNd(DimsHW{2, 2});

    // CR
    auto* conv3 = network->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3}, weightMap["features.6.weight"], weightMap["features.6.bias"]);
    auto* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(conv3 && relu3);
    conv3->setPaddingNd(DimsHW{1, 1});

    // CR
    auto* conv4 = network->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.8.weight"], weightMap["features.8.bias"]);
    auto* relu4 = network->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    assert(conv4 && relu4);
    conv4->setPaddingNd(DimsHW{1, 1});

    // CRP
    auto* conv5 = network->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.10.weight"], weightMap["features.10.bias"]);
    auto* relu5 = network->addActivation(*conv5->getOutput(0), ActivationType::kRELU);    assert(conv5);
    auto* pool3 = network->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv5 && relu5 && pool3);
    conv5->setPaddingNd(DimsHW{1, 1});
    pool3->setStrideNd(DimsHW{2, 2});

    // adaptive avgerage pooling
    auto* adaptive_pool = network->addPoolingNd(*pool3->getOutput(0), PoolingType::kAVERAGE, DimsHW{1, 1});
    assert(adaptive_pool);
    IShuffleLayer* shuffle = network->addShuffle(*adaptive_pool->getOutput(0));
    assert(shuffle);
    shuffle->setReshapeDimensions(Dims2{N, -1}); // "-1" means "256 * 6 * 6"

    // all classifier tensors
    auto* fc1w = network->addConstant(DimsHW{4096, 256 * 6 * 6}, weightMap["classifier.1.weight"])->getOutput(0);
    auto* fc1b = network->addConstant(DimsHW{1, 4096}, weightMap["classifier.1.bias"])->getOutput(0);
    auto* fc2w = network->addConstant(DimsHW{4096, 4096}, weightMap["classifier.4.weight"])->getOutput(0);
    auto* fc2b = network->addConstant(DimsHW{1, 4096}, weightMap["classifier.4.bias"])->getOutput(0);
    auto* fc3w = network->addConstant(DimsHW{OUTPUT_SIZE, 4096}, weightMap["classifier.6.weight"])->getOutput(0);
    auto* fc3b = network->addConstant(DimsHW{1, OUTPUT_SIZE}, weightMap["classifier.6.bias"])->getOutput(0);
    assert(fc1w && fc1b && fc2w && fc2b && fc3w && fc3b);

    // all layers in classifier
    auto* fc1_0 = network->addMatrixMultiply(*shuffle->getOutput(0), MatrixOperation::kNONE, *fc1w, MatrixOperation::kTRANSPOSE);
    auto* fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, ElementWiseOperation::kSUM);
    auto* relu6 = network->addActivation(*fc1_1->getOutput(0), ActivationType::kRELU);
    assert(fc1_0 && fc1_1 && relu6);
    fc1_0->setName("fc1_0"); // set name here, only for debug purpose
    auto* fc2_0 = network->addMatrixMultiply(*relu6->getOutput(0), MatrixOperation::kNONE, *fc2w, MatrixOperation::kTRANSPOSE);
    auto* fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, ElementWiseOperation::kSUM);
    auto* relu7 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    assert(fc2_0 && fc2_1 && relu7);
    fc2_0->setName("fc2_0");
    auto* fc3_0 = network->addMatrixMultiply(*relu7->getOutput(0), MatrixOperation::kNONE, *fc3w, MatrixOperation::kTRANSPOSE);
    auto* fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, ElementWiseOperation::kSUM);
    assert(fc3_0 && fc3_1);
    fc3_0->setName("fc3_0");

    // clang-format on
    fc3_1->getOutput(0)->setName(OUTPUT_NAME);
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
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

#if TRT_VERSION >= 8000
    int32_t nIO = engine.getNbIOTensors();
    const int32_t inputIndex = 0;
    const int32_t outputIndex = nIO - 1;
#else
    int32_t nIO = engine.getNbBindings();
    const int32_t inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int32_t outputIndex = engine.getBindingIndex(OUTPUT_NAME);
#endif
    assert(nIO == 2);  // AlexNet has 1 input and 1 output

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::vector<void*> buffers(2, nullptr);
    size_t inputSize = batchSize * 3 * INPUT_SIZE * sizeof(float);
    size_t outputSize = batchSize * OUTPUT_SIZE * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));

#if TRT_VERSION >= 8000
    for (int32_t i = 0; i < nIO; ++i) {
        auto* tensor_name = engine.getIOTensorName(i);
        context.setTensorAddress(tensor_name, buffers[i]);
    }
    context.enqueueV3(stream);
#else
    context.enqueueV2(buffers.data(), stream, nullptr);
#endif

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    for (auto* buf : buffers) {
        CHECK(cudaFree(buf));
    }
    CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
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

    // dummy input/output cpu data
    std::vector<float> data(3 * INPUT_SIZE, 1.f);
    std::vector<float> prob(OUTPUT_SIZE, std::nanf("nan"));

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    std::cout << "Execution time: ";
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data.data(), prob.data(), 1);
        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << time << "ms " << std::flush;
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
    std::cout << "\n\n";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << prob[i] << ", " << std::flush;
    }
    int res = std::max_element(prob.begin(), prob.end()) - prob.begin();
    std::cout << "\n\ndummy classification result is: " << res << std::endl;

    return 0;
}
