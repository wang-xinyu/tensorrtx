#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "logging.h"
#include "utils.h"

// stuff we know about vgg
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_SIZE (3 * INPUT_H * INPUT_W)
#define OUTPUT_SIZE 1000
#define INPUT_NAME "data"
#define OUTPUT_NAME "prob"

#define WTS_PATH "../models/vgg.wts"
#define ENGINE_PATH "../models/vgg.engine"

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;

static Logger gLogger;

/**
 * @brief create a Conv-Relu layer
 *
 * @param net network definition from TensorRT API
 * @param map weight map read from `.wts` file
 * @param input input tensor
 * @param out output channels for convolution
 * @param w_name convolution weight key name
 * @param b_name convolution bias key name
 * @param k convolution kernel size
 * @param act_type activation type, default is ActivationType::kRELU
 * @return IActivationLayer*
 */
IActivationLayer* CR(INetworkDefinition* network, WeightMap& map, ITensor& input, int32_t out, std::string key_name,
                     DimsHW k, DimsHW p, ActivationType act_type = ActivationType::kRELU) {
    auto* conv = network->addConvolutionNd(input, out, k, map[key_name + ".weight"], map[key_name + ".bias"]);
    auto* relu = network->addActivation(*conv->getOutput(0), act_type);
    assert(conv && relu);
    conv->setPaddingNd(p);
    conv->setName(key_name.c_str());
    return relu;
}

/**
 * @brief create a Conv-Relu-Pool layer
 *
 * @param net network definition from TensorRT API
 * @param s stride for pooling layer
 * @param p_type pooling layer type, default is PoolingType::kMAX
 * @return IPoolingLayer*
 */
IPoolingLayer* CRP(INetworkDefinition* network, WeightMap& map, ITensor& input, int32_t out, std::string key_name,
                   DimsHW k, DimsHW p, DimsHW s, PoolingType p_type = PoolingType::kMAX,
                   ActivationType act_type = ActivationType::kRELU) {
    auto* cr = CR(network, map, input, out, key_name, k, p, act_type);
    auto* pool = network->addPoolingNd(*cr->getOutput(0), p_type, DimsHW{2, 2});
    assert(cr && pool);
    pool->setStrideNd(s);
    pool->setName((key_name + ".pool").c_str());
    return pool;
}

/**
 * @brief Create a Engine object using only the API and not any parser.
 *
 * @param N max batch size
 * @param runtime
 * @param builder
 * @param config
 * @param dt data type
 * @return ICudaEngine*
 */
ICudaEngine* createEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(1u);
    WeightMap m = loadWeights(WTS_PATH);

    // Create input tensor of shape { N, 3, INPUT_H, INPUT_W } with name INPUT_NAME
    ITensor* data = network->addInput(INPUT_NAME, dt, Dims4{N, 3, INPUT_H, INPUT_W});
    assert(data);

    auto* crp1 = CRP(network, m, *data, 64, "features.0", {3, 3}, {1, 1}, {2, 2});
    auto* crp2 = CRP(network, m, *crp1->getOutput(0), 128, "features.3", {3, 3}, {1, 1}, {2, 2});
    auto cr1 = CR(network, m, *crp2->getOutput(0), 256, "features.6", {3, 3}, {1, 1});
    auto* crp3 = CRP(network, m, *cr1->getOutput(0), 256, "features.8", {3, 3}, {1, 1}, {2, 2});
    auto* cr2 = CR(network, m, *crp3->getOutput(0), 512, "features.11", {3, 3}, {1, 1});
    auto* crp4 = CRP(network, m, *cr2->getOutput(0), 512, "features.13", {3, 3}, {1, 1}, {2, 2});
    auto* cr3 = CR(network, m, *crp4->getOutput(0), 512, "features.16", {3, 3}, {1, 1});
    auto* crp5 = CRP(network, m, *cr3->getOutput(0), 512, "features.18", {3, 3}, {1, 1}, {2, 2});

    auto* avg_pool = network->addPoolingNd(*crp5->getOutput(0), PoolingType::kAVERAGE, Dims2{1, 1});
    auto* flatten = network->addShuffle(*avg_pool->getOutput(0));
    assert(avg_pool && flatten);
    flatten->setReshapeDimensions(Dims2{N, -1});  // "-1" means "512 * 7 * 7"

    // tensors for 3 FC layers
    auto* fc1w = network->addConstant(Dims2{4096, 512 * 7 * 7}, m["classifier.0.weight"])->getOutput(0);
    auto* fc1b = network->addConstant(Dims2{1, 4096}, m["classifier.0.bias"])->getOutput(0);
    auto* fc2w = network->addConstant(Dims2{4096, 4096}, m["classifier.3.weight"])->getOutput(0);
    auto* fc2b = network->addConstant(Dims2{1, 4096}, m["classifier.3.bias"])->getOutput(0);
    auto* fc3w = network->addConstant(Dims2{1000, 4096}, m["classifier.6.weight"])->getOutput(0);
    auto* fc3b = network->addConstant(Dims2{1, 1000}, m["classifier.6.bias"])->getOutput(0);
    assert(fc1w && fc1b && fc2w && fc2b && fc3w && fc3b);
    // clang-format off

    // FC1
    auto* fc1_0 = network->addMatrixMultiply(*flatten->getOutput(0), MatrixOperation::kNONE, *fc1w, MatrixOperation::kTRANSPOSE);
    auto* fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, ElementWiseOperation::kSUM);
    auto* relu1 = network->addActivation(*fc1_1->getOutput(0), ActivationType::kRELU);
    // FC2
    auto* fc2_0 = network->addMatrixMultiply(*relu1->getOutput(0), MatrixOperation::kNONE, *fc2w, MatrixOperation::kTRANSPOSE);
    auto* fc2_1 = network->addElementWise(*fc2_0->getOutput(0), *fc2b, ElementWiseOperation::kSUM);
    auto* relu2 = network->addActivation(*fc2_1->getOutput(0), ActivationType::kRELU);
    // FC3
    auto* fc3_0 = network->addMatrixMultiply(*relu2->getOutput(0), MatrixOperation::kNONE, *fc3w, MatrixOperation::kTRANSPOSE);
    auto* fc3_1 = network->addElementWise(*fc3_0->getOutput(0), *fc3b, ElementWiseOperation::kSUM);
    // clang-format on

    fc3_1->getOutput(0)->setName(OUTPUT_NAME);
    network->markOutput(*fc3_1->getOutput(0));

#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
#endif

    std::cout << "build out" << std::endl;

#if TRT_VERSION >= 8000
    delete network;
#else
    network->destroy();
#endif

    // Release host memory
    for (auto& mem : m) {
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

void doInference(IExecutionContext& context, float* input, float* output, int N) {
    const ICudaEngine& engine = context.getEngine();

#if TRT_VERSION >= 8000
    const int32_t nIO = engine.getNbIOTensors();
    const int32_t inputIndex = 0;
    const int32_t outputIndex = nIO - 1;
#else
    const int32_t nIO = engine.getNbBindings();
    const int32_t inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int32_t outputIndex = engine.getBindingIndex(OUTPUT_NAME);
#endif
    assert(nIO == 2);
    std::vector<void*> buffers(nIO, nullptr);

    const size_t inputSize = N * INPUT_SIZE * sizeof(float);
    const size_t outputSize = N * OUTPUT_SIZE * sizeof(float);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));

#if TRT_VERSION >= 8000
    for (int32_t i = 0; i < nIO; ++i) {
        auto* tensor_name = engine.getIOTensorName(i);
        context.setTensorAddress(tensor_name, buffers[i]);
    }
    context.enqueueV3(stream);
#else
    context.enqueueV2(N, buffers, stream, nullptr);
#endif

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./vgg -s   // serialize model to plan file" << std::endl;
        std::cerr << "./vgg -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    IRuntime* runtime = createInferRuntime(gLogger);
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
        return 1;
    }

    std::vector<float> data(INPUT_SIZE, 1.f);
    std::vector<float> prob(OUTPUT_SIZE, std::nanf(""));

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data.data(), prob.data(), 1);
        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Execution time: " << time << "ms" << std::endl;
    }

    delete[] trtModelStream;
#if TRT_VERSION >= 8000
    delete context;
    delete engine;
    delete runtime;
#else
    context->destroy();
    engine->destroy();
    runtime->destroy();
#endif

    std::cout << "\nOutput:\n\n";
    for (int32_t i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << prob[i] << ", " << std::flush;
    }
    int32_t classId = std::max_element(prob.begin(), prob.end()) - prob.begin();
    std::cout << std::endl << "dummy classification result: " << classId << std::endl;

    return 0;
}
