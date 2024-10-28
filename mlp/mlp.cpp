#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include "logging.h"
#include "utils.h"

using namespace nvinfer1;

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1
#define INPUT_NAME "data"
#define OUTPUT_NAME "out"

// Logger from TRT API
static Logger gLogger;

/**
 * Create a single-layer "MLP" using the TRT Builder and Configurations
 *
 * @param N: max batch size for built TRT model
 * @param builder: to build engine and networks
 * @param config: configuration related to Hardware
 * @param dt: datatype for model layers
 * @return engine: TRT model
 */
ICudaEngine* createMLPEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    std::cout << "[INFO]: Creating MLP using TensorRT..." << std::endl;

    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights("./mlp.wts");

    // Create an empty network
    INetworkDefinition* network = builder->createNetworkV2(1u);

    // Create an input with proper name
    ITensor* data = network->addInput(INPUT_NAME, DataType::kFLOAT, Dims4{N, 1, 1, 1});
    assert(data);

    // all tensors
    auto* fc1w = network->addConstant(Dims4{1, 1, 1, 1}, weightMap["linear.weight"])->getOutput(0);
    auto* fc1b = network->addConstant(Dims4{1, 1, 1, 1}, weightMap["linear.bias"])->getOutput(0);
    assert(fc1w && fc1b);
    // fc layer
    auto* fc1_0 = network->addMatrixMultiply(*data, MatrixOperation::kNONE, *fc1w, MatrixOperation::kTRANSPOSE);
    auto* fc1_1 = network->addElementWise(*fc1_0->getOutput(0), *fc1b, ElementWiseOperation::kSUM);
    assert(fc1_0 && fc1_1);
    fc1_0->setName("fc1_0");

    // set output with name
    auto* output = fc1_1->getOutput(0);
    output->setName(OUTPUT_NAME);

    // mark the output
    network->markOutput(*output);

#if TRT_VERSION >= 8000
    IHostMemory* serialized_mem = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serialized_mem->data(), serialized_mem->size());
    delete network;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif
    assert(engine != nullptr);

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(int32_t maxBatchSize, IRuntime* runtime, IHostMemory** modelStream) {
    /**
     * Create engine using TensorRT APIs
     *
     * @param maxBatchSize: for the deployed model configs
     * @param modelStream: shared memory to store serialized model
     */

    // Create builder with the logger
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Build an engine
    ICudaEngine* engine = createMLPEngine(maxBatchSize, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
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

void doInference(IExecutionContext& ctx, float* input, float* output, int batchSize = 1) {
    /**
     * Perform inference using the CUDA ctx
     *
     * @param ctx: context created by engine
     * @param input: input from the host
     * @param output: output to save on host
     * @param batchSize: batch size for TRT model
     */
    // Get engine from the ctx
    const ICudaEngine& engine = ctx.getEngine();

#if TRT_VERSION >= 8000
    int32_t nIO = engine.getNbIOTensors();
    const int inputIndex = 0;
    const int outputIndex = engine.getNbIOTensors() - 1;
#else
    int32_t nIO = engine.getNbBindings();
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);
#endif
    assert(nIO == 2);  // mlp contains 1 input and 1 output

    // create cuda stream for aync cuda operations
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // create GPU buffers on cuda device and copy input data from host
    std::vector<void*> buffers(2, nullptr);
    size_t inputSize = batchSize * INPUT_SIZE * sizeof(float);
    size_t outputSize = batchSize * OUTPUT_SIZE * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));

    // execute inference using ctx provided by engine
#if TRT_VERSION >= 8000
    for (int32_t i = 0; i < engine.getNbIOTensors(); i++) {
        auto const name = engine.getIOTensorName(i);
        auto dims = ctx.getTensorShape(name);
        auto total = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
        std::cout << name << "\t" << total << std::endl;
        ctx.setTensorAddress(name, buffers[i]);
    }
    assert(ctx.enqueueV3(stream));
#else
    assert(ctx.enqueueV2(buffers.data(), stream, nullptr));
#endif

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    for (auto& buffer : buffers) {
        CHECK(cudaFree(buffer));
    }
    CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "[ERROR]: Arguments not right!" << std::endl;
        std::cerr << "./mlp -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mlp -d   // deserialize plan file and run inference" << std::endl;
        return 1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, runtime, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("./mlp.engine", std::ios::binary | std::ios::trunc);
        if (!p.good()) {
            std::cerr << "could not open plan output file" << std::endl;
            return 1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

#if TRT_VERSION >= 8000
        delete modelStream;
#else
        modelStream->destroy();
#endif
        std::cout << "[INFO]: Successfully created TensorRT engine." << std::endl;
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("mlp.engine", std::ios::binary);

        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    delete[] trtModelStream;

    IExecutionContext* ctx = engine->createExecutionContext();
    assert(ctx != nullptr);

    float output[1] = {-1.f};
    float input[1] = {12.0f};

    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        doInference(*ctx, input, output);
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Execution time: " << time << "us\t"
                  << "output: " << output[0] << std::endl;
    }

#if TRT_VERSION >= 8000
    delete ctx;
    delete engine;
    delete runtime;
#else
    ctx->destroy();
    engine->destroy();
    runtime->destroy();
#endif

    return 0;
}
