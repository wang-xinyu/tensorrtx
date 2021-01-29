#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// stuff we know about the network and the input/output blobs
#define MAX_BATCH_SIZE 1
const std::vector<int> groupwise_layers{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26};
const std::map<std::string, int> groupwise_counts = {
    {"RepVGG-A0", 1},
    {"RepVGG-A1", 1},
    {"RepVGG-A2", 1},
    {"RepVGG-B0", 1},
    {"RepVGG-B1", 1},
    {"RepVGG-B1g2", 2},
    {"RepVGG-B1g4", 4},
    {"RepVGG-B2", 1},
    {"RepVGG-B2g2", 2},
    {"RepVGG-B2g4", 4},
    {"RepVGG-B3", 1},
    {"RepVGG-B3g2", 2},
    {"RepVGG-B3g4", 4}};
const std::map<std::string, std::vector<int>> num_blocks = {
    {"RepVGG-A0", {2, 4, 14, 1}},
    {"RepVGG-A1", {2, 4, 14, 1}},
    {"RepVGG-A2", {2, 4, 14, 1}},
    {"RepVGG-B0", {4, 6, 16, 1}},
    {"RepVGG-B1", {4, 6, 16, 1}},
    {"RepVGG-B1g2", {4, 6, 16, 1}},
    {"RepVGG-B1g4", {4, 6, 16, 1}},
    {"RepVGG-B2", {4, 6, 16, 1}},
    {"RepVGG-B2g2", {4, 6, 16, 1}},
    {"RepVGG-B2g4", {4, 6, 16, 1}},
    {"RepVGG-B3", {4, 6, 16, 1}},
    {"RepVGG-B3g2", {4, 6, 16, 1}},
    {"RepVGG-B3g4", {4, 6, 16, 1}}};
const std::map<std::string, std::vector<float>> width_multiplier = {
    {"RepVGG-A0", {0.75, 0.75, 0.75, 2.5}},
    {"RepVGG-A1", {1, 1, 1, 2.5}},
    {"RepVGG-A2", {1.5, 1.5, 1.5, 2.75}},
    {"RepVGG-B0", {1, 1, 1, 2.5}},
    {"RepVGG-B1", {2, 2, 2, 4}},
    {"RepVGG-B1g2", {2, 2, 2, 4}},
    {"RepVGG-B1g4", {2, 2, 2, 4}},
    {"RepVGG-B2", {2.5, 2.5, 2.5, 5}},
    {"RepVGG-B2g2", {2.5, 2.5, 2.5, 5}},
    {"RepVGG-B2g4", {2.5, 2.5, 2.5, 5}},
    {"RepVGG-B3", {3, 3, 3, 5}},
    {"RepVGG-B3g2", {3, 3, 3, 5}},
    {"RepVGG-B3g4", {3, 3, 3, 5}}};

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout << "Finished Load weights: " << file << std::endl;
    return weightMap;
}

IActivationLayer *RepVGGBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int stride, int groups, std::string lname)
{
    IConvolutionLayer *conv = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "rbr_reparam.weight"], weightMap[lname + "rbr_reparam.bias"]);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{1, 1});
    conv->setNbGroups(groups);
    assert(conv);
    IActivationLayer *relu = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

IActivationLayer *makeStage(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, int &layer_idx, const int group_count, ITensor &input, int inch, int outch, int stride, int blocks, std::string lname)
{
    IActivationLayer *layer;
    for (int i = 0; i < blocks; ++i)
    {
        int group = 1;
        if (std::find(groupwise_layers.begin(), groupwise_layers.end(), layer_idx) != groupwise_layers.end())
            group = group_count;
        if (i == 0)
            layer = RepVGGBlock(network, weightMap, input, inch, outch, 2, group, lname + std::to_string(i) + ".");
        else
            layer = RepVGGBlock(network, weightMap, *layer->getOutput(0), inch, outch, 1, group, lname + std::to_string(i) + ".");
        layer_idx += 1;
    }
    return layer;
}
// Creat the engine using only the API and not any parser.
ICudaEngine *createEngine(std::string netName, unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    const std::vector<int> blocks = num_blocks.at(netName);
    const std::vector<float> widths = width_multiplier.at(netName);
    const int group_count = groupwise_counts.at(netName);
    int layer_idx = 1;

    std::map<std::string, Weights> weightMap = loadWeights("../" + netName + ".wts");

    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    int in_planes = std::min(64, int(64 * widths[0]));
    auto stage0 = RepVGGBlock(network, weightMap, *data, 3, in_planes, 2, 1, "stage0.");
    assert(stage0);

    auto stage1 = makeStage(network, weightMap, layer_idx, group_count, *stage0->getOutput(0), in_planes, int(64 * widths[0]), 2, blocks[0], "stage1.");
    assert(stage1);
    auto stage2 = makeStage(network, weightMap, layer_idx, group_count, *stage1->getOutput(0), int(64 * widths[0]), int(128 * widths[1]), 2, blocks[1], "stage2.");
    assert(stage2);
    auto stage3 = makeStage(network, weightMap, layer_idx, group_count, *stage2->getOutput(0), int(128 * widths[1]), int(256 * widths[2]), 2, blocks[2], "stage3.");
    assert(stage3);
    auto stage4 = makeStage(network, weightMap, layer_idx, group_count, *stage3->getOutput(0), int(256 * widths[2]), int(512 * widths[3]), 2, blocks[3], "stage4.");
    assert(stage4);

    IPoolingLayer *pool = network->addPoolingNd(*stage4->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    pool->setStrideNd(DimsHW{7, 7});
    pool->setPaddingNd(DimsHW{0, 0});
    assert(pool);

    IFullyConnectedLayer *linear = network->addFullyConnected(*pool->getOutput(0), 1000, weightMap["linear.weight"], weightMap["linear.bias"]);
    assert(linear);

    linear->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*linear->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}

void APIToModel(std::string netName, unsigned int maxBatchSize, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(netName, maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./repvgg -s  RepVGG-B1g2 // serialize model to plan file" << std::endl;
        std::cerr << "./repvgg -d  RepVGG-B1g2 // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s")
    {
        std::string netName = std::string(argv[2]);
        IHostMemory *modelStream{nullptr};
        APIToModel(netName, MAX_BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(netName + ".engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    }
    else if (std::string(argv[1]) == "-d")
    {
        std::string netName = std::string(argv[2]);
        std::ifstream file(netName + ".engine", std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        return -1;
    }

    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
