#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

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
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createLenetEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
    assert(data);

    // Add convolution layer with 6 outputs and a 5x5 filter.
    std::map<std::string, Weights> weightMap = loadWeights("../lenet5.wts");
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 6, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    // Add second convolution layer with 16 outputs and a 5x5 filter.
    IConvolutionLayer* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 16, DimsHW{5, 5}, weightMap["conv2.weight"], weightMap["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x2>
    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});

    // Add fully connected layer
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 120, weightMap["fc1.weight"], weightMap["fc1.bias"]);
    assert(fc1);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu3 = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    // Add second fully connected layer
    IFullyConnectedLayer* fc2 = network->addFullyConnected(*relu3->getOutput(0), 84, weightMap["fc2.weight"], weightMap["fc2.bias"]);
    assert(fc2);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu4 = network->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    // Add third fully connected layer
    IFullyConnectedLayer* fc3 = network->addFullyConnected(*relu4->getOutput(0), OUTPUT_SIZE, weightMap["fc3.weight"], weightMap["fc3.bias"]);
    assert(fc3);

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*fc3->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createLenetEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./lenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./lenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("lenet5.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("lenet5.engine", std::ios::binary);
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


    // Subtract mean from image
    float data[INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
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

    return 0;
}
