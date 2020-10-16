#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>
#include <cmath>

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
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

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

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* basicConv2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn", 1e-3);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

IConcatenationLayer* inception(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
    int ch1x1, int ch3x3red, int ch3x3, int ch5x5red, int ch5x5, int pool_proj) {
    IActivationLayer* relu1 = basicConv2d(network, weightMap, input, ch1x1, 1, 1, 0, lname + "branch1.");

    IActivationLayer* relu2 = basicConv2d(network, weightMap, input, ch3x3red, 1, 1, 0, lname + "branch2.0.");
    IActivationLayer* relu3 = basicConv2d(network, weightMap, *relu2->getOutput(0), ch3x3, 3, 1, 1, lname + "branch2.1.");

    IActivationLayer* relu4 = basicConv2d(network, weightMap, input, ch5x5red, 1, 1, 0, lname + "branch3.0.");
    IActivationLayer* relu5 = basicConv2d(network, weightMap, *relu4->getOutput(0), ch5x5, 3, 1, 1, lname + "branch3.1.");

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu6 = basicConv2d(network, weightMap, *pool1->getOutput(0), pool_proj, 1, 1, 0, lname + "branch4.1.");

    ITensor* inputTensors[] = {relu1->getOutput(0), relu3->getOutput(0), relu5->getOutput(0), relu6->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors, 4);
    assert(cat1);
    return cat1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../googlenet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    float shval[3] = {(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5};
    float scval[3] = {0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5};
    float pval[3] = {1.0, 1.0, 1.0};
    Weights shift{DataType::kFLOAT, shval, 3};
    Weights scale{DataType::kFLOAT, scval, 3};
    Weights power{DataType::kFLOAT, pval, 3};
    IScaleLayer* scale1 = network->addScale(*data, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale1);

    IActivationLayer* relu1 = basicConv2d(network, weightMap, *scale1->getOutput(0), 64, 7, 2, 3, "conv1.");
    IPaddingLayer* pad1 = network->addPaddingNd(*relu1->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    assert(pad1);
    IPoolingLayer* pool1 = network->addPoolingNd(*pad1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});

    IActivationLayer* relu2 = basicConv2d(network, weightMap, *pool1->getOutput(0), 64, 1, 1, 0, "conv2.");
    IActivationLayer* relu3 = basicConv2d(network, weightMap, *relu2->getOutput(0), 192, 3, 1, 1, "conv3.");
    IPaddingLayer* pad2 = network->addPaddingNd(*relu3->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    assert(pad2);
    IPoolingLayer* pool2 = network->addPoolingNd(*pad2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});

    IConcatenationLayer* cat1 = inception(network, weightMap, *pool2->getOutput(0), "inception3a.", 64, 96, 128, 16, 32, 32);
    IConcatenationLayer* cat2 = inception(network, weightMap, *cat1->getOutput(0), "inception3b.", 128, 128, 192, 32, 96, 64);

    IPaddingLayer* pad3 = network->addPaddingNd(*cat2->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    assert(pad3);
    IPoolingLayer* pool3 = network->addPoolingNd(*pad3->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool3);
    pool3->setStrideNd(DimsHW{2, 2});

    IConcatenationLayer* cat3 = inception(network, weightMap, *pool3->getOutput(0), "inception4a.", 192, 96, 208, 16, 48, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4b.", 160, 112, 224, 24, 64, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4c.", 128, 128, 256, 24, 64, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4d.", 112, 144, 288, 32, 64, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4e.", 256, 160, 320, 32, 128, 128);

    IPoolingLayer* pool4 = network->addPoolingNd(*cat3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool4);
    pool4->setStrideNd(DimsHW{2, 2});

    cat3 = inception(network, weightMap, *pool4->getOutput(0), "inception5a.", 256, 160, 320, 32, 128, 128);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception5b.", 384, 192, 384, 48, 128, 128);

    IPoolingLayer* pool5 = network->addPoolingNd(*cat3->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool5);

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool5->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

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
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
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

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./googlenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./googlenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("googlenet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("googlenet.engine", std::ios::binary);
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
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
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
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
