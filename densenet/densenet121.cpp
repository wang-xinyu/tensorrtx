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

IConvolutionLayer* addDenseLayer(INetworkDefinition* network, ITensor* input, std::map<std::string, Weights>& weightMap, std::string lname, float eps)
{
    // add Batchnorm
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *input, lname + ".norm1", eps);

    // add relu
    IActivationLayer* relu1 = network -> addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // add conv
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network -> addConvolutionNd(*relu1->getOutput(0), 128, DimsHW{1, 1}, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1 -> setStrideNd(DimsHW{1, 1});

    // add Batchnorm
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv1 -> getOutput(0), lname + ".norm2", eps);

    // add relu
    IActivationLayer* relu2 = network -> addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // add conv
    IConvolutionLayer* conv2 = network -> addConvolutionNd(*relu2->getOutput(0), 32, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2 -> setStrideNd(DimsHW{1, 1});
    conv2 -> setPaddingNd(DimsHW{1, 1});
    return conv2;
}


IPoolingLayer* addTransition(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap, int outch, std::string lname, float eps)
{
    // add batch norm
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap,input, lname + ".norm", eps);

    // add relu activation
    IActivationLayer* relu1 = network -> addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // add convolution layer
    // empty weights for no bias
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network -> addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1 -> setStrideNd(DimsHW{1, 1});

    // add pooling
    IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1 -> setStrideNd(DimsHW{2, 2});
    pool1 -> setPaddingNd(DimsHW{0,0});
    return pool1;
}


IConcatenationLayer* addDenseBlock(INetworkDefinition* network, ITensor* input, std::map<std::string, Weights>& weightMap, int numDenseLayers, std::string lname, float eps)
{
    IConvolutionLayer* c{nullptr};
    IConcatenationLayer* concat{nullptr};
    ITensor* inputTensors[numDenseLayers+1];
    inputTensors[0] = input;

    c = addDenseLayer(network, input, weightMap, lname + ".denselayer" + std::to_string(1), eps);
    int i;
    for(i=1; i<numDenseLayers; i++)
    {
        // inch += 32;
        inputTensors[i] = c -> getOutput(0);
        concat = network -> addConcatenation(inputTensors, i+1);
        assert(concat);
        c = addDenseLayer(network, concat->getOutput(0), weightMap, lname + ".denselayer" + std::to_string(i+1), eps);
    }
    inputTensors[numDenseLayers] = c -> getOutput(0);
    concat = network -> addConcatenation(inputTensors, numDenseLayers+1);
    assert(concat);
    return concat;
}


/**
 * Uses the TensorRT API to create the network engine.  
**/
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    // Initialize NetworkDefinition
    INetworkDefinition* network = builder -> createNetworkV2(0U);

    auto data = network -> addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../densenet121.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    auto conv0 = network -> addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["features.conv0.weight"], emptywts);
    assert(conv0);
    conv0 -> setStrideNd(DimsHW{2, 2});
    conv0 -> setPaddingNd(DimsHW{3, 3});

    auto norm0 = addBatchNorm2d(network, weightMap, *conv0 -> getOutput(0), "features.norm0", 1e-5);

    auto relu0 = network -> addActivation(*norm0 -> getOutput(0), ActivationType::kRELU);
    assert(relu0);

    auto pool0 = network -> addPoolingNd(*relu0 -> getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool0);
    pool0 -> setStrideNd(DimsHW{2, 2});
    pool0 -> setPaddingNd(DimsHW{1, 1});
    
    auto dense1 = addDenseBlock(network, pool0 -> getOutput(0), weightMap, 6, "features.denseblock1", 1e-5);
    auto transition1 = addTransition(network, *dense1 -> getOutput(0), weightMap, 128, "features.transition1", 1e-5);

    auto dense2 = addDenseBlock(network, transition1 -> getOutput(0), weightMap, 12, "features.denseblock2", 1e-5);
    auto transition2 = addTransition(network, *dense2 -> getOutput(0), weightMap, 256, "features.transition2", 1e-5);

    auto dense3 = addDenseBlock(network, transition2 -> getOutput(0), weightMap, 24, "features.denseblock3", 1e-5);
    auto transition3 = addTransition(network, *dense3 -> getOutput(0), weightMap, 512, "features.transition3", 1e-5);

    auto dense4 = addDenseBlock(network, transition3 -> getOutput(0), weightMap, 16, "features.denseblock4", 1e-5);

    auto bn5 = addBatchNorm2d(network, weightMap, *dense4 -> getOutput(0), "features.norm5", 1e-5);
    auto relu5 = network -> addActivation(*bn5 -> getOutput(0), ActivationType::kRELU);

    // adaptive average pool => pytorch (F.adaptive_avg_pool2d(input, (1, 1)))
    auto pool5 = network -> addPoolingNd(*relu5 -> getOutput(0), PoolingType::kAVERAGE, DimsHW{7,7});

    auto fc1 = network -> addFullyConnected(*pool5 -> getOutput(0), 1000, weightMap["classifier.weight"], weightMap["classifier.bias"]);
    assert(fc1);

    // set ouput blob name
    fc1 -> getOutput(0) -> setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;

    // mark the output
    network -> markOutput(*fc1 -> getOutput(0));

    // set batchsize and workspace size
    builder -> setMaxBatchSize(maxBatchSize);
    config -> setMaxWorkspaceSize(1 << 28); // 256 MiB

    // build engine
    ICudaEngine* engine = builder -> buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;
    
    // destroy
    network -> destroy();

    // fere host mem
    for(auto& mem: weightMap)
    {
        free((void*)(mem.second.values));
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
    config->destroy();
}

/**
 * Performs inference on the given input and 
 * writes the output from device to host memory.
**/
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
        std::cerr << "./densenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./densenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("densenet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("densenet.engine", std::ios::binary);
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
