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
static const int BS = 1;

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

IScaleLayer* addBatchNorm(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
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

ILayer* hSwish(INetworkDefinition *network, ITensor& input, std::string name) {
    auto hsig = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    ILayer* hsw = network->addElementWise(input, *hsig->getOutput(0),ElementWiseOperation::kPROD);
    assert(hsw);
    return hsw;
}

ILayer* convBnHswish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (ksize - 1) / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);

    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    ILayer* hsw = hSwish(network, *bn1->getOutput(0), lname+"2");
    assert(hsw);
    return hsw;
}

ILayer* seLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c, int w, std::string lname) {
    int h = w;
    IPoolingLayer* l1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW(w, h));
    assert(l1);
    l1->setStrideNd(DimsHW{w, h});
    IFullyConnectedLayer* l2 = network->addFullyConnected(*l1->getOutput(0), BS*c/4, weightMap[lname+"fc.0.weight"], weightMap[lname+"fc.0.bias"]);
    IActivationLayer* relu1 = network->addActivation(*l2->getOutput(0), ActivationType::kRELU);
    IFullyConnectedLayer* l4 = network->addFullyConnected(*relu1->getOutput(0), BS*c, weightMap[lname+"fc.2.weight"], weightMap[lname+"fc.2.bias"]);

    auto hsig = network->addActivation(*l4->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);

    ILayer* se = network->addElementWise(input, *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(se);
    return se;
}

ILayer* convSeq1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int output, int hdim, int k, int s, bool use_se, bool use_hs, int w, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (k - 1) / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, hdim, DimsHW{k, k}, weightMap[lname + "0.weight"], emptywts);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(hdim);

    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    ITensor *tensor3, *tensor4;
    tensor3 = nullptr;
    tensor4 = nullptr;
    if (use_hs) {
        ILayer* hsw = hSwish(network, *bn1->getOutput(0), lname+"2");
        tensor3 = hsw->getOutput(0);
    } else {
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        tensor3 = relu1->getOutput(0);
    }
    if (use_se) {
         ILayer* se1 = seLayer(network, weightMap, *tensor3, hdim, w, lname + "3.");
         tensor4 = se1->getOutput(0);
    } else {
         tensor4 = tensor3;
    }
    IConvolutionLayer* conv2 = network->addConvolutionNd(*tensor4, output, DimsHW{1, 1}, weightMap[lname + "4.weight"], emptywts);
    IScaleLayer* bn2 = addBatchNorm(network, weightMap, *conv2->getOutput(0), lname + "5", 1e-5);
    assert(bn2);
    return bn2;
}

ILayer* convSeq2(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int output, int hdim, int k, int s, bool use_se, bool use_hs, int w, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = (k - 1) / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, hdim, DimsHW{1, 1}, weightMap[lname + "0.weight"], emptywts);
    IScaleLayer* bn1 = addBatchNorm(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    ITensor *tensor3, *tensor6, *tensor7;
    tensor3 = nullptr;
    tensor6 = nullptr;
    tensor7 = nullptr;
    if (use_hs) {
        ILayer* hsw1 = hSwish(network, *bn1->getOutput(0), lname + "2");
        tensor3 = hsw1->getOutput(0);
    } else {
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        tensor3 = relu1->getOutput(0);
    }
    IConvolutionLayer* conv2 = network->addConvolutionNd(*tensor3, hdim, DimsHW{k, k}, weightMap[lname + "3.weight"], emptywts);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{p, p});
    conv2->setNbGroups(hdim);
    IScaleLayer* bn2 = addBatchNorm(network, weightMap, *conv2->getOutput(0), lname + "4", 1e-5);
    if (use_se) {
         ILayer* se1 = seLayer(network, weightMap, *bn2->getOutput(0), hdim, w, lname + "5.");
         tensor6 = se1->getOutput(0);
    } else {
         tensor6 = bn2->getOutput(0);
    }
    if (use_hs) {
        ILayer* hsw2 = hSwish(network, *tensor6, lname + "6");
        tensor7 = hsw2->getOutput(0);
    } else {
        IActivationLayer* relu2 = network->addActivation(*tensor6, ActivationType::kRELU);
        tensor7 = relu2->getOutput(0);
    }
    IConvolutionLayer* conv3 = network->addConvolutionNd(*tensor7, output, DimsHW{1, 1}, weightMap[lname + "7.weight"], emptywts);
    IScaleLayer* bn3 = addBatchNorm(network, weightMap, *conv3->getOutput(0), lname + "8", 1e-5);
    assert(bn3);
    return bn3;
}

ILayer* invertedRes(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s, int hidden, int k, bool use_se, bool use_hs, int w) {
    bool use_res_connect = (s == 1 && inch == outch);
    ILayer *conv = nullptr;
    if (inch == hidden) {
        conv = convSeq1(network, weightMap, input, outch, hidden, k, s, use_se, use_hs, w, lname + "conv.");
    } else {
        conv = convSeq2(network, weightMap, input, outch, hidden, k, s, use_se, use_hs, w, lname + "conv.");
    }

    if (!use_res_connect) return conv;
    IElementWiseLayer* ew3 = network->addElementWise(input, *conv->getOutput(0), ElementWiseOperation::kSUM);
    assert(ew3);
    return ew3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngineSmall(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../mbv3_small.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    //auto test1 = network->addActivation(*data, ActivationType::kRELU);
    auto ew1 = convBnHswish(network, weightMap, *data, 16, 3, 2, 1, "features.0.");
    auto ir1 = invertedRes(network, weightMap, *ew1->getOutput(0), "features.1.", 16, 16, 2, 16, 3, 1, 0, 56);
    auto ir2 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.2.", 16, 24, 2, 72, 3, 0, 0, 28);
    auto ir3 = invertedRes(network, weightMap, *ir2->getOutput(0), "features.3.", 24, 24, 1, 88, 3, 0, 0, 28);
    auto ir4 = invertedRes(network, weightMap, *ir3->getOutput(0), "features.4.", 24, 40, 2, 96, 5, 1, 1, 14);
    auto ir5 = invertedRes(network, weightMap, *ir4->getOutput(0), "features.5.", 40, 40, 1, 240, 5, 1, 1, 14);
    auto ir6 = invertedRes(network, weightMap, *ir5->getOutput(0), "features.6.", 40, 40, 1, 240, 5, 1, 1, 14);
    auto ir7 = invertedRes(network, weightMap, *ir6->getOutput(0), "features.7.", 40, 48, 1, 120, 5, 1, 1, 14);
    auto ir8 = invertedRes(network, weightMap, *ir7->getOutput(0), "features.8.", 48, 48, 1, 144, 5, 1, 1, 14);
    auto ir9 = invertedRes(network, weightMap, *ir8->getOutput(0), "features.9.", 48, 96, 2, 288, 5, 1, 1, 7);
    auto ir10 = invertedRes(network, weightMap, *ir9->getOutput(0), "features.10.", 96, 96, 1, 576, 5, 1, 1, 7);
    auto ir11 = invertedRes(network, weightMap, *ir10->getOutput(0), "features.11.", 96, 96, 1, 576, 5, 1, 1, 7);
    ILayer* ew2 = convBnHswish(network, weightMap, *ir11->getOutput(0), 576, 1, 1, 1, "conv.0.");
    ILayer* se1 = seLayer(network, weightMap, *ew2->getOutput(0), 576, 7, "conv.1.");

    IPoolingLayer* pool1 = network->addPoolingNd(*se1->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool1);
    pool1->setStrideNd(DimsHW{7, 7});
    ILayer* sw1 = hSwish(network, *pool1->getOutput(0), "hSwish.0");

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*sw1->getOutput(0), 1280, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    ILayer* bn1 = addBatchNorm(network, weightMap, *fc1->getOutput(0), "classifier.1", 1e-5);
    ILayer* sw2 = hSwish(network, *bn1->getOutput(0), "hSwish.1");
    IFullyConnectedLayer* fc2 = network->addFullyConnected(*sw2->getOutput(0), 1000, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);
    ILayer* bn2 = addBatchNorm(network, weightMap, *fc2->getOutput(0), "classifier.4", 1e-5);
    ILayer* sw3 = hSwish(network, *bn2->getOutput(0), "hSwish.2");

    sw3->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*sw3->getOutput(0));

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

ICudaEngine* createEngineLarge(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../mbv3_large.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    //auto test1 = network->addActivation(*data, ActivationType::kRELU);
    auto ew1 = convBnHswish(network, weightMap, *data, 16, 3, 2, 1, "features.0.");
    auto ir1 = invertedRes(network, weightMap, *ew1->getOutput(0), "features.1.", 16, 16, 1, 16, 3, 0, 0, 112);
    auto ir2 = invertedRes(network, weightMap, *ir1->getOutput(0), "features.2.", 16, 24, 2, 64, 3, 0, 0, 56);
    auto ir3 = invertedRes(network, weightMap, *ir2->getOutput(0), "features.3.", 24, 24, 1, 72, 3, 0, 0, 56);
    auto ir4 = invertedRes(network, weightMap, *ir3->getOutput(0), "features.4.", 24, 40, 2, 72, 5, 1, 0, 28);
    auto ir5 = invertedRes(network, weightMap, *ir4->getOutput(0), "features.5.", 40, 40, 1, 120, 5, 1, 0, 28);
    auto ir6 = invertedRes(network, weightMap, *ir5->getOutput(0), "features.6.", 40, 40, 1, 120, 5, 1, 0, 28);
    auto ir7 = invertedRes(network, weightMap, *ir6->getOutput(0), "features.7.", 40, 80, 2, 240, 3, 0, 1, 14);
    auto ir8 = invertedRes(network, weightMap, *ir7->getOutput(0), "features.8.", 80, 80, 1, 200, 3, 0, 1, 14);
    auto ir9 = invertedRes(network, weightMap, *ir8->getOutput(0), "features.9.", 80, 80, 1, 184, 3, 0, 1, 14);
    auto ir10 = invertedRes(network, weightMap, *ir9->getOutput(0), "features.10.", 80, 80, 1, 184, 3, 0, 1, 14);
    auto ir11 = invertedRes(network, weightMap, *ir10->getOutput(0), "features.11.", 80, 112, 1, 480, 3, 1, 1, 14);
    auto ir12 = invertedRes(network, weightMap, *ir11->getOutput(0), "features.12.", 112, 112, 1, 672, 3, 1, 1, 14);
    auto ir13 = invertedRes(network, weightMap, *ir12->getOutput(0), "features.13.", 112, 160, 1, 672, 5, 1, 1, 14);
    auto ir14 = invertedRes(network, weightMap, *ir13->getOutput(0), "features.14.", 160, 160, 2, 672, 5, 1, 1, 7);
    auto ir15 = invertedRes(network, weightMap, *ir14->getOutput(0), "features.15.", 160, 160, 1, 960, 5, 1, 1, 7);
    ILayer* ew2 = convBnHswish(network, weightMap, *ir15->getOutput(0), 960, 1, 1, 1, "conv.0.");

    IPoolingLayer* pool1 = network->addPoolingNd(*ew2->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool1);
    pool1->setStrideNd(DimsHW{7, 7});
    ILayer* sw1 = hSwish(network, *pool1->getOutput(0), "hSwish.0");

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*sw1->getOutput(0), 1280, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    ILayer* sw2 = hSwish(network, *fc1->getOutput(0), "hSwish.1");
    IFullyConnectedLayer* fc2 = network->addFullyConnected(*sw2->getOutput(0), 1000, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);

    fc2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc2->getOutput(0));

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

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string mode)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine;

    if (mode == "small") {
        std::cout << "create engine small" << std::endl;
        engine = createEngineSmall(maxBatchSize, builder, config, DataType::kFLOAT);
    } else if (mode == "large") {
        engine = createEngineLarge(maxBatchSize, builder, config, DataType::kFLOAT);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
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
    if (argc != 3) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mobilenet -s small  // serialize small model to plan file" << std::endl;
        std::cerr << "./mobilenet -s large  // serialize large model to plan file" << std::endl;
        std::cerr << "./mobilenet -d small  // deserialize small model plan file and run inference" << std::endl;
        std::cerr << "./mobilenet -d large  // deserialize large model plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::string mode = std::string(argv[2]);
    std::cout << mode << std::endl;

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream, mode);
        assert(modelStream != nullptr);

        std::ofstream p("mobilenetv3_" + mode + ".engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("mobilenetv3_" + mode + ".engine", std::ios::binary);
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
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {
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
        //if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
