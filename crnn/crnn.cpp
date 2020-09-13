#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

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

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 32;
static const int INPUT_W = 100;
static const int OUTPUT_SIZE = 26 * 37;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

const int ks[] = {3, 3, 3, 3, 3, 3, 2};
const int ps[] = {1, 1, 1, 1, 1, 1, 0};
const int ss[] = {1, 1, 1, 1, 1, 1, 1};
const int nm[] = {64, 128, 256, 256, 512, 512, 512};
const std::string alphabet = "-0123456789abcdefghijklmnopqrstuvwxyz";

using namespace nvinfer1;

std::string strDecode(std::vector<int>& preds, bool raw) {
    std::string str;
    if (raw) {
        for (auto v: preds) {
            str.push_back(alphabet[v]);
        }
    } else {
        for (size_t i = 0; i < preds.size(); i++) {
            if (preds[i] == 0 || (i > 0 && preds[i - 1] == preds[i])) continue;
            str.push_back(alphabet[preds[i]]);
        }
    }
    return str;
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

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

ILayer* convRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int i, bool use_bn = false) {
    int nOut = nm[i];
    IConvolutionLayer* conv = network->addConvolutionNd(input, nOut, DimsHW{ks[i], ks[i]}, weightMap["cnn.conv" + std::to_string(i) + ".weight"], weightMap["cnn.conv" + std::to_string(i) + ".bias"]);
    assert(conv);
    conv->setStrideNd(DimsHW{ss[i], ss[i]});
    conv->setPaddingNd(DimsHW{ps[i], ps[i]});
    ILayer *tmp = conv;
    if (use_bn) {
        tmp = addBatchNorm2d(network, weightMap, *conv->getOutput(0), "cnn.batchnorm" + std::to_string(i), 1e-5);
    }
    auto relu = network->addActivation(*tmp->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

void splitLstmWeights(std::map<std::string, Weights>& weightMap, std::string lname) {
    int weight_size = weightMap[lname].count;
    for (int i = 0; i < 4; i++) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        wt.count = weight_size / 4;
        float *val = reinterpret_cast<float*>(malloc(sizeof(float) * wt.count));
        memcpy(val, (float*)weightMap[lname].values + wt.count * i, sizeof(float) * wt.count);
        wt.values = val;
        weightMap[lname + std::to_string(i)] = wt;
    }
}

ILayer* addLSTM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int nHidden, std::string lname) {
    splitLstmWeights(weightMap, lname + ".weight_ih_l0");
    splitLstmWeights(weightMap, lname + ".weight_hh_l0");
    splitLstmWeights(weightMap, lname + ".bias_ih_l0");
    splitLstmWeights(weightMap, lname + ".bias_hh_l0");
    splitLstmWeights(weightMap, lname + ".weight_ih_l0_reverse");
    splitLstmWeights(weightMap, lname + ".weight_hh_l0_reverse");
    splitLstmWeights(weightMap, lname + ".bias_ih_l0_reverse");
    splitLstmWeights(weightMap, lname + ".bias_hh_l0_reverse");
    Dims dims = input.getDimensions();
    std::cout << "lstm input shape: " << dims.nbDims << " [" << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << "]"<< std::endl;
    auto lstm = network->addRNNv2(input, 1, nHidden, dims.d[1], RNNOperation::kLSTM);
    lstm->setDirection(RNNDirection::kBIDIRECTION);
    lstm->setWeightsForGate(0, RNNGateType::kINPUT, true, weightMap[lname + ".weight_ih_l00"]);
    lstm->setWeightsForGate(0, RNNGateType::kFORGET, true, weightMap[lname + ".weight_ih_l01"]);
    lstm->setWeightsForGate(0, RNNGateType::kCELL, true, weightMap[lname + ".weight_ih_l02"]);
    lstm->setWeightsForGate(0, RNNGateType::kOUTPUT, true, weightMap[lname + ".weight_ih_l03"]);

    lstm->setWeightsForGate(0, RNNGateType::kINPUT, false, weightMap[lname + ".weight_hh_l00"]);
    lstm->setWeightsForGate(0, RNNGateType::kFORGET, false, weightMap[lname + ".weight_hh_l01"]);
    lstm->setWeightsForGate(0, RNNGateType::kCELL, false, weightMap[lname + ".weight_hh_l02"]);
    lstm->setWeightsForGate(0, RNNGateType::kOUTPUT, false, weightMap[lname + ".weight_hh_l03"]);

    lstm->setBiasForGate(0, RNNGateType::kINPUT, true, weightMap[lname + ".bias_ih_l00"]);
    lstm->setBiasForGate(0, RNNGateType::kFORGET, true, weightMap[lname + ".bias_ih_l01"]);
    lstm->setBiasForGate(0, RNNGateType::kCELL, true, weightMap[lname + ".bias_ih_l02"]);
    lstm->setBiasForGate(0, RNNGateType::kOUTPUT, true, weightMap[lname + ".bias_ih_l03"]);

    lstm->setBiasForGate(0, RNNGateType::kINPUT, false, weightMap[lname + ".bias_hh_l00"]);
    lstm->setBiasForGate(0, RNNGateType::kFORGET, false, weightMap[lname + ".bias_hh_l01"]);
    lstm->setBiasForGate(0, RNNGateType::kCELL, false, weightMap[lname + ".bias_hh_l02"]);
    lstm->setBiasForGate(0, RNNGateType::kOUTPUT, false, weightMap[lname + ".bias_hh_l03"]);

    lstm->setWeightsForGate(1, RNNGateType::kINPUT, true, weightMap[lname + ".weight_ih_l0_reverse0"]);
    lstm->setWeightsForGate(1, RNNGateType::kFORGET, true, weightMap[lname + ".weight_ih_l0_reverse1"]);
    lstm->setWeightsForGate(1, RNNGateType::kCELL, true, weightMap[lname + ".weight_ih_l0_reverse2"]);
    lstm->setWeightsForGate(1, RNNGateType::kOUTPUT, true, weightMap[lname + ".weight_ih_l0_reverse3"]);

    lstm->setWeightsForGate(1, RNNGateType::kINPUT, false, weightMap[lname + ".weight_hh_l0_reverse0"]);
    lstm->setWeightsForGate(1, RNNGateType::kFORGET, false, weightMap[lname + ".weight_hh_l0_reverse1"]);
    lstm->setWeightsForGate(1, RNNGateType::kCELL, false, weightMap[lname + ".weight_hh_l0_reverse2"]);
    lstm->setWeightsForGate(1, RNNGateType::kOUTPUT, false, weightMap[lname + ".weight_hh_l0_reverse3"]);

    lstm->setBiasForGate(1, RNNGateType::kINPUT, true, weightMap[lname + ".bias_ih_l0_reverse0"]);
    lstm->setBiasForGate(1, RNNGateType::kFORGET, true, weightMap[lname + ".bias_ih_l0_reverse1"]);
    lstm->setBiasForGate(1, RNNGateType::kCELL, true, weightMap[lname + ".bias_ih_l0_reverse2"]);
    lstm->setBiasForGate(1, RNNGateType::kOUTPUT, true, weightMap[lname + ".bias_ih_l0_reverse3"]);

    lstm->setBiasForGate(1, RNNGateType::kINPUT, false, weightMap[lname + ".bias_hh_l0_reverse0"]);
    lstm->setBiasForGate(1, RNNGateType::kFORGET, false, weightMap[lname + ".bias_hh_l0_reverse1"]);
    lstm->setBiasForGate(1, RNNGateType::kCELL, false, weightMap[lname + ".bias_hh_l0_reverse2"]);
    lstm->setBiasForGate(1, RNNGateType::kOUTPUT, false, weightMap[lname + ".bias_hh_l0_reverse3"]);
    return lstm;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {C, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../crnn.wts");

    // cnn
    auto x = convRelu(network, weightMap, *data, 0);
    auto p = network->addPoolingNd(*x->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    p->setStrideNd(DimsHW{2, 2});
    x = convRelu(network, weightMap, *p->getOutput(0), 1);
    p = network->addPoolingNd(*x->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    p->setStrideNd(DimsHW{2, 2});
    x = convRelu(network, weightMap, *p->getOutput(0), 2, true);
    x = convRelu(network, weightMap, *x->getOutput(0), 3);
    p = network->addPoolingNd(*x->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    p->setStrideNd(DimsHW{2, 1});
    p->setPaddingNd(DimsHW{0, 1});
    x = convRelu(network, weightMap, *p->getOutput(0), 4, true);
    x = convRelu(network, weightMap, *x->getOutput(0), 5);
    p = network->addPoolingNd(*x->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    p->setStrideNd(DimsHW{2, 1});
    p->setPaddingNd(DimsHW{0, 1});
    x = convRelu(network, weightMap, *p->getOutput(0), 6, true);

    auto sfl = network->addShuffle(*x->getOutput(0));
    sfl->setFirstTranspose(Permutation{1, 2, 0});

    // rnn
    auto lstm0 = addLSTM(network, weightMap, *sfl->getOutput(0), 256, "rnn.0.rnn");
    auto sfl0 = network->addShuffle(*lstm0->getOutput(0));
    sfl0->setReshapeDimensions(Dims4{26, 1, 1, 512});
    auto fc0 = network->addFullyConnected(*sfl0->getOutput(0), 256, weightMap["rnn.0.embedding.weight"], weightMap["rnn.0.embedding.bias"]);

    sfl = network->addShuffle(*fc0->getOutput(0));
    sfl->setFirstTranspose(Permutation{2, 3, 0, 1});
    sfl->setReshapeDimensions(Dims3{1, 26, 256});

    auto lstm1 = addLSTM(network, weightMap, *sfl->getOutput(0), 256, "rnn.1.rnn");
    auto sfl1 = network->addShuffle(*lstm1->getOutput(0));
    sfl1->setReshapeDimensions(Dims4{26, 1, 1, 512});
    auto fc1 = network->addFullyConnected(*sfl1->getOutput(0), 37, weightMap["rnn.1.embedding.weight"], weightMap["rnn.1.embedding.bias"]);
    Dims dims = fc1->getOutput(0)->getDimensions();
    std::cout << "fc1 shape " << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << std::endl;

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
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

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 1 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("crnn.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("crnn.engine", std::ios::binary);
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
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./crnn -s  // serialize model to plan file" << std::endl;
        std::cerr << "./crnn -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 1 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 1 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 1 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cv::Mat img = cv::imread("demo.png");
    if (img.empty()) {
        std::cerr << "demo.png not found !!!" << std::endl;
        return -1;
    }
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)img.at<uchar>(i) / 255.0 - 0.5) * 2.0;
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<int> preds;
    for (int i = 0; i < 26; i++) {
        int maxj = 0;
        for (int j = 1; j < 37; j++) {
            if (prob[37 * i + j] > prob[37 * i + maxj]) maxj = j;
        }
        preds.push_back(maxj);
    }
    std::cout << "raw: " << strDecode(preds, true) << std::endl;
    std::cout << "sim: " << strDecode(preds, false) << std::endl;

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
