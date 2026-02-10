#include <NvInfer.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include "logging.h"
#include "utils.h"

using WeightMap = std::map<std::string, Weights>;
using M = nvinfer1::MatrixOperation;
using E = nvinfer1::ElementWiseOperation;
using NDCF = nvinfer1::NetworkDefinitionCreationFlag;

static Logger gLogger;

// stuff we know about googlenet
static constexpr const std::size_t N = 1;
static constexpr const int32_t INPUT_H = 224;
static constexpr const int32_t INPUT_W = 224;
static constexpr const std::array<int32_t, 2> SIZES = {3 * INPUT_H * INPUT_W, 1000};
static constexpr const std::array<const char*, 2> NAMES = {"data", "prob"};
static constexpr const bool TRT_PREPROCESS = TRT_VERSION >= 8510 ? true : false;
static constexpr const char* WTS_PATH = "../models/googlenet.wts";
static constexpr const char* ENGINE_PATH = "../models/googlenet.engine";
static constexpr const char* LABELS_PATH = "../assets/imagenet1000_clsidx_to_labels.txt";
static constexpr const std::array<const float, 3> mean = {0.485f, 0.456f, 0.406f};
static constexpr const std::array<const float, 3> stdv = {0.229f, 0.224f, 0.225f};

auto addBatchNorm2d(INetworkDefinition* network, WeightMap& m, ITensor& input, const std::string& lname,
                    float eps = 1e-3) -> ILayer* {
    static Weights none{DataType::kFLOAT, nullptr, 0ll};
    float* gamma = (float*)m[lname + ".weight"].values;
    float* beta = (float*)m[lname + ".bias"].values;
    float* mean = (float*)m[lname + ".running_mean"].values;
    float* var = (float*)m[lname + ".running_var"].values;
    int64_t len = m[lname + ".running_var"].count;

    auto* scval = static_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    auto* shift_val = static_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shift_val[i] = beta[i] - (mean[i] * scval[i]);
    }
    Weights shift{DataType::kFLOAT, shift_val, len};

    m[lname + ".scale"] = scale;
    m[lname + ".shift"] = shift;
    m[lname + ".power"] = none;
    auto* bn = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, none);
    assert(bn);
    bn->setName(lname.c_str());
    return bn;
}

/**
 * @brief A basic conv2d+bn+relu layer from googlenet
 *
 * @param network network definition from TensorRT API
 * @param weightMap weight map
 * @param input input tensor
 * @param outch output channels
 * @param k kernel size for convolution
 * @param s stride size for convolution
 * @param p padding size for convolution
 * @param lname layer name from weight map
 * @return ILayer*
 */
ILayer* basicConv2d(INetworkDefinition* network, WeightMap& weightMap, ITensor& input, const std::string& lname,
                    int32_t outch, int k, int s = 1, int p = 0) {
    static const Weights none{DataType::kFLOAT, nullptr, 0ll};
    auto* conv = network->addConvolutionNd(input, outch, DimsHW{k, k}, weightMap[lname + ".conv.weight"], none);
    auto* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn");
    auto* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(conv && bn && relu);
    conv->setName(lname.c_str());
    bn->setName((lname + ".bn").c_str());
    relu->setName((lname + ".relu").c_str());
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    return relu;
}

/**
 * @brief Inception module from googlenet implementation in torchvision, see:
 * https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/googlenet.py#L184
 *
 * @param network network definition from TensorRT API
 * @param weightMap weight map
 * @param input input tensor
 * @param lname layer name from weight map
 * @param ch1x1
 * @param ch3x3red
 * @param ch3x3
 * @param ch5x5red
 * @param ch5x5
 * @param pool_proj
 * @return IConcatenationLayer*
 */
IConcatenationLayer* inception(INetworkDefinition* network, WeightMap& weightMap, ITensor& input,
                               const std::string& lname, int ch1x1, int ch3x3red, int ch3x3, int ch5x5red, int ch5x5,
                               int pool_proj) {
    // "cbr" means "Conv-Batchnorm-Relu"
    auto* cbr1 = basicConv2d(network, weightMap, input, lname + "branch1", ch1x1, 1);
    auto* cbr2 = basicConv2d(network, weightMap, input, lname + "branch2.0", ch3x3red, 1);
    auto* cbr3 = basicConv2d(network, weightMap, *cbr2->getOutput(0), lname + "branch2.1", ch3x3, 3, 1, 1);
    auto* cbr4 = basicConv2d(network, weightMap, input, lname + "branch3.0", ch5x5red, 1);
    auto* cbr5 = basicConv2d(network, weightMap, *cbr4->getOutput(0), lname + "branch3.1", ch5x5, 3, 1, 1);
    auto* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{3, 3});
    auto* cbr6 = basicConv2d(network, weightMap, *pool1->getOutput(0), lname + "branch4.1", pool_proj, 1);
    assert(cbr1 && cbr2 && cbr3 && cbr4 && cbr5 && pool1 && cbr6);
    pool1->setStrideNd(DimsHW{1, 1});
    pool1->setPaddingNd(DimsHW{1, 1});
    pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

    std::array<ITensor*, 4> inputTensors = {cbr1->getOutput(0), cbr3->getOutput(0), cbr5->getOutput(0),
                                            cbr6->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors.data(), 4);
    assert(cat1);
    return cat1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    WeightMap weightMap = loadWeights(WTS_PATH);

#if TRT_VERSION >= 11200
    auto flag = 1U << static_cast<int>(NDCF::kSTRONGLY_TYPED);
#elif TRT_VERSION >= 10000
    auto flag = 0U;
#else
    auto flag = 1U << static_cast<int>(NDCF::kEXPLICIT_BATCH);
#endif
    auto* network = builder->createNetworkV2(flag);

    ITensor* input{nullptr};
    if constexpr (TRT_PREPROCESS) {
        dt = DataType::kUINT8;
        input = network->addInput(NAMES[0], dt, Dims4{N, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(network, *input, true, mean, stdv);
        input = trans->getOutput(0);
    } else {
        input = network->addInput(NAMES[0], dt, Dims4{N, 3, INPUT_H, INPUT_W});
    }
    assert(input);

    auto* relu1 = basicConv2d(network, weightMap, *input, "conv1", 64, 7, 2, 3);
    auto* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    pool1->setName("pool1");

    auto* relu2 = basicConv2d(network, weightMap, *pool1->getOutput(0), "conv2", 64, 1);
    auto* relu3 = basicConv2d(network, weightMap, *relu2->getOutput(0), "conv3", 192, 3, 1, 1);
    auto* pool2 = network->addPoolingNd(*relu3->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});
    pool2->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    pool2->setName("pool2");

    auto* cat1 = inception(network, weightMap, *pool2->getOutput(0), "inception3a.", 64, 96, 128, 16, 32, 32);
    auto* cat2 = inception(network, weightMap, *cat1->getOutput(0), "inception3b.", 128, 128, 192, 32, 96, 64);
    auto* pool3 = network->addPoolingNd(*cat2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool3);
    pool3->setStrideNd(DimsHW{2, 2});
    pool3->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    pool3->setName("pool3");

    auto* cat3 = inception(network, weightMap, *pool3->getOutput(0), "inception4a.", 192, 96, 208, 16, 48, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4b.", 160, 112, 224, 24, 64, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4c.", 128, 128, 256, 24, 64, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4d.", 112, 144, 288, 32, 64, 64);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception4e.", 256, 160, 320, 32, 128, 128);

    IPoolingLayer* pool4 = network->addPoolingNd(*cat3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool4);
    pool4->setStrideNd(DimsHW{2, 2});
    pool4->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
    pool4->setName("pool4");

    cat3 = inception(network, weightMap, *pool4->getOutput(0), "inception5a.", 256, 160, 320, 32, 128, 128);
    cat3 = inception(network, weightMap, *cat3->getOutput(0), "inception5b.", 384, 192, 384, 48, 128, 128);

    // this is a AdaptiveAvgPool2d in pytorch implementation
    IPoolingLayer* pool5 = network->addPoolingNd(*cat3->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    auto* shuffle = network->addShuffle(*pool5->getOutput(0));
    assert(pool5 && shuffle);
    shuffle->setName("shuffle");
    shuffle->setReshapeDimensions(Dims2{1, -1});  // "-1" means "1024"

    auto* fcw = network->addConstant(Dims2{1000, 1024}, weightMap["fc.weight"])->getOutput(0);
    auto* fcb = network->addConstant(Dims2{1, 1000}, weightMap["fc.bias"])->getOutput(0);
    auto* fc0 = network->addMatrixMultiply(*shuffle->getOutput(0), M::kNONE, *fcw, M::kTRANSPOSE);
    auto* fc1 = network->addElementWise(*fc0->getOutput(0), *fcb, E::kSUM);

    fc1->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*fc1->getOutput(0));
    // Build engine
#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete network;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif

    std::cout << "build finished\n";
    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)mem.second.values);
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
    checkTrtEnv();
    if (argc != 2) {
        std::cerr << "arguments not right!\n";
        std::cerr << "./googlenet -s   // serialize model to plan file\n";
        std::cerr << "./googlenet -d   // deserialize plan file and run inference\n";
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
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

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    const std::string img_path = "../assets/cats.jpg";
    void* input = nullptr;
    std::vector<float> flat_img;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);

    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }
    assert(input);

    for (int32_t i = 0; i < 100; ++i) {
        auto _start = std::chrono::system_clock::now();
        auto prob = doInference(*context, input, 1);
        auto _end = std::chrono::system_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
        std::cout << "Execution time: " << _time << "us\n";

        for (const auto& vector : prob) {
            int idx = 0;
            for (auto v : vector) {
                std::cout << std::setprecision(4) << v << ", " << std::flush;
                if (++idx > 20) {
                    std::cout << "\n====\n";
                    break;
                }
            }
        }

        if (i == 99) {
            std::cout << "prediction result:\n";
            auto labels = loadImagenetLabelMap(LABELS_PATH);
            int _top = 0;
            for (auto& [idx, logits] : topk(prob[0], 3)) {
                std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits
                          << ", label: " << labels[idx] << "\n";
            }
        }
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

    return 0;
}
