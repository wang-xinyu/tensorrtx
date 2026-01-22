#include <NvInfer.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "logging.h"

#include "utils.h"

// stuff we know about mnasnet and the input/output blobs
static constexpr const int INPUT_H = 224;
static constexpr const int INPUT_W = 224;
static constexpr const int OUTPUT_SIZE = 1000;
static constexpr int N = 1;
static constexpr const char* NAMES[] = {"data", "prob"};
static constexpr const int SIZES[] = {3 * INPUT_H * INPUT_W, OUTPUT_SIZE};
static const std::string WTS_PATH = "../models/mnasnet0_5.wts";
static const std::string ENGINE_PATH = "../models/mnasnet0_5.engine";
static constexpr const char LABELS_PATH[] = "../assets/imagenet1000_clsidx_to_labels.txt";
static constexpr const bool TRT_PREPROCESS = TRT_VERSION >= 8510 ? true : false;
static constexpr const float mean[3] = {0.485f, 0.456f, 0.406f};
static constexpr const float stdv[3] = {0.229f, 0.224f, 0.225f};

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;
using M = nvinfer1::MatrixOperation;
using E = nvinfer1::ElementWiseOperation;

static Logger gLogger;

ILayer* addBatchNorm2d(INetworkDefinition* network, WeightMap& weightMap, ITensor& input, std::string lname,
                       float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << lname << " running_var's len: " << len << std::endl;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
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

ILayer* CBR(INetworkDefinition* net, WeightMap& map, const std::string& name, ITensor& input, int o, int k, int s,
            int p, int d, int g, float eps = 1e-5f, int start_index = 0, bool has_relu = true) {
    Weights bias{DataType::kFLOAT, nullptr, 0};

    // conv -> bn -> relu
    auto conv_name = name + "." + std::to_string(start_index++) + ".weight";
    if (map.find(conv_name) == map.end()) {
        throw std::invalid_argument("KeyError: " + name + "is not in weight map");
    }
    auto* conv = net->addConvolutionNd(input, o, DimsHW{k, k}, map[conv_name], bias);
    if (conv == nullptr) {
        throw std::runtime_error("build conv layer failed in " + name);
    }
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    conv->setDilationNd(DimsHW{d, d});
    conv->setNbGroups(g);
    conv->setName(conv_name.c_str());

    std::string bn_name = name + "." + std::to_string(start_index);
    auto* bn = addBatchNorm2d(net, map, *conv->getOutput(0), bn_name, eps);
    if (has_relu) {
        auto* relu = net->addActivation(*bn->getOutput(0), ActivationType::kRELU);
        if (relu == nullptr) {
            throw std::runtime_error("build relu layer failed in " + name);
        }
        return relu;
    } else {
        return bn;
    }
}

ILayer* invertedRes(INetworkDefinition* network, WeightMap& w, ITensor& input, std::string lname, int inch, int outch,
                    int ksize, int s, int exp) {
    std::cout << "Building layer: " << lname << std::endl;
    static const Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int midch = inch * exp;
    auto* conv1 = network->addConvolutionNd(input, midch, DimsHW{1, 1}, w[lname + "layers.0.weight"], emptywts);
    assert(conv1);
    auto* bn1 = addBatchNorm2d(network, w, *conv1->getOutput(0), lname + "layers.1", 1e-5);
    auto* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    auto* conv2 = network->addConvolutionNd(*relu1->getOutput(0), midch, DimsHW{ksize, ksize},
                                            w[lname + "layers.3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{ksize / 2, ksize / 2});
    conv2->setNbGroups(midch);
    auto* bn2 = addBatchNorm2d(network, w, *conv2->getOutput(0), lname + "layers.4", 1e-5);
    auto* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    auto* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{1, 1}, w[lname + "layers.6.weight"],
                                            emptywts);
    assert(conv3);
    auto* bn3 = addBatchNorm2d(network, w, *conv3->getOutput(0), lname + "layers.7", 1e-5);

    if (inch == outch && s == 1) {
        auto* ew1 = network->addElementWise(*bn3->getOutput(0), input, ElementWiseOperation::kSUM);
        assert(ew1);
        return ew1;
    }
    return bn3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config,
                          DataType dt) {
    auto weightMap = loadWeights(WTS_PATH);
#if TRT_VERSION >= 10000
    auto* network = builder->createNetworkV2(0);
#else
    auto* network = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif
    ITensor* data{nullptr};
    if constexpr (TRT_PREPROCESS) {
#if TRT_VERSION > 8510
        dt = DataType::kUINT8;
#else
        dt = DataType::kINT8;
#endif
        data = network->addInput(NAMES[0], dt, Dims4{N, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(network, *data, true, mean, stdv);
        data = trans->getOutput(0);
    } else {
        data = network->addInput(NAMES[0], dt, Dims4{N, 3, INPUT_H, INPUT_W});
    }
    assert(data);

    static const Weights emptywts{DataType::kFLOAT, nullptr, 0};

    int start_idx = 0;
    auto* cbr_0 = CBR(network, weightMap, "layers", *data, 16, 3, 2, 1, 1, 1, 1e-5, start_idx, true);
    start_idx += 3;
    auto* cbr_1 = CBR(network, weightMap, "layers", *cbr_0->getOutput(0), 16, 3, 1, 1, 1, 16, 1e-5, start_idx, true);
    start_idx += 3;
    auto* cbr_2 = CBR(network, weightMap, "layers", *cbr_1->getOutput(0), 8, 1, 1, 1, 1, 1, 1, start_idx, false);

    ILayer* ir1 = invertedRes(network, weightMap, *cbr_2->getOutput(0), "layers.8.0.", 8, 16, 3, 2, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.8.1.", 16, 16, 3, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.8.2.", 16, 16, 3, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.9.0.", 16, 24, 5, 2, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.9.1.", 24, 24, 5, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.9.2.", 24, 24, 5, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.10.0.", 24, 40, 5, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.10.1.", 40, 40, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.10.2.", 40, 40, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.11.0.", 40, 48, 3, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.11.1.", 48, 48, 3, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.0.", 48, 96, 5, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.1.", 96, 96, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.2.", 96, 96, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.3.", 96, 96, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.13.0.", 96, 160, 3, 1, 6);

    auto* cbr_3 = CBR(network, weightMap, "layers", *ir1->getOutput(0), 1280, 1, 1, 0, 1, 1, 1e-5, 14, true);

    auto* avg = network->addReduce(*cbr_3->getOutput(0), ReduceOperation::kAVG, 0xc, false);
    auto* _fcw = network->addConstant(DimsHW{1000, 1280}, weightMap["classifier.1.weight"]);
    auto* _fcb = network->addConstant(DimsHW{1, 1000}, weightMap["classifier.1.bias"]);
    auto* _fc1 = network->addMatrixMultiply(*avg->getOutput(0), M::kNONE, *_fcw->getOutput(0), M::kTRANSPOSE);
    auto* fc1 = network->addElementWise(*_fc1->getOutput(0), *_fcb->getOutput(0), E::kSUM);
    assert(fc1);

    fc1->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*fc1->getOutput(0));

    // Build engine
#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    auto* _serialized = builder->buildSerializedNetwork(*network, *config);
    auto* engine = runtime->deserializeCudaEngine(_serialized->data(), _serialized->size());
    delete _serialized;
    delete network;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    auto* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif
    std::cout << "build out" << std::endl;

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IRuntime* runtime, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
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

std::vector<std::vector<float>> do_inference(IExecutionContext& context, void* input, int32_t batch_size) {
    const ICudaEngine& engine = context.getEngine();
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
        size = s * batch_size * SIZES[i];
        CHECK(cudaMalloc(&buffers[i], size));
        if (i == 0) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
        context.setTensorAddress(tensor_name, buffers[i]);
#else
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        auto s = getSize(engine.getBindingDataType(idx));
        assert(idx == i);
        size = s * batch_size * SIZES[i];
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
        std::vector<float> tmp(batch_size * SIZES[i], std::nan(""));
        std::size_t size = batch_size * SIZES[i] * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(tmp);
    }
    CHECK(cudaStreamSynchronize(stream));

    cudaStreamDestroy(stream);
    for (auto i = 0; i < nIO; ++i) {
        CHECK(cudaFree(buffers[i]));
    }
    return prob;
}

int main(int argc, char** argv) {
    checkTrtEnv();
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mnasnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mnasnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    auto* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // create a model using the API directly and serialize it to a stream
    char* trt_model_stream{nullptr};
    std::size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(N, runtime, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(ENGINE_PATH, std::ios::binary);
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
            trt_model_stream = new char[size];
            assert(trt_model_stream);
            file.read(trt_model_stream, size);
            file.close();
        }
    } else {
        return -1;
    }

#if TRT_VERSION >= 8000
    auto* engine = runtime->deserializeCudaEngine(trt_model_stream, size);
#else
    auto* engine = runtime->deserializeCudaEngine(trt_model_stream, size, nullptr);
#endif
    assert(engine != nullptr);
    auto* context = engine->createExecutionContext();
    assert(context != nullptr);

    void* input = nullptr;
    std::vector<float> flat_img;
    cv::Mat img;
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        img = cv::imread("../assets/cats.jpg", cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        img = cv::imread("../assets/cats.jpg", cv::IMREAD_COLOR);
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }

    for (int32_t i = 0; i < 100; ++i) {
        auto _start = std::chrono::system_clock::now();
        auto prob = do_inference(*context, input, 1);
        auto _end = std::chrono::system_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count();
        std::cout << "Execution time: " << _time << "ms" << std::endl;

        for (auto vector : prob) {
            int idx = 0;
            for (auto v : vector) {
                std::cout << std::setprecision(4) << v << ", " << std::flush;
                if (++idx > 20) {
                    std::cout << "\n====" << std::endl;
                    break;
                }
            }
        }

        if (i == 99) {
            std::cout << "prediction result: " << std::endl;
            auto labels = loadImagenetLabelMap(LABELS_PATH);
            int _top = 0;
            for (auto& [idx, logits] : topk(prob[0], 3)) {
                std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits
                          << ", label: " << labels[idx] << std::endl;
            }
        }
    }

    delete[] trt_model_stream;
    return 0;
}
