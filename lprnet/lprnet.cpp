#include <NvInfer.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "logging.h"
#include "utils.h"

using namespace nvinfer1;

using WeightMap = std::map<std::string, Weights>;

static Logger gLogger;

#define DEVICE 0
static constexpr const int32_t BATCH_SIZE = 1;
static constexpr const char* WTS_PATH = "../models/LPRNet.wts";
static constexpr const char* ENGINE_PATH = "../models/LPRNet.engine";
// stuff we know about the network and the input/output blobs
static constexpr const int32_t INPUT_H = 24;
static constexpr const int32_t INPUT_W = 94;
static constexpr const char* NAMES[2] = {"data", "prob"};
static constexpr const int32_t SIZES[2] = {3 * INPUT_H * INPUT_W, 18 * 68};
static constexpr const bool TRT_PREPROCESS = TRT_VERSION >= 8510 ? true : false;
static constexpr const float mean[3] = {0.5f, 0.5f, 0.5f};
static constexpr const float stdv[3] = {1.f, 1.f, 1.f};

const std::string alphabet[] = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽",
                                "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
                                "青", "宁", "新", "0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "A",
                                "B",  "C",  "D",  "E",  "F",  "G",  "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",
                                "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z",  "I",  "O",  "-"};

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, WeightMap& weightMap, ITensor& input, std::string lname,
                            float eps = 1e-5) {
    const float* gamma = reinterpret_cast<const float*>(weightMap[lname + ".weight"].values);
    const float* beta = reinterpret_cast<const float*>(weightMap[lname + ".bias"].values);
    const float* mean = reinterpret_cast<const float*>(weightMap[lname + ".running_mean"].values);
    const float* var = reinterpret_cast<const float*>(weightMap[lname + ".running_var"].values);
    int64_t len = weightMap[lname + ".running_var"].count;

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
        pval[i] = 1.0f;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    scale_1->setName(lname.c_str());
    return scale_1;
}

IConvolutionLayer* smallBasicBlock(INetworkDefinition* network, WeightMap& w, ITensor& input, int ch_out,
                                   std::string lname) {
    int o = ch_out / 4, i = 0;
    ITensor* cur_input = &input;
    IConvolutionLayer* ret{nullptr};
    struct {
        DimsHW k_dim, p_dim;
        int ch_out;
        std::string w_name, b_name;
    } conv_params[] = {
            {DimsHW{1, 1}, DimsHW{0, 0}, o, lname + ".block.0.weight", lname + ".block.0.bias"},
            {DimsHW{3, 1}, DimsHW{1, 0}, o, lname + ".block.2.weight", lname + ".block.2.bias"},
            {DimsHW{1, 3}, DimsHW{0, 1}, o, lname + ".block.4.weight", lname + ".block.4.bias"},
            {DimsHW{1, 1}, DimsHW{0, 0}, ch_out, lname + ".block.6.weight", lname + ".block.6.bias"},
    };
    for (const auto& param : conv_params) {
        ret = network->addConvolutionNd(*cur_input, param.ch_out, param.k_dim, w[param.w_name], w[param.b_name]);
        assert(ret);
        ret->setPaddingNd(param.p_dim);
        ret->setName((lname + ".block." + std::to_string(i++)).c_str());
        if (i != 4) {
            auto* relu = network->addActivation(*ret->getOutput(0), ActivationType::kRELU);
            assert(relu);
            cur_input = relu->getOutput(0);
        } else {
            cur_input = ret->getOutput(0);
        }
    }
    return ret;
}

ICudaEngine* createEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    const int nc = 68;
    WeightMap w = loadWeights(WTS_PATH);

#if TRT_VERSION >= 10000
    auto* network = builder->createNetworkV2(0);
#else
    auto* network = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

    ITensor* data{nullptr};
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
#if TRT_VERSION > 8510
        dt = DataType::kUINT8;
#else
        dt = DataType::kINT8;
#endif
        auto* input = network->addInput(NAMES[0], dt, Dims4{N, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(network, *input, false, mean, stdv);
        data = trans->getOutput(0);
    } else {
        data = network->addInput(NAMES[0], dt, Dims4{N, 3, INPUT_H, INPUT_W});
    }
    assert(data);

    // CBR (Conv-BatchNorm-ReLU)
    auto* c0 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, w["backbone.0.weight"], w["backbone.0.bias"]);
    auto* bn0 = addBatchNorm2d(network, w, *c0->getOutput(0), "backbone.1");
    auto* relu0 = network->addActivation(*bn0->getOutput(0), ActivationType::kRELU);

    auto* f0 = network->addPoolingNd(*relu0->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    f0->setStrideNd(DimsHW{1, 1});
    assert(c0 && bn0 && relu0);

    auto* sm0 = smallBasicBlock(network, w, *f0->getOutput(0), 128, "backbone.4");
    auto* bn1 = addBatchNorm2d(network, w, *sm0->getOutput(0), "backbone.5");
    auto* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(sm0 && bn1 && relu1);

    // need to unsqueeze to 5D tensor for 3D pooling
    auto* to5d0 = network->addShuffle(*relu1->getOutput(0));
    to5d0->setReshapeDimensions({5, {BATCH_SIZE, 1, 128, 20, 90}});
    auto* f1 = network->addPoolingNd(*to5d0->getOutput(0), PoolingType::kMAX, Dims3{1, 3, 3});
    f1->setStrideNd(Dims3{2, 1, 2});
    f1->setName("MaxPool3d_1");
    auto* to5d1 = network->addShuffle(*f1->getOutput(0));
    to5d1->setReshapeDimensions(Dims4{BATCH_SIZE, 64, 18, 44});

    auto* sm1 = smallBasicBlock(network, w, *to5d1->getOutput(0), 256, "backbone.8");
    auto* bn2 = addBatchNorm2d(network, w, *sm1->getOutput(0), "backbone.9");
    auto* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    auto* sm2 = smallBasicBlock(network, w, *relu2->getOutput(0), 256, "backbone.11");
    auto* bn3 = addBatchNorm2d(network, w, *sm2->getOutput(0), "backbone.12");
    auto* relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);

    // need to unsqueeze to 5D tensor for 3D pooling
    auto* to5d2 = network->addShuffle(*relu3->getOutput(0));
    to5d2->setReshapeDimensions({5, {BATCH_SIZE, 1, 256, 18, 44}});
    auto* f2 = network->addPoolingNd(*to5d2->getOutput(0), PoolingType::kMAX, Dims3{1, 3, 3});
    f2->setStrideNd(Dims3{4, 1, 2});
    f2->setName("MaxPool3d_2");
    auto* to5d3 = network->addShuffle(*f2->getOutput(0));
    to5d3->setReshapeDimensions(Dims4{BATCH_SIZE, 64, 16, 21});

    // CBR (Conv-BatchNorm-ReLU)
    c0 = network->addConvolutionNd(*to5d3->getOutput(0), 256, DimsHW{1, 4}, w["backbone.16.weight"],
                                   w["backbone.16.bias"]);
    auto* bn4 = addBatchNorm2d(network, w, *c0->getOutput(0), "backbone.17");
    auto* relu5 = network->addActivation(*bn4->getOutput(0), ActivationType::kRELU);

    // CBR (Conv-BatchNorm-ReLU)
    c0 = network->addConvolutionNd(*relu5->getOutput(0), nc, DimsHW{13, 1}, w["backbone.20.weight"],
                                   w["backbone.20.bias"]);
    auto* bn5 = addBatchNorm2d(network, w, *c0->getOutput(0), "backbone.21");
    auto* backbone = network->addActivation(*bn5->getOutput(0), ActivationType::kRELU);

    int pow_idx = 0;
    auto makeGlobalContext = [&](ITensor* feat, bool pool5, bool pool4x10) -> ITensor* {
        static int j = 0;
        ITensor* t = feat;
        if (pool5) {
            auto* pool = network->addPoolingNd(*t, PoolingType::kAVERAGE, DimsHW{5, 5});
            assert(pool);
            pool->setStrideNd(DimsHW{5, 5});
            auto _name = "global5." + std::to_string(j);
            pool->setName(_name.c_str());
            t = pool->getOutput(0);
        }
        if (pool4x10) {
            auto* pool = network->addPoolingNd(*t, PoolingType::kAVERAGE, DimsHW{4, 10});
            assert(pool);
            pool->setStrideNd(DimsHW{4, 2});
            auto _name = "global4x10." + std::to_string(j);
            pool->setName(_name.c_str());
            t = pool->getOutput(0);
        }

        // pow
        Dims dims = t->getDimensions();
        int64_t size = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3];
        void* data = malloc(sizeof(float) * size);
        for (int i = 0; i < size; ++i) {
            reinterpret_cast<float*>(data)[i] = 2.0f;
        }
        auto name = "pow." + std::to_string(j);
        w[name] = {DataType::kFLOAT, data, size};
        auto* pow_const = network->addConstant(dims, w[name]);
        auto* pow = network->addElementWise(*t, *pow_const->getOutput(0), ElementWiseOperation::kPOW);
        assert(pow);
        pow->setName(name.c_str());

        // mean
        int32_t mask = (1 << dims.nbDims) - 1;
        auto* mean = network->addReduce(*pow->getOutput(0), ReduceOperation::kAVG, mask, true);
        auto _mean_name = "mean." + std::to_string(j);
        mean->setName(_mean_name.c_str());

        // div
        auto* div = network->addElementWise(*t, *mean->getOutput(0), ElementWiseOperation::kDIV);
        auto _div_name = "div." + std::to_string(j);
        div->setName(_div_name.c_str());
        ++j;
        return div->getOutput(0);
    };

    auto* gc0 = makeGlobalContext(relu0->getOutput(0), true, false);
    auto* gc1 = makeGlobalContext(relu1->getOutput(0), true, false);
    auto* gc2 = makeGlobalContext(relu3->getOutput(0), false, true);
    auto* gc3 = makeGlobalContext(backbone->getOutput(0), false, false);
    ITensor* const gcs[] = {gc0, gc1, gc2, gc3};
    auto* cat = network->addConcatenation(gcs, 4);
    assert(cat);
    cat->setAxis(1);

    auto* c = network->addConvolutionNd(*cat->getOutput(0), nc, DimsHW{1, 1}, w["container.0.weight"],
                                        w["container.0.bias"]);
    auto* logits = network->addReduce(*c->getOutput(0), ReduceOperation::kAVG, 0x04, false);
    logits->getOutput(0)->setName(NAMES[1]);

    network->markOutput(*logits->getOutput(0));

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

    std::cout << "build finished" << std::endl;
    // Release host memory
    for (auto& mem : w) {
        free((void*)mem.second.values);
    }

    return engine;
}

void APIToModel(int32_t N, IRuntime* runtime, IHostMemory** modelStream) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    ICudaEngine* engine = createEngine(N, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

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

std::vector<std::vector<float>> doInference(IExecutionContext& context, void* input, int batchSize) {
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
        std::vector<float> tmp(batchSize * SIZES[i], std::nan(""));
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
    cudaSetDevice(DEVICE);
    checkTrtEnv(DEVICE);
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./LPRnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./LPRnet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, runtime, &modelStream);
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

    void* input = nullptr;
    std::vector<float> data;
    cv::Mat img = cv::imread("../assets/car_plate.jpg");
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_CUBIC);
        input = static_cast<void*>(img.data);
    } else {
        data = preprocess_img(img, false, mean, stdv, BATCH_SIZE, INPUT_H, INPUT_W);
        input = data.data();
    }

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    for (int32_t i = 0; i < 100; ++i) {
        auto _start = std::chrono::system_clock::now();
        auto prob = doInference(*context, input, 1);
        auto _end = std::chrono::system_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
        std::cout << "Execution time: " << _time << "us" << std::endl;

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
            int prev = 67;
            std::string str;
            for (int t = 0; t < 18; ++t) {
                std::array<float, 68> scores{};
                for (int c = 0; c < 68; ++c) {
                    scores[c] = prob[0][t + 18 * c];
                }
                int best =
                        static_cast<int>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
                if (best != prev && best != 67)
                    str += alphabet[best];
                prev = best;
            }
            std::cout << "result: " << str << std::endl;
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
