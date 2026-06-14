#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>
#include "logging.h"
#include "macros.h"
#include "utils.h"

using WeightMap = std::map<std::string, Weights>;
using M = nvinfer1::MatrixOperation;
using E = nvinfer1::ElementWiseOperation;
using LayerConfig = std::vector<std::string>;
using LayerConfigMap = std::map<std::string, LayerConfig>;

static Logger g_logger;

struct VggVariant {
    std::string name;
    std::string cfg_name;
    bool batch_norm;
    std::string wts_path;
    std::string engine_path;
};

static constexpr int N = 1;
static constexpr const int32_t INPUT_H = 224;
static constexpr const int32_t INPUT_W = 224;
static constexpr const std::array<int32_t, 2> SIZES = {3 * INPUT_H * INPUT_W, 1000};
static constexpr const std::array<const char*, 2> NAMES = {"data", "prob"};
static constexpr const char* LABELS_PATH = "assets/imagenet1000_clsidx_to_labels.txt";
static constexpr const bool TRT_PREPROCESS = TRT_VERSION_GE(8, 5, 1);
static constexpr const std::array<const float, 3> mean = {0.485f, 0.456f, 0.406f};
static constexpr const std::array<const float, 3> stdv = {0.229f, 0.224f, 0.225f};

const LayerConfigMap CFGS = {
        {"A", {"64", "M", "128", "M", "256", "256", "M", "512", "512", "M", "512", "512", "M"}},
        {"B", {"64", "64", "M", "128", "128", "M", "256", "256", "M", "512", "512", "M", "512", "512", "M"}},
        {"D",
         {"64", "64", "M", "128", "128", "M", "256", "256", "256", "M", "512", "512", "512", "M", "512", "512", "512",
          "M"}},
        {"E", {"64",  "64",  "M",   "128", "128", "M",   "256", "256", "256", "256", "M",
               "512", "512", "512", "512", "M",   "512", "512", "512", "512", "M"}}};

static auto normalizeModelName(const std::string& raw_name) -> std::string {
    std::string name;
    name.reserve(raw_name.size());
    for (char ch : raw_name) {
        const auto c = static_cast<unsigned char>(ch);
        name.push_back(ch == '-' ? '_' : static_cast<char>(std::tolower(c)));
    }
    return name;
}

static auto getVariantConfig(const std::string& raw_name) -> VggVariant {
    const auto name = normalizeModelName(raw_name);
    static const std::map<std::string, std::pair<std::string, bool>> variants = {
            {"vgg11", {"A", false}}, {"vgg11_bn", {"A", true}}, {"vgg13", {"B", false}}, {"vgg13_bn", {"B", true}},
            {"vgg16", {"D", false}}, {"vgg16_bn", {"D", true}}, {"vgg19", {"E", false}}, {"vgg19_bn", {"E", true}},
    };
    const auto iter = variants.find(name);
    if (iter == variants.end()) {
        std::cerr << "Unknown VGG variant: " << raw_name
                  << " (expected vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn)\n";
        std::abort();
    }

    return VggVariant{name, iter->second.first, iter->second.second, "models/" + name + ".wts",
                      "models/" + name + ".engine"};
}

static auto addBatchNorm2d(INetworkDefinition* network, WeightMap& w, ITensor& input, const std::string& lname,
                           float eps = 1e-5) -> IScaleLayer* {
    const float* gamma = static_cast<const float*>(w[lname + ".weight"].values);
    const float* beta = static_cast<const float*>(w[lname + ".bias"].values);
    const float* mean = static_cast<const float*>(w[lname + ".running_mean"].values);
    const float* var = static_cast<const float*>(w[lname + ".running_var"].values);
    int64_t len = w[lname + ".running_var"].count;

    auto* scval = new float[len];
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    auto* shval = new float[len];
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    auto* pval = new float[len];
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0f;
    }
    Weights power{DataType::kFLOAT, pval, len};

    w[lname + ".scale"] = scale;
    w[lname + ".shift"] = shift;
    w[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    scale_1->setName(lname.c_str());
    return scale_1;
}

static auto make_layers(INetworkDefinition* net, WeightMap& w, ITensor& input, const LayerConfig& cfg,
                        bool use_bn) -> ITensor* {
    auto* tensor = &input;
    int32_t idx = 0;
    std::string name = "features.";

    for (const auto& _v : cfg) {
        if (_v == "M") {
            auto* pool = net->addPoolingNd(*tensor, PoolingType::kMAX, DimsHW{2, 2});
            pool->setStrideNd(DimsHW{2, 2});
            pool->setName((name + std::to_string(idx)).c_str());
            tensor = pool->getOutput(0);
            ++idx;
            continue;
        }
        const int32_t v = std::stoi(_v);
        auto _name = "features." + std::to_string(idx);
        auto* _conv = net->addConvolutionNd(*tensor, v, DimsHW{3, 3}, w.at(_name + ".weight"), w.at(_name + ".bias"));
        _conv->setPaddingNd(DimsHW{1, 1});
        _conv->setName(_name.c_str());
        tensor = _conv->getOutput(0);
        ++idx;
        if (use_bn) {
            auto _bn_name = "features." + std::to_string(idx);
            auto* bn = addBatchNorm2d(net, w, *tensor, _bn_name);
            tensor = bn->getOutput(0);
            ++idx;
        }
        auto* relu = net->addActivation(*tensor, ActivationType::kRELU);
        relu->setName(("features." + std::to_string(idx)).c_str());
        tensor = relu->getOutput(0);
        ++idx;
    }

    return tensor;
}

auto create_engine(const VggVariant& variant, int32_t batch_size, IRuntime* runtime, IBuilder* builder,
                   IBuilderConfig* config, DataType dt) -> ICudaEngine* {
    auto w = loadWeights(variant.wts_path);
#if TRT_VERSION_GE(10, 0, 0)
    auto* net = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
#else
    auto* net = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

    ITensor* input{nullptr};
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        dt = DataType::kUINT8;
        input = net->addInput(NAMES[0], dt, Dims4{batch_size, INPUT_H, INPUT_W, 3});
        auto* trans = addTransformLayer(net, *input, true, mean, stdv);
        input = trans->getOutput(0);
    } else {
        input = net->addInput(NAMES[0], dt, Dims4{batch_size, 3, INPUT_H, INPUT_W});
    }
    assert(input);

    auto* features = make_layers(net, w, *input, CFGS.at(variant.cfg_name), variant.batch_norm);

    auto* _avg_pool = net->addPoolingNd(*features, PoolingType::kAVERAGE, Dims2{1, 1});
    auto* _flatten = net->addShuffle(*_avg_pool->getOutput(0));
    assert(_avg_pool && _flatten);
    _flatten->setReshapeDimensions(Dims2{batch_size, -1});

    auto* _fc1w =
            net->addConstant(Dims2{4096, static_cast<int64_t>(512 * 7 * 7)}, w["classifier.0.weight"])->getOutput(0);
    auto* _fc1b = net->addConstant(Dims2{1, 4096}, w["classifier.0.bias"])->getOutput(0);
    auto* _fc2w = net->addConstant(Dims2{4096, 4096}, w["classifier.3.weight"])->getOutput(0);
    auto* _fc2b = net->addConstant(Dims2{1, 4096}, w["classifier.3.bias"])->getOutput(0);
    auto* _fc3w = net->addConstant(Dims2{1000, 4096}, w["classifier.6.weight"])->getOutput(0);
    auto* _fc3b = net->addConstant(Dims2{1, 1000}, w["classifier.6.bias"])->getOutput(0);
    assert(_fc1w && _fc1b && _fc2w && _fc2b && _fc3w && _fc3b);

    auto* _fc1_0 = net->addMatrixMultiply(*_flatten->getOutput(0), M::kNONE, *_fc1w, M::kTRANSPOSE);
    auto* _fc1_1 = net->addElementWise(*_fc1_0->getOutput(0), *_fc1b, E::kSUM);
    auto* _relu1 = net->addActivation(*_fc1_1->getOutput(0), ActivationType::kRELU);

    auto* _fc2_0 = net->addMatrixMultiply(*_relu1->getOutput(0), M::kNONE, *_fc2w, M::kTRANSPOSE);
    auto* _fc2_1 = net->addElementWise(*_fc2_0->getOutput(0), *_fc2b, E::kSUM);
    auto* _relu2 = net->addActivation(*_fc2_1->getOutput(0), ActivationType::kRELU);

    auto* _fc3_0 = net->addMatrixMultiply(*_relu2->getOutput(0), M::kNONE, *_fc3w, M::kTRANSPOSE);
    auto* _fc3_1 = net->addElementWise(*_fc3_0->getOutput(0), *_fc3b, E::kSUM);

    _fc3_1->getOutput(0)->setName(NAMES[1]);
    net->markOutput(*_fc3_1->getOutput(0));

#if TRT_VERSION_GE(8, 0, 0)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    auto* _serialized = builder->buildSerializedNetwork(*net, *config);
    auto* _engine = runtime->deserializeCudaEngine(_serialized->data(), _serialized->size());
    delete _serialized;
    delete net;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    auto* _engine = builder->buildEngineWithConfig(*net, *config);
    net->destroy();
#endif
    std::cout << "build out" << '\n';

    releaseWeights(w);

    return _engine;
}

void APIToModel(const VggVariant& variant, int32_t batch_size, IRuntime* runtime, IHostMemory** model_stream) {
    auto* builder = createInferBuilder(g_logger);
    auto* config = builder->createBuilderConfig();

    auto* engine = create_engine(variant, batch_size, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    (*model_stream) = engine->serialize();

#if TRT_VERSION_GE(8, 0, 0)
    delete engine;
    delete config;
    delete builder;
#else
    engine->destroy();
    config->destroy();
    builder->destroy();
#endif
}

auto doInference(IExecutionContext& context, void* input, std::size_t batch_size) -> std::vector<std::vector<float>> {
    const ICudaEngine& engine = context.getEngine();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<void*> buffers;

#if TRT_VERSION_GE(8, 0, 0)
    const int32_t nIO = engine.getNbIOTensors();
#else
    const int32_t nIO = engine.getNbBindings();
#endif

    buffers.resize(nIO);
    for (auto i = 0; i < nIO; ++i) {
        std::size_t size = 0;
#if TRT_VERSION_GE(8, 0, 0)
        auto* tensor_name = engine.getIOTensorName(i);
        const std::string name = tensor_name;
        auto s = getSize(engine.getTensorDataType(tensor_name));
        size = s * batch_size * (name == NAMES[0] ? SIZES[0] : SIZES[1]);
        CHECK(cudaMalloc(&buffers[i], size));
        if (name == NAMES[0]) {
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
        }
        if (!context.setTensorAddress(tensor_name, buffers[i])) {
            std::cerr << "setTensorAddress failed\n";
            std::abort();
        }
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

#if TRT_VERSION_GE(8, 0, 0)
    if (!context.enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed\n";
        std::abort();
    }
#else
    if (!context.enqueueV2(buffers.data(), stream, nullptr)) {
        std::cerr << "enqueueV2 failed\n";
        std::abort();
    }
#endif

    std::vector<std::vector<float>> prob;
    for (int i = 0; i < nIO; ++i) {
#if TRT_VERSION_GE(8, 0, 0)
        const std::string name = engine.getIOTensorName(i);
        if (name == NAMES[0]) {
            continue;
        }
        constexpr auto output_size = SIZES[1];
#else
        if (i == 0) {
            continue;
        }
        const auto output_size = SIZES[i];
#endif
        std::vector<float> tmp(batch_size * output_size, std::nanf(""));
        std::size_t size = batch_size * output_size * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(tmp);
    }
    CHECK(cudaStreamSynchronize(stream));

    for (auto i = 0; i < nIO; ++i) {
        CHECK(cudaFree(buffers[i]));
    }
    CHECK(cudaStreamDestroy(stream));
    return prob;
}

auto main(int argc, char** argv) -> int {
    checkTrtEnv();
    if (argc < 2 || argc > 3) {
        std::cerr << "arguments not right!" << '\n';
        std::cerr << "./vgg -s [model]   // serialize model to plan file" << '\n';
        std::cerr << "./vgg -d [model]   // deserialize plan file and run inference" << '\n';
        std::cerr << "model choices: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn" << '\n';
        return -1;
    }
    const auto variant = getVariantConfig(argc == 3 ? argv[2] : "vgg11");
    std::cout << "Using VGG variant: " << variant.name << '\n';

    auto* runtime = createInferRuntime(g_logger);
    assert(runtime != nullptr);
    char* trt_model_stream{nullptr};
    std::streamsize size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* mem{nullptr};
        APIToModel(variant, 1, runtime, &mem);
        assert(mem != nullptr);

        std::ofstream _plan(variant.engine_path, std::ios::binary | std::ios::trunc);
        if (!_plan) {
            std::cerr << "could not open plan output file" << '\n';
            return -1;
        }
        if (mem->size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::cerr << "this model is too large to serialize\n";
            return -1;
        }
        const auto* data_ptr = reinterpret_cast<const char*>(mem->data());
        auto data_size = static_cast<std::streamsize>(mem->size());
        _plan.write(data_ptr, data_size);
#if TRT_VERSION_GE(8, 0, 0)
        delete mem;
        delete runtime;
#else
        mem->destroy();
        runtime->destroy();
#endif
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream _file(variant.engine_path, std::ios::binary);
        if (_file.good()) {
            _file.seekg(0, _file.end);
            size = _file.tellg();
            _file.seekg(0, _file.beg);
            trt_model_stream = new char[size];
            assert(trt_model_stream);
            _file.read(trt_model_stream, size);
            _file.close();
        } else {
            std::cerr << "could not open engine file" << '\n';
            return -1;
        }
    } else {
        return 1;
    }

#if TRT_VERSION_GE(8, 0, 0)
    auto* engine = runtime->deserializeCudaEngine(trt_model_stream, size);
#else
    auto* engine = runtime->deserializeCudaEngine(trt_model_stream, size, nullptr);
#endif
    assert(engine != nullptr);
    auto* context = engine->createExecutionContext();
    assert(context != nullptr);

    const std::string img_path = "assets/cats.jpg";
    void* input = nullptr;
    std::vector<float> flat_img;
    cv::Mat img;
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
        img = cv::imread(img_path, cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        img = cv::imread(img_path, cv::IMREAD_COLOR);
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }
    assert(input);

    auto firstProb = doInference(*context, input, 1);
    printFirstOutputs("vgg", firstProb[0].data(), firstProb[0].size());
    std::cout << "prediction result:\n";
    auto labels = loadImagenetLabelMap(LABELS_PATH);
    int _top = 0;
    for (auto& [idx, logits] : topk(firstProb[0], 3)) {
        std::cout << "Top: " << _top++ << " idx: " << idx << ", logits: " << logits << ", label: " << labels[idx]
                  << '\n';
    }

    std::vector<double> latencies;
    latencies.reserve(kBenchmarkRuns);
    for (int32_t i = 0; i < kBenchmarkRuns; ++i) {
        auto _start = std::chrono::steady_clock::now();
        (void)doInference(*context, input, 1);
        auto _end = std::chrono::steady_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
        latencies.push_back(static_cast<double>(_time) / 1000.0);
    }
    printBenchmark("vgg", latencies);

    delete[] trt_model_stream;
#if TRT_VERSION_GE(8, 0, 0)
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
