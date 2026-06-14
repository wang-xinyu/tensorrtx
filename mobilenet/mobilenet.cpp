#include <NvInfer.h>

#include <array>
#include <cassert>
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
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "utils.h"

struct MobileNetVariant {
    std::string name;
    std::string wts_path;
    std::string engine_path;
    bool is_v3_small;
};

struct V2BlockParams {
    int32_t index;
    int32_t in_channels;
    int32_t out_channels;
    int32_t stride;
    int32_t expansion;
};

struct V3BlockParams {
    int32_t index;
    int32_t in_channels;
    int32_t out_channels;
    int32_t hidden_channels;
    int32_t kernel_size;
    int32_t stride;
    bool use_se;
    bool use_hs;
};

static constexpr const std::size_t WORKSPACE_SIZE = 16 << 20;
static constexpr const int64_t N = 1;
static constexpr const int32_t INPUT_H = 224;
static constexpr const int32_t INPUT_W = 224;
static constexpr const std::array<int32_t, 2> SIZES = {3 * INPUT_H * INPUT_W, 1000};
static constexpr const std::array<const char*, 2> NAMES = {"data", "logits"};
static constexpr const bool TRT_PREPROCESS = TRT_VERSION_GE(8, 5, 1);
static constexpr const std::array<const float, 3> mean = {0.485f, 0.456f, 0.406f};
static constexpr const std::array<const float, 3> stdv = {0.229f, 0.224f, 0.225f};
static constexpr const char* LABELS_PATH = "assets/imagenet1000_clsidx_to_labels.txt";

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;
using M = MatrixOperation;
using E = ElementWiseOperation;
using NDCF = NetworkDefinitionCreationFlag;

static Logger gLogger;

static auto getVariantConfig(const std::string& name) -> MobileNetVariant {
    if (name == "mobilenet_v2" || name == "v2") {
        return {"mobilenet_v2", "models/mobilenet_v2.wts", "models/mobilenet_v2.engine", false};
    }
    if (name == "mobilenet_v3_small" || name == "v3_small") {
        return {"mobilenet_v3_small", "models/mobilenet_v3_small.wts", "models/mobilenet_v3_small.engine", true};
    }
    std::cerr << "Unsupported MobileNet variant: " << name << "\n";
    std::cerr << "Choose one of: mobilenet_v2, mobilenet_v3_small\n";
    std::abort();
}

static auto emptyWeights() -> Weights {
    return Weights{DataType::kFLOAT, nullptr, 0ll};
}

static auto getWeight(const WeightMap& weight_map, const std::string& key) -> Weights {
    const auto iter = weight_map.find(key);
    if (iter == weight_map.end()) {
        std::cerr << "Missing MobileNet weight: " << key << "\n";
        std::abort();
    }
    return iter->second;
}

static auto addBatchNorm2d(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& lname,
                           float eps = 1e-5f) -> ILayer* {
    const auto* gamma = static_cast<const float*>(getWeight(weight_map, lname + ".weight").values);
    const auto* beta = static_cast<const float*>(getWeight(weight_map, lname + ".bias").values);
    const auto* bn_mean = static_cast<const float*>(getWeight(weight_map, lname + ".running_mean").values);
    const auto* var = static_cast<const float*>(getWeight(weight_map, lname + ".running_var").values);
    const auto len = getWeight(weight_map, lname + ".running_var").count;

    auto* scale_values = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    auto* shift_values = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    if (scale_values == nullptr || shift_values == nullptr) {
        std::cerr << "batchnorm weight allocation failed\n";
        std::abort();
    }
    for (int64_t i = 0; i < len; ++i) {
        scale_values[i] = gamma[i] / std::sqrt(var[i] + eps);
        shift_values[i] = beta[i] - bn_mean[i] * gamma[i] / std::sqrt(var[i] + eps);
    }

    Weights scale{DataType::kFLOAT, scale_values, len};
    Weights shift{DataType::kFLOAT, shift_values, len};
    static const Weights power{DataType::kFLOAT, nullptr, 0ll};

    weight_map[lname + ".scale"] = scale;
    weight_map[lname + ".shift"] = shift;
    weight_map[lname + ".power"] = power;
    auto* scale_layer = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_layer);
    return scale_layer;
}

static auto addHardSigmoid(INetworkDefinition* network, ITensor& input, const std::string& name) -> IActivationLayer* {
    auto* hsigmoid = network->addActivation(input, ActivationType::kHARD_SIGMOID);
    assert(hsigmoid);
    hsigmoid->setAlpha(1.0f / 6.0f);
    hsigmoid->setBeta(0.5f);
    hsigmoid->setName(name.c_str());
    return hsigmoid;
}

static auto addHardSwish(INetworkDefinition* network, ITensor& input, const std::string& name) -> ILayer* {
    auto* hsigmoid = addHardSigmoid(network, input, name + ".hsigmoid");
    auto* hswish = network->addElementWise(input, *hsigmoid->getOutput(0), E::kPROD);
    assert(hswish);
    hswish->setName(name.c_str());
    return hswish;
}

static auto addRelu6(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                     const std::string& name) -> ILayer* {
    auto* relu = network->addActivation(input, ActivationType::kRELU);
    assert(relu);
    relu->setName((name + ".relu").c_str());

    auto* six = static_cast<float*>(std::malloc(sizeof(float)));
    if (six == nullptr) {
        std::cerr << "relu6 constant allocation failed\n";
        std::abort();
    }
    *six = 6.0f;
    const std::string key = name + ".relu6.max";
    weight_map[key] = Weights{DataType::kFLOAT, six, 1ll};
    auto* c = network->addConstant(Dims4{1, 1, 1, 1}, weight_map[key]);
    assert(c);
    auto* clipped = network->addElementWise(*relu->getOutput(0), *c->getOutput(0), E::kMIN);
    assert(clipped);
    clipped->setName(name.c_str());
    return clipped;
}

static auto addActivation(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& name,
                          bool hard_swish, bool relu6) -> ILayer* {
    if (hard_swish) {
        return addHardSwish(network, input, name);
    }
    if (relu6) {
        return addRelu6(network, weight_map, input, name);
    }
    auto* relu = network->addActivation(input, ActivationType::kRELU);
    assert(relu);
    relu->setName(name.c_str());
    return relu;
}

static auto addConvBnAct(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                         const std::string& conv_key, const std::string& bn_key, int32_t out_channels,
                         int32_t kernel_size, int32_t stride, int32_t padding, int32_t groups, bool with_act,
                         bool hard_swish = false, bool relu6 = false, float eps = 1e-5f) -> ILayer* {
    auto* conv = network->addConvolutionNd(input, out_channels, DimsHW{kernel_size, kernel_size},
                                           getWeight(weight_map, conv_key), emptyWeights());
    assert(conv);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});
    conv->setNbGroups(groups);
    conv->setName(conv_key.c_str());

    auto* bn = addBatchNorm2d(network, weight_map, *conv->getOutput(0), bn_key, eps);
    bn->setName((bn_key + ".bn").c_str());
    if (!with_act) {
        return bn;
    }
    return addActivation(network, weight_map, *bn->getOutput(0), bn_key + ".act", hard_swish, relu6);
}

static auto addLinear(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& weight_key,
                      const std::string& bias_key, int32_t in_features, int32_t out_features,
                      const std::string& name) -> ILayer* {
    auto* fcw = network->addConstant(DimsHW{out_features, in_features}, getWeight(weight_map, weight_key));
    auto* fcb = network->addConstant(DimsHW{1, out_features}, getWeight(weight_map, bias_key));
    assert(fcw);
    assert(fcb);
    auto* fc_matmul = network->addMatrixMultiply(input, M::kNONE, *fcw->getOutput(0), M::kTRANSPOSE);
    assert(fc_matmul);
    auto* fc = network->addElementWise(*fc_matmul->getOutput(0), *fcb->getOutput(0), E::kSUM);
    assert(fc);
    fc->setName(name.c_str());
    return fc;
}

static auto addSqueezeExcitation(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                                 const std::string& lname, int32_t channels) -> ILayer* {
    auto* pool = network->addReduce(input, ReduceOperation::kAVG, 0xc, true);
    assert(pool);
    pool->setName((lname + ".avgpool").c_str());

    const auto fc1_weight = getWeight(weight_map, lname + ".fc1.weight");
    const auto squeeze_channels = toI32(fc1_weight.count / channels);
    auto* fc1 = network->addConvolutionNd(*pool->getOutput(0), squeeze_channels, DimsHW{1, 1}, fc1_weight,
                                          getWeight(weight_map, lname + ".fc1.bias"));
    assert(fc1);
    fc1->setName((lname + ".fc1").c_str());
    auto* relu = network->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu);
    relu->setName((lname + ".relu").c_str());
    auto* fc2 = network->addConvolutionNd(*relu->getOutput(0), channels, DimsHW{1, 1},
                                          getWeight(weight_map, lname + ".fc2.weight"),
                                          getWeight(weight_map, lname + ".fc2.bias"));
    assert(fc2);
    fc2->setName((lname + ".fc2").c_str());
    auto* scale = addHardSigmoid(network, *fc2->getOutput(0), lname + ".scale");
    auto* out = network->addElementWise(input, *scale->getOutput(0), E::kPROD);
    assert(out);
    out->setName((lname + ".mul").c_str());
    return out;
}

static auto addV2Block(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                       const V2BlockParams& params) -> ILayer* {
    const std::string lname = "features." + std::to_string(params.index) + ".conv.";
    const int32_t hidden = params.in_channels * params.expansion;
    ILayer* layer = nullptr;
    if (params.expansion != 1) {
        layer = addConvBnAct(network, weight_map, input, lname + "0.0.weight", lname + "0.1", hidden, 1, 1, 0, 1, true,
                             false, true);
        layer = addConvBnAct(network, weight_map, *layer->getOutput(0), lname + "1.0.weight", lname + "1.1", hidden, 3,
                             params.stride, 1, hidden, true, false, true);
        layer = addConvBnAct(network, weight_map, *layer->getOutput(0), lname + "2.weight", lname + "3",
                             params.out_channels, 1, 1, 0, 1, false);
    } else {
        layer = addConvBnAct(network, weight_map, input, lname + "0.0.weight", lname + "0.1", hidden, 3, params.stride,
                             1, hidden, true, false, true);
        layer = addConvBnAct(network, weight_map, *layer->getOutput(0), lname + "1.weight", lname + "2",
                             params.out_channels, 1, 1, 0, 1, false);
    }
    if (params.stride == 1 && params.in_channels == params.out_channels) {
        auto* ew = network->addElementWise(input, *layer->getOutput(0), E::kSUM);
        assert(ew);
        ew->setName((lname + "residual").c_str());
        return ew;
    }
    return layer;
}

static auto addV3Block(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                       const V3BlockParams& params) -> ILayer* {
    const std::string lname = "features." + std::to_string(params.index) + ".block.";
    ITensor* tensor = &input;
    int32_t depthwise_index = 0;
    int32_t project_index = 2;
    if (params.in_channels != params.hidden_channels) {
        auto* layer = addConvBnAct(network, weight_map, input, lname + "0.0.weight", lname + "0.1",
                                   params.hidden_channels, 1, 1, 0, 1, true, params.use_hs, false, 1e-3f);
        tensor = layer->getOutput(0);
        depthwise_index = 1;
        project_index = params.use_se ? 3 : 2;
    }

    auto* layer = addConvBnAct(network, weight_map, *tensor, lname + std::to_string(depthwise_index) + ".0.weight",
                               lname + std::to_string(depthwise_index) + ".1", params.hidden_channels,
                               params.kernel_size, params.stride, params.kernel_size / 2, params.hidden_channels, true,
                               params.use_hs, false, 1e-3f);
    if (params.use_se) {
        layer = addSqueezeExcitation(network, weight_map, *layer->getOutput(0),
                                     lname + std::to_string(depthwise_index + 1), params.hidden_channels);
    }
    layer = addConvBnAct(network, weight_map, *layer->getOutput(0), lname + std::to_string(project_index) + ".0.weight",
                         lname + std::to_string(project_index) + ".1", params.out_channels, 1, 1, 0, 1, false, false,
                         false, 1e-3f);
    if (params.stride == 1 && params.in_channels == params.out_channels) {
        auto* ew = network->addElementWise(input, *layer->getOutput(0), E::kSUM);
        assert(ew);
        ew->setName((lname + "residual").c_str());
        return ew;
    }
    return layer;
}

static auto buildMobileNetV2(INetworkDefinition* network, WeightMap& weight_map, ITensor& input) -> ITensor* {
    auto* layer = addConvBnAct(network, weight_map, input, "features.0.0.weight", "features.0.1", 32, 3, 2, 1, 1, true,
                               false, true);
    const std::array<V2BlockParams, 17> blocks = {{{1, 32, 16, 1, 1},
                                                   {2, 16, 24, 2, 6},
                                                   {3, 24, 24, 1, 6},
                                                   {4, 24, 32, 2, 6},
                                                   {5, 32, 32, 1, 6},
                                                   {6, 32, 32, 1, 6},
                                                   {7, 32, 64, 2, 6},
                                                   {8, 64, 64, 1, 6},
                                                   {9, 64, 64, 1, 6},
                                                   {10, 64, 64, 1, 6},
                                                   {11, 64, 96, 1, 6},
                                                   {12, 96, 96, 1, 6},
                                                   {13, 96, 96, 1, 6},
                                                   {14, 96, 160, 2, 6},
                                                   {15, 160, 160, 1, 6},
                                                   {16, 160, 160, 1, 6},
                                                   {17, 160, 320, 1, 6}}};
    for (const auto& block : blocks) {
        layer = addV2Block(network, weight_map, *layer->getOutput(0), block);
    }
    layer = addConvBnAct(network, weight_map, *layer->getOutput(0), "features.18.0.weight", "features.18.1", 1280, 1, 1,
                         0, 1, true, false, true);
    auto* pool = network->addReduce(*layer->getOutput(0), ReduceOperation::kAVG, 0xc, false);
    assert(pool);
    pool->setName("avgpool");
    return addLinear(network, weight_map, *pool->getOutput(0), "classifier.1.weight", "classifier.1.bias", 1280, 1000,
                     "classifier.1")
            ->getOutput(0);
}

static auto buildMobileNetV3Small(INetworkDefinition* network, WeightMap& weight_map, ITensor& input) -> ITensor* {
    auto* layer = addConvBnAct(network, weight_map, input, "features.0.0.weight", "features.0.1", 16, 3, 2, 1, 1, true,
                               true, false, 1e-3f);
    const std::array<V3BlockParams, 11> blocks = {{{1, 16, 16, 16, 3, 2, true, false},
                                                   {2, 16, 24, 72, 3, 2, false, false},
                                                   {3, 24, 24, 88, 3, 1, false, false},
                                                   {4, 24, 40, 96, 5, 2, true, true},
                                                   {5, 40, 40, 240, 5, 1, true, true},
                                                   {6, 40, 40, 240, 5, 1, true, true},
                                                   {7, 40, 48, 120, 5, 1, true, true},
                                                   {8, 48, 48, 144, 5, 1, true, true},
                                                   {9, 48, 96, 288, 5, 2, true, true},
                                                   {10, 96, 96, 576, 5, 1, true, true},
                                                   {11, 96, 96, 576, 5, 1, true, true}}};
    for (const auto& block : blocks) {
        layer = addV3Block(network, weight_map, *layer->getOutput(0), block);
    }
    layer = addConvBnAct(network, weight_map, *layer->getOutput(0), "features.12.0.weight", "features.12.1", 576, 1, 1,
                         0, 1, true, true, false, 1e-3f);
    auto* pool = network->addReduce(*layer->getOutput(0), ReduceOperation::kAVG, 0xc, false);
    assert(pool);
    pool->setName("avgpool");
    layer = addLinear(network, weight_map, *pool->getOutput(0), "classifier.0.weight", "classifier.0.bias", 576, 1024,
                      "classifier.0");
    layer = addHardSwish(network, *layer->getOutput(0), "classifier.1");
    return addLinear(network, weight_map, *layer->getOutput(0), "classifier.3.weight", "classifier.3.bias", 1024, 1000,
                     "classifier.3")
            ->getOutput(0);
}

static auto createEngine(int32_t batch_size, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt,
                         const MobileNetVariant& variant) -> ICudaEngine* {
    WeightMap weight_map = loadWeights(variant.wts_path);

#if TRT_VERSION_GE(10, 12, 0)
    auto flag = 1U << static_cast<int>(NDCF::kSTRONGLY_TYPED);
#elif TRT_VERSION_GE(10, 0, 0)
    auto flag = 0U;
#else
    auto flag = 1U << static_cast<int>(NDCF::kEXPLICIT_BATCH);
#endif
    auto* network = builder->createNetworkV2(flag);
    assert(network);

    ITensor* input = nullptr;
    if constexpr (TRT_PREPROCESS) {
        dt = DataType::kUINT8;
        input = network->addInput(NAMES[0], dt, Dims4{batch_size, INPUT_H, INPUT_W, 3});
        auto* transform = addTransformLayer(network, *input, true, mean, stdv);
        input = transform->getOutput(0);
    } else {
        input = network->addInput(NAMES[0], dt, Dims4{batch_size, 3, INPUT_H, INPUT_W});
    }
    assert(input);

    ITensor* logits = variant.is_v3_small ? buildMobileNetV3Small(network, weight_map, *input)
                                          : buildMobileNetV2(network, weight_map, *input);
    logits->setName(NAMES[1]);
    network->markOutput(*logits);

#if TRT_VERSION_GE(8, 0, 0)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    assert(mem);
    ICudaEngine* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete mem;
    delete network;
#else
    builder->setMaxBatchSize(batch_size);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif
    std::cout << "build finished\n";

    for (auto& mem : weight_map) {
        std::free(const_cast<void*>(mem.second.values));
    }
    return engine;
}

static void APIToModel(int32_t batch_size, IRuntime* runtime, IHostMemory** model_stream,
                       const MobileNetVariant& variant) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = createEngine(batch_size, runtime, builder, config, DataType::kFLOAT, variant);
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

static auto doInference(IExecutionContext& context, void* input,
                        int64_t batch_size) -> std::vector<std::vector<float>> {
    ICudaEngine const& engine = context.getEngine();
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
        auto element_size = getSize(engine.getTensorDataType(tensor_name));
        size = element_size * static_cast<std::size_t>(batch_size) * (name == NAMES[0] ? SIZES[0] : SIZES[1]);
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
        auto element_size = getSize(engine.getBindingDataType(idx));
        assert(idx == i);
        size = element_size * static_cast<std::size_t>(batch_size) * SIZES[i];
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
        const auto size = static_cast<std::size_t>(batch_size) * output_size * sizeof(float);
        CHECK(cudaMemcpyAsync(tmp.data(), buffers[i], size, cudaMemcpyDeviceToHost, stream));
        prob.emplace_back(std::move(tmp));
    }
    CHECK(cudaStreamSynchronize(stream));

    for (auto& buffer : buffers) {
        CHECK(cudaFree(buffer));
    }
    CHECK(cudaStreamDestroy(stream));
    return prob;
}

auto main(int argc, char** argv) -> int {
    checkTrtEnv();
    if (argc < 2 || argc > 3) {
        std::cerr << "arguments not right!\n";
        std::cerr << "./mobilenet -s [model]   // serialize model to plan file\n";
        std::cerr << "./mobilenet -d [model]   // deserialize plan file and run inference\n";
        std::cerr << "model: mobilenet_v2 | mobilenet_v3_small\n";
        return -1;
    }

    const auto variant = getVariantConfig(argc == 3 ? argv[2] : "mobilenet_v2");
    std::cout << "Using MobileNet variant: " << variant.name << "\n";

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trt_model_stream{nullptr};
    std::streamsize size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* model_stream{nullptr};
        APIToModel(N, runtime, &model_stream, variant);
        assert(model_stream != nullptr);

        std::ofstream plan(variant.engine_path, std::ios::binary | std::ios::trunc);
        if (!plan) {
            std::cerr << "could not open plan output file\n";
            return -1;
        }
        if (model_stream->size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            std::cerr << "this model is too large to serialize\n";
            return -1;
        }
        const auto* data_ptr = reinterpret_cast<const char*>(model_stream->data());
        const auto data_size = static_cast<std::streamsize>(model_stream->size());
        plan.write(data_ptr, data_size);
#if TRT_VERSION_GE(8, 0, 0)
        delete model_stream;
        delete runtime;
#else
        model_stream->destroy();
        runtime->destroy();
#endif
        return 0;
    }
    if (std::string(argv[1]) == "-d") {
        std::ifstream file(variant.engine_path, std::ios::binary);
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

#if TRT_VERSION_GE(8, 0, 0)
    ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream, size, nullptr);
#endif
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trt_model_stream;

    void* input = nullptr;
    std::vector<float> flat_img;
    cv::Mat img = cv::imread("assets/cats.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "failed to read image: assets/cats.jpg\n";
        return -1;
    }
    if constexpr (TRT_PREPROCESS) {
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }

    auto firstProb = doInference(*context, input, N);
    printFirstOutputs("mobilenet", firstProb[0].data(), firstProb[0].size());
    std::cout << "prediction result:\n";
    auto labels = loadImagenetLabelMap(LABELS_PATH);
    if (labels.empty()) {
        std::cerr << "failed to load labels from " << LABELS_PATH << "\n";
        std::abort();
    }
    int top = 0;
    for (auto& [idx, logits] : topk(firstProb[0], 3)) {
        std::cout << "Top: " << top++ << " idx: " << idx << ", logits: " << std::setprecision(4) << logits
                  << ", label: " << labels[idx] << "\n";
    }

    std::vector<double> latencies;
    latencies.reserve(kBenchmarkRuns);
    for (int i = 0; i < kBenchmarkRuns; ++i) {
        auto start = std::chrono::steady_clock::now();
        (void)doInference(*context, input, N);
        auto end = std::chrono::steady_clock::now();
        auto period = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(static_cast<double>(period.count()) / 1000.0);
    }
    printBenchmark("mobilenet", latencies, N);
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
