#include <NvInfer.h>
#include <array>
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

struct ShuffleNetV2Variant {
    std::string name;
    std::array<int32_t, 3> repeat;
    std::array<int32_t, 5> output_chn;
    std::string wts_path;
    std::string engine_path;
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
using NDCF = nvinfer1::NetworkDefinitionCreationFlag;

static Logger gLogger;

static auto getVariantConfig(const std::string& name) -> ShuffleNetV2Variant {
    static const std::map<std::string, std::pair<std::array<int32_t, 3>, std::array<int32_t, 5>>> variants = {
            {"shufflenet_v2_x0_5", {{4, 8, 4}, {24, 48, 96, 192, 1024}}},
            {"shufflenet_v2_x1_0", {{4, 8, 4}, {24, 116, 232, 464, 1024}}},
            {"shufflenet_v2_x1_5", {{4, 8, 4}, {24, 176, 352, 704, 1024}}},
            {"shufflenet_v2_x2_0", {{4, 8, 4}, {24, 244, 488, 976, 2048}}},
    };

    const auto iter = variants.find(name);
    if (iter == variants.end()) {
        std::cerr << "Unsupported ShuffleNetV2 variant: " << name << "\n";
        std::cerr << "Choose one of: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0\n";
        std::abort();
    }

    return ShuffleNetV2Variant{name, iter->second.first, iter->second.second, "models/" + name + ".wts",
                               "models/" + name + ".engine"};
}

static auto addBatchNorm2d(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& lname,
                           float eps = 1e-3f) -> ILayer* {
    const auto* gamma = static_cast<const float*>(weight_map[lname + ".weight"].values);
    const auto* beta = static_cast<const float*>(weight_map[lname + ".bias"].values);
    const auto* bn_mean = static_cast<const float*>(weight_map[lname + ".running_mean"].values);
    const auto* var = static_cast<const float*>(weight_map[lname + ".running_var"].values);
    const auto len = weight_map[lname + ".running_var"].count;

    auto* scale_values = reinterpret_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    auto* shift_values = reinterpret_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    if (scale_values == nullptr || shift_values == nullptr) {
        std::cerr << "batchnorm weight allocation failed\n";
        std::abort();
    }
    for (int64_t i = 0; i < len; i++) {
        scale_values[i] = gamma[i] / std::sqrt(var[i] + eps);
        shift_values[i] = beta[i] - bn_mean[i] * gamma[i] / std::sqrt(var[i] + eps);
    }

    Weights scale{DataType::kFLOAT, scale_values, len};
    Weights shift{DataType::kFLOAT, shift_values, len};
    static const Weights power{DataType::kFLOAT, nullptr, 0ll};

    weight_map[lname + ".scale"] = scale;
    weight_map[lname + ".shift"] = shift;
    weight_map[lname + ".power"] = power;
    IScaleLayer* scale_layer = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_layer);
    return scale_layer;
}

static auto addConvBnRelu(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& lname,
                          int32_t out_channels, int32_t kernel_size, int32_t stride = 1, int32_t padding = 0,
                          int32_t groups = 1, bool with_relu = true, int32_t start_index = 0) -> ILayer* {
    static const Weights empty_weights{DataType::kFLOAT, nullptr, 0ll};
    const auto conv_name = lname + "." + std::to_string(start_index++);
    auto* conv = network->addConvolutionNd(input, out_channels, DimsHW{kernel_size, kernel_size},
                                           weight_map[conv_name + ".weight"], empty_weights);
    assert(conv);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});
    conv->setNbGroups(groups);
    conv->setName(conv_name.c_str());

    const auto bn_name = lname + "." + std::to_string(start_index++);
    auto* bn = addBatchNorm2d(network, weight_map, *conv->getOutput(0), bn_name, 1e-5f);
    bn->setName((bn_name + ".bn").c_str());

    if (!with_relu) {
        return bn;
    }

    auto* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(relu);
    relu->setName((lname + "." + std::to_string(start_index) + ".relu").c_str());
    return relu;
}

static auto addInvertedResidual(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                                const std::string& lname, int32_t in_channels, int32_t out_channels,
                                int32_t stride) -> ILayer* {
    if (stride < 1 || stride > 3) {
        std::cerr << "stride must be in [1, 3]\n";
        std::abort();
    }

    const int32_t branch_features = out_channels / 2;
    ITensor* branch1_output = nullptr;
    ITensor* branch2_input = nullptr;

    if (stride == 1) {
        const auto dims = input.getDimensions();
        const Dims4 half{dims.d[0], dims.d[1] / 2, dims.d[2], dims.d[3]};
        auto* slice1 = network->addSlice(input, Dims4{0, 0, 0, 0}, half, Dims4{1, 1, 1, 1});
        auto* slice2 = network->addSlice(input, Dims4{0, dims.d[1] / 2, 0, 0}, half, Dims4{1, 1, 1, 1});
        assert(slice1);
        assert(slice2);
        branch1_output = slice1->getOutput(0);
        branch2_input = slice2->getOutput(0);
    } else {
        auto* branch1 = addConvBnRelu(network, weight_map, input, lname + ".branch1", in_channels, 3, stride, 1,
                                      in_channels, false, 0);
        branch1 = addConvBnRelu(network, weight_map, *branch1->getOutput(0), lname + ".branch1", branch_features, 1, 1,
                                0, 1, true, 2);
        branch1_output = branch1->getOutput(0);
        branch2_input = &input;
    }

    auto* branch2 = addConvBnRelu(network, weight_map, *branch2_input, lname + ".branch2", branch_features, 1, 1, 0, 1,
                                  true, 0);
    branch2 = addConvBnRelu(network, weight_map, *branch2->getOutput(0), lname + ".branch2", branch_features, 3, stride,
                            1, branch_features, false, 3);
    branch2 = addConvBnRelu(network, weight_map, *branch2->getOutput(0), lname + ".branch2", branch_features, 1, 1, 0,
                            1, true, 5);

    std::array<ITensor*, 2> cat_tensors = {branch1_output, branch2->getOutput(0)};
    auto* cat = network->addConcatenation(cat_tensors.data(), toI32(cat_tensors.size()));
    assert(cat);
    cat->setName((lname + ".cat").c_str());
    cat->setAxis(1);

    auto* shuffle1 = network->addShuffle(*cat->getOutput(0));
    assert(shuffle1);
    shuffle1->setName((lname + ".shuffle.1").c_str());
    const auto dims = cat->getOutput(0)->getDimensions();
    shuffle1->setReshapeDimensions(Dims{5, {dims.d[0], 2, dims.d[1] / 2, dims.d[2], dims.d[3]}});
    shuffle1->setSecondTranspose({0, 2, 1, 3, 4});

    auto* shuffle2 = network->addShuffle(*shuffle1->getOutput(0));
    assert(shuffle2);
    shuffle2->setName((lname + ".shuffle.2").c_str());
    shuffle2->setReshapeDimensions(dims);
    return shuffle2;
}

static auto createEngine(int32_t batch_size, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt,
                         const ShuffleNetV2Variant& variant) -> ICudaEngine* {
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

    auto* conv1 = addConvBnRelu(network, weight_map, *input, "conv1", variant.output_chn[0], 3, 2, 1);
    auto* pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    ILayer* layer = pool1;
    int32_t in_channels = variant.output_chn[0];
    for (int32_t stage = 2; stage < 5; ++stage) {
        const int32_t out_channels = variant.output_chn[stage - 1];
        const std::string lname = "stage" + std::to_string(stage);
        layer = addInvertedResidual(network, weight_map, *layer->getOutput(0), lname + ".0", in_channels, out_channels,
                                    2);
        for (int32_t block = 1; block < variant.repeat[stage - 2]; ++block) {
            layer = addInvertedResidual(network, weight_map, *layer->getOutput(0), lname + "." + std::to_string(block),
                                        out_channels, out_channels, 1);
        }
        in_channels = out_channels;
    }

    auto* conv5 = addConvBnRelu(network, weight_map, *layer->getOutput(0), "conv5", variant.output_chn[4], 1, 1, 0);
    auto* global_pool = network->addReduce(*conv5->getOutput(0), ReduceOperation::kAVG, 0xc, false);
    assert(global_pool);
    global_pool->setName("global_pool");
    auto* fcw = network->addConstant(DimsHW{1000, variant.output_chn[4]}, weight_map["fc.weight"]);
    auto* fcb = network->addConstant(DimsHW{1, 1000}, weight_map["fc.bias"]);
    auto* fc_matmul =
            network->addMatrixMultiply(*global_pool->getOutput(0), M::kNONE, *fcw->getOutput(0), M::kTRANSPOSE);
    auto* fc = network->addElementWise(*fc_matmul->getOutput(0), *fcb->getOutput(0), ElementWiseOperation::kSUM);
    assert(fc);
    fc->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*fc->getOutput(0));

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
                       const ShuffleNetV2Variant& variant) {
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
        std::cerr << "./shufflenetv2 -s [model]   // serialize model to plan file\n";
        std::cerr << "./shufflenetv2 -d [model]   // deserialize plan file and run inference\n";
        return -1;
    }

    const auto variant = getVariantConfig(argc == 3 ? argv[2] : "shufflenet_v2_x0_5");
    std::cout << "Using ShuffleNetV2 variant: " << variant.name << "\n";

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trt_model_stream{nullptr};
    std::streamsize size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* model_stream{nullptr};
        APIToModel(1, runtime, &model_stream, variant);
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
    if constexpr (TRT_PREPROCESS) {
        cv::resize(img, img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
        input = static_cast<void*>(img.data);
    } else {
        flat_img = preprocess_img(img, true, mean, stdv, N, INPUT_H, INPUT_W);
        input = flat_img.data();
    }

    auto firstProb = doInference(*context, input, N);
    printFirstOutputs("shufflenetv2", firstProb[0].data(), firstProb[0].size());
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
    printBenchmark("shufflenetv2", latencies, N);
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
