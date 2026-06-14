#include <NvInfer.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "logging.h"
#include "utils.h"

static constexpr int32_t INPUT_H = 224;
static constexpr int32_t INPUT_W = 224;
static constexpr int32_t OUTPUT_SIZE = 1000;
static constexpr int32_t N = 1;
static constexpr std::size_t WORKSPACE_SIZE = 16 << 20;
static constexpr std::array<const char*, 2> NAMES = {"data", "prob"};
static constexpr std::array<int32_t, 2> SIZES = {3 * INPUT_H * INPUT_W, OUTPUT_SIZE};
static constexpr const char* LABELS_PATH = "assets/imagenet1000_clsidx_to_labels.txt";
static constexpr bool TRT_PREPROCESS = TRT_VERSION_GE(8, 5, 1);
static constexpr std::array<const float, 3> mean = {0.485f, 0.456f, 0.406f};
static constexpr std::array<const float, 3> stdv = {0.229f, 0.224f, 0.225f};

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;
using M = nvinfer1::MatrixOperation;
using NDCF = nvinfer1::NetworkDefinitionCreationFlag;

static Logger gLogger;

struct ResNetVariant {
    const char* name;
    const char* wts_path;
    const char* engine_path;
    std::array<int32_t, 4> layers;
    bool bottleneck;
    int32_t groups;
    int32_t width_per_group;
};

static auto getVariantConfig(const std::string& name) -> ResNetVariant {
    if (name == "resnet18") {
        return {"resnet18", "models/resnet18.wts", "models/resnet18.engine", {2, 2, 2, 2}, false, 1, 64};
    }
    if (name == "resnet34") {
        return {"resnet34", "models/resnet34.wts", "models/resnet34.engine", {3, 4, 6, 3}, false, 1, 64};
    }
    if (name == "resnet50") {
        return {"resnet50", "models/resnet50.wts", "models/resnet50.engine", {3, 4, 6, 3}, true, 1, 64};
    }
    if (name == "resnext50_32x4d") {
        return {"resnext50_32x4d",
                "models/resnext50_32x4d.wts",
                "models/resnext50_32x4d.engine",
                {3, 4, 6, 3},
                true,
                32,
                4};
    }
    if (name == "wide_resnet50_2") {
        return {"wide_resnet50_2",
                "models/wide_resnet50_2.wts",
                "models/wide_resnet50_2.engine",
                {3, 4, 6, 3},
                true,
                1,
                128};
    }
    std::cerr << "unsupported resnet variant: " << name << "\n";
    std::cerr << "model: resnet18 | resnet34 | resnet50 | resnext50_32x4d | wide_resnet50_2\n";
    std::abort();
}

static auto addBatchNorm2d(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& lname,
                           float eps) -> IScaleLayer* {
    const auto* gamma = static_cast<const float*>(weight_map[lname + ".weight"].values);
    const auto* beta = static_cast<const float*>(weight_map[lname + ".bias"].values);
    const auto* mean_ptr = static_cast<const float*>(weight_map[lname + ".running_mean"].values);
    const auto* var = static_cast<const float*>(weight_map[lname + ".running_var"].values);
    const auto len = weight_map[lname + ".running_var"].count;

    auto* scval = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    auto* shval = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    auto* pval = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(len)));
    for (int64_t i = 0; i < len; ++i) {
        scval[i] = gamma[i] / std::sqrt(var[i] + eps);
        shval[i] = beta[i] - mean_ptr[i] * gamma[i] / std::sqrt(var[i] + eps);
        pval[i] = 1.0f;
    }

    Weights scale{DataType::kFLOAT, scval, len};
    Weights shift{DataType::kFLOAT, shval, len};
    Weights power{DataType::kFLOAT, pval, len};
    weight_map[lname + ".scale"] = scale;
    weight_map[lname + ".shift"] = shift;
    weight_map[lname + ".power"] = power;

    auto* layer = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(layer);
#if TRT_VERSION_GE(8, 0, 0)
    layer->setChannelAxis(1);
#endif
    return layer;
}

static auto addConv2d(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& name,
                      int32_t out_channels, int32_t kernel, int32_t stride, int32_t padding,
                      int32_t groups = 1) -> IConvolutionLayer* {
    static constexpr Weights empty{DataType::kFLOAT, nullptr, 0};
    auto* conv =
            network->addConvolutionNd(input, out_channels, DimsHW{kernel, kernel}, weight_map[name + ".weight"], empty);
    assert(conv);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});
    conv->setNbGroups(groups);
    conv->setName(name.c_str());
    return conv;
}

static auto addRelu(INetworkDefinition* network, ITensor& input) -> IActivationLayer* {
    auto* relu = network->addActivation(input, ActivationType::kRELU);
    assert(relu);
    return relu;
}

static auto addBasicBlock(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, int32_t in_channels,
                          int32_t planes, int32_t stride, const std::string& lname) -> ITensor* {
    auto* conv1 = addConv2d(network, weight_map, input, lname + "conv1", planes, 3, stride, 1);
    auto* bn1 = addBatchNorm2d(network, weight_map, *conv1->getOutput(0), lname + "bn1", 1e-5f);
    auto* relu1 = addRelu(network, *bn1->getOutput(0));

    auto* conv2 = addConv2d(network, weight_map, *relu1->getOutput(0), lname + "conv2", planes, 3, 1, 1);
    auto* bn2 = addBatchNorm2d(network, weight_map, *conv2->getOutput(0), lname + "bn2", 1e-5f);

    ITensor* shortcut = &input;
    if (stride != 1 || in_channels != planes) {
        auto* conv3 = addConv2d(network, weight_map, input, lname + "downsample.0", planes, 1, stride, 0);
        auto* bn3 = addBatchNorm2d(network, weight_map, *conv3->getOutput(0), lname + "downsample.1", 1e-5f);
        shortcut = bn3->getOutput(0);
    }

    auto* sum = network->addElementWise(*shortcut, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    assert(sum);
    return addRelu(network, *sum->getOutput(0))->getOutput(0);
}

static auto addBottleneck(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, int32_t in_channels,
                          int32_t planes, int32_t stride, int32_t groups, int32_t width_per_group,
                          const std::string& lname) -> ITensor* {
    constexpr int32_t expansion = 4;
    const int32_t width = planes * width_per_group / 64 * groups;
    const int32_t out_channels = planes * expansion;

    auto* conv1 = addConv2d(network, weight_map, input, lname + "conv1", width, 1, 1, 0);
    auto* bn1 = addBatchNorm2d(network, weight_map, *conv1->getOutput(0), lname + "bn1", 1e-5f);
    auto* relu1 = addRelu(network, *bn1->getOutput(0));

    auto* conv2 = addConv2d(network, weight_map, *relu1->getOutput(0), lname + "conv2", width, 3, stride, 1, groups);
    auto* bn2 = addBatchNorm2d(network, weight_map, *conv2->getOutput(0), lname + "bn2", 1e-5f);
    auto* relu2 = addRelu(network, *bn2->getOutput(0));

    auto* conv3 = addConv2d(network, weight_map, *relu2->getOutput(0), lname + "conv3", out_channels, 1, 1, 0);
    auto* bn3 = addBatchNorm2d(network, weight_map, *conv3->getOutput(0), lname + "bn3", 1e-5f);

    ITensor* shortcut = &input;
    if (stride != 1 || in_channels != out_channels) {
        auto* conv4 = addConv2d(network, weight_map, input, lname + "downsample.0", out_channels, 1, stride, 0);
        auto* bn4 = addBatchNorm2d(network, weight_map, *conv4->getOutput(0), lname + "downsample.1", 1e-5f);
        shortcut = bn4->getOutput(0);
    }

    auto* sum = network->addElementWise(*shortcut, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    assert(sum);
    return addRelu(network, *sum->getOutput(0))->getOutput(0);
}

static auto addLayer(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, int32_t& in_channels,
                     const ResNetVariant& variant, int32_t layer_index, int32_t planes, int32_t blocks,
                     int32_t stride) -> ITensor* {
    ITensor* x = &input;
    for (int32_t i = 0; i < blocks; ++i) {
        const int32_t block_stride = i == 0 ? stride : 1;
        const std::string lname = "layer" + std::to_string(layer_index) + "." + std::to_string(i) + ".";
        if (variant.bottleneck) {
            x = addBottleneck(network, weight_map, *x, in_channels, planes, block_stride, variant.groups,
                              variant.width_per_group, lname);
            in_channels = planes * 4;
        } else {
            x = addBasicBlock(network, weight_map, *x, in_channels, planes, block_stride, lname);
            in_channels = planes;
        }
    }
    return x;
}

static auto addLinear(INetworkDefinition* network, WeightMap& weight_map, ITensor& input, const std::string& lname,
                      int32_t in_features, int32_t out_features) -> ITensor* {
    auto* reshape = network->addShuffle(input);
    assert(reshape);
    reshape->setReshapeDimensions(Dims2{N, in_features});

    auto* kernel = network->addConstant(Dims2{out_features, in_features}, weight_map[lname + ".weight"]);
    assert(kernel);
    auto* matmul = network->addMatrixMultiply(*reshape->getOutput(0), M::kNONE, *kernel->getOutput(0), M::kTRANSPOSE);
    assert(matmul);

    auto* bias = network->addConstant(Dims2{1, out_features}, weight_map[lname + ".bias"]);
    assert(bias);
    auto* sum = network->addElementWise(*matmul->getOutput(0), *bias->getOutput(0), ElementWiseOperation::kSUM);
    assert(sum);
    return sum->getOutput(0);
}

static auto buildResNet(INetworkDefinition* network, WeightMap& weight_map, ITensor& input,
                        const ResNetVariant& variant) -> ITensor* {
    auto* conv1 = addConv2d(network, weight_map, input, "conv1", 64, 7, 2, 3);
    auto* bn1 = addBatchNorm2d(network, weight_map, *conv1->getOutput(0), "bn1", 1e-5f);
    auto* relu1 = addRelu(network, *bn1->getOutput(0));

    auto* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    int32_t in_channels = 64;
    ITensor* x = addLayer(network, weight_map, *pool1->getOutput(0), in_channels, variant, 1, 64, variant.layers[0], 1);
    x = addLayer(network, weight_map, *x, in_channels, variant, 2, 128, variant.layers[1], 2);
    x = addLayer(network, weight_map, *x, in_channels, variant, 3, 256, variant.layers[2], 2);
    x = addLayer(network, weight_map, *x, in_channels, variant, 4, 512, variant.layers[3], 2);

    auto* pool2 = network->addPoolingNd(*x, PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});
    return addLinear(network, weight_map, *pool2->getOutput(0), "fc", in_channels, OUTPUT_SIZE);
}

static auto createEngine(int32_t batch_size, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt,
                         const ResNetVariant& variant) -> ICudaEngine* {
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

    ITensor* logits = buildResNet(network, weight_map, *input, variant);
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
                       const ResNetVariant& variant) {
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
    for (int32_t i = 0; i < nIO; ++i) {
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
        std::vector<float> tmp(static_cast<std::size_t>(batch_size) * output_size, std::nanf(""));
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
        std::cerr << "./resnet -s [model]   // serialize model to plan file\n";
        std::cerr << "./resnet -d [model]   // deserialize plan file and run inference\n";
        std::cerr << "model: resnet18 | resnet34 | resnet50 | resnext50_32x4d | wide_resnet50_2\n";
        return -1;
    }

    const auto variant = getVariantConfig(argc == 3 ? argv[2] : "resnet18");
    std::cout << "Using ResNet variant: " << variant.name << "\n";

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

    auto first_prob = doInference(*context, input, N);
    printFirstOutputs("resnet", first_prob[0].data(), first_prob[0].size());
    std::cout << "prediction result:\n";
    auto labels = loadImagenetLabelMap(LABELS_PATH);
    if (labels.empty()) {
        std::cerr << "failed to load labels from " << LABELS_PATH << "\n";
        std::abort();
    }
    int32_t top = 0;
    for (auto& [idx, logits] : topk(first_prob[0], 3)) {
        std::cout << "Top: " << top++ << " idx: " << idx << ", logits: " << std::setprecision(4) << logits
                  << ", label: " << labels[idx] << "\n";
    }

    std::vector<double> latencies;
    latencies.reserve(kBenchmarkRuns);
    for (int32_t i = 0; i < kBenchmarkRuns; ++i) {
        auto start = std::chrono::steady_clock::now();
        (void)doInference(*context, input, N);
        auto end = std::chrono::steady_clock::now();
        auto period = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(static_cast<double>(period.count()) / 1000.0);
    }
    printBenchmark("resnet", latencies, N);
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
