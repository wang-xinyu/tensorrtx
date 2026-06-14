#include <NvInferRuntimeCommon.h>
#include <logging.h>
#include <chrono>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <variant>

#include "NvInfer.h"
#include "utils.h"

using LayerConfig = std::vector<std::variant<int32_t, std::string>>;
using WeightMap = std::map<std::string, nvinfer1::Weights>;
using O = nvinfer1::OptProfileSelector;

constexpr static const std::array<const char*, 2> NAMES = {"data", "prob"};
constexpr static const char* WTS_PATH = "../models/csrnet.wts";
constexpr static const char* ENGINE_PATH = "../models/csrnet.engine";

// for simplicity, always use BatchSize == 1
constexpr static const int64_t maxBatchSize = 1;
constexpr static const int64_t MAX_INPUT_SIZE = 1440;
constexpr static const int64_t MIN_INPUT_SIZE = 608;

/** @note: we use this value on purpose to simplify preprocess in this demo.
 * To set your own H, W value, be sure to make them divisible by 32 */
constexpr static const int64_t OPT_INPUT_W = 1024;
constexpr static const int64_t OPT_INPUT_H = 768;

constexpr static int64_t kMaxInputImageSize = MAX_INPUT_SIZE * MAX_INPUT_SIZE * 3;
constexpr static int64_t kMaxOutputProbSize = (MAX_INPUT_SIZE * MAX_INPUT_SIZE) >> 6;

static const LayerConfig frontend_cfg = {64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512};
static const LayerConfig backend_cfg = {512, 512, 512, 256, 128, 64};

constexpr static const std::array<const float, 3> mean = {0.406, 0.456, 0.485};
constexpr static const std::array<const float, 3> stdv = {0.225, 0.224, 0.229};

static Logger gLogger;

ILayer* addBatchNorm2d(INetworkDefinition* network, WeightMap& m, ITensor& input, const std::string& lname,
                       float eps = 1e-3) {
    static Weights none{DataType::kFLOAT, nullptr, 0ll};
    const auto* gamma = reinterpret_cast<const float*>(m[lname + ".weight"].values);
    const auto* beta = reinterpret_cast<const float*>(m[lname + ".bias"].values);
    const auto* mean = reinterpret_cast<const float*>(m[lname + ".running_mean"].values);
    const auto* var = reinterpret_cast<const float*>(m[lname + ".running_var"].values);
    auto len = m[lname + ".running_var"].count;

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

static auto make_layers(INetworkDefinition* net, WeightMap& w, ITensor& input, const LayerConfig& cfg,
                        const std::string& name = "frontend.", bool use_bn = false,
                        bool use_dilation = false) -> ILayer* {
    auto* tensor = &input;
    int32_t idx = 0;
    ILayer* ret = nullptr;

    for (const auto& _v : cfg) {
        if (std::holds_alternative<std::string>(_v)) {
            auto* pool = net->addPoolingNd(*tensor, PoolingType::kMAX, DimsHW{2, 2});
            pool->setStrideNd(DimsHW{2, 2});
            pool->setName((name + std::to_string(idx++)).c_str());
            tensor = pool->getOutput(0);
            ret = pool;
            assert(ret);
        } else {
            assert(std::holds_alternative<int32_t>(_v));
            const int32_t v = std::get<int32_t>(_v);
            auto _name = name + std::to_string(idx++);
            auto* conv = net->addConvolutionNd(*tensor, v, DimsHW{3, 3}, w[_name + ".weight"], w[_name + ".bias"]);
            auto d_rate = use_dilation ? DimsHW{2, 2} : DimsHW{1, 1};
            conv->setPaddingNd(d_rate);
            conv->setDilationNd(d_rate);
            conv->setName(_name.c_str());
            tensor = conv->getOutput(0);
            if (use_bn) {
                auto _name = name + std::to_string(idx++);
                auto* bn = addBatchNorm2d(net, w, *tensor, _name, 1e-5f);
                tensor = bn->getOutput(0);
                assert(bn);
            }
            auto* relu = net->addActivation(*tensor, ActivationType::kRELU);
            relu->setName((name + std::to_string(idx++)).c_str());
            tensor = relu->getOutput(0);
            ret = relu;
            assert(ret);
        }
    }
    return ret;
}

auto createEngine(int32_t maxBatchSize, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config,
                  DataType dt) -> ICudaEngine* {
    WeightMap w = loadWeights(WTS_PATH);

#if TRT_VERSION_GE(10, 0, 0)
    auto* network = builder->createNetworkV2(0);
#else
    auto* network = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

    ITensor* data = network->addInput(NAMES[0], dt, Dims4{1, 3, -1, -1});
    assert(data);

    auto* frontend = make_layers(network, w, *data, frontend_cfg, "frontend.", false, false);
    auto* backend = make_layers(network, w, *frontend->getOutput(0), backend_cfg, "backend.", false, true);
    auto conv = network->addConvolutionNd(*backend->getOutput(0), 1, DimsHW{1, 1}, w["output_layer.weight"],
                                          w["output_layer.bias"]);
    assert(conv);

    conv->setStrideNd(DimsHW{1, 1});
    conv->getOutput(0)->setName(NAMES[1]);
    network->markOutput(*conv->getOutput(0));

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(NAMES[0], O::kMIN, Dims4(maxBatchSize, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
    profile->setDimensions(NAMES[0], O::kOPT, Dims4(maxBatchSize, 3, OPT_INPUT_H, OPT_INPUT_W));
    profile->setDimensions(NAMES[0], O::kMAX, Dims4(maxBatchSize, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
    config->addOptimizationProfile(profile);

#if TRT_VERSION_GE(8, 0, 0)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    auto* serialized = builder->buildSerializedNetwork(*network, *config);
    auto* engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
    delete serialized;
    delete network;
#else
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    auto* engine = builder->buildEngineWithConfig(*network, *config);
    network->destroy();
#endif
    std::cout << "build out\n";

    for (auto& _mem : w) {
        free(const_cast<void*>(_mem.second.values));
    }

    return engine;
}

void APIToModel(int32_t batch_size, IRuntime* runtime, IHostMemory** model_stream) {
    auto* builder = createInferBuilder(gLogger);
    auto* config = builder->createBuilderConfig();

    auto* engine = createEngine(batch_size, runtime, builder, config, DataType::kFLOAT);
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

std::vector<DummyTensor> doInference(IExecutionContext& context, void* input, int batchSize) {
    const auto& engine = context.getEngine();
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    std::vector<void*> buffers;

#if TRT_VERSION_GE(8, 0, 0)
    const int32_t nIO = engine.getNbIOTensors();
#else
    const int32_t nIO = engine.getNbBindings();
#endif

    buffers.resize(nIO);
    Dims out_shape;
    std::size_t out_size;
    context.setOptimizationProfileAsync(0, stream);
    for (auto i = 0; i < nIO; ++i) {
#if TRT_VERSION_GE(8, 0, 0)
        const auto* tensor_name = engine.getIOTensorName(i);
        auto s = getSize(engine.getTensorDataType(tensor_name));
        if (i == 0) {
            std::size_t size = s * batchSize * 3 * OPT_INPUT_W * OPT_INPUT_H;
            CHECK(cudaMalloc(&buffers[i], size));
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
            // since input config is dynamic, must set input shape to let TensorRT deduce the output shape
            context.setInputShape(tensor_name, Dims4{batchSize, 3, OPT_INPUT_H, OPT_INPUT_W});
            context.setInputTensorAddress(tensor_name, buffers[i]);
        }

        // we have already known nIO==2, so make it simple here
        if (i > 0) {
            out_shape = context.getTensorShape(tensor_name);
            out_size = std::accumulate(out_shape.d, out_shape.d + out_shape.nbDims, 1ULL, std::multiplies<>());
            CHECK(cudaMalloc(&buffers[i], s * out_size));
#if TRT_VERSION_GE(10, 0, 0)
            if (!context.setOutputTensorAddress(tensor_name, buffers[i])) {
                std::cerr << "setOutputTensorAddress failed\n";
                std::abort();
            }
#else
            if (!context.setTensorAddress(tensor_name, buffers[i])) {
                std::cerr << "setTensorAddress failed\n";
                std::abort();
            }
#endif
        }
#else
        const int32_t idx = engine.getBindingIndex(NAMES[i]);
        auto s = getSize(engine.getBindingDataType(idx));
        assert(idx == i);
        if (engine.bindingIsInput(idx)) {
            std::size_t size = s * batchSize * 3 * OPT_INPUT_W * OPT_INPUT_H;
            CHECK(cudaMalloc(&buffers[i], size));
            CHECK(cudaMemcpyAsync(buffers[i], input, size, cudaMemcpyHostToDevice, stream));
            context.setBindingDimensions(idx, Dims4{batchSize, 3, OPT_INPUT_H, OPT_INPUT_W});
        } else {
            assert(context.allInputDimensionsSpecified());
            out_shape = context.getBindingDimensions(idx);
            out_size = std::accumulate(out_shape.d, out_shape.d + out_shape.nbDims, 1, std::multiplies<std::size_t>());
            CHECK(cudaMalloc(&buffers[i], s * out_size));
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

    std::vector<DummyTensor> prob;
    prob.reserve(static_cast<std::size_t>(nIO > 0 ? nIO - 1 : 0));
    for (int i = 1; i < nIO; ++i) {
        prob.emplace_back(out_shape, DataType::kFLOAT, -1);
        CHECK(cudaMemcpyAsync(prob.back().data, buffers[i], out_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    for (auto& buffer : buffers) {
        CHECK(cudaFree(buffer));
    }
    CHECK(cudaStreamDestroy(stream));
    return prob;
}

static auto mainImpl(int argc, char** argv) -> int {
    checkTrtEnv();
    if (argc != 2) {
        std::cerr << "arguments not right!\n" << std::flush;
        std::cerr << "./csrnet -s  // serialize model to plan file\n" << std::flush;
        std::cerr << "./csrnet -d  // deserialize plan file and run inference\n" << std::flush;
        return -1;
    }
    char* trtModelStream{nullptr};
    std::streamsize size{0};
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(maxBatchSize, runtime, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(ENGINE_PATH, std::ios::binary);
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
#if TRT_VERSION_GE(8, 0, 0)
        delete modelStream;
#else
        modelStream->destroy();
#endif
        return 0;
    }
    if (std::string(argv[1]) == "-d") {
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
        return -1;
    }

#if TRT_VERSION_GE(8, 0, 0)
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // read and preprocess input data
    cv::Mat img_32f, rgb;
    auto img = cv::imread("../assets/IMG_1.jpg", cv::IMREAD_COLOR);
    img.convertTo(img_32f, CV_32FC3, 1.0 / 255.0, 0);
    img_32f = (img_32f - cv::Scalar(mean[0], mean[1], mean[2])) / cv::Scalar(stdv[0], stdv[1], stdv[2]);
    cv::cvtColor(img_32f, rgb, cv::COLOR_BGR2RGB);
    static std::size_t img_size = OPT_INPUT_H * OPT_INPUT_W * img.channels();
    std::vector<float> data(img_size);
    for (int i = 0; i < maxBatchSize; ++i) {
        // to NCHW (N == 1)
        for (int y = 0; y < OPT_INPUT_H; ++y) {
            for (int x = 0; x < OPT_INPUT_W; ++x) {
                const cv::Vec3f v = img_32f.at<cv::Vec3f>(y, x);
                data[i * img_size + 0 * OPT_INPUT_H * OPT_INPUT_W + y * OPT_INPUT_W + x] = v[0];
                data[i * img_size + 1 * OPT_INPUT_H * OPT_INPUT_W + y * OPT_INPUT_W + x] = v[1];
                data[i * img_size + 2 * OPT_INPUT_H * OPT_INPUT_W + y * OPT_INPUT_W + x] = v[2];
            }
        }
    }

    auto output = doInference(*context, data.data(), 1);
    const auto* data_ptr = reinterpret_cast<const float*>(output[0].data);

    DummyTensor& t = output[0];
    const auto out_h = t.dims.d[2];
    const auto out_w = t.dims.d[3];
    auto stride = static_cast<std::ptrdiff_t>(out_h) * out_w;
    printFirstOutputs("csrnet", data_ptr, static_cast<std::size_t>(stride));
    float num = std::accumulate(data_ptr, data_ptr + stride, 0.0f);
    cv::Mat density((int)out_h, (int)out_w, CV_32FC1, t.data);
    cv::Mat scaled, heatmap, save;
    cv::normalize(density, scaled, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(scaled, heatmap, cv::COLORMAP_JET);
    cv::resize(heatmap, heatmap, img.size(), 0., 0., cv::INTER_LINEAR);
    cv::addWeighted(img, 0.7, heatmap, 0.3, 0, save);
    cv::imwrite("../assets/csrnet_output_tensorrt.jpg", save);
    std::cout << "approximate people num: " << std::ceil(num) << "\nsave to ../assets/csrnet_output_tensorrt.jpg\n";

    std::vector<double> latencies;
    latencies.reserve(kBenchmarkRuns);
    for (int32_t i = 0; i < kBenchmarkRuns; ++i) {
        auto _start = std::chrono::steady_clock::now();
        (void)doInference(*context, data.data(), 1);
        auto _end = std::chrono::steady_clock::now();
        auto _time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
        latencies.push_back(static_cast<double>(_time) / 1000.0);
    }
    printBenchmark("csrnet", latencies);

    return 0;
}

auto main(int argc, char** argv) -> int {
    try {
        return mainImpl(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
}
