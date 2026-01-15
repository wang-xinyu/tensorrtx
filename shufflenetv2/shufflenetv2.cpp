#include <NvInfer.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "logging.h"
#include "utils.h"

struct ShuffleNetV2Params {
    int repeat[3];
    int output_chn[5];
};

static constexpr ShuffleNetV2Params v2_x0_5 = {{4, 8, 4}, {24, 48, 96, 192, 1024}};
static constexpr ShuffleNetV2Params v2_x1_0 = {{4, 8, 4}, {24, 116, 232, 464, 1024}};
static constexpr ShuffleNetV2Params v2_x1_5 = {{4, 8, 4}, {24, 176, 352, 704, 1024}};
static constexpr ShuffleNetV2Params v2_x2_0 = {{4, 8, 4}, {24, 244, 488, 976, 2048}};

// stuff we know about shufflenet-v2
constexpr const int32_t N = 1;
constexpr const int32_t INPUT_H = 224;
constexpr const int32_t INPUT_W = 224;
constexpr const int32_t SIZES[] = {3 * INPUT_H * INPUT_W, 24 * 56 * 56};  // 1000
constexpr const char* NAMES[] = {"data", "logits"};
static constexpr const bool TRT_PREPROCESS = TRT_VERSION >= 8510 ? true : false;
static constexpr const float mean[3] = {0.485f, 0.456f, 0.406f};
static constexpr const float stdv[3] = {0.229f, 0.224f, 0.225f};

constexpr char WTS_PATH[] = "../models/shufflenet_v2_x0_5.wts";
constexpr char ENGINE_PATH[] = "../models/shufflenet.engine";
const char LABELS_PATH[] = "../assets/imagenet1000_clsidx_to_labels.txt";

using namespace nvinfer1;
using WeightMap = std::map<std::string, Weights>;
using M = MatrixOperation;

static Logger gLogger;

Dims _debug_shape(const ILayer* l) {
    Dims dims = l->getOutput(0)->getDimensions();
    std::cout << l->getOutput(0)->getName() << ":\t[";
    for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << ", ";
    }
    std::cout << ']' << std::endl;
    return dims;
}

ILayer* addBatchNorm2d(INetworkDefinition* network, WeightMap& weightMap, ITensor& input, std::string lname,
                       float eps = 1e-3f) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << lname << " running_var len: " << len << std::endl;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float* shval = static_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};
    static const Weights power{DataType::kFLOAT, nullptr, 0ll};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

/**
 * @brief a basic convolution+bn layer with an optional relu layer
 *
 * @param network network definition
 * @param m weight map
 * @param input input tensor
 * @param lname layer name
 * @param ch output channels
 * @param k kernel
 * @param s stride
 * @param p padding
 * @param g groups
 * @param with_relu true if with relu
 * @return ILayer*
 */
ILayer* CBR(INetworkDefinition* network, WeightMap& m, ITensor& input, std::string lname, int ch, int k, int s = 1,
            int p = 0, int g = 1, bool with_relu = true, int start_index = 0) {
    static const Weights emptywts{DataType::kFLOAT, nullptr, 0ll};
    auto conv_name = lname + "." + std::to_string(start_index++);
    auto* conv = network->addConvolutionNd(input, ch, DimsHW{k, k}, m[conv_name + ".weight"], emptywts);

    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    conv->setNbGroups(g);
    conv->setName(conv_name.c_str());

    auto bn_name = lname + "." + std::to_string(start_index++);
    auto* bn = addBatchNorm2d(network, m, *conv->getOutput(0), bn_name, 1e-5f);
    bn->setName((bn_name + ".bn").c_str());

    if (with_relu) {
        auto* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
        auto relu_name = lname + "." + std::to_string(start_index) + ".relu";
        assert(relu);
        relu->setName(relu_name.c_str());
        return relu;
    }
    return bn;
}

/**
 * @brief invered residual block
 *
 * @param network network definition
 * @param m weight map
 * @param input input tensor
 * @param lname layer name
 * @param inch input channels
 * @param outch output channels
 * @param s stride
 * @return ILayer*
 */
ILayer* invertedRes(INetworkDefinition* net, WeightMap& m, ITensor& input, std::string lname, int inch, int outch,
                    int s) {
    static const Weights emptywts{DataType::kFLOAT, nullptr, 0ll};
    if (s < 1 || s > 3) {
        throw std::invalid_argument("stride must be in [1, 3]");
    }
    int32_t bf /* branch features */ = outch / 2;
    ITensor *x1{nullptr}, *x2{nullptr};

    if (s == 1) {
        auto d = input.getDimensions();
        Dims4 stride{1, 1, 1, 1};
        Dims4 half{d.d[0], d.d[1] / 2, d.d[2], d.d[3]};
        auto* s1 = net->addSlice(input, Dims4{0, 0, 0, 0}, half, stride);
        auto* s2 = net->addSlice(input, Dims4{0, d.d[1] / 2, 0, 0}, half, stride);
        _debug_shape(s2);
        x1 = s1->getOutput(0);
        x2 = s2->getOutput(0);
    } else {
        if (s > 1) {
            auto* b1 = CBR(net, m, input, lname + ".branch1", inch, 3, s, 1, inch, false, 0);
            b1 = CBR(net, m, *b1->getOutput(0), lname + ".branch1", inch, 1, 1, 0, 1, true, 2);
            x1 = b1->getOutput(0);
            _debug_shape(b1);
        } else {
            x1 = &input;
        }
        x2 = &input;
    }

    auto* b2 = CBR(net, m, *x2, lname + ".branch2", bf, 1, 1, 0, 1, true, 0);
    b2 = CBR(net, m, *b2->getOutput(0), lname + ".branch2", bf, 3, s, 1, bf, false, 3);
    b2 = CBR(net, m, *b2->getOutput(0), lname + ".branch2", bf, 1, 1, 0, 1, true, 5);
    _debug_shape(b2);

    ITensor* cat_tensors[] = {x1, b2->getOutput(0)};
    auto* cat = net->addConcatenation(cat_tensors, 2);
    auto cat_name = lname + ".cat";
    assert(cat);
    cat->setName(cat_name.c_str());
    cat->setAxis(1);
    auto dims = _debug_shape(cat);

    auto* sf1 = net->addShuffle(*cat->getOutput(0));
    assert(sf1);
    sf1->setName((lname + ".shuffle.1").c_str());
    auto d = cat->getOutput(0)->getDimensions();
    auto dim_sf1 = Dims{5, {d.d[0], 2, d.d[1] / 2, d.d[2], d.d[3]}};
    sf1->setReshapeDimensions(dim_sf1);
    sf1->setSecondTranspose({0, 2, 1, 3, 4});

    auto* sf2 = net->addShuffle(*sf1->getOutput(0));
    assert(sf2);
    sf2->setName((lname + ".shuffle.2").c_str());
    sf2->setReshapeDimensions(d);

    return sf2;
}

/**
 * @brief Create a Engine object
 * 
 * @param N max batch size
 * @param runtime runtime
 * @param builder builder
 * @param config config
 * @param dt data type
 * @param param the type of model to be built
 * @return ICudaEngine* 
 */
ICudaEngine* createEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt,
                          ShuffleNetV2Params param = v2_x0_5) {
    WeightMap m = loadWeights(WTS_PATH);
    static const Weights emptywts{DataType::kFLOAT, nullptr, 0};

#if TRT_VERSION >= 10000
    auto* net = builder->createNetworkV2(0);
#else
    auto* net = builder->createNetworkV2(1u << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

    int32_t in_ch = 3;
    ITensor* input{nullptr};
    if constexpr (TRT_PREPROCESS) {
        // for simplicity, resize image on cpu side
#if TRT_VERSION > 8510
        dt = DataType::kUINT8;
#else
        dt = DataType::kINT8;
#endif
        input = net->addInput(NAMES[0], dt, Dims4{N, INPUT_H, INPUT_W, in_ch});
        auto* trans = addTransformLayer(net, *input, true, mean, stdv);
        input = trans->getOutput(0);
    } else {
        input = net->addInput(NAMES[0], dt, Dims4{N, in_ch, INPUT_H, INPUT_W});
    }
    assert(input);

    /** conv1 and maxpool */
    auto* cbr1 = CBR(net, m, *input, "conv1", param.output_chn[0], 3, 2, 1);
    auto* pool1 = net->addPoolingNd(*cbr1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});
    _debug_shape(pool1);

    /** stage 2, 3, 4 */
    ILayer* _layer = pool1;
    in_ch = param.output_chn[0];
    for (int stage = 2; stage < 5; ++stage) {
        int32_t out_ch = param.output_chn[stage - 1];
        std::string lname = "stage" + std::to_string(stage);
        std::cout << "================ " << lname << " ================" << std::endl;
        _layer = invertedRes(net, m, *_layer->getOutput(0), lname + ".0", in_ch, out_ch, 2);
        _debug_shape(_layer);
        for (int j = 1; j < param.repeat[stage - 2]; ++j) {
            _layer = invertedRes(net, m, *_layer->getOutput(0), lname + "." + std::to_string(j), out_ch, out_ch, 1);
        }
        in_ch = out_ch;
    }

    /** conv5, mean and fully connected layer */
    auto* conv5 = CBR(net, m, *_layer->getOutput(0), "conv5", param.output_chn[4], 1, 1, 0);
    auto* mean = net->addReduce(*conv5->getOutput(0), ReduceOperation::kAVG, 0xc, false);
    mean->setName("global_pool(mean)");
    auto* fcw = net->addConstant(DimsHW{1000, 1024}, m["fc.weight"]);
    auto* fcb = net->addConstant(DimsHW{1, 1000}, m["fc.bias"]);
    auto* _fc = net->addMatrixMultiply(*mean->getOutput(0), M::kNONE, *fcw->getOutput(0), M::kTRANSPOSE);
    auto* fc = net->addElementWise(*_fc->getOutput(0), *fcb->getOutput(0), ElementWiseOperation::kSUM);
    fc->getOutput(0)->setName(NAMES[1]);
    _debug_shape(fc);

    net->markOutput(*fc->getOutput(0));

#if TRT_VERSION >= 8000
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
    IHostMemory* mem = builder->buildSerializedNetwork(*net, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(mem->data(), mem->size());
    delete net;
#else
    builder->setMaxBatchSize(N);
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    ICudaEngine* engine = builder->buildEngineWithConfig(*net, *config);
    net->destroy();
#endif
    std::cout << "build out" << std::endl;

    // Release host memory
    for (auto& mem : m) {
        free((void*)(mem.second.values));
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

std::vector<std::vector<float>> doInference(IExecutionContext& context, void* input, int batchSize) {
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
    // Release stream and buffers
    CHECK(cudaStreamDestroy(stream));
    for (auto& buffer : buffers) {
        CHECK(cudaFree(buffer));
    }
    return prob;
}

int main(int argc, char** argv) {
    checkTrtEnv();
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./shufflenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./shufflenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, runtime, &modelStream);
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
        return 1;
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
        return -1;
    }

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
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
    for (int i = 0; i < 100; ++i) {
        auto start = std::chrono::system_clock::now();
        auto prob = doInference(*context, input, N);
        auto end = std::chrono::system_clock::now();
        auto period = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << period.count() << "us" << std::endl;

        for (auto& vector : prob) {
            int idx = 0;
            for (auto& v : vector) {
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
