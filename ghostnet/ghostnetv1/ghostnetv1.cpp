#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace std;

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 256;
static const int INPUT_W = 320;
static const int OUTPUT_SIZE = 1000;
static const int batchSize = 32;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    if (!input.is_open()) {
        std::cerr << "Unable to load weight file." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read number of weight blobs
    int32_t count;
    input >> count;
    if (count <= 0) {
        std::cerr << "Invalid weight map file." << std::endl;
        exit(EXIT_FAILURE);
    }

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

int _make_divisible(int v, int divisor, int min_value = -1) {
    if (min_value == -1) {
        min_value = divisor;
    }

    int new_v = std::max(min_value, (v + divisor / 2) / divisor * divisor);

    if (new_v < static_cast<int>(0.9 * v)) {
        new_v += divisor;
    }

    return new_v;
}

ILayer* hardSigmoid(INetworkDefinition* network, ITensor& input) {

    IActivationLayer* scale_layer = network->addActivation(input, ActivationType::kHARD_SIGMOID);

    return scale_layer;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                            std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

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

IActivationLayer* convBnReluStem(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                                 int outch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});   // Stride = 2
    conv1->setPaddingNd(DimsHW{1, 1});  // Padding = 1

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;
}

ILayer* convBnAct(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                  int out_channels, std::string lname, ActivationType actType = ActivationType::kRELU) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv =
            network->addConvolutionNd(input, out_channels, DimsHW{1, 1}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});

    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* act = network->addActivation(*bn->getOutput(0), actType);
    assert(act);

    return act;
}

ILayer* squeezeExcite(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap,
                      int in_chs, float se_ratio = 0.25, std::string lname = "", float eps = 1e-5) {

    IReduceLayer* avg_pool = network->addReduce(input, ReduceOperation::kAVG, 1 << 2 | 1 << 3, true);
    assert(avg_pool);

    // Reduce channels with 1x1 convolution
    int reduced_chs = _make_divisible(static_cast<int>(in_chs * se_ratio), 4);
    IConvolutionLayer* conv_reduce =
            network->addConvolutionNd(*avg_pool->getOutput(0), reduced_chs, DimsHW{1, 1},
                                      weightMap[lname + ".conv_reduce.weight"], weightMap[lname + ".conv_reduce.bias"]);
    assert(conv_reduce);

    IActivationLayer* relu1 = network->addActivation(*conv_reduce->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Expand channels back with another 1x1 convolution
    IConvolutionLayer* conv_expand =
            network->addConvolutionNd(*relu1->getOutput(0), in_chs, DimsHW{1, 1},
                                      weightMap[lname + ".conv_expand.weight"], weightMap[lname + ".conv_expand.bias"]);
    assert(conv_expand);
    cout << "SE conv_expand -> " << printTensorShape(conv_expand->getOutput(0)) << endl;

    // Apply hardSigmoid function
    ILayer* hard_sigmoid = hardSigmoid(network, *conv_expand->getOutput(0));
    cout << "hard_sigmoid conv_expand -> " << printTensorShape(hard_sigmoid->getOutput(0)) << endl;

    // Elementwise multiplication of input and gated SE output
    IElementWiseLayer* scale = network->addElementWise(input, *hard_sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(scale);

    return scale;
}

ILayer* ghostModule(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap, int inp,
                    int oup, int kernel_size = 1, int ratio = 2, int dw_size = 3, int stride = 1, bool relu = true,
                    std::string lname = "") {
    int init_channels = std::ceil(oup / ratio);
    int new_channels = init_channels * (ratio - 1);

    // Primary convolution
    IConvolutionLayer* primary_conv = network->addConvolutionNd(input, init_channels, DimsHW{kernel_size, kernel_size},
                                                                weightMap[lname + ".primary_conv.0.weight"], Weights{});
    primary_conv->setStrideNd(DimsHW{stride, stride});
    primary_conv->setPaddingNd(DimsHW{kernel_size / 2, kernel_size / 2});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *primary_conv->getOutput(0), lname + ".primary_conv.1", 1e-5);

    // Cheap operation (Depthwise Convolution)
    IConvolutionLayer* cheap_conv =
            network->addConvolutionNd(*bn1->getOutput(0), new_channels, DimsHW{dw_size, dw_size},
                                      weightMap[lname + ".cheap_operation.0.weight"], Weights{});
    cheap_conv->setStrideNd(DimsHW{1, 1});
    cheap_conv->setPaddingNd(DimsHW{dw_size / 2, dw_size / 2});
    cheap_conv->setNbGroups(init_channels);
    IScaleLayer* bn2 =
            addBatchNorm2d(network, weightMap, *cheap_conv->getOutput(0), lname + ".cheap_operation.1", 1e-5);

    // Define relu1 and relu2
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    // Initialize inputs array based on the `relu` flag
    std::vector<ITensor*> inputs_vec;
    if (relu) {
        inputs_vec = {relu1->getOutput(0), relu2->getOutput(0)};
    } else {
        inputs_vec = {bn1->getOutput(0), bn2->getOutput(0)};
    }

    ITensor* inputs[] = {inputs_vec[0], inputs_vec[1]};
    IConcatenationLayer* concat = network->addConcatenation(inputs, 2);
    std::cout << printTensorShape(concat->getOutput(0)) << std::endl;

    // Slice the output to keep only the first `oup` channels
    Dims start{4, {0, 0, 0, 0}};  // Starting from batch=0, channel=0, height=0, width=0
    Dims size{4,
              {concat->getOutput(0)->getDimensions().d[0], oup, concat->getOutput(0)->getDimensions().d[2],
               concat->getOutput(0)
                       ->getDimensions()
                       .d[3]}};     // Keep all batches, first `oup` channels, all heights and widths
    Dims stride_{4, {1, 1, 1, 1}};  // Stride is 1 for all dimensions

    ISliceLayer* slice = network->addSlice(*concat->getOutput(0), start, size, stride_);
    cout << "slice" << printTensorShape(slice->getOutput(0)) << endl;

    return slice;
}

ILayer* ghostBottleneck(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap,
                        int in_chs, int mid_chs, int out_chs, int dw_kernel_size = 3, int stride = 1,
                        float se_ratio = 0.0f, std::string lname = "") {
    ILayer* ghost1 = ghostModule(network, input, weightMap, in_chs, mid_chs, 1, 2, 3, 1, true, lname + ".ghost1");

    ILayer* depthwise_conv = ghost1;
    if (stride > 1) {
        IConvolutionLayer* conv_dw =
                network->addConvolutionNd(*ghost1->getOutput(0), mid_chs, DimsHW{dw_kernel_size, dw_kernel_size},
                                          weightMap[lname + ".conv_dw.weight"], Weights{});
        conv_dw->setStrideNd(DimsHW{stride, stride});
        conv_dw->setPaddingNd(DimsHW{(dw_kernel_size - 1) / 2, (dw_kernel_size - 1) / 2});
        conv_dw->setNbGroups(mid_chs);  // Depth-wise convolution
        IScaleLayer* bn_dw = addBatchNorm2d(network, weightMap, *conv_dw->getOutput(0), lname + ".bn_dw", 1e-5);
        depthwise_conv = bn_dw;
    }

    ILayer* se_layer = depthwise_conv;
    if (se_ratio > 0.0f) {
        se_layer = squeezeExcite(network, *depthwise_conv->getOutput(0), weightMap, mid_chs, se_ratio, lname + ".se");
    }

    ILayer* ghost2 = ghostModule(network, *se_layer->getOutput(0), weightMap, mid_chs, out_chs, 1, 2, 3, 1, false,
                                 lname + ".ghost2");

    ILayer* shortcut_layer = nullptr;
    if (in_chs == out_chs && stride == 1) {
        shortcut_layer = network->addIdentity(input);
    } else {
        IConvolutionLayer* conv_shortcut_dw =
                network->addConvolutionNd(input, in_chs, DimsHW{dw_kernel_size, dw_kernel_size},
                                          weightMap[lname + ".shortcut.0.weight"], Weights{});

        conv_shortcut_dw->setStrideNd(DimsHW{stride, stride});
        conv_shortcut_dw->setPaddingNd(DimsHW{(dw_kernel_size - 1) / 2, (dw_kernel_size - 1) / 2});
        conv_shortcut_dw->setNbGroups(in_chs);  // Depth-wise convolution
        IScaleLayer* bn_shortcut_dw =
                addBatchNorm2d(network, weightMap, *conv_shortcut_dw->getOutput(0), lname + ".shortcut.1", 1e-5);

        IConvolutionLayer* conv_shortcut_pw =
                network->addConvolutionNd(*bn_shortcut_dw->getOutput(0), out_chs, DimsHW{1, 1},
                                          weightMap[lname + ".shortcut.2.weight"], Weights{});
        IScaleLayer* bn_shortcut_pw =
                addBatchNorm2d(network, weightMap, *conv_shortcut_pw->getOutput(0), lname + ".shortcut.3", 1e-5);
        shortcut_layer = bn_shortcut_pw;
    }

    IElementWiseLayer* ew_sum =
            network->addElementWise(*ghost2->getOutput(0), *shortcut_layer->getOutput(0), ElementWiseOperation::kSUM);

    return ew_sum;
}

ICudaEngine* createEngine(IBuilder* builder, IBuilderConfig* config, DataType dt) {

    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // Create input tensor of shape {batchSize, 3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{batchSize, 3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../ghostnetv1.weights");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Conv Stem
    IActivationLayer* conv_stem = convBnReluStem(network, weightMap, *data, 16, "conv_stem");

    ILayer* current_layer = conv_stem;
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 16, 16, 16, 3, 1, 0, "blocks.0.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 16, 48, 24, 3, 2, 0, "blocks.1.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 24, 72, 24, 3, 1, 0, "blocks.2.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 24, 72, 40, 5, 2, 0.25, "blocks.3.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 40, 120, 40, 5, 1, 0.25, "blocks.4.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 40, 240, 80, 3, 2, 0, "blocks.5.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 200, 80, 3, 1, 0, "blocks.6.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 184, 80, 3, 1, 0, "blocks.6.1");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 184, 80, 3, 1, 0, "blocks.6.2");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 480, 112, 3, 1, 0.25, "blocks.6.3");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 112, 672, 112, 3, 1, 0.25, "blocks.6.4");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 112, 672, 160, 5, 2, 0.25, "blocks.7.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0, "blocks.8.0");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0.25, "blocks.8.1");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0, "blocks.8.2");
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0.25, "blocks.8.3");

    // Apply ConvBnAct
    current_layer = convBnAct(network, weightMap, *current_layer->getOutput(0), 960, "blocks.9.0");
    // Global Average Pooling
    IReduceLayer* global_pool =
            network->addReduce(*current_layer->getOutput(0), ReduceOperation::kAVG, 1 << 2 | 1 << 3, true);
    assert(global_pool);

    // Conv Head
    IConvolutionLayer* conv_head = network->addConvolutionNd(
            *global_pool->getOutput(0), 1280, DimsHW{1, 1}, weightMap["conv_head.weight"], weightMap["conv_head.bias"]);
    IActivationLayer* act2 = network->addActivation(*conv_head->getOutput(0), ActivationType::kRELU);

    // Fully Connected Layer (Classifier)
    IFullyConnectedLayer* classifier = network->addFullyConnected(
            *act2->getOutput(0), 1000, weightMap["classifier.weight"], weightMap["classifier.bias"]);
    classifier->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*classifier->getOutput(0));

    // Build engine
    config->setMaxWorkspaceSize(1 << 24);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Pointers to input and output device buffers to pass to engine.
    void* buffers[2];

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./ghostnetv1 -s   // serialize model to plan file" << std::endl;
        std::cerr << "./ghostnetv1 -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(&modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("ghostnetv1.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("ghostnetv1.engine", std::ios::binary);
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

    float* data = new float[batchSize * 3 * INPUT_H * INPUT_W];
    for (int i = 0; i < batchSize * 3 * INPUT_H * INPUT_W; i++)
        data[i] = 10.0;

    float* prob = new float[batchSize * OUTPUT_SIZE];

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    doInference(*context, data, prob, batchSize);

    std::cout << "\nOutput:\n\n";
    for (int i = 0; i < batchSize; i++) {
        std::cout << "Batch " << i << ":\n";
        for (unsigned int j = 0; j < OUTPUT_SIZE; j++) {
            std::cout << prob[i * OUTPUT_SIZE + j] << ", ";
            if (j % 10 == 0)
                std::cout << j / 10 << std::endl;
        }
        std::cout << "\n";
    }

    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete[] data;
    delete[] prob;

    return 0;
}
