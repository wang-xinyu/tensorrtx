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

// Define input/output parameters
static const int INPUT_H = 256;
static const int INPUT_W = 320;
static const int OUTPUT_SIZE = 1000;
static const int batchSize = 32;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
using namespace nvinfer1;

static Logger gLogger;

// Load weight file
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open the weight file
    std::ifstream input(file);
    if (!input.is_open()) {
        std::cerr << "Unable to load weight file." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read the number of weights
    int32_t count;
    input >> count;
    if (count <= 0) {
        std::cerr << "Invalid weight map file." << std::endl;
        exit(EXIT_FAILURE);
    }

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read the name and size
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load weight data
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
    // If min_value is not specified, set it to divisor
    if (min_value == -1) {
        min_value = divisor;
    }

    // Calculate new channel size to be divisible by divisor
    int new_v = std::max(min_value, (v + divisor / 2) / divisor * divisor);

    // Ensure rounding down does not reduce by more than 10%
    if (new_v < static_cast<int>(0.9 * v)) {
        new_v += divisor;
    }

    return new_v;
}

ILayer* hardSigmoid(INetworkDefinition* network, ITensor& input) {
    // Apply Hard Sigmoid activation function
    IActivationLayer* scale_layer = network->addActivation(input, ActivationType::kHARD_SIGMOID);

    // Return the output after activation
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

    // Step 1: Convolution layer
    IConvolutionLayer* conv1 =
            network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});   // Stride of 2
    conv1->setPaddingNd(DimsHW{1, 1});  // Padding of 1

    // Step 2: Batch normalization layer
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Step 3: ReLU activation
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    return relu1;  // Return the result after activation
}

ILayer* convBnAct(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input,
                  int out_channels, std::string lname, ActivationType actType = ActivationType::kRELU) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Add convolution layer
    IConvolutionLayer* conv =
            network->addConvolutionNd(input, out_channels, DimsHW{1, 1}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv);
    conv->setStrideNd(DimsHW{1, 1});

    // Add batch normalization layer
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn1", 1e-5);

    // Add activation layer (default is ReLU)
    IActivationLayer* act = network->addActivation(*bn->getOutput(0), actType);
    assert(act);

    return act;
}

ILayer* squeezeExcite(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap,
                      int in_chs, float se_ratio = 0.25, std::string lname = "", float eps = 1e-5) {
    // Step 1: Global average pooling
    IReduceLayer* avg_pool = network->addReduce(input, ReduceOperation::kAVG, 1 << 2 | 1 << 3, true);
    assert(avg_pool);

    // Step 2: 1x1 convolution for dimension reduction
    int reduced_chs = _make_divisible(static_cast<int>(in_chs * se_ratio), 4);
    IConvolutionLayer* conv_reduce =
            network->addConvolutionNd(*avg_pool->getOutput(0), reduced_chs, DimsHW{1, 1},
                                      weightMap[lname + ".conv_reduce.weight"], weightMap[lname + ".conv_reduce.bias"]);
    assert(conv_reduce);

    // Step 3: ReLU activation
    IActivationLayer* relu1 = network->addActivation(*conv_reduce->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Step 4: 1x1 convolution for dimension expansion
    IConvolutionLayer* conv_expand =
            network->addConvolutionNd(*relu1->getOutput(0), in_chs, DimsHW{1, 1},
                                      weightMap[lname + ".conv_expand.weight"], weightMap[lname + ".conv_expand.bias"]);
    assert(conv_expand);

    // Step 5: Hard Sigmoid activation
    ILayer* hard_sigmoid = hardSigmoid(network, *conv_expand->getOutput(0));

    // Step 6: Multiply input by the output of SE module
    IElementWiseLayer* scale = network->addElementWise(input, *hard_sigmoid->getOutput(0), ElementWiseOperation::kPROD);
    assert(scale);

    return scale;
}

ILayer* ghostModuleV2(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap, int inp,
                      int oup, int kernel_size = 1, int ratio = 2, int dw_size = 3, int stride = 1, bool relu = true,
                      std::string lname = "", std::string mode = "original") {
    int init_channels = std::ceil(oup / ratio);
    int new_channels = init_channels * (ratio - 1);

    // Primary convolution
    IConvolutionLayer* primary_conv = network->addConvolutionNd(input, init_channels, DimsHW{kernel_size, kernel_size},
                                                                weightMap[lname + ".primary_conv.0.weight"], Weights{});
    primary_conv->setStrideNd(DimsHW{stride, stride});
    primary_conv->setPaddingNd(DimsHW{kernel_size / 2, kernel_size / 2});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *primary_conv->getOutput(0), lname + ".primary_conv.1", 1e-5);

    ITensor* act1_output = bn1->getOutput(0);
    if (relu) {
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        act1_output = relu1->getOutput(0);
    }

    // Cheap operation
    IConvolutionLayer* cheap_conv =
            network->addConvolutionNd(*act1_output, new_channels, DimsHW{dw_size, dw_size},
                                      weightMap[lname + ".cheap_operation.0.weight"], Weights{});
    cheap_conv->setStrideNd(DimsHW{1, 1});
    cheap_conv->setPaddingNd(DimsHW{dw_size / 2, dw_size / 2});
    cheap_conv->setNbGroups(init_channels);

    IScaleLayer* bn2 =
            addBatchNorm2d(network, weightMap, *cheap_conv->getOutput(0), lname + ".cheap_operation.1", 1e-5);

    ITensor* act2_output = bn2->getOutput(0);
    if (relu) {
        IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        act2_output = relu2->getOutput(0);
    }

    // Concatenate
    ITensor* concat_inputs[] = {act1_output, act2_output};
    IConcatenationLayer* concat = network->addConcatenation(concat_inputs, 2);

    // Slice to oup channels
    Dims start{4, {0, 0, 0, 0}};
    Dims size = concat->getOutput(0)->getDimensions();
    size.d[1] = oup;
    Dims stride_{4, {1, 1, 1, 1}};

    ISliceLayer* slice = network->addSlice(*concat->getOutput(0), start, size, stride_);

    ITensor* out = slice->getOutput(0);

    if (mode == "original") {
        return slice;
    } else if (mode == "attn") {
        // Attention mechanism
        // Average pooling
        IPoolingLayer* avg_pool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{2, 2});
        avg_pool->setStrideNd(DimsHW{2, 2});

        ITensor* avg_pooled = avg_pool->getOutput(0);

        // Short convolution branch
        IConvolutionLayer* short_conv1 =
                network->addConvolutionNd(*avg_pooled, oup, DimsHW{kernel_size, kernel_size},
                                          weightMap[lname + ".short_conv.0.weight"], Weights{});
        short_conv1->setStrideNd(DimsHW{1, 1});
        short_conv1->setPaddingNd(DimsHW{kernel_size / 2, kernel_size / 2});
        IScaleLayer* short_bn1 =
                addBatchNorm2d(network, weightMap, *short_conv1->getOutput(0), lname + ".short_conv.1", 1e-5);

        // Conv with kernel size (1,5)
        IConvolutionLayer* short_conv2 = network->addConvolutionNd(
                *short_bn1->getOutput(0), oup, DimsHW{1, 5}, weightMap[lname + ".short_conv.2.weight"], Weights{});
        short_conv2->setStrideNd(DimsHW{1, 1});
        short_conv2->setPaddingNd(DimsHW{0, 2});
        short_conv2->setNbGroups(oup);
        IScaleLayer* short_bn2 =
                addBatchNorm2d(network, weightMap, *short_conv2->getOutput(0), lname + ".short_conv.3", 1e-5);

        // Conv with kernel size (5,1)
        IConvolutionLayer* short_conv3 = network->addConvolutionNd(
                *short_bn2->getOutput(0), oup, DimsHW{5, 1}, weightMap[lname + ".short_conv.4.weight"], Weights{});
        short_conv3->setStrideNd(DimsHW{1, 1});
        short_conv3->setPaddingNd(DimsHW{2, 0});
        short_conv3->setNbGroups(oup);
        IScaleLayer* short_bn3 =
                addBatchNorm2d(network, weightMap, *short_conv3->getOutput(0), lname + ".short_conv.5", 1e-5);

        ITensor* res = short_bn3->getOutput(0);

        // Sigmoid activation
        IActivationLayer* gate = network->addActivation(*res, ActivationType::kSIGMOID);

        // Upsample to the same size as out
        IResizeLayer* gate_upsampled = network->addResize(*gate->getOutput(0));
        gate_upsampled->setResizeMode(ResizeMode::kNEAREST);
        Dims out_dims = out->getDimensions();
        gate_upsampled->setOutputDimensions(out_dims);

        // Element-wise multiplication
        IElementWiseLayer* scaled_out =
                network->addElementWise(*out, *gate_upsampled->getOutput(0), ElementWiseOperation::kPROD);

        return scaled_out;
    } else {
        std::cerr << "Invalid mode: " << mode << " in ghostModuleV2" << std::endl;
        return nullptr;
    }
}

ILayer* ghostBottleneck(INetworkDefinition* network, ITensor& input, std::map<std::string, Weights>& weightMap,
                        int in_chs, int mid_chs, int out_chs, int dw_kernel_size = 3, int stride = 1,
                        float se_ratio = 0.0f, std::string lname = "", int layer_id = 0) {
    // Determine mode based on layer_id
    std::string mode = (layer_id <= 1) ? "original" : "attn";

    // ghost1
    ILayer* ghost1 =
            ghostModuleV2(network, input, weightMap, in_chs, mid_chs, 1, 2, 3, 1, true, lname + ".ghost1", mode);

    ILayer* depthwise_conv = ghost1;
    if (stride > 1) {
        IConvolutionLayer* conv_dw =
                network->addConvolutionNd(*ghost1->getOutput(0), mid_chs, DimsHW{dw_kernel_size, dw_kernel_size},
                                          weightMap[lname + ".conv_dw.weight"], Weights{});
        conv_dw->setStrideNd(DimsHW{stride, stride});
        conv_dw->setPaddingNd(DimsHW{(dw_kernel_size - 1) / 2, (dw_kernel_size - 1) / 2});
        conv_dw->setNbGroups(mid_chs);
        IScaleLayer* bn_dw = addBatchNorm2d(network, weightMap, *conv_dw->getOutput(0), lname + ".bn_dw", 1e-5);
        depthwise_conv = bn_dw;
    }

    ILayer* se_layer = depthwise_conv;
    if (se_ratio > 0.0f) {
        se_layer = squeezeExcite(network, *depthwise_conv->getOutput(0), weightMap, mid_chs, se_ratio, lname + ".se");
    }

    // ghost2 uses original mode
    ILayer* ghost2 = ghostModuleV2(network, *se_layer->getOutput(0), weightMap, mid_chs, out_chs, 1, 2, 3, 1, false,
                                   lname + ".ghost2", "original");

    ILayer* shortcut_layer = nullptr;
    if (in_chs == out_chs && stride == 1) {
        shortcut_layer = network->addIdentity(input);
    } else {
        IConvolutionLayer* conv_shortcut_dw =
                network->addConvolutionNd(input, in_chs, DimsHW{dw_kernel_size, dw_kernel_size},
                                          weightMap[lname + ".shortcut.0.weight"], Weights{});
        conv_shortcut_dw->setStrideNd(DimsHW{stride, stride});
        conv_shortcut_dw->setPaddingNd(DimsHW{(dw_kernel_size - 1) / 2, (dw_kernel_size - 1) / 2});
        conv_shortcut_dw->setNbGroups(in_chs);
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
    // Use explicit batch mode
    INetworkDefinition* network =
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    // Create input tensor
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{batchSize, 3, INPUT_H, INPUT_W});
    assert(data);

    // Load weights
    std::map<std::string, Weights> weightMap = loadWeights("../ghostnetv2.weights");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Step 1: Conv Stem
    IActivationLayer* conv_stem = convBnReluStem(network, weightMap, *data, 16, "conv_stem");

    ILayer* current_layer = conv_stem;

    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 16, 16, 16, 3, 1, 0.0f, "blocks.0.0", 0);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 16, 48, 24, 3, 2, 0.0f, "blocks.1.0", 1);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 24, 72, 24, 3, 1, 0.0f, "blocks.2.0", 2);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 24, 72, 40, 5, 2, 0.25f, "blocks.3.0", 3);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 40, 120, 40, 5, 1, 0.25f,
                                    "blocks.4.0", 4);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 40, 240, 80, 3, 2, 0.0f, "blocks.5.0", 5);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 200, 80, 3, 1, 0.0f, "blocks.6.0", 6);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 184, 80, 3, 1, 0.0f, "blocks.6.1", 7);
    current_layer =
            ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 184, 80, 3, 1, 0.0f, "blocks.6.2", 8);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 80, 480, 112, 3, 1, 0.25f,
                                    "blocks.6.3", 9);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 112, 672, 112, 3, 1, 0.25f,
                                    "blocks.6.4", 10);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 112, 672, 160, 5, 2, 0.25f,
                                    "blocks.7.0", 11);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0.0f,
                                    "blocks.8.0", 12);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0.25f,
                                    "blocks.8.1", 13);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0.0f,
                                    "blocks.8.2", 14);
    current_layer = ghostBottleneck(network, *current_layer->getOutput(0), weightMap, 160, 960, 160, 5, 1, 0.25f,
                                    "blocks.8.3", 15);

    // Apply ConvBnAct
    current_layer = convBnAct(network, weightMap, *current_layer->getOutput(0), 960, "blocks.9.0");

    // Global average pooling
    IReduceLayer* global_pool =
            network->addReduce(*current_layer->getOutput(0), ReduceOperation::kAVG, 1 << 2 | 1 << 3, true);
    assert(global_pool);

    // Conv Head
    IConvolutionLayer* conv_head = network->addConvolutionNd(
            *global_pool->getOutput(0), 1280, DimsHW{1, 1}, weightMap["conv_head.weight"], weightMap["conv_head.bias"]);
    IActivationLayer* act2 = network->addActivation(*conv_head->getOutput(0), ActivationType::kRELU);

    // Fully connected layer (classifier)
    IFullyConnectedLayer* classifier = network->addFullyConnected(
            *act2->getOutput(0), 1000, weightMap["classifier.weight"], weightMap["classifier.bias"]);
    classifier->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*classifier->getOutput(0));

    // Build the engine
    config->setMaxWorkspaceSize(1 << 24);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Destroy the network
    network->destroy();

    // Free memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model and serialize
    ICudaEngine* engine = createEngine(builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Release resources
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Input and output buffers
    void* buffers[2];

    // Create buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Copy input data to device, execute inference, and copy output back to host
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
        std::cerr << "./ghostnetv2 -s   // serialize model to plan file" << std::endl;
        std::cerr << "./ghostnetv2 -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create model and serialize
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(&modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("ghostnetv2.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("ghostnetv2.engine", std::ios::binary);
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

    // Allocate input and output data
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

    // Execute inference
    doInference(*context, data, prob, batchSize);

    // Print output results
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

    // Release resources
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete[] data;
    delete[] prob;

    return 0;
}
