#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

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
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                            const std::string& lname, ITensor& input, float eps = 1e-5) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;

    int len = weightMap[lname + ".running_var"].count;
    std::cout << "[BatchNorm] " << lname << " channels = " << len << std::endl;

    // scale = gamma / sqrt(var + eps)
    float* scaleVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        scaleVal[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scaleVal, len};

    // shift = beta - mean * scale
    float* shiftVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        shiftVal[i] = beta[i] - mean[i] * scaleVal[i];
    }
    Weights shift{DataType::kFLOAT, shiftVal, len};

    // power = 1.0
    float* powerVal = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; ++i) {
        powerVal[i] = 1.0f;
    }
    Weights power{DataType::kFLOAT, powerVal, len};

    // save in weightMap in case of memory leak.
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;

    IScaleLayer* bn = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(bn);

    return bn;
}

IActivationLayer* convBnRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                             const std::string& lname, ITensor& input, int out_ch, int ksize, int stride, int padding,
                             int groups = 1) {
    // 1. conv
    std::string convWeightName = lname + ".0.weight";
    IConvolutionLayer* conv =
            network->addConvolutionNd(input, out_ch, DimsHW{ksize, ksize}, weightMap[convWeightName], Weights{});
    assert(conv);

    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{padding, padding});
    conv->setNbGroups(groups);

    // 2. BatchNorm
    std::string bnName = lname + ".1";
    IScaleLayer* bn = addBatchNorm2d(network, weightMap, bnName, *conv->getOutput(0), 1e-5f);
    assert(bn);

    // 3. relu
    IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
}

IActivationLayer* addResBottleneckBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap,
                                        const std::string& lname, ITensor& input, int in_ch, int out_ch, int stride,
                                        int groups) {
    ITensor* in_tensor = &input;

    // 1. calculate bottleneck channel.
    // f.a conv weight shape = [bottleneck, in_ch, 1, 1]
    float* wa = (float*)weightMap[lname + ".f.a.0.weight"].values;
    int wa_count = weightMap[lname + ".f.a.0.weight"].count;

    int bottleneck = wa_count / in_ch;  // 1x1 conv
    std::cout << "[Block] " << lname << " bottleneck = " << bottleneck << std::endl;

    // f.c out_ch
    float* wc = (float*)weightMap[lname + ".f.c.0.weight"].values;
    int wc_count = weightMap[lname + ".f.c.0.weight"].count;

    out_ch = wc_count / bottleneck;
    std::cout << "[Block] " << lname << " out_ch = " << out_ch << std::endl;

    // 2. projection branch
    ITensor* proj_out = nullptr;

    if (stride != 1 || in_ch != out_ch) {
        // 1×1 conv
        IConvolutionLayer* proj_conv = network->addConvolutionNd(*in_tensor, out_ch, DimsHW{1, 1},
                                                                 weightMap[lname + ".proj.0.weight"], Weights{});
        assert(proj_conv);
        proj_conv->setStrideNd(DimsHW{stride, stride});

        // BN
        IScaleLayer* proj_bn = addBatchNorm2d(network, weightMap, lname + ".proj.1", *proj_conv->getOutput(0));
        proj_out = proj_bn->getOutput(0);
    } else {
        // identity
        proj_out = in_tensor;
    }

    // 3. f.a 1×1 conv + BN + ReLU
    IActivationLayer* a = convBnRelu(network, weightMap, lname + ".f.a", *in_tensor, bottleneck,
                                     1,  // ksize
                                     1,  // stride
                                     0   // padding
    );

    // 4. f.b 3×3 conv (groups) + BN + ReLU
    IActivationLayer* b = convBnRelu(network, weightMap, lname + ".f.b", *a->getOutput(0), bottleneck,
                                     3,       // ksize
                                     stride,  // stride (block stride)
                                     1,       // padding
                                     groups);

    // 5. f.c 1×1 conv + BN
    IConvolutionLayer* c_conv = network->addConvolutionNd(*b->getOutput(0), out_ch, DimsHW{1, 1},
                                                          weightMap[lname + ".f.c.0.weight"], Weights{});
    assert(c_conv);

    IScaleLayer* c_bn = addBatchNorm2d(network, weightMap, lname + ".f.c.1", *c_conv->getOutput(0));

    // 6. elementwise sum
    IElementWiseLayer* ew = network->addElementWise(*c_bn->getOutput(0), *proj_out, ElementWiseOperation::kSUM);
    assert(ew);

    // 7. ReLU
    IActivationLayer* out = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
    assert(out);

    return out;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    // 1. Create Network
    INetworkDefinition* network =
            builder->createNetworkV2(1U << (unsigned int)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    assert(network);

    // 2. Input
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{1, 3, INPUT_H, INPUT_W});
    assert(data);
    std::cout << "Input shape = (1,3," << INPUT_H << "," << INPUT_W << ")" << std::endl;

    // 3. Load weights
    std::map<std::string, Weights> weightMap = loadWeights("../models/regnet_x_400mf.wts");

    std::cout << "Loaded " << weightMap.size() << " weights." << std::endl;

    // 4. Stem  (Conv-BN-ReLU)
    // stem: 3×3, stride=2, out=32
    IActivationLayer* stem = convBnRelu(network, weightMap, "stem", *data,
                                        32,  // out_ch
                                        3,   // ksize
                                        2,   // stride
                                        1    // padding
    );

    // 5. Block1 (1 block)
    // channels: 32 → 32, stride=2, groups=2
    std::cout << "----- Block1 -----" << std::endl;
    IActivationLayer* x = addResBottleneckBlock(network, weightMap, "trunk_output.block1.block1-0", *stem->getOutput(0),
                                                32,  // in_ch
                                                32,  // out_ch
                                                2,   // stride
                                                2    // groups
    );

    // 6. Block2 (2 blocks)
    // channels: 32 → 64, stride=2, groups=4
    std::cout << "----- Block2 -----" << std::endl;
    for (int i = 0; i < 2; i++) {
        int stride = (i == 0 ? 2 : 1);
        int in_ch = (i == 0 ? 32 : 64);

        x = addResBottleneckBlock(network, weightMap, "trunk_output.block2.block2-" + std::to_string(i),
                                  *x->getOutput(0), in_ch, 64, stride,
                                  4  // groups
        );
    }

    // 7. Block3 (7 blocks)
    // channels: 64 → 160, stride=2, groups=10
    std::cout << "----- Block3 -----" << std::endl;
    for (int i = 0; i < 7; i++) {
        int stride = (i == 0 ? 2 : 1);
        int in_ch = (i == 0 ? 64 : 160);

        x = addResBottleneckBlock(network, weightMap, "trunk_output.block3.block3-" + std::to_string(i),
                                  *x->getOutput(0), in_ch, 160, stride,
                                  10  // groups
        );
    }

    // 8. Block4 (12 blocks)
    // channels: 160 → 400, stride=2, groups=25
    std::cout << "----- Block4 -----" << std::endl;
    for (int i = 0; i < 12; i++) {
        int stride = (i == 0 ? 2 : 1);
        int in_ch = (i == 0 ? 160 : 400);

        x = addResBottleneckBlock(network, weightMap, "trunk_output.block4.block4-" + std::to_string(i),
                                  *x->getOutput(0), in_ch, 400, stride,
                                  25  // groups
        );
    }

    // 9. AvgPool 7×7 (stride=7)
    IPoolingLayer* pool = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool);
    pool->setStrideNd(DimsHW{7, 7});

    std::cout << "After avg pool: " << pool->getOutput(0)->getDimensions().d[0] << ","
              << pool->getOutput(0)->getDimensions().d[1] << "," << pool->getOutput(0)->getDimensions().d[2]
              << std::endl;

    // 10. Flatten
    IShuffleLayer* flatten = network->addShuffle(*pool->getOutput(0));
    flatten->setReshapeDimensions(Dims2{1, 400});  // shape -> (1, 400)
    assert(flatten);

    // 11. FC Layer (MatrixMultiply + Bias)
    // FC weight: [1000, 400]
    // reshape fc weight
    Weights& fcw = weightMap["fc.weight"];
    std::cout << "fc weight count = " << fcw.count << std::endl;

    // constant for FC weight
    IConstantLayer* fc_weight_tensor = network->addConstant(Dims2{1000, 400}, fcw);
    assert(fc_weight_tensor);

    // MatMul: (1×400) × (400×1000) → (1×1000)
    IMatrixMultiplyLayer* fc_mm =
            network->addMatrixMultiply(*flatten->getOutput(0), MatrixOperation::kNONE, *fc_weight_tensor->getOutput(0),
                                       MatrixOperation::kTRANSPOSE);
    assert(fc_mm);

    // Bias (1×1000)
    IConstantLayer* fc_bias_tensor = network->addConstant(Dims2{1, 1000}, weightMap["fc.bias"]);
    assert(fc_bias_tensor);

    IElementWiseLayer* fc =
            network->addElementWise(*fc_mm->getOutput(0), *fc_bias_tensor->getOutput(0), ElementWiseOperation::kSUM);
    assert(fc);

    // 12. Output
    fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*fc->getOutput(0));

    // 13. Build engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    std::cout << "Engine build completed!" << std::endl;

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();
}

void doInference(IExecutionContext& context, float* input, float* output, float* pool3FlattenLayerOutput,
                 int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    const char* inputName = INPUT_BLOB_NAME;
    const char* outputName = OUTPUT_BLOB_NAME;

    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};

    // Create GPU buffers on device
    CHECK(cudaMalloc(&deviceInput, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(deviceInput, input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice,
                          stream));

    context.setTensorAddress(inputName, deviceInput);
    context.setTensorAddress(outputName, deviceOutput);

    context.enqueueV3(stream);

    CHECK(cudaMemcpyAsync(output, deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./regnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./regnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("../models/regnet_x_400mf.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("../models/regnet_x_400mf.engine", std::ios::binary);
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

    // Subtract mean from image
    float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    float prob[OUTPUT_SIZE];
    float pool3FlattenLayerOutput[9216];
    doInference(*context, data, prob, pool3FlattenLayerOutput, 1);

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0)
            std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
