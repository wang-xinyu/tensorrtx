#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 400;
static const int NUM_SEGMENTS = 8;
static const int SHIFT_DIV = 8;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const char* WEIGHTS_PATH = "../tsm_r50_kinetics400_mmaction2.wts";
const char* ENGINE_PATH = "./tsm_r50_kinetics400_mmaction2_cpp.trt";
const char* RESULT_PATH = "./result.txt";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

void print(char* name, ITensor* tensor) {
    Dims dim = tensor->getDimensions();
    std::cout << name << " " << dim.d[0] << " " << dim.d[1] << " " << dim.d[2] << " " << dim.d[3] <<std::endl;
}

IConcatenationLayer* addShift(INetworkDefinition *network, ITensor& input, Dims4 inputShape, int numSegments, int shiftDiv) {
    int fold = int(inputShape.d[1] / shiftDiv);
    float* zeros = reinterpret_cast<float*>(malloc(sizeof(zeros) * fold*inputShape.d[2]*inputShape.d[3]));
    memset(zeros, 0, sizeof(zeros) * fold*inputShape.d[2]*inputShape.d[3]);
    Weights zeros_weights{DataType::kFLOAT, zeros, fold*inputShape.d[2]*inputShape.d[3]};

    // left
    ISliceLayer* left1 = network->addSlice(input, Dims4{1, 0, 0, 0}, Dims4{numSegments - 1, fold, inputShape.d[2], inputShape.d[3]}, Dims4{1, 1, 1, 1});
    IConstantLayer* left2 = network->addConstant(Dims4{1, fold, inputShape.d[2], inputShape.d[3]}, zeros_weights);
    ITensor* tensorsLeft[] = {left1->getOutput(0), left2->getOutput(0)};
    IConcatenationLayer* left = network->addConcatenation(tensorsLeft, 2);
    left->setAxis(0);

    // mid
    IConstantLayer* mid1 = network->addConstant(Dims4{1, fold, inputShape.d[2], inputShape.d[3]}, zeros_weights);
    ISliceLayer* mid2 = network->addSlice(input, Dims4{0, fold, 0, 0}, Dims4{numSegments - 1, fold, inputShape.d[2], inputShape.d[3]}, Dims4{1, 1, 1, 1});
    ITensor* tensorsMid[] = {mid1->getOutput(0), mid2->getOutput(0)};
    IConcatenationLayer* mid = network->addConcatenation(tensorsMid, 2);
    mid->setAxis(0);

    // right
    ISliceLayer* right = network->addSlice(input, Dims4{0, 2 * fold, 0, 0}, Dims4{numSegments, inputShape.d[1] - 2 * fold, inputShape.d[2], inputShape.d[3]}, Dims4{1, 1, 1, 1});

    // concatenate left/mid/right
    ITensor* tensors[] = {left->getOutput(0), mid->getOutput(0), right->getOutput(0)};
    IConcatenationLayer* concat = network->addConcatenation(tensors, 3);
    concat->setAxis(1);
    return concat;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
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

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname, Dims4 inputShape) {
    IConcatenationLayer* shift = addShift(network, input, inputShape, NUM_SEGMENTS, SHIFT_DIV);
    assert(shift);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(*shift->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt)
{
    INetworkDefinition* network = builder->createNetwork();

    // Create input tensor of shape {NUM_SEGMENTS, 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{NUM_SEGMENTS, 3, INPUT_H, INPUT_W});
    assert(data);
    print("input", data);

    std::map<std::string, Weights> weightMap = loadWeights(WEIGHTS_PATH);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});
    pool1->setPadding(DimsHW{1, 1});
    
    int curHeight = int(INPUT_H / 4);
    int curWidth = int(INPUT_W / 4);
    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.", Dims4{NUM_SEGMENTS, 64, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.", Dims4{NUM_SEGMENTS, 256, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.", Dims4{NUM_SEGMENTS, 256, curHeight, curWidth});
    
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0.", Dims4{NUM_SEGMENTS, 256, curHeight, curWidth});
    curHeight = int(INPUT_H / 8);
    curWidth = int(INPUT_W / 8);
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1.", Dims4{NUM_SEGMENTS, 512, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2.", Dims4{NUM_SEGMENTS, 512, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3.", Dims4{NUM_SEGMENTS, 512, curHeight, curWidth});
    
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0.", Dims4{NUM_SEGMENTS, 512, curHeight, curWidth});
    curHeight = int(INPUT_H / 16);
    curWidth = int(INPUT_W / 16);
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1.", Dims4{NUM_SEGMENTS, 1024, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2.", Dims4{NUM_SEGMENTS, 1024, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3.", Dims4{NUM_SEGMENTS, 1024, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4.", Dims4{NUM_SEGMENTS, 1024, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5.", Dims4{NUM_SEGMENTS, 1024, curHeight, curWidth});

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.", Dims4{NUM_SEGMENTS, 1024, curHeight, curWidth});
    curHeight = int(INPUT_H / 32);
    curWidth = int(INPUT_W / 32);
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.", Dims4{NUM_SEGMENTS, 2048, curHeight, curWidth});
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2.", Dims4{NUM_SEGMENTS, 2048, curHeight, curWidth});

    IPoolingLayer* pool2 = network->addPooling(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{curHeight, curWidth});
    assert(pool2);
    pool2->setStride(DimsHW{1, 1});
    
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), OUTPUT_SIZE, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    IReduceLayer* reduce = network->addReduce(*fc1->getOutput(0), ReduceOperation::kAVG, 1, false);
    assert(reduce);

    ISoftMaxLayer* softmax = network->addSoftMax(*reduce->getOutput(0));
    assert(softmax);
    softmax->setAxes(1);

    softmax->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*softmax->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    ICudaEngine* engine = builder->buildCudaEngine(*network);

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./tsm_r50 -s   // serialize model to plan file" << std::endl;
        std::cerr << "./tsm_r50 -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(ENGINE_PATH, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
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


    // Subtract mean from image
    static float data[NUM_SEGMENTS * 3 * INPUT_H * INPUT_W];
    for (int i = 0; i < NUM_SEGMENTS * 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
    }
    std::cout << std::endl;
    std::fstream writer(RESULT_PATH, std::ios::out);

    writer << prob[0];
    for(int i = 1; i < OUTPUT_SIZE ; i++) {
        writer << " " << prob[i];
    }
    writer.close();

    return 0;
}
